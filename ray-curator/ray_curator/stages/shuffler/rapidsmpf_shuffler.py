# Copyright (c) 2025, NVIDIA CORPORATION.
"""
This module implements a bulk-synchronous shuffle class using UCXX communication compatible with Ray Actors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cudf
import pylibcudf as plc
import rmm.mr
from rapidsmpf.buffer.buffer import MemoryType
from rapidsmpf.buffer.resource import BufferResource, LimitAvailableMemory
from rapidsmpf.shuffler import partition_and_pack, unpack_and_concat
from rapidsmpf.statistics import Statistics
from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table, pylibcudf_to_cudf_dataframe
from rapidsmpf.utils.ray_utils import BaseShufflingActor

from ray_curator.stages.deduplication.gpu_utils import align_down_to_256

if TYPE_CHECKING:
    from collections.abc import Iterator

    from rapidsmpf.shuffler import Shuffler


class BulkRapidsMPFShuffler(BaseShufflingActor):
    """
    Class that performs a bulk shuffle operation.
    This class is compatible with Ray Actors communicating with each other using UCXX communication.
    Parameters
    ----------
    nranks
        Number of ranks in the communication group.
    total_nparts
        Total number of output partitions.
    shuffle_on
        List of column names to shuffle on.
    output_path
        Path to write output files.
    rmm_pool_size
        Size of the RMM memory pool in bytes.
    spill_memory_limit
        Device memory limit in bytes for spilling to host.
        If "auto", the limit is set to 80% of the RMM pool size.
        If None spilling is disabled.
    enable_statistics
        Whether to collect shuffle statistics.
    """

    def __init__(  # noqa: PLR0913
        self,
        nranks: int,
        total_nparts: int,
        shuffle_on: list[str],
        output_path: str = "./",
        rmm_pool_size: int = 1024 * 1024 * 1024,
        spill_memory_limit: int | Literal["auto"] | None = "auto",
        *,
        enable_statistics: bool = False,
    ):
        super().__init__(nranks)
        self.shuffle_on = shuffle_on
        self.output_path = output_path
        self.total_nparts = total_nparts
        self.rmm_pool_size = align_down_to_256(rmm_pool_size)

        if isinstance(spill_memory_limit, int):
            self.spill_memory_limit = align_down_to_256(spill_memory_limit)
        elif spill_memory_limit == "auto":
            self.spill_memory_limit = align_down_to_256(0.8 * self.rmm_pool_size)
        elif spill_memory_limit is None:
            self.spill_memory_limit = None
        else:
            err_msg = f"Invalid spill_memory_limit: {spill_memory_limit}"
            raise ValueError(err_msg)

        self.enable_statistics = enable_statistics

    def setup_worker(self, root_address_bytes: bytes) -> None:
        """
        Setup the UCXX communication and a shuffle operation.

        Parameters
        ----------
        root_address_bytes
            Address of the root worker for UCXX initialization.
        """
        super().setup_worker(root_address_bytes)

        # Initialize the RMM memory resource
        mr = rmm.mr.StatisticsResourceAdaptor(
            rmm.mr.PoolMemoryResource(
                rmm.mr.CudaMemoryResource(),
                initial_pool_size=self.rmm_pool_size,
                maximum_pool_size=None,
            )
        )
        rmm.mr.set_current_device_resource(mr)
        # Create a buffer resource that limits device memory if spill_memory_limit is set
        memory_available = (
            None
            if self.spill_memory_limit is None
            else {MemoryType.DEVICE: LimitAvailableMemory(mr, limit=self.spill_memory_limit)}
        )
        br = BufferResource(mr, memory_available)
        # Create a statistics object
        self.stats = Statistics(self.enable_statistics)
        # Create a shuffler
        self.shuffler: Shuffler = self.create_shuffler(
            0,
            total_num_partitions=self.total_nparts,
            buffer_resource=br,
            statistics=self.stats,
        )

    def cleanup(self) -> None:
        """Cleanup the UCXX communication and the shuffle operation."""
        if self.enable_statistics and self.stats is not None:
            self.comm.logger.info(self.stats.report())
        if self.shuffler is not None:
            self.shuffler.shutdown()

    def read_batch(self, paths: list[str]) -> tuple[plc.Table, list[str]]:
        """
        Read a single batch of Parquet files.

        Parameters
        ----------
        paths
            List of file paths to the Parquet files.

        Returns
        -------
            A tuple containing the read in table and the column names.
        """
        options = plc.io.parquet.ParquetReaderOptions.builder(plc.io.SourceInfo(paths)).build()
        tbl_w_meta = plc.io.parquet.read_parquet(options)
        return (tbl_w_meta.tbl, tbl_w_meta.column_names(include_children=False))

    def write_table(
        self,
        table: plc.Table,
        output_path: str,
        partition_id: int | str,
        column_names: list[str],
    ) -> None:
        """
        Write a pylibcudf Table to a Parquet file.

        Parameters
        ----------
        table
            The table to write.
        output_path
            The path to write the table to.
        id
            Partition id used for naming the output file.
        column_names
            The column names of the table.
        """
        path = f"{output_path}/part.{partition_id}.parquet"
        pylibcudf_to_cudf_dataframe(
            table,
            column_names=column_names,
        ).to_parquet(path)

    def insert_chunk(self, table: plc.Table | cudf.DataFrame, column_names: list[str]) -> None:
        """
        Insert a pylibcudf Table into the shuffler.

        Parameters
        ----------
        table
            The table to insert.
        column_names
            The column names of the table.
        """
        from rmm.pylibrmm.stream import DEFAULT_STREAM

        if isinstance(table, cudf.DataFrame):
            table = cudf_to_pylibcudf_table(table)
        columns_to_hash = tuple(column_names.index(val) for val in self.shuffle_on)
        packed_inputs = partition_and_pack(
            table,
            columns_to_hash=columns_to_hash,
            num_partitions=self.total_nparts,
            stream=DEFAULT_STREAM,
            device_mr=rmm.mr.get_current_device_resource(),
        )
        self.shuffler.insert_chunks(packed_inputs)

    def read_and_insert(self, paths: list[str], batchsize: int = 1) -> list[str]:
        """
        Read the list of parquet files every batchsize and insert the partitions into the shuffler.

        Parameters
        ----------
        paths
            List of file paths to the Parquet files.

        Returns
        -------
            The column names of the table.
        """
        for i in range(0, len(paths), batchsize):
            tbl, column_names = self.read_batch(paths[i : i + batchsize])
            self.insert_chunk(tbl, column_names)
        self.insert_finished()
        return column_names

    def insert_finished(self) -> None:
        """Tell the shuffler that we are done inserting data."""
        for pid in range(self.total_nparts):
            self.shuffler.insert_finished(pid)
        self.comm.logger.info("Insert finished")

    def extract(self) -> Iterator[tuple[int, plc.Table]]:
        """
        Extract shuffled partitions as they become ready.

        Returns
        -------
            An iterator over the shuffled partitions.
        """
        from rmm.pylibrmm.stream import DEFAULT_STREAM

        while not self.shuffler.finished():
            partition_id = self.shuffler.wait_any()
            packed_chunks = self.shuffler.extract(partition_id)
            partition = unpack_and_concat(
                packed_chunks,
                stream=DEFAULT_STREAM,
                device_mr=rmm.mr.get_current_device_resource(),
            )
            yield partition_id, partition

    def extract_and_write(self, column_names: list[str]) -> None:
        """
        Extract and write shuffled partitions.

        Parameters
        ----------
        column_names
            The column names of the table.
        """
        for partition_id, partition in self.extract():
            self.write_table(partition, self.output_path, partition_id, column_names)
