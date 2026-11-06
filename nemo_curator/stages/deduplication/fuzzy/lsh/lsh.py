# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal

import cudf
import pyarrow.parquet as pq
from loguru import logger

from nemo_curator.stages.deduplication.fuzzy.banding import (
    CURATOR_MINHASH_BAND_FIELD_PREFIX,
    get_band_columns,
    melt_band_columns,
    minhash_to_band_columns,
)
from nemo_curator.stages.deduplication.fuzzy.utils import CURATOR_DEFAULT_MINHASH_FIELD, CURATOR_LSH_BUCKET_FIELD
from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from nemo_curator.stages.deduplication.shuffle_utils.rapidsmpf_shuffler import (
    BulkRapidsMPFShuffler,
    pylibcudf_to_cudf_dataframe,
)
from nemo_curator.utils.file_utils import get_fs

if TYPE_CHECKING:
    from collections.abc import Iterator


class LSHActor(BulkRapidsMPFShuffler):
    """
    Actor that performs LSH operations and shuffling using Ray.

    Parameters
    ----------
    nranks
        Number of ranks in the communication group.
    total_nparts
        Total number of output partitions.
    num_bands
        Number of LSH bands.
    minhashes_per_band
        Number of minhashes per band.
    id_field
        Name of the ID field in input data.
    minhash_field
        Name of the minhash field in raw input data.
    input_format
        Whether input contains raw signatures, precomputed band columns, or should be
        detected automatically from the Parquet schema.
    band_field_prefix
        Prefix used for precomputed band columns.
    output_path
        Path to write output files.
    rmm_pool_size
        Size of the RMM GPU memory pool in bytes.
        If "auto", the memory pool is set to 90% of the free GPU memory.
        If None, the memory pool is set to 50% of the free GPU memory that can expand if needed.
    rmm_async
        Whether to use a CUDA asynchronous memory resource for the shuffle.
    spill_memory_limit
        Device memory limit in bytes for spilling to host.
        If "auto", the limit is set to 80% of the RMM pool size.
        If None spilling is disabled.
    enable_statistics
        Whether to collect statistics.
    read_kwargs
        Keyword arguments for the read method.
    write_kwargs
        Keyword arguments for the write method.

    Notes
    -----
    Architecture and Processing Flow:

    This implementation follows a clean separation of responsibilities with distinct methods
    for each part of the pipeline:

    Input Phase:
    - `read_minhash`: Reads minhash files and returns a DataFrame

    Processing Phase:
    - `minhash_to_bands`: Transforms a single minhash DataFrame into LSH bands
    - `read_and_insert`: Orchestrates reading, band creation, and insertion

    Output Phase:
    - `extract_and_group`: Extracts and groups shuffled data, yielding results as a generator
    - `extract_and_write`: Processes each yielded result and writes to output files immediately

    1. Files are read using `read_minhash`
    2. Data is processed with `minhash_to_bands` to extract LSH bucket IDs
    3. Processed data is immediately inserted into the shuffler
    4. Results are extracted and processed one partition at a time using generators
    5. Each partition is written to disk as soon as it's processed, without accumulating in memory
    """

    def __init__(  # noqa: PLR0913
        self,
        nranks: int,
        total_nparts: int,
        num_bands: int,
        minhashes_per_band: int,
        id_field: str = CURATOR_DEDUP_ID_STR,
        minhash_field: str = CURATOR_DEFAULT_MINHASH_FIELD,
        output_path: str = "./",
        rmm_pool_size: int | Literal["auto"] | None = "auto",
        spill_memory_limit: int | Literal["auto"] | None = "auto",
        rmm_async: bool = True,
        *,
        input_format: Literal["auto", "raw", "banded"] = "auto",
        band_field_prefix: str = CURATOR_MINHASH_BAND_FIELD_PREFIX,
        enable_statistics: bool = False,
        read_kwargs: dict[str, Any] | None = None,
        write_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(
            nranks=nranks,
            total_nparts=total_nparts,
            shuffle_on=[CURATOR_LSH_BUCKET_FIELD],
            output_path=output_path,
            rmm_pool_size=rmm_pool_size,
            rmm_async=rmm_async,
            spill_memory_limit=spill_memory_limit,
            enable_statistics=enable_statistics,
        )
        self.num_bands = num_bands
        self.minhashes_per_band = minhashes_per_band
        self.id_field = id_field
        self.minhash_field = minhash_field
        self.read_kwargs = read_kwargs if read_kwargs is not None else {}
        self.write_kwargs = write_kwargs if write_kwargs is not None else {}
        if input_format not in ("auto", "raw", "banded"):
            msg = f"input_format must be 'auto', 'raw', or 'banded', got {input_format!r}"
            raise ValueError(msg)
        self.input_format = input_format
        self.band_field_prefix = band_field_prefix

    def _resolve_input_format(self, filepaths: list[str]) -> Literal["raw", "banded"]:
        if self.input_format != "auto":
            return self.input_format

        input_fs = get_fs(filepaths[0], self.read_kwargs.get("storage_options"))
        schema = pq.read_schema(filepaths[0], filesystem=input_fs)
        schema_columns = set(schema.names)
        band_columns = get_band_columns(
            (0, self.num_bands),
            band_field_prefix=self.band_field_prefix,
        )
        has_raw_minhashes = self.minhash_field in schema_columns
        has_all_bands = set(band_columns).issubset(schema_columns)

        if has_raw_minhashes and has_all_bands:
            msg = (
                "Could not infer the MinHash input format because both the raw "
                f"{self.minhash_field!r} column and all band columns are present. "
                "Set input_format explicitly."
            )
            raise ValueError(msg)
        if has_raw_minhashes:
            return "raw"
        if has_all_bands:
            return "banded"

        msg = (
            "Could not infer the MinHash input format. Expected either the raw "
            f"{self.minhash_field!r} column or band columns {band_columns}."
        )
        raise ValueError(msg)

    def read_minhash(
        self,
        filepaths: list[str],
        band_range: tuple[int, int] | None = None,
    ) -> cudf.DataFrame:
        """Read only the raw signature or band columns needed by this iteration."""
        if self.read_kwargs.get("columns", None) is not None:
            err_msg = "Columns cannot be set in read_kwargs for LSHActor"
            raise ValueError(err_msg)

        band_range = band_range or (0, self.num_bands)
        input_format = self._resolve_input_format(filepaths)
        if input_format == "raw":
            columns = [self.id_field, self.minhash_field]
        else:
            columns = [
                self.id_field,
                *get_band_columns(
                    band_range,
                    band_field_prefix=self.band_field_prefix,
                ),
            ]
        return cudf.read_parquet(filepaths, columns=columns, **self.read_kwargs)

    def minhash_to_bands(
        self,
        minhash_df: cudf.DataFrame,
        band_range: tuple[int, int],
    ) -> cudf.DataFrame | None:
        """Convert raw or pre-banded MinHash input into rows for the LSH shuffle."""
        if minhash_df is None or len(minhash_df) == 0:
            return None

        band_columns = get_band_columns(
            band_range,
            band_field_prefix=self.band_field_prefix,
        )
        if self.minhash_field in minhash_df.columns:
            computed_bands = minhash_to_band_columns(
                minhash_df[self.minhash_field],
                num_bands=self.num_bands,
                minhashes_per_band=self.minhashes_per_band,
                band_range=band_range,
                band_field_prefix=self.band_field_prefix,
            )
            band_df = minhash_df[[self.id_field]].copy()
            for column in band_columns:
                band_df[column] = computed_bands[column]
        else:
            band_df = minhash_df

        return melt_band_columns(
            band_df,
            id_field=self.id_field,
            band_columns=band_columns,
        )

    def read_and_insert(self, filepaths: list[str], band_range: tuple[int, int]) -> None:
        """
        Read minhashes from files, create LSH bands, and insert into the shuffler.

        This method orchestrates the full processing pipeline:
        1. Reads minhash data from parquet files in batches
        2. Processes each batch to extract LSH bands
        3. Inserts the bands into the shuffler for distribution

        Parameters
        ----------
        filepaths
            List of paths to minhash files.
        band_range
            Tuple of (start_band, end_band) to process.

        Returns
        -------
        None
        """
        if not filepaths:
            return

        if band_range[0] < 0 or band_range[1] > self.num_bands or band_range[0] >= band_range[1]:
            msg = f"Invalid band range: {band_range}, must be in range [0, {self.num_bands}]"
            raise ValueError(msg)

        # Process files in batches
        minhash_df = self.read_minhash(filepaths, band_range)
        # Skip processing if the batch is empty
        if minhash_df is None or len(minhash_df) == 0:
            logger.info("Skipping empty batch")
            return

        # Process this batch of minhashes to get band data
        band_df = self.minhash_to_bands(minhash_df, band_range)

        # Call parent's insert_chunk method
        self.insert_chunk(band_df, list(band_df.columns))
        # Clear memory after processing a batch
        del minhash_df, band_df

    def group_by_bucket(self, df: cudf.DataFrame, include_singles: bool = False) -> cudf.DataFrame:
        """
        Group items by bucket ID and aggregate IDs into lists.

        Parameters
        ----------
        df
            DataFrame containing bucket IDs and document IDs.
        include_singles
            If True, include buckets with only one document. Default is False, which
            excludes single-document buckets as they cannot form duplicates. Set to True
            when building an LSH index that needs to maintain all documents.

        Returns
        -------
            DataFrame with bucket IDs and lists of document IDs.
        """
        if len(df) == 0:
            return df
        if not include_singles:
            # TODO: Add support for generating LSH index with single-document buckets that can be reused in incremental runs
            # Find bucket_ids that appear more than once (have multiple documents)
            # Keep only rows with buckets that are duplicated
            df = df[df[CURATOR_LSH_BUCKET_FIELD].duplicated(keep=False)]
        # Group by bucket_id and aggregate document IDs
        return df.groupby(CURATOR_LSH_BUCKET_FIELD)[self.id_field].agg(list).list.sort_values().reset_index()

    def extract_and_group(self) -> Iterator[tuple[int, cudf.DataFrame]]:
        """
        Extract shuffled partitions and group by bucket ID, yielding results one by one.

        This generator approach allows processing each partition immediately after it's ready,
        which is more memory-efficient than collecting all partitions first.

        Yields
        ------
        tuple
            A tuple of (partition_id, grouped_df) where grouped_df contains bucket IDs
            and their corresponding document ID lists.
        """
        # Fixed column names for pylibcudf conversion
        column_names = [self.id_field, CURATOR_LSH_BUCKET_FIELD]
        for partition_id, partition in self.extract():
            # Convert to cuDF DataFrame
            df = pylibcudf_to_cudf_dataframe(partition, column_names=column_names)
            # Group by bucket ID
            grouped_df = self.group_by_bucket(df)

            # Yield the result immediately instead of collecting in a list
            yield partition_id, grouped_df
            # Clean up memory
            del df, grouped_df

    def extract_and_write(self) -> list[dict[str, Any]]:
        """
        Extract shuffled partitions, group by bucket ID, and write results to files.

        This method orchestrates the post-processing pipeline:
        1. Extracts partitioned data from the shuffler using extract_and_group
        2. Writes each grouped partition to a parquet file as soon as it's available

        This generator-based approach is more memory-efficient since it processes
        one partition at a time rather than collecting all partitions in memory.

        Returns
        -------
        list[dict[str, Any]]
            A list of dictionaries containing partition information.
            Each dictionary contains:
            - partition_id: The ID of the partition
            - path: The path to the partition file
            - num_docs: The number of documents in the partition
        """
        partition_paths = []
        write_kwargs = self.write_kwargs.copy()
        write_kwargs["index"] = write_kwargs.get("index", False)
        # Process each partition as it becomes available
        for partition_id, grouped_df in self.extract_and_group():
            path = f"{self.output_path}/part.{partition_id}.parquet"

            # Write to file immediately
            grouped_df.to_parquet(path, **write_kwargs)
            partition_paths.append({"partition_id": partition_id, "path": path, "num_docs": len(grouped_df)})
            # Clean up to release memory
            del grouped_df

        return partition_paths
