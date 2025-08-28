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

import math
from typing import TYPE_CHECKING, Any

import cudf
from cugraph.dask.comms.comms_wrapper import init_subcomms as c_init_subcomms
from loguru import logger
from pylibcugraph import GraphProperties, MGGraph, ResourceHandle
from pylibcugraph import weakly_connected_components as pylibcugraph_wcc

from nemo_curator.backends.experimental.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.deduplication.fuzzy.utils import CURATOR_FUZZY_DUPLICATE_GROUP_FIELD
from nemo_curator.stages.deduplication.id_generator import (
    CURATOR_DEDUP_ID_STR,
)
from nemo_curator.stages.deduplication.io_utils import DeduplicationIO
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks.file_group import FileGroupTask
from nemo_curator.utils.file_utils import create_or_overwrite_dir, get_fs

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata


class ConnectedComponentsStage(ProcessingStage[FileGroupTask, FileGroupTask], DeduplicationIO):
    def __init__(
        self,
        output_path: str,
        source_field: str = f"{CURATOR_DEDUP_ID_STR}_x",
        destination_field: str = f"{CURATOR_DEDUP_ID_STR}_y",
        read_kwargs: dict | None = None,
        write_kwargs: dict | None = None,
    ):
        """
        Args:
            output_path: The path to write the resulting connected components to.
            source_field: The field name containing the document ids of the source of the edge.
            destination_field: The field name containing the document ids of the destination of the edge.
            read_kwargs: Keyword arguments to pass for reading the input files.
            write_kwargs: Keyword arguments to pass for writing the output files.
        """

        self.source_field = source_field
        self.destination_field = destination_field
        self.read_kwargs = read_kwargs if read_kwargs is not None else {}
        self.write_kwargs = write_kwargs if write_kwargs is not None else {}

        self._name = self.__class__.__name__
        self._resources = Resources(cpus=1.0, gpus=1.0)
        self._batch_size = None

        # Handle output directory cleanup logic
        self.output_fs = get_fs(output_path, self.write_kwargs.get("storage_options"))
        self.output_path = self.output_fs.sep.join([output_path, self.name])
        create_or_overwrite_dir(self.output_path, fs=self.output_fs)

    def setup(self, _worker_metadata: "WorkerMetadata | None" = None) -> None:
        if not hasattr(self, "_raft_handle"):
            msg = "RAFT handle not found. Make sure the stage is initialized with RAFT"
            raise ValueError(msg)

        self._setup_post()

    def ray_stage_spec(self) -> dict[str, Any]:
        return {
            RayStageSpecKeys.IS_RAFT_ACTOR: True,
        }

    def __get_2D_div(self, ngpus: int) -> tuple[int, int]:  # noqa: N802
        """
        Cugraph 2d partitioning number of rows and columns
        """
        # Taken from https://github.com/rapidsai/cugraph/blob/branch-25.06/python/cugraph/cugraph/dask/comms/comms.py#L41-L45
        prows = int(math.sqrt(ngpus))
        while ngpus % prows != 0:
            prows = prows - 1
        return prows, int(ngpus / prows)

    def _setup_post(self) -> None:
        """Setup the sub-communicator for cuGraph communications.

        This method is specific to cuGraph comms and is used to initialize the
        sub-communicator.
        """
        if not hasattr(self, "_raft_handle") or not hasattr(self, "_actor_pool_size"):
            msg = "RAFT handle or actor pool size not found. Make sure the stage is initialized with RAFT"
            raise ValueError(msg)

        logger.debug("     Setting up cuGraph-subcom...")
        row_comm_size, _ = self.__get_2D_div(self._actor_pool_size)
        c_init_subcomms(self._raft_handle, row_comm_size)

    def weakly_connected_components(self, df: cudf.DataFrame, src_col: str, dst_col: str) -> None:
        """Compute the weakly connected components of a graph.

        This method loads a chunk of the graph, creates a cuGraph object, and
        computes the weakly connected components using the MGGraph library.

        Parameters
        ----------
        start: int
            The start index of the chunk.
        stop: int
            The stop index of the chunk.
        """

        src_array = df[src_col]
        dst_array = df[dst_col]

        rhandle = ResourceHandle(self._raft_handle.getHandle())

        graph_props = GraphProperties(
            is_multigraph=False,
            is_symmetric=True,
        )
        logger.debug("Running graph creation")
        plc_graph = MGGraph(
            resource_handle=rhandle,
            graph_properties=graph_props,
            src_array=[src_array],
            dst_array=[dst_array],
            edge_id_array=None,
            edge_type_array=None,
            num_arrays=1,
            store_transposed=False,
            symmetrize=False,
            do_expensive_check=False,
            drop_multi_edges=True,
        )
        logger.debug("Running weakly connected components")
        res = pylibcugraph_wcc(
            resource_handle=rhandle,
            graph=plc_graph,
            offsets=None,
            indices=None,
            weights=None,
            labels=None,
            do_expensive_check=False,
        )
        logger.info("Computing weakly connected components completed successfully!")
        return res

    def process(self, task: FileGroupTask) -> FileGroupTask:
        err_msg = "ConnectedComponentsStage only support process batch"
        raise NotImplementedError(err_msg)

    def process_batch(self, tasks: list[FileGroupTask]) -> list[FileGroupTask]:
        """
        Process a batch of input files containing edges between documents.
        Compute the weakly connected components of the graph and write a mapping of document ids to their connected component id.

        Parameters
        ----------
        tasks: list[FileGroupTask]
            A list of FileGroupTasks containing the input files.
        Returns
        -------
        list[FileGroupTask]
            A list of FileGroupTasks containing the output doc_id to connected component id mapping.
        """
        input_files = []
        for task in tasks:
            input_files.extend(task.data)
        output_file = self.output_fs.sep.join([self.output_path, f"{tasks[0]._uuid}.parquet"])
        edgelist_columns = [self.source_field, self.destination_field]
        dfs = []
        for input_file in input_files:
            dfs.append(self.read_parquet(input_file, columns=edgelist_columns, **self.read_kwargs))
        df = cudf.concat(dfs)
        # remove duplicate edges
        df = df.drop_duplicates(subset=edgelist_columns, ignore_index=True)
        vertices, labels = self.weakly_connected_components(df, self.source_field, self.destination_field)
        df = cudf.DataFrame(
            {
                CURATOR_DEDUP_ID_STR: vertices,
                CURATOR_FUZZY_DUPLICATE_GROUP_FIELD: labels,
            }
        )
        self.write_parquet(df=df, filepath=output_file, index=False, **self.write_kwargs)
        return [
            FileGroupTask(
                dataset_name=tasks[0].dataset_name,
                task_id=tasks[0].task_id,
                data=[output_file],
                _metadata={
                    "storage_options": self.write_kwargs.get("storage_options"),
                    "num_vertices": len(vertices),
                },
            )
        ]
