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

from itertools import pairwise
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask
from nemo_curator.utils.file_utils import create_or_overwrite_dir, get_fs


class BucketsToEdgesStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    """
    Stage that takes in a file consiting of LSH bucket ids and document ids belonging to the bucket
    and outputs a file consisting of edges between documents with the same bucket id.

    Args:
        doc_id_field: The field name containing the document ids for each bucket.
        output_path: The directory to write the output file to.
        read_kwargs: Keyword arguments to pass for reading the input files.
            Only the storage_options key is supported for now.
        write_kwargs: Keyword arguments to pass for writing the output files.
            Only the storage_options key is supported for now.
    """

    _name = "BucketsToEdgesStage"
    _resources = Resources(cpus=1.0)

    def __init__(
        self,
        output_path: str,
        doc_id_field: str = CURATOR_DEDUP_ID_STR,
        read_kwargs: dict[str, Any] | None = None,
        write_kwargs: dict[str, Any] | None = None,
    ):
        self.doc_id_field = doc_id_field
        self._check_io_kwargs(read_kwargs)
        self._check_io_kwargs(write_kwargs)
        self.read_storage_options = read_kwargs.get("storage_options") if read_kwargs is not None else None
        self.write_storage_options = write_kwargs.get("storage_options") if write_kwargs is not None else None

        self.output_fs = get_fs(output_path, self.write_storage_options)
        self.output_path = self.output_fs.sep.join([output_path, self.name])

        # Handle output directory cleanup logic
        create_or_overwrite_dir(self.output_path, fs=self.output_fs)

    def _check_io_kwargs(self, kwargs: dict[str, Any] | None) -> None:
        if kwargs is not None:
            unused_keys = set(kwargs.keys()) - {"storage_options"}
            if len(unused_keys) > 0:
                logger.warning(f"{unused_keys} will be ignored as this stage only supports 'storage_options'.")

    def process(self, task: FileGroupTask) -> FileGroupTask:
        input_fs = get_fs(task.data[0], self.read_storage_options)
        df = pq.read_table(task.data, filesystem=input_fs)
        edges = []
        for bucket_docs in df[self.doc_id_field]:
            edges.extend(pairwise(bucket_docs))
        edges = [list(edge) for edge in edges]
        edges = pa.Table.from_pandas(pd.DataFrame(edges, columns=[f"{self.doc_id_field}_x", f"{self.doc_id_field}_y"]))

        output_path = self.output_fs.sep.join([self.output_path, f"{task._uuid}.parquet"])
        pq.write_table(edges, output_path, filesystem=self.output_fs)
        return FileGroupTask(
            task_id=f"{task.task_id}",
            dataset_name=f"{task.dataset_name}_edges",
            data=[output_path],
            _metadata={**task._metadata, "storage_options": self.write_storage_options},
            _stage_perf=task._stage_perf,
        )
