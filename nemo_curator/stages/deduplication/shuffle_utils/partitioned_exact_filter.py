# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import os
import posixpath
from typing import TYPE_CHECKING, Any, Literal

from fsspec.core import split_protocol

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask
from nemo_curator.utils.file_utils import check_disallowed_kwargs, get_fs

if TYPE_CHECKING:
    import cudf


ExactFilterMode = Literal["leftsemi", "leftanti"]

_ROW_SUBSETTING_READ_KWARGS = ["columns", "filters", "row_groups", "nrows", "skip_rows"]
_CONTROLLED_WRITE_KWARGS = ["index"]
_NON_SINGLE_FILE_WRITE_KWARGS = [
    "metadata_file_path",
    "partition_cols",
    "partition_file_name",
    "partition_offsets",
]


def _path_identity(path: str) -> tuple[str, str]:
    """Return a comparison identity without changing the path used for I/O."""
    protocol, stripped_path = split_protocol(path)
    normalized_protocol = (protocol or "file").casefold()
    if normalized_protocol == "file":
        normalized_path = os.path.realpath(stripped_path)
    else:
        normalized_path = posixpath.normpath(stripped_path)
    return normalized_protocol, normalized_path


class PartitionedExactFilterStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    """Filter complete shuffled partitions against matching reference partitions.

    Both inputs must have been partitioned with the same key columns, hash
    function, and partition count. Each input task is matched to
    ``part.<partition_index>.parquet`` below ``reference_path``. The stage
    performs a cuDF left-semi or left-anti join and therefore preserves the
    multiplicity of rows on the left; duplicate reference keys cannot multiply
    the output.

    This stage deliberately does not perform a shuffle. It is the bucket-local
    reduction step to compose after two compatible :class:`ShuffleStage`
    operations.

    Parameters
    ----------
    key_fields
        Exact key column or columns shared by both shuffled datasets.
    reference_path
        Directory containing reference files named
        ``part.<partition_index>.parquet``.
    output_path
        Directory for the filtered partition files.
    total_partitions
        Expected partition count for both shuffled datasets.
    mode
        ``"leftsemi"`` keeps matching left rows and ``"leftanti"`` keeps
        non-matching left rows.
    read_kwargs
        Keyword arguments for reading complete left-side Parquet files with
        cuDF. Row projection, filtering, and row-subsetting options are not
        accepted.
    reference_read_kwargs
        Keyword arguments for reading complete reference Parquet files with
        cuDF. Row projection, filtering, and row-subsetting options are not
        accepted.
    write_kwargs
        Keyword arguments for writing one result Parquet file with cuDF.
        The stage controls ``index``. Dataset-partitioning and
        auxiliary-metadata output options are not accepted.
    """

    name = "PartitionedExactFilter"
    resources = Resources(gpus=1.0)

    def __init__(  # noqa: PLR0913
        self,
        key_fields: str | list[str],
        reference_path: str,
        output_path: str,
        total_partitions: int,
        mode: ExactFilterMode = "leftanti",
        read_kwargs: dict[str, Any] | None = None,
        reference_read_kwargs: dict[str, Any] | None = None,
        write_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        fields = [key_fields] if isinstance(key_fields, str) else list(key_fields)
        if not fields or any(not isinstance(field, str) or not field for field in fields):
            msg = "key_fields must contain at least one non-empty column name"
            raise ValueError(msg)
        if len(set(fields)) != len(fields):
            msg = "key_fields must not contain duplicate column names"
            raise ValueError(msg)
        if mode not in ("leftsemi", "leftanti"):
            msg = "mode must be 'leftsemi' or 'leftanti'"
            raise ValueError(msg)
        if isinstance(total_partitions, bool) or not isinstance(total_partitions, int) or total_partitions <= 0:
            msg = "total_partitions must be a positive integer"
            raise ValueError(msg)

        reference_root = reference_path.rstrip("/")
        output_root = output_path.rstrip("/")
        if not reference_root or not output_root:
            msg = "reference_path and output_path must be non-root directory paths"
            raise ValueError(msg)
        if _path_identity(reference_root) == _path_identity(output_root):
            msg = "output_path must be distinct from reference_path"
            raise ValueError(msg)

        self.key_fields = fields
        self.reference_path = reference_root
        self.output_path = output_root
        self.total_partitions = total_partitions
        self.mode = mode
        self.read_kwargs = dict(read_kwargs or {})
        self.reference_read_kwargs = dict(reference_read_kwargs or {})
        self.write_kwargs = dict(write_kwargs or {})

        check_disallowed_kwargs(self.read_kwargs, _ROW_SUBSETTING_READ_KWARGS)
        check_disallowed_kwargs(self.reference_read_kwargs, _ROW_SUBSETTING_READ_KWARGS)
        check_disallowed_kwargs(self.write_kwargs, _CONTROLLED_WRITE_KWARGS)
        check_disallowed_kwargs(self.write_kwargs, _NON_SINGLE_FILE_WRITE_KWARGS)

        self.reference_fs = get_fs(
            self.reference_path,
            storage_options=self.reference_read_kwargs.get("storage_options", {}),
        )
        self.output_fs = get_fs(self.output_path, storage_options=self.write_kwargs.get("storage_options", {}))
        self.output_fs.mkdirs(self.output_path, exist_ok=True)

    def inputs(self) -> tuple[list[str], list[str]]:
        return (["data", "_metadata"], [])

    def outputs(self) -> tuple[list[str], list[str]]:
        return (["data", "_metadata"], [])

    def _partition_index(self, task: FileGroupTask) -> int:
        partition_index = task._metadata.get("partition_index")
        task_total = task._metadata.get("total_partitions")
        if isinstance(partition_index, bool) or not isinstance(partition_index, int):
            msg = "input task metadata must contain an integer partition_index"
            raise TypeError(msg)
        if partition_index < 0 or partition_index >= self.total_partitions:
            msg = f"partition_index {partition_index} is outside [0, {self.total_partitions})"
            raise ValueError(msg)
        if task_total != self.total_partitions:
            msg = (
                f"input task total_partitions {task_total!r} does not match the configured "
                f"value {self.total_partitions}"
            )
            raise ValueError(msg)
        return partition_index

    def _reference_partition_path(self, partition_index: int) -> str:
        return self.reference_fs.sep.join([self.reference_path, f"part.{partition_index}.parquet"])

    def _output_partition_path(self, partition_index: int) -> str:
        return self.output_fs.sep.join([self.output_path, f"part.{partition_index}.parquet"])

    def process(self, task: FileGroupTask) -> FileGroupTask:
        partition_index = self._partition_index(task)
        output_file = self._output_partition_path(partition_index)
        output_identity = _path_identity(output_file)
        if any(_path_identity(input_file) == output_identity for input_file in task.data):
            msg = "output partition would overwrite its left input partition"
            raise ValueError(msg)
        reference_file = self._reference_partition_path(partition_index)
        if not self.reference_fs.isfile(reference_file):
            msg = f"matching reference partition is missing: {reference_file}"
            raise FileNotFoundError(msg)

        import cudf
        import pyarrow.parquet as pq

        with self.reference_fs.open(reference_file, "rb") as reference_stream:
            reference_schema = pq.read_schema(reference_stream)
        self._require_column_names(reference_schema.names, "reference")
        left = cudf.read_parquet(task.data, **self.read_kwargs)
        reference = cudf.read_parquet(
            reference_file,
            columns=self.key_fields,
            **self.reference_read_kwargs,
        )
        self._require_key_fields(left, "left")

        reference_rows = len(reference)
        output = left.merge(reference, how=self.mode, on=self.key_fields)
        if len(output) > len(left):
            msg = "exact semi/anti join unexpectedly increased left-row multiplicity"
            raise RuntimeError(msg)

        output.to_parquet(output_file, index=False, **self.write_kwargs)

        self._log_metrics(
            {
                "input_rows": len(left),
                "reference_rows": reference_rows,
                "output_rows": len(output),
            }
        )
        return FileGroupTask(
            dataset_name=f"{task.dataset_name}_{self.name}",
            data=[output_file],
            _metadata={
                **task._metadata,
                "partition_index": partition_index,
                "total_partitions": self.total_partitions,
                "key_fields": self.key_fields,
                "filter_mode": self.mode,
                "input_rows": len(left),
                "reference_rows": reference_rows,
                "output_rows": len(output),
            },
            _stage_perf=task._stage_perf,
        )

    def _require_key_fields(self, frame: cudf.DataFrame, side: str) -> None:
        self._require_column_names(frame.columns, side)

    def _require_column_names(self, column_names: Any, side: str) -> None:
        missing = [field for field in self.key_fields if field not in column_names]
        if missing:
            msg = f"{side} partition is missing exact key columns: {missing}"
            raise ValueError(msg)
