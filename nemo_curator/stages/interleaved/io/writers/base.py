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

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from fsspec.core import url_to_fs
from loguru import logger

import nemo_curator.stages.text.io.writer.utils as writer_utils
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.utils import materialize_task_binary_content
from nemo_curator.tasks import FileGroupTask, InterleavedBatch
from nemo_curator.utils.client_utils import is_remote_url
from nemo_curator.utils.file_utils import check_output_mode

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class BaseInterleavedWriter(ProcessingStage[InterleavedBatch, FileGroupTask], ABC):
    """Base class for interleaved writers.

    Handles filesystem setup, deterministic file naming, optional binary
    materialization, and process() orchestration.  Subclasses implement
    ``_write_dataframe`` for format-specific output.
    """

    path: str
    file_extension: str
    write_kwargs: dict[str, Any] = field(default_factory=dict)
    materialize_on_write: bool = True
    name: str = "base_interleaved_writer"
    mode: Literal["ignore", "overwrite", "append", "error"] = "ignore"
    append_mode_implemented: bool = False

    def __post_init__(self) -> None:
        self.storage_options = (self.write_kwargs or {}).get("storage_options", {})
        self.fs, self._fs_path = url_to_fs(self.path, **self.storage_options)
        check_output_mode(self.mode, self.fs, self._fs_path, append_mode_implemented=self.append_mode_implemented)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    # -- materialization --

    def _materialize_dataframe(self, task: InterleavedBatch) -> pd.DataFrame:
        out = task.to_pandas()
        image_mask = (out["modality"] == "image") & (out["binary_content"].isna())
        self._log_metrics(
            {
                "rows_out": float(len(out)),
                "image_rows": float((out["modality"] == "image").sum()),
                "image_rows_missing_binary": float(image_mask.sum()),
            }
        )
        if not self.materialize_on_write or not image_mask.any():
            return out

        with self._time_metric("materialize_fetch_binary_s"):
            out = materialize_task_binary_content(task, io_kwargs=self.write_kwargs).to_pandas()
        if "materialize_error" in out.columns:
            self._log_metric("materialize_errors", float(out["materialize_error"].notna().sum()))
        return out

    # -- write pipeline --

    @abstractmethod
    def _write_dataframe(self, df: pd.DataFrame, file_path: str, write_kwargs: dict[str, Any]) -> None:
        """Format-specific DataFrame writer. Subclasses implement this."""

    def write_data(self, task: InterleavedBatch, file_path: str) -> None:
        with self._time_metric("materialize_dataframe_total_s"):
            df = self._materialize_dataframe(task)
        write_kwargs: dict[str, Any] = dict(self.write_kwargs)
        write_kwargs["index"] = False
        self._write_dataframe(df, file_path, write_kwargs)

    def process(self, task: InterleavedBatch) -> FileGroupTask:
        if source_files := task._metadata.get("source_files"):
            filename = writer_utils.get_deterministic_hash(source_files, task.task_id)
        else:
            logger.warning("The task does not have source_files in metadata, using UUID for base filename")
            filename = uuid.uuid4().hex

        file_path = self.fs.sep.join([self._fs_path, f"{filename}.{self.file_extension}"])
        file_path_with_protocol = self.fs.unstrip_protocol(file_path) if is_remote_url(self.path) else file_path

        self.write_data(task, file_path_with_protocol)
        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=[file_path_with_protocol],
            _metadata={**task._metadata, "format": self.file_extension},
            _stage_perf=task._stage_perf,
        )
