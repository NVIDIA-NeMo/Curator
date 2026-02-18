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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nemo_curator.stages.multimodal.utils import materialize_task_binary_content

from .base import BaseMultimodalWriter

if TYPE_CHECKING:
    import pandas as pd

    from nemo_curator.tasks import MultiBatchTask


@dataclass
class BaseMultimodalTabularWriter(BaseMultimodalWriter, ABC):
    """Shared multimodal tabular writer with optional image materialization."""

    write_kwargs: dict[str, Any] = field(default_factory=dict)
    materialize_on_write: bool = True
    name: str = "base_multimodal_tabular_writer"

    def _materialize_dataframe(self, task: MultiBatchTask) -> pd.DataFrame:
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

    @abstractmethod
    def _write_dataframe(self, df: pd.DataFrame, file_path: str, write_kwargs: dict[str, Any]) -> None:
        """Format-specific writer implementation."""

    def write_data(self, task: MultiBatchTask, file_path: str) -> None:
        with self._time_metric("materialize_dataframe_total_s"):
            df = self._materialize_dataframe(task)
        write_kwargs = {"index": None}
        write_kwargs.update(self.write_kwargs)
        self._write_dataframe(df, file_path, write_kwargs)


@dataclass
class MultimodalParquetWriterStage(BaseMultimodalTabularWriter):
    """Thin parquet writer on top of the tabular multimodal base."""

    file_extension: str = "parquet"
    name: str = "multimodal_parquet_writer"

    def _write_dataframe(self, df: pd.DataFrame, file_path: str, write_kwargs: dict[str, Any]) -> None:
        # Empirically best default from current benchmark sweep.
        # Note: row_group_size is in rows; 128_000 rows is a typical Parquet best-practice default.
        # Callers can override this via write_kwargs["row_group_size"] if needed.
        write_kwargs.setdefault("compression", "snappy")
        write_kwargs.setdefault("row_group_size", 128_000)
        with self._time_metric("parquet_write_s"):
            df.to_parquet(file_path, **write_kwargs)
