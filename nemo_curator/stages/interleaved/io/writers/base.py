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
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import pyarrow as pa
import pyarrow.compute as pc
from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.utils import materialize_task_binary_content
from nemo_curator.stages.interleaved.utils.schema import align_interleaved_table, resolve_schema
from nemo_curator.tasks import FileGroupTask, InterleavedBatch
from nemo_curator.utils.client_utils import is_remote_url
from nemo_curator.utils.file_utils import check_output_mode
from nemo_curator.utils.hash_utils import get_deterministic_hash

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class BaseInterleavedWriter(ProcessingStage[InterleavedBatch, FileGroupTask], ABC):
    """Base class for interleaved writers.

    Handles filesystem setup, deterministic file naming, optional binary
    materialization, schema alignment, and process() orchestration.
    Subclasses implement ``_write_table`` for format-specific output.

    If *schema* is set, every output table is aligned to it (missing columns
    become typed nulls, extra columns are dropped, types are reconciled).
    By default (``schema=None``) extra user columns are preserved and only
    reserved-column types are reconciled via ``reconcile_schema``.

    Use *schema* or *schema_overrides* only when strict column control is needed
    (e.g. to prevent heterogeneous-schema crashes).
    """

    path: str
    file_extension: str
    write_kwargs: dict[str, Any] = field(default_factory=dict)
    materialize_on_write: bool = True
    name: str = "base_interleaved_writer"
    mode: Literal["ignore", "overwrite", "append", "error"] = "ignore"
    append_mode_implemented: bool = False
    on_materialize_error: Literal["error", "warn", "drop_row", "drop_sample"] = "error"
    schema: pa.Schema | None = None
    schema_overrides: dict[str, pa.DataType] | None = None

    def __post_init__(self) -> None:
        if self.schema is not None or self.schema_overrides is not None:
            self.schema = resolve_schema(self.schema, self.schema_overrides)
        self.storage_options = (self.write_kwargs or {}).get("storage_options", {})
        self.fs, self._fs_path = url_to_fs(self.path, **self.storage_options)
        check_output_mode(self.mode, self.fs, self._fs_path, append_mode_implemented=self.append_mode_implemented)
        self._effective_write_kwargs = {k: v for k, v in self.write_kwargs.items() if k != "storage_options"}
        self._effective_write_kwargs["index"] = False  # pandas index must never leak into output files

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def _materialize_dataframe(self, task: InterleavedBatch) -> pd.DataFrame:
        out = task.to_pandas()
        image_rows = out["modality"] == "image"
        self._log_metrics({"rows_out": float(len(out)), "image_rows": float(image_rows.sum())})
        if self.materialize_on_write:
            image_mask = (image_rows & out["binary_content"].isna()) if "binary_content" in out.columns else image_rows
            self._log_metric("image_rows_missing_binary", float(image_mask.sum()))
            if image_mask.any():
                with self._time_metric("materialize_fetch_binary_s"):
                    out = materialize_task_binary_content(task, io_kwargs=self.write_kwargs).to_pandas()

        # Apply on_materialize_error policy to any errors — whether set by the
        # fetch step above or by an upstream stage (e.g. ImageValidationStage).
        if "materialize_error" not in out.columns:
            return out
        error_mask = out["materialize_error"].notna()
        n_errors = int(error_mask.sum())
        self._log_metric("materialize_errors", float(n_errors))
        if n_errors == 0:
            return out
        if self.on_materialize_error == "error":
            first_err = out.loc[error_mask, "materialize_error"].iloc[0]
            msg = f"Materialization failed ({n_errors} errors). First: {first_err}"
            raise RuntimeError(msg)
        if self.on_materialize_error == "warn":
            logger.warning("materialize: {} errors (mode=warn, keeping rows)", n_errors)
        elif self.on_materialize_error == "drop_row":
            out = out[~error_mask].reset_index(drop=True)
            logger.info("materialize: dropped {} error rows", n_errors)
        elif self.on_materialize_error == "drop_sample":
            bad_samples = set(out.loc[error_mask, "sample_id"])
            out = out[~out["sample_id"].isin(bad_samples)].reset_index(drop=True)
            logger.info("materialize: dropped {} samples with errors", len(bad_samples))
        return out

    def _align_output(self, df: pd.DataFrame) -> pa.Table:
        """Reconcile or align *df* to the declared schema."""
        table = pa.Table.from_pandas(df, preserve_index=False)
        return align_interleaved_table(table, self.schema)

    def _prepare_table(self, task: InterleavedBatch) -> pa.Table:
        table = task.data
        error_index = table.schema.get_field_index("materialize_error") if isinstance(table, pa.Table) else -1
        if (
            isinstance(table, pa.Table)
            and not self.materialize_on_write
            and (error_index < 0 or table.column(error_index).null_count == table.num_rows)
        ):
            table = align_interleaved_table(table, self.schema)
            image_rows = pc.sum(pc.equal(table.column("modality"), "image")).as_py() or 0
            self._log_metrics({"rows_out": float(table.num_rows), "image_rows": float(image_rows)})
            if error_index >= 0:
                self._log_metric("materialize_errors", 0.0)
            return table

        with self._time_metric("materialize_dataframe_total_s"):
            return self._align_output(self._materialize_dataframe(task))

    def _write_table(self, table: pa.Table, file_path: str, write_kwargs: dict[str, Any]) -> None:
        """Format-specific Arrow table writer. Subclasses must implement this.

        Subclasses that override ``write_data()`` or ``process()`` directly
        (e.g. writers that do not follow the one-file-per-task pattern) may
        override this method as a no-op instead.
        """
        msg = (
            f"{type(self).__name__} must override `_write_table()`, or override "
            "`write_data()` / `process()` so that output data is actually written."
        )
        raise NotImplementedError(msg)

    def write_data(self, task: InterleavedBatch, file_path: str) -> None:
        self._write_table(self._prepare_table(task), file_path, self._effective_write_kwargs)

    def process(self, task: InterleavedBatch) -> FileGroupTask:
        if source_files := task._metadata.get("source_files"):
            filename = get_deterministic_hash(source_files, task.task_id)
        else:
            logger.warning("The task does not have source_files in metadata, using UUID for base filename")
            filename = uuid.uuid4().hex

        file_path = self.fs.sep.join([self._fs_path, f"{filename}.{self.file_extension}"])
        file_path_with_protocol = self.fs.unstrip_protocol(file_path) if is_remote_url(self.path) else file_path

        self.write_data(task, file_path_with_protocol)
        return FileGroupTask(
            dataset_name=task.dataset_name,
            data=[file_path_with_protocol],
            _metadata={**task._metadata, "format": self.file_extension},
            _stage_perf=task._stage_perf,
        )
