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

import tarfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import fsspec
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .base import BaseMultimodalWriter

if TYPE_CHECKING:
    from nemo_curator.tasks import MultiBatchTask


@dataclass
class _MaterializationBuffers:
    binary_values: list[object]
    error_values: list[str | None]


@dataclass
class BaseMultimodalTabularWriter(BaseMultimodalWriter, ABC):
    """Shared multimodal tabular writer with optional image materialization."""

    write_kwargs: dict[str, Any] = field(default_factory=dict)
    materialize_on_write: bool = True
    name: str = "base_multimodal_tabular_writer"

    @staticmethod
    def _set_errors(error_values: list[str | None], indices: list[int], error: str | None) -> None:
        for idx in indices:
            error_values[idx] = error

    @staticmethod
    def _set_payload(
        binary_values: list[object], error_values: list[str | None], indices: list[int], payload: bytes
    ) -> None:
        for idx in indices:
            binary_values[idx] = payload
            error_values[idx] = None

    @staticmethod
    def _key_to_indices(df: pd.DataFrame, keyed_idxs: list[int]) -> dict[str, list[int]]:
        key_to_indices: dict[str, list[int]] = {}
        for idx in keyed_idxs:
            key = str(df.loc[idx, "_src_content_key"])
            key_to_indices.setdefault(key, []).append(idx)
        return key_to_indices

    def _materialize_group(
        self,
        df: pd.DataFrame,
        content_path: object,
        idxs: list[int],
        storage_options: dict[str, Any],
        buffers: _MaterializationBuffers,
    ) -> None:
        binary_values = buffers.binary_values
        error_values = buffers.error_values
        if not content_path:
            self._set_errors(error_values, idxs, "missing content_path")
            return

        keyed_idxs = [idx for idx in idxs if df.loc[idx, "_src_content_key"]]
        direct_idxs = [idx for idx in idxs if not df.loc[idx, "_src_content_key"]]
        try:
            with fsspec.open(str(content_path), mode="rb", **storage_options) as fobj:
                if keyed_idxs:
                    key_to_indices = self._key_to_indices(df, keyed_idxs)
                    with tarfile.open(fileobj=fobj, mode="r:*") as tf:
                        for key, key_indices in key_to_indices.items():
                            try:
                                extracted = tf.extractfile(key)
                            except KeyError:
                                extracted = None
                            if extracted is None:
                                self._set_errors(error_values, key_indices, f"missing content_key '{key}'")
                                continue
                            self._set_payload(binary_values, error_values, key_indices, extracted.read())
                if direct_idxs:
                    if keyed_idxs:
                        with fsspec.open(str(content_path), mode="rb", **storage_options) as fresh:
                            payload = fresh.read()
                    else:
                        payload = fobj.read()
                    self._set_payload(binary_values, error_values, direct_idxs, payload)
        except Exception as e:  # noqa: BLE001
            self._set_errors(error_values, idxs, str(e))

    def _materialize_dataframe(self, task: MultiBatchTask) -> pd.DataFrame:
        if not self.materialize_on_write:
            with self._time_metric("to_pandas_s"):
                out = task.to_pandas()
            self._log_metric("rows_out", float(len(out)))
            return out

        with self._time_metric("parse_source_columns_s"):
            out = task.with_parsed_source_columns(prefix="_src_").reset_index(drop=True)
        if "materialize_error" in out.columns:
            error_values = out["materialize_error"].astype("object").tolist()
        else:
            error_values = [None] * len(out)
        binary_values = out["binary_content"].astype("object").tolist()

        image_mask = (out["modality"] == "image") & (out["binary_content"].isna())
        self._log_metrics(
            {
                "rows_out": float(len(out)),
                "image_rows": float((out["modality"] == "image").sum()),
                "image_rows_missing_binary": float(image_mask.sum()),
            }
        )
        if not image_mask.any():
            return out.drop(columns=[c for c in out.columns if c.startswith("_src_")], errors="ignore")

        source_storage_options = task._metadata.get("source_storage_options", {})
        if isinstance(source_storage_options, dict):
            storage_options = source_storage_options or (self.write_kwargs or {}).get("storage_options", {})
        else:
            storage_options = (self.write_kwargs or {}).get("storage_options", {})
        pending = out[image_mask]
        buffers = _MaterializationBuffers(binary_values=binary_values, error_values=error_values)
        with self._time_metric("materialize_fetch_binary_s"):
            for content_path, idxs in pending.groupby("_src_content_path").groups.items():
                self._materialize_group(
                    df=out,
                    content_path=content_path,
                    idxs=list(idxs),
                    storage_options=storage_options,
                    buffers=buffers,
                )

        out["binary_content"] = pd.Series(binary_values, dtype="object")
        out["materialize_error"] = pd.Series(error_values, dtype="object")
        self._log_metric("materialize_errors", float(sum(v is not None for v in error_values)))
        return out.drop(columns=[c for c in out.columns if c.startswith("_src_")], errors="ignore")

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
        write_kwargs.setdefault("compression", "snappy")
        write_kwargs.setdefault("row_group_size", 128)
        writer_backend = str(write_kwargs.pop("writer_backend", "pandas")).lower()
        if writer_backend == "pyarrow":
            write_kwargs.pop("index", None)
            write_kwargs.pop("storage_options", None)
            with self._time_metric("parquet_write_s"):
                table = pa.Table.from_pandas(df, preserve_index=False)
                pq.write_table(table, file_path, **write_kwargs)
            return
        with self._time_metric("parquet_write_s"):
            df.to_parquet(file_path, **write_kwargs)
