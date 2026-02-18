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

import io
import tarfile
from dataclasses import dataclass, field
from typing import Any

import fsspec
import pandas as pd
from PIL import Image  # type: ignore[import-not-found]

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import MultiBatchTask


@dataclass
class BasicMultimodalFilterStage(ProcessingStage[MultiBatchTask, MultiBatchTask]):
    """Validation/filter stage for multimodal rows with optional JPEG aspect-ratio checks."""

    drop_invalid_rows: bool = True
    validate_jpeg_aspect_ratio: bool = False
    min_aspect_ratio: float = 0.2
    max_aspect_ratio: float = 5.0
    jpeg_content_types: tuple[str, ...] = ("image/jpeg", "image/jpg")
    read_kwargs: dict[str, Any] = field(default_factory=dict)
    name: str = "basic_multimodal_filter"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    @staticmethod
    def _image_aspect_ratio(image_bytes: bytes) -> float | None:
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                width, height = image.size
        except Exception:  # noqa: BLE001
            return None
        if height <= 0:
            return None
        return float(width) / float(height)

    @staticmethod
    def _load_image_bytes_from_source(
        source_value: str | None,
        storage_options: dict[str, Any],
        byte_cache: dict[tuple[str, str], bytes | None],
    ) -> bytes | None:
        source = MultiBatchTask.parse_metadata_source(source_value)
        content_path = source.get("content_path")
        content_key = source.get("content_key")
        if not content_path:
            return None

        cache_key = (str(content_path), str(content_key or ""))
        if cache_key in byte_cache:
            return byte_cache[cache_key]

        try:
            with fsspec.open(str(content_path), mode="rb", **storage_options) as fobj:
                if content_key:
                    with tarfile.open(fileobj=fobj, mode="r:*") as tf:
                        try:
                            extracted = tf.extractfile(content_key)
                        except KeyError:
                            extracted = None
                        payload = extracted.read() if extracted is not None else None
                        byte_cache[cache_key] = payload
                        return payload
                payload = fobj.read()
                byte_cache[cache_key] = payload
                return payload
        except Exception:  # noqa: BLE001
            byte_cache[cache_key] = None
            return None

    def _filter_jpeg_aspect_ratio(self, task: MultiBatchTask, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if "modality" not in df.columns or "content_type" not in df.columns:
            return df

        jpeg_mask = (df["modality"] == "image") & (df["content_type"].isin(self.jpeg_content_types))
        if not jpeg_mask.any():
            return df

        storage_options = task._metadata.get("source_storage_options")
        if not isinstance(storage_options, dict):
            storage_options = (self.read_kwargs or {}).get("storage_options", {})

        byte_cache: dict[tuple[str, str], bytes | None] = {}
        keep_mask = pd.Series(True, index=df.index)

        for idx in df[jpeg_mask].index.tolist():
            image_bytes = df.loc[idx, "binary_content"]
            if not isinstance(image_bytes, (bytes, bytearray)):
                image_bytes = self._load_image_bytes_from_source(
                    source_value=df.loc[idx, "metadata_source"] if "metadata_source" in df.columns else None,
                    storage_options=storage_options,
                    byte_cache=byte_cache,
                )
            if not isinstance(image_bytes, (bytes, bytearray)):
                keep_mask.loc[idx] = False
                continue
            aspect_ratio = self._image_aspect_ratio(bytes(image_bytes))
            if aspect_ratio is None:
                keep_mask.loc[idx] = False
                continue
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                keep_mask.loc[idx] = False

        return df[keep_mask]

    def process(self, task: MultiBatchTask) -> MultiBatchTask:
        df = task.to_pandas().copy()
        if df.empty:
            return task

        if self.drop_invalid_rows:
            allowed = {"text", "image", "metadata"}
            df = df[df["modality"].isin(allowed)]
            # Keep metadata rows at sentinel position -1; content rows should be non-negative.
            valid_pos = (df["modality"] == "metadata") & (df["position"] == -1)
            valid_pos = valid_pos | ((df["modality"] != "metadata") & (df["position"] >= 0))
            df = df[valid_pos]

        if self.validate_jpeg_aspect_ratio:
            df = self._filter_jpeg_aspect_ratio(task, df)

        return MultiBatchTask(
            task_id=f"{task.task_id}_{self.name}",
            dataset_name=task.dataset_name,
            data=df.reset_index(drop=True),
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )
