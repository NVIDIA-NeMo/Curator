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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.multimodal.utils import materialize_task_binary_content
from nemo_curator.tasks import MultiBatchTask

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class BaseMultimodalAnnotatorStage(ProcessingStage[MultiBatchTask, MultiBatchTask], ABC):
    """Base stage for row-wise multimodal annotation/filter transforms."""

    name: str = "base_multimodal_annotator"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    @abstractmethod
    def annotate(self, task: MultiBatchTask, df: pd.DataFrame) -> pd.DataFrame:
        """Apply annotation/filter logic and return transformed dataframe."""

    def process(self, task: MultiBatchTask) -> MultiBatchTask:
        df = task.to_pandas().copy()
        if df.empty:
            return task
        out_df = self.annotate(task, df)
        return MultiBatchTask(
            task_id=f"{task.task_id}_{self.name}",
            dataset_name=task.dataset_name,
            data=out_df.reset_index(drop=True),
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


@dataclass
class BaseMultimodalFilterStage(BaseMultimodalAnnotatorStage, ABC):
    """Base stage for multimodal filtering based on a keep-mask."""

    drop_invalid_rows: bool = True
    name: str = "base_multimodal_filter"

    @abstractmethod
    def content_keep_mask(self, task: MultiBatchTask, df: pd.DataFrame) -> pd.Series:
        """Return content-specific boolean keep-mask aligned to dataframe index."""

    @staticmethod
    def _basic_row_validity_mask(df: pd.DataFrame) -> pd.Series:
        keep_mask = pd.Series(True, index=df.index, dtype=bool)
        allowed = {"text", "image", "metadata"}
        keep_mask &= df["modality"].isin(allowed)
        metadata_pos = (df["modality"] == "metadata") & (df["position"] == -1)
        content_pos = (df["modality"] != "metadata") & (df["position"] >= 0)
        keep_mask &= metadata_pos | content_pos
        return keep_mask

    def keep_mask(self, task: MultiBatchTask, df: pd.DataFrame) -> pd.Series:
        keep_mask = pd.Series(True, index=df.index, dtype=bool)
        if self.drop_invalid_rows:
            keep_mask &= self._basic_row_validity_mask(df)
        keep_mask &= self.content_keep_mask(task, df)
        return keep_mask

    def iter_materialized_bytes(
        self, task: MultiBatchTask, df: pd.DataFrame, row_mask: pd.Series
    ) -> Iterator[tuple[int, bytes | None]]:
        """Yield (row_index, bytes) for masked rows using shared materialization logic."""
        materialized_df = materialize_task_binary_content(task).to_pandas()
        for idx in df[row_mask].index.tolist():
            row_bytes = materialized_df.loc[idx, "binary_content"] if "binary_content" in materialized_df.columns else None
            yield idx, bytes(row_bytes) if isinstance(row_bytes, (bytes, bytearray)) else None

    def annotate(self, task: MultiBatchTask, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.keep_mask(task, df)]


@dataclass
class MultimodalJpegAspectRatioFilterStage(BaseMultimodalFilterStage):
    """Filter multimodal rows and enforce JPEG aspect-ratio bounds."""

    min_aspect_ratio: float = 0.2
    max_aspect_ratio: float = 5.0
    jpeg_content_types: tuple[str, ...] = ("image/jpeg", "image/jpg")
    name: str = "multimodal_jpeg_aspect_ratio_filter"

    @staticmethod
    def _image_aspect_ratio(image_bytes: bytes) -> float | None:
        from PIL import Image  # type: ignore[import-not-found]  # noqa: PLC0415

        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                width, height = image.size
        except Exception:  # noqa: BLE001
            return None
        if height <= 0:
            return None
        return float(width) / float(height)

    def _jpeg_keep_mask(self, task: MultiBatchTask, df: pd.DataFrame) -> pd.Series:
        keep_mask = pd.Series(True, index=df.index, dtype=bool)
        jpeg_mask = (df["modality"] == "image") & (df["content_type"].isin(self.jpeg_content_types))
        if not jpeg_mask.any():
            return keep_mask
        for idx, image_bytes in self.iter_materialized_bytes(task=task, df=df, row_mask=jpeg_mask):
            if image_bytes is None:
                keep_mask.loc[idx] = False
                continue
            aspect_ratio = self._image_aspect_ratio(image_bytes)
            if aspect_ratio is None:
                keep_mask.loc[idx] = False
                continue
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                keep_mask.loc[idx] = False
        return keep_mask

    def content_keep_mask(self, task: MultiBatchTask, df: pd.DataFrame) -> pd.Series:
        return self._jpeg_keep_mask(task, df)
