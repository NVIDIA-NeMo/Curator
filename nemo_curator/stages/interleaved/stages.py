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
from nemo_curator.stages.interleaved.utils import materialize_task_binary_content
from nemo_curator.tasks import InterleavedBatch

if TYPE_CHECKING:
    from collections.abc import Iterator

try:
    from PIL import Image
except ImportError:
    Image = None


@dataclass
class BaseInterleavedAnnotatorStage(ProcessingStage[InterleavedBatch, InterleavedBatch], ABC):
    """Base stage for row-wise interleaved annotation/filter transforms."""

    name: str = "base_interleaved_annotator"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    @abstractmethod
    def annotate(self, task: InterleavedBatch, df: pd.DataFrame) -> pd.DataFrame:
        """Apply annotation/filter logic and return transformed dataframe."""

    def process(self, task: InterleavedBatch) -> InterleavedBatch:
        df = task.to_pandas().copy()
        if df.empty:
            return task
        out_df = self.annotate(task, df)
        return InterleavedBatch(
            task_id=f"{task.task_id}_{self.name}",
            dataset_name=task.dataset_name,
            data=out_df.reset_index(drop=True),
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


@dataclass
class BaseInterleavedFilterStage(BaseInterleavedAnnotatorStage, ABC):
    """Base stage for interleaved filtering based on a keep-mask."""

    drop_invalid_rows: bool = True
    name: str = "base_interleaved_filter"

    @abstractmethod
    def content_keep_mask(self, task: InterleavedBatch, df: pd.DataFrame) -> pd.Series:
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

    def keep_mask(self, task: InterleavedBatch, df: pd.DataFrame) -> pd.Series:
        keep_mask = pd.Series(True, index=df.index, dtype=bool)
        if self.drop_invalid_rows:
            keep_mask &= self._basic_row_validity_mask(df)
        keep_mask &= self.content_keep_mask(task, df)
        return keep_mask

    def iter_materialized_bytes(
        self, task: InterleavedBatch, df: pd.DataFrame, row_mask: pd.Series
    ) -> Iterator[tuple[int, bytes | None]]:
        """Yield ``(row_index, bytes)`` for masked rows after materialization.

        Only the masked subset is materialized, avoiding redundant I/O for
        the full task.
        """
        masked_indices = df[row_mask].index.tolist()
        if not masked_indices:
            return
        temp_task = InterleavedBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=df.loc[masked_indices],
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )
        materialized_df = materialize_task_binary_content(temp_task).to_pandas().reset_index(drop=True)
        if "binary_content" not in materialized_df.columns:
            for idx in masked_indices:
                yield idx, None
            return
        for i, idx in enumerate(masked_indices):
            row_bytes = materialized_df.iloc[i]["binary_content"]
            yield idx, bytes(row_bytes) if isinstance(row_bytes, (bytes, bytearray)) else None

    def annotate(self, task: InterleavedBatch, df: pd.DataFrame) -> pd.DataFrame:
        filtered = df[self.keep_mask(task, df)].copy()
        content_mask = filtered["modality"] != "metadata"
        if content_mask.any():
            content_by_position = filtered[content_mask].sort_values("position")
            reindexed = content_by_position.groupby("sample_id", sort=False).cumcount()
            filtered.loc[content_mask, "position"] = reindexed.astype(filtered["position"].dtype)
        content_sample_ids = set(filtered.loc[content_mask, "sample_id"])
        orphan_mask = (~content_mask) & (~filtered["sample_id"].isin(content_sample_ids))
        filtered = filtered[~orphan_mask]
        return filtered.sort_values(["sample_id", "position"])


@dataclass
class BaseInterleavedScoreFilterStage(BaseInterleavedAnnotatorStage, ABC):
    """Base stage that only adds score columns; no row dropping or reordering in :meth:`annotate`.

    Subclasses implement :meth:`annotation_columns` to return ``dict[column_name, Series]``
    aligned to ``df.index``. :meth:`annotate` copies the frame, assigns score columns, and leaves
    all original columns and row order unchanged.

    For an optional boolean pass mask (e.g. annotation metadata), use
    :func:`nemo_curator.stages.interleaved.annotation.pass_mask.interleaved_score_pass_mask`.
    """

    name: str = "base_interleaved_score_filter"

    @abstractmethod
    def annotation_columns(self, task: InterleavedBatch, df: pd.DataFrame) -> dict[str, pd.Series]:
        """Per-row score columns aligned to ``df.index``."""

    def annotate(self, task: InterleavedBatch, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col_name, series in self.annotation_columns(task, out).items():
            out[col_name] = series.reindex(out.index)
        return out

    def iter_materialized_bytes(
        self, task: InterleavedBatch, df: pd.DataFrame, row_mask: pd.Series
    ) -> Iterator[tuple[int, bytes | None]]:
        """Yield ``(row_index, bytes)`` for masked rows after materialization.

        Only the masked subset is materialized, avoiding redundant I/O for
        the full task.
        """
        masked_indices = df[row_mask].index.tolist()
        if not masked_indices:
            return
        temp_task = InterleavedBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=df.loc[masked_indices],
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )
        materialized_df = materialize_task_binary_content(temp_task).to_pandas().reset_index(drop=True)
        if "binary_content" not in materialized_df.columns:
            for idx in masked_indices:
                yield idx, None
            return
        for i, idx in enumerate(masked_indices):
            row_bytes = materialized_df.iloc[i]["binary_content"]
            yield idx, bytes(row_bytes) if isinstance(row_bytes, (bytes, bytearray)) else None


@dataclass
class InterleavedAspectRatioFilterStage(BaseInterleavedScoreFilterStage):
    """Attach per-image aspect ratio (``{name}_aspect_ratio``); ``<NA>`` on non-images / failures."""

    min_aspect_ratio: float = 1.0
    max_aspect_ratio: float = 2.0
    name: str = "interleaved_aspect_ratio_filter"

    @staticmethod
    def _image_aspect_ratio(image_bytes: bytes) -> float | None:
        if Image is None:
            msg = (
                "Pillow is required for InterleavedAspectRatioFilterStage. "
                "Install dependency group `image_cpu` (or `pillow`)."
            )
            raise RuntimeError(msg)
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                width, height = image.size
        except (OSError, SyntaxError, ValueError):
            return None
        if height <= 0:
            return None
        return float(width) / float(height)

    def _aspect_ratio_series(self, task: InterleavedBatch, df: pd.DataFrame) -> pd.Series:
        s = pd.Series(pd.NA, index=df.index, dtype="Float64")
        image_mask = df["modality"] == "image"
        if not image_mask.any():
            return s
        for idx, image_bytes in self.iter_materialized_bytes(task=task, df=df, row_mask=image_mask):
            if image_bytes is None:
                continue
            aspect_ratio = self._image_aspect_ratio(image_bytes)
            if aspect_ratio is not None:
                s.loc[idx] = aspect_ratio
        return s

    def annotation_columns(self, task: InterleavedBatch, df: pd.DataFrame) -> dict[str, pd.Series]:
        return {f"{self.name}_aspect_ratio": self._aspect_ratio_series(task, df)}
