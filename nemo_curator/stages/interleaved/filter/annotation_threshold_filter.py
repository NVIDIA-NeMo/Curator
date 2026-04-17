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

"""Filter interleaved batches using score columns produced by interleaved annotator stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from nemo_curator.stages.interleaved.filter.blur_filter import DEFAULT_BLUR_SCORE_THRESHOLD
from nemo_curator.stages.interleaved.filter.clip_score_filter import DEFAULT_CLIP_MIN_SCORE
from nemo_curator.stages.interleaved.filter.image_to_text_ratio_filter import (
    DEFAULT_IMAGE_TO_TEXT_MAX_RATIO,
    DEFAULT_IMAGE_TO_TEXT_MIN_RATIO,
)
from nemo_curator.stages.interleaved.filter.qrcode_filter import DEFAULT_QRCODE_SCORE_THRESHOLD
from nemo_curator.stages.interleaved.stages import BaseInterleavedFilterStage

if TYPE_CHECKING:
    from nemo_curator.tasks import InterleavedBatch

DEFAULT_BLUR_SCORE_COLUMN: str | None = "sharpness"
DEFAULT_CLIP_SCORES_COLUMN: str | None = "clip_scores"
DEFAULT_QRCODE_RATIO_COLUMN: str | None = "qr_area_ratio"
DEFAULT_IMAGE_TEXT_IMAGE_NUM_COLUMN: str | None = "image_num"
DEFAULT_IMAGE_TEXT_WORD_NUM_COLUMN: str | None = "text_word_num"


def _clip_cell_max_score(cell: object) -> float | None:
    """Return max CLIP score from an annotator cell (dict or Series), or None if unusable."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return None
    if isinstance(cell, pd.Series):
        cell = {int(k): float(v) for k, v in cell.items() if pd.notna(v)}
    if not isinstance(cell, dict) or not cell:
        return None
    return max(float(v) for v in cell.values())


@dataclass
class InterleavedAnnotationThresholdFilterStage(BaseInterleavedFilterStage):
    """Drop rows using thresholds on columns written by interleaved annotator stages.

    Expects column names matching ``InterleavedBlurAnnotatorStage``, ``InterleavedCLIPScoreAnnotatorStage``,
    ``InterleavedQRCodeAnnotatorStage``, and ``InterleavedImageToTextRatioAnnotatorStage`` when their
    default ``name`` is unchanged. Set any ``*_column`` to ``None`` to skip that criterion.

    Blur: image rows need Laplacian sharpness ``>= blur_min_sharpness``.
    CLIP: image rows need ``max(clip_scores dict values) >= clip_min_score``.
    QR code: image rows need QR area ratio ``< qrcode_max_area_ratio``.
    Image-text: per ``sample_id``, ratio ``image_count / max(text_word_count, 1)`` must lie in
    ``[image_text_min_ratio, image_text_max_ratio]``, using annotator count columns (typically
    non-null only on ``position == 0``).
    """

    blur_min_sharpness: float = DEFAULT_BLUR_SCORE_THRESHOLD
    clip_min_score: float = DEFAULT_CLIP_MIN_SCORE
    qrcode_max_area_ratio: float = DEFAULT_QRCODE_SCORE_THRESHOLD
    image_text_min_ratio: float = DEFAULT_IMAGE_TO_TEXT_MIN_RATIO
    image_text_max_ratio: float = DEFAULT_IMAGE_TO_TEXT_MAX_RATIO

    blur_score_column: str | None = DEFAULT_BLUR_SCORE_COLUMN
    clip_scores_column: str | None = DEFAULT_CLIP_SCORES_COLUMN
    qrcode_ratio_column: str | None = DEFAULT_QRCODE_RATIO_COLUMN
    image_text_image_num_column: str | None = DEFAULT_IMAGE_TEXT_IMAGE_NUM_COLUMN
    image_text_word_num_column: str | None = DEFAULT_IMAGE_TEXT_WORD_NUM_COLUMN

    name: str = "interleaved_annotation_threshold_filter"

    def _blur_mask(self, df: pd.DataFrame, image_rows: pd.Series) -> pd.Series:
        if self.blur_score_column not in df.columns:
            return ~image_rows
        sharp = df[self.blur_score_column]
        return ~image_rows | (sharp.notna() & (sharp >= self.blur_min_sharpness))

    def _qrcode_mask(self, df: pd.DataFrame, image_rows: pd.Series) -> pd.Series:
        if self.qrcode_ratio_column not in df.columns:
            return ~image_rows
        qr = df[self.qrcode_ratio_column]
        return ~image_rows | (qr.notna() & (qr < self.qrcode_max_area_ratio))

    def _clip_mask(self, df: pd.DataFrame, image_rows: pd.Series) -> pd.Series:
        if self.clip_scores_column not in df.columns:
            return ~image_rows
        keep = pd.Series(True, index=df.index, dtype=bool)
        scores = df[self.clip_scores_column]
        for idx in df.index[image_rows]:
            mx = _clip_cell_max_score(scores.loc[idx])
            keep.loc[idx] = mx is not None and mx >= self.clip_min_score
        return keep

    def _image_text_mask(self, df: pd.DataFrame) -> pd.Series:
        img_col = self.image_text_image_num_column
        word_col = self.image_text_word_num_column
        if img_col not in df.columns or word_col not in df.columns or "sample_id" not in df.columns:
            return pd.Series(False, index=df.index, dtype=bool)
        sample_keep: dict[object, bool] = {}
        for sample_id, group in df.groupby("sample_id"):
            im = group[img_col].max()
            w = group[word_col].max()
            if pd.isna(im) or pd.isna(w):
                sample_keep[sample_id] = False
            else:
                ratio = float(im) / max(int(w), 1)
                sample_keep[sample_id] = self.image_text_min_ratio <= ratio <= self.image_text_max_ratio
        return df["sample_id"].map(sample_keep).fillna(True).astype(bool)

    def content_keep_mask(self, task: InterleavedBatch, df: pd.DataFrame) -> pd.Series:  # noqa: ARG002
        keep_mask = pd.Series(True, index=df.index, dtype=bool)
        image_rows = df["modality"] == "image"
        if self.blur_score_column:
            keep_mask &= self._blur_mask(df, image_rows)
        if self.qrcode_ratio_column:
            keep_mask &= self._qrcode_mask(df, image_rows)
        if self.clip_scores_column:
            keep_mask &= self._clip_mask(df, image_rows)
        if self.image_text_image_num_column and self.image_text_word_num_column:
            keep_mask &= self._image_text_mask(df)
        return keep_mask
