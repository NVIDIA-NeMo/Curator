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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from nemo_curator.stages.interleaved.stages import BaseInterleavedScoreFilterStage

if TYPE_CHECKING:
    from nemo_curator.tasks import InterleavedBatch

DEFAULT_IMAGE_TO_TEXT_MIN_RATIO: float = 0.0
DEFAULT_IMAGE_TO_TEXT_MAX_RATIO: float = float("inf")


def _text_word_count(text: str | None) -> int:
    """Count words in text by splitting on whitespace."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return 0
    return len(str(text).split())


def per_row_image_word_counts_broadcast(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Per-sample image count and text word count, broadcast to every row index.

    Used by :func:`~nemo_curator.stages.interleaved.annotation.pass_mask.interleaved_score_pass_mask`
    so ratio thresholds apply to every row (e.g. image rows) even though stored columns are
    only filled at ``position == 0``.
    """
    if "sample_id" not in df.columns:
        na_i = pd.Series(pd.NA, index=df.index, dtype="Int64")
        return na_i, na_i.copy()
    img_per: dict[Any, int] = {}
    words_per: dict[Any, int] = {}
    for sample_id, group in df.groupby("sample_id"):
        image_count = int((group["modality"] == "image").sum())
        text_mask = group["modality"] == "text"
        text_word_count = 0
        if text_mask.any() and "text_content" in group.columns:
            text_word_count = sum(_text_word_count(t) for t in group.loc[text_mask, "text_content"].tolist())
        img_per[sample_id] = image_count
        words_per[sample_id] = text_word_count
    image_num = df["sample_id"].map(img_per).astype("Int64")
    word_num = df["sample_id"].map(words_per).astype("Int64")
    return image_num, word_num


@dataclass
class InterleavedImageToTextRatioAnnotatorStage(BaseInterleavedScoreFilterStage):
    """Add per-sample image count and text word count only on rows with ``position == 0``.

    Other rows get null counts. Per-sample values are still derived from the whole sample
    (all modalities). ``min_ratio`` / ``max_ratio`` apply to image_count / max(text_word_count, 1)
    in :func:`~nemo_curator.stages.interleaved.annotation.pass_mask.interleaved_score_pass_mask`
    (which uses full-sample counts on every row); that ratio is not stored as a column.
    """

    min_ratio: float = DEFAULT_IMAGE_TO_TEXT_MIN_RATIO
    max_ratio: float = DEFAULT_IMAGE_TO_TEXT_MAX_RATIO
    name: str = "interleaved_image_to_text_ratio_annotator"

    def annotation_columns(self, task: InterleavedBatch, df: pd.DataFrame) -> dict[str, pd.Series]:
        image_full, word_full = per_row_image_word_counts_broadcast(df)
        pos0 = df["position"] == 0
        return {
            f"{self.name}_image_num": image_full.where(pos0, pd.NA).astype("Int64"),
            f"{self.name}_text_word_num": word_full.where(pos0, pd.NA).astype("Int64"),
        }
