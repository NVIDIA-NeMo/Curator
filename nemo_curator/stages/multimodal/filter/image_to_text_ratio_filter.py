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

import pandas as pd

from nemo_curator.stages.multimodal.stages import BaseMultimodalFilterStage
from nemo_curator.tasks import MultiBatchTask


def _text_word_count(text: str | None) -> int:
    """Count words in text by splitting on whitespace."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return 0
    return len(str(text).split())


@dataclass
class MultimodalImageToTextRatioFilterStage(BaseMultimodalFilterStage):
    """Filter multimodal samples by image-to-text ratio (images per word).

    Groups rows by sample_id. For each sample:
    - image_count = number of rows with modality == 'image'
    - text_word_count = sum of len(text_content.split()) over text rows
    - ratio = image_count / max(text_word_count, 1)

    Samples with ratio outside [min_ratio, max_ratio] are dropped (all their rows).
    """

    min_ratio: float = 0.0
    max_ratio: float = float("inf")
    name: str = "multimodal_image_to_text_ratio_filter"

    def content_keep_mask(self, task: MultiBatchTask, df: pd.DataFrame) -> pd.Series:
        keep_mask = pd.Series(True, index=df.index, dtype=bool)
        if "sample_id" not in df.columns:
            return keep_mask

        sample_keep: dict[str, bool] = {}
        for sample_id, group in df.groupby("sample_id"):
            image_count = int((group["modality"] == "image").sum())
            text_mask = group["modality"] == "text"
            text_word_count = 0
            if text_mask.any() and "text_content" in group.columns:
                text_word_count = sum(_text_word_count(t) for t in group.loc[text_mask, "text_content"].tolist())
            ratio = image_count / max(text_word_count, 1)
            sample_keep[sample_id] = self.min_ratio <= ratio <= self.max_ratio

        keep_mask = df["sample_id"].map(sample_keep)
        keep_mask = keep_mask.fillna(True)
        return keep_mask.astype(bool)
