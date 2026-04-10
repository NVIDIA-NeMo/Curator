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

"""Optional pass/fail mask derived from interleaved score stages (for metadata / filtering)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from nemo_curator.stages.interleaved.filter.blur_filter import InterleavedBlurFilterStage
from nemo_curator.stages.interleaved.filter.clip_score_filter import InterleavedCLIPScoreFilterStage
from nemo_curator.stages.interleaved.filter.image_to_text_ratio_filter import (
    InterleavedImageToTextRatioFilterStage,
    per_row_image_word_counts_broadcast,
)
from nemo_curator.stages.interleaved.filter.qrcode_filter import InterleavedQRCodeFilterStage
from nemo_curator.stages.interleaved.stages import (
    BaseInterleavedFilterStage,
    BaseInterleavedScoreFilterStage,
    InterleavedAspectRatioFilterStage,
)

if TYPE_CHECKING:
    from nemo_curator.tasks import InterleavedBatch


def basic_interleaved_row_validity_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows with allowed modality and valid metadata/content positions."""
    return BaseInterleavedFilterStage._basic_row_validity_mask(df)


def interleaved_score_pass_mask(
    stage: BaseInterleavedScoreFilterStage,
    task: InterleavedBatch,
    df: pd.DataFrame,
    *,
    drop_invalid_rows: bool = True,
) -> pd.Series:
    """Return a boolean mask aligned to ``df.index`` from score columns and stage thresholds.

    This does not mutate ``df``. It calls :meth:`~BaseInterleavedScoreFilterStage.annotation_columns`
    once per invocation, except for :class:`~InterleavedImageToTextRatioFilterStage`, which uses
    full-sample counts broadcast to every row (stored score columns are only non-null at
    ``position == 0``).
    """
    idx = df.index
    out = pd.Series(True, index=idx, dtype=bool)
    if drop_invalid_rows:
        out &= basic_interleaved_row_validity_mask(df)

    if isinstance(stage, InterleavedImageToTextRatioFilterStage):
        img, words = per_row_image_word_counts_broadcast(df)
        wf = words.astype("float64").fillna(0.0)
        denom = wf.where(wf >= 1.0, 1.0)
        ratio = img.astype("float64") / denom
        ok = (ratio >= stage.min_ratio) & (ratio <= stage.max_ratio)
        return out & ok.fillna(True).astype(bool)

    cols = stage.annotation_columns(task, df)

    if isinstance(stage, InterleavedBlurFilterStage):
        sharp = cols[f"{stage.name}_sharpness"]
        image = df["modality"] == "image"
        out &= ~image | (sharp.notna() & (sharp >= stage.score_threshold))
        return out
    if isinstance(stage, InterleavedQRCodeFilterStage):
        qr = cols[f"{stage.name}_qr_area_ratio"]
        image = df["modality"] == "image"
        out &= ~image | (qr.notna() & (qr < stage.score_threshold))
        return out
    if isinstance(stage, InterleavedCLIPScoreFilterStage):
        scores = cols[f"{stage.name}_clip_scores_by_text_position"]
        image = df["modality"] == "image"
        for i in idx[image]:
            cell = scores.loc[i]
            if isinstance(cell, pd.Series):
                cell = {int(k): float(v) for k, v in cell.items() if pd.notna(v)}
            if not isinstance(cell, dict) or not cell:
                out.loc[i] = False
            else:
                out.loc[i] = max(cell.values()) >= stage.min_score
        return out
    if isinstance(stage, InterleavedAspectRatioFilterStage):
        ar = cols[f"{stage.name}_aspect_ratio"]
        image = df["modality"] == "image"
        out &= ~image | (ar.notna() & (ar >= stage.min_aspect_ratio) & (ar <= stage.max_aspect_ratio))
        return out

    msg = (
        f"interleaved_score_pass_mask does not know how to combine scores for {type(stage).__name__!r}. "
        "Compute a mask from annotation_columns(...) in your own code."
    )
    raise TypeError(msg)
