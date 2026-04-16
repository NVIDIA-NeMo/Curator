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

"""Optional pass/fail mask derived from interleaved annotator stages (for metadata / filtering)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from nemo_curator.stages.interleaved.annotation.blur_annotator import InterleavedBlurAnnotatorStage
from nemo_curator.stages.interleaved.annotation.clip_score_annotator import InterleavedCLIPScoreAnnotatorStage
from nemo_curator.stages.interleaved.annotation.image_to_text_ratio_annotator import (
    InterleavedImageToTextRatioAnnotatorStage,
    per_row_image_word_counts_broadcast,
)
from nemo_curator.stages.interleaved.annotation.qrcode_annotator import InterleavedQRCodeAnnotatorStage
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
    once per invocation, except for :class:`~InterleavedImageToTextRatioAnnotatorStage`, which uses
    full-sample counts broadcast to every row (stored score columns are only non-null at
    ``position == 0``).
    """
    idx = df.index
    out = pd.Series(True, index=idx, dtype=bool)
    if drop_invalid_rows:
        out &= basic_interleaved_row_validity_mask(df)

    if isinstance(stage, InterleavedImageToTextRatioAnnotatorStage):
        img, words = per_row_image_word_counts_broadcast(df)
        img_f = img.astype("float64")
        wf = words.astype("float64").fillna(0.0)
        denom = wf.where(wf >= 1.0, 1.0)
        ratio = img_f / denom
        # NaN ratio means sample_id was absent; treat as pass
        ok = ratio.isna() | ((ratio >= stage.min_ratio) & (ratio <= stage.max_ratio))
        return out & ok.astype(bool)

    cols = stage.annotation_columns(task, df)

    if isinstance(stage, InterleavedBlurAnnotatorStage):
        sharp = cols[f"{stage.name}_sharpness"]
        image = df["modality"] == "image"
        out &= ~image | (sharp.notna() & (sharp >= stage.score_threshold))
        return out
    if isinstance(stage, InterleavedQRCodeAnnotatorStage):
        qr = cols[f"{stage.name}_qr_area_ratio"]
        image = df["modality"] == "image"
        out &= ~image | (qr.notna() & (qr < stage.score_threshold))
        return out
    if isinstance(stage, InterleavedCLIPScoreAnnotatorStage):
        scores = cols[f"{stage.name}_clip_scores"]
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
