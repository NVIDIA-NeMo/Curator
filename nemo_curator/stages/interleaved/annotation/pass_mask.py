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

from nemo_curator.stages.interleaved.annotation.blur_annotator import (
    DEFAULT_BLUR_SCORE_THRESHOLD,
    InterleavedBlurAnnotatorStage,
)
from nemo_curator.stages.interleaved.annotation.clip_score_annotator import (
    DEFAULT_CLIP_MIN_SCORE,
    InterleavedCLIPScoreAnnotatorStage,
)
from nemo_curator.stages.interleaved.annotation.image_to_text_ratio_annotator import (
    DEFAULT_IMAGE_TO_TEXT_MAX_RATIO,
    DEFAULT_IMAGE_TO_TEXT_MIN_RATIO,
    InterleavedImageToTextRatioAnnotatorStage,
    per_row_image_word_counts_broadcast,
)
from nemo_curator.stages.interleaved.annotation.qrcode_annotator import (
    DEFAULT_QRCODE_SCORE_THRESHOLD,
    InterleavedQRCodeAnnotatorStage,
)
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


def interleaved_score_pass_mask(  # noqa: PLR0913
    stage: BaseInterleavedScoreFilterStage,
    task: InterleavedBatch,
    df: pd.DataFrame,
    *,
    drop_invalid_rows: bool = True,
    score_threshold: float | None = None,
    min_score: float | None = None,
    min_ratio: float | None = None,
    max_ratio: float | None = None,
) -> pd.Series:
    """Return a boolean mask aligned to ``df.index`` from score columns and thresholds.

    This does not mutate ``df``. It calls :meth:`~BaseInterleavedScoreFilterStage.annotation_columns`
    once per invocation, except for :class:`~InterleavedImageToTextRatioAnnotatorStage`, which uses
    full-sample counts broadcast to every row (stored score columns are only non-null at
    ``position == 0``).

    Threshold parameters default to the stage-specific defaults when ``None``:
    ``score_threshold`` for blur (min sharpness) and qrcode (max QR area ratio),
    ``min_score`` for CLIP, ``min_ratio`` / ``max_ratio`` for image-to-text ratio.
    """
    idx = df.index
    out = pd.Series(True, index=idx, dtype=bool)
    if drop_invalid_rows:
        out &= basic_interleaved_row_validity_mask(df)

    if isinstance(stage, InterleavedImageToTextRatioAnnotatorStage):
        actual_min_ratio = min_ratio if min_ratio is not None else DEFAULT_IMAGE_TO_TEXT_MIN_RATIO
        actual_max_ratio = max_ratio if max_ratio is not None else DEFAULT_IMAGE_TO_TEXT_MAX_RATIO
        img, words = per_row_image_word_counts_broadcast(df)
        img_f = img.astype("float64")
        wf = words.astype("float64").fillna(0.0)
        denom = wf.where(wf >= 1.0, 1.0)
        ratio = img_f / denom
        # NaN ratio means sample_id was absent; treat as pass
        ok = ratio.isna() | ((ratio >= actual_min_ratio) & (ratio <= actual_max_ratio))
        return out & ok.astype(bool)

    cols = stage.annotation_columns(task, df)

    if isinstance(stage, InterleavedBlurAnnotatorStage):
        threshold = score_threshold if score_threshold is not None else DEFAULT_BLUR_SCORE_THRESHOLD
        sharp = cols[f"{stage.name}_sharpness"]
        image = df["modality"] == "image"
        out &= ~image | (sharp.notna() & (sharp >= threshold))
        return out
    if isinstance(stage, InterleavedQRCodeAnnotatorStage):
        threshold = score_threshold if score_threshold is not None else DEFAULT_QRCODE_SCORE_THRESHOLD
        qr = cols[f"{stage.name}_qr_area_ratio"]
        image = df["modality"] == "image"
        out &= ~image | (qr.notna() & (qr < threshold))
        return out
    if isinstance(stage, InterleavedCLIPScoreAnnotatorStage):
        actual_min_score = min_score if min_score is not None else DEFAULT_CLIP_MIN_SCORE
        scores = cols[f"{stage.name}_clip_scores"]
        image = df["modality"] == "image"
        for i in idx[image]:
            cell = scores.loc[i]
            if isinstance(cell, pd.Series):
                cell = {int(k): float(v) for k, v in cell.items() if pd.notna(v)}
            if not isinstance(cell, dict) or not cell:
                out.loc[i] = False
            else:
                out.loc[i] = max(cell.values()) >= actual_min_score
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
