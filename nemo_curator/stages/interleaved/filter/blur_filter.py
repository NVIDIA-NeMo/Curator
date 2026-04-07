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
from typing import TYPE_CHECKING

import cv2
import pandas as pd

from nemo_curator.stages.interleaved.stages import BaseInterleavedFilterStage
from nemo_curator.stages.interleaved.utils import image_bytes_to_array

if TYPE_CHECKING:
    import numpy as np

    from nemo_curator.tasks import InterleavedBatch

DEFAULT_BLUR_SCORE_THRESHOLD: float = 100.0


def _sharpness_score(image: np.ndarray) -> float:
    """Compute Laplacian variance as sharpness score; higher is sharper."""
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


@dataclass
class InterleavedBlurFilterStage(BaseInterleavedFilterStage):
    """Filter interleaved image rows by sharpness (Laplacian variance); drop blurry images."""

    score_threshold: float = DEFAULT_BLUR_SCORE_THRESHOLD
    name: str = "interleaved_blur_filter"

    def content_keep_mask(self, task: InterleavedBatch, df: pd.DataFrame) -> pd.Series:
        keep_mask = pd.Series(True, index=df.index, dtype=bool)
        image_mask = df["modality"] == "image"
        if not image_mask.any():
            return keep_mask
        for idx, image_bytes in self.iter_materialized_bytes(task=task, df=df, row_mask=image_mask):
            if image_bytes is None:
                keep_mask.loc[idx] = False
                continue
            try:
                image = image_bytes_to_array(image_bytes)
                sharpness = _sharpness_score(image)
                keep_mask.loc[idx] = sharpness >= self.score_threshold
            except OSError:
                keep_mask.loc[idx] = False
        return keep_mask
