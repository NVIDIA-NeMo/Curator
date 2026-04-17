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
from loguru import logger

from nemo_curator.stages.interleaved.stages import BaseInterleavedScoreFilterStage
from nemo_curator.stages.interleaved.utils import image_bytes_to_array

if TYPE_CHECKING:
    from collections.abc import Hashable

    import numpy as np

    from nemo_curator.tasks import InterleavedBatch


def _sharpness_score(image: np.ndarray, row_index: Hashable | None = None) -> float:
    """Compute Laplacian variance as sharpness score; higher is sharper."""
    try:
        return float(cv2.Laplacian(image, cv2.CV_64F).var())
    except cv2.error as e:
        logger.debug(
            "cv2.Laplacian failed (row_index={} image_shape={}): {}",
            row_index,
            getattr(image, "shape", None),
            e,
        )
        return 0.0


@dataclass
class InterleavedBlurAnnotatorStage(BaseInterleavedScoreFilterStage):
    """Add Laplacian sharpness per image row as ``sharpness`` (``<NA>`` on non-images / failures)."""

    name: str = "interleaved_blur_annotator"

    def _sharpness_series(self, task: InterleavedBatch, df: pd.DataFrame) -> pd.Series:
        sharp = pd.Series(pd.NA, index=df.index, dtype="Float64")
        image_mask = df["modality"] == "image"
        if not image_mask.any():
            return sharp
        for idx, image_bytes in self.iter_materialized_bytes(task=task, df=df, row_mask=image_mask):
            if image_bytes is None:
                continue
            image = image_bytes_to_array(image_bytes, row_index=idx)
            if image is None:
                continue
            sharp.loc[idx] = _sharpness_score(image, row_index=idx)
        return sharp

    def annotation_columns(self, task: InterleavedBatch, df: pd.DataFrame) -> dict[str, pd.Series]:
        return {"sharpness": self._sharpness_series(task, df)}
