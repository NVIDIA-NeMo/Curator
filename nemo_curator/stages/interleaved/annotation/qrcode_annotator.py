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
import numpy as np
import pandas as pd
from loguru import logger

from nemo_curator.stages.interleaved.stages import BaseInterleavedScoreFilterStage
from nemo_curator.stages.interleaved.utils import image_bytes_to_array

if TYPE_CHECKING:
    from collections.abc import Hashable

    from nemo_curator.tasks import InterleavedBatch

DEFAULT_QRCODE_SCORE_THRESHOLD: float = 0.05


def _qr_code_ratio(image: np.ndarray, row_index: Hashable | None = None) -> float:
    """Return the ratio of image area covered by all detected QR code(s), in [0, 1]."""
    img_shape = image.shape
    height, width = img_shape[:2]
    img_area = float(height * width)
    if img_area <= 0:
        return 0.0
    try:
        detector = cv2.QRCodeDetector()
        retval, _decoded_info, points, _ = detector.detectAndDecodeMulti(image)
        if not retval or points is None or points.size == 0:
            return 0.0
        points = np.asarray(points, dtype=np.float32)
        total_qr_area = 0.0
        for i in range(len(points)):
            pts = points[i].reshape(-1, 1, 2)
            total_qr_area += cv2.contourArea(pts)
        return total_qr_area / img_area
    except cv2.error as e:
        logger.debug(
            "cv2 QR code ratio computation failed (row_index={} image_shape={}): {}",
            row_index,
            img_shape,
            e,
        )
        return 0.0


@dataclass
class InterleavedQRCodeAnnotatorStage(BaseInterleavedScoreFilterStage):
    """Add QR code area ratio per image row as ``{name}_qr_area_ratio`` (``<NA>`` on non-images)."""

    score_threshold: float = DEFAULT_QRCODE_SCORE_THRESHOLD
    name: str = "interleaved_qrcode_annotator"

    def _qr_ratio_series(self, task: InterleavedBatch, df: pd.DataFrame) -> pd.Series:
        ratios = pd.Series(pd.NA, index=df.index, dtype="Float64")
        image_mask = df["modality"] == "image"
        if not image_mask.any():
            return ratios
        for idx, image_bytes in self.iter_materialized_bytes(task=task, df=df, row_mask=image_mask):
            if image_bytes is None:
                continue
            image = image_bytes_to_array(image_bytes, row_index=idx)
            if image is None:
                continue
            ratios.loc[idx] = _qr_code_ratio(image, row_index=idx)
        return ratios

    def annotation_columns(self, task: InterleavedBatch, df: pd.DataFrame) -> dict[str, pd.Series]:
        return {f"{self.name}_qr_area_ratio": self._qr_ratio_series(task, df)}
