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

from nemo_curator.stages.image.filters.qrcode_filter import _qr_code_ratio
from nemo_curator.stages.multimodal.filter.blur_filter import _image_bytes_to_array
from nemo_curator.stages.multimodal.stages import BaseMultimodalFilterStage
from nemo_curator.tasks import MultiBatchTask


@dataclass
class MultimodalQRCodeFilterStage(BaseMultimodalFilterStage):
    """Filter multimodal rows by QR code area ratio; drop images with high QR coverage."""

    score_threshold: float = 0.05
    image_content_types: tuple[str, ...] = ("image/jpeg", "image/jpg", "image/png")
    name: str = "multimodal_qrcode_filter"

    def content_keep_mask(self, task: MultiBatchTask, df: pd.DataFrame) -> pd.Series:
        keep_mask = pd.Series(True, index=df.index, dtype=bool)
        image_mask = (df["modality"] == "image") & (df["content_type"].isin(self.image_content_types))
        if not image_mask.any():
            return keep_mask
        for idx, image_bytes in self.iter_materialized_bytes(task=task, df=df, row_mask=image_mask):
            if image_bytes is None:
                keep_mask.loc[idx] = False
                continue
            image = _image_bytes_to_array(image_bytes)
            if image is None:
                keep_mask.loc[idx] = False
                continue
            qr_ratio = _qr_code_ratio(image)
            keep_mask.loc[idx] = qr_ratio < self.score_threshold
        return keep_mask
