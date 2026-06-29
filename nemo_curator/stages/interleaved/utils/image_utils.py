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

from typing import TYPE_CHECKING

import cv2
import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Hashable


def image_bytes_to_array(image_bytes: bytes, *, row_index: Hashable | None = None) -> np.ndarray | None:
    """Decode image bytes to RGB numpy array for OpenCV."""
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except cv2.error as e:
        logger.debug("cv2 image decode failed (row_index={}): {}", row_index, e)
        return None
