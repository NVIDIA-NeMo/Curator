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

from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from nemo_curator.stages.interleaved.utils.image_utils import image_bytes_to_array


@patch("nemo_curator.stages.interleaved.utils.image_utils.cv2.imdecode")
def test_image_bytes_to_array_cv2_imdecode_error_returns_none(mock_imdecode: MagicMock) -> None:
    mock_imdecode.side_effect = cv2.error("mock imdecode failure")
    assert image_bytes_to_array(b"\x00\x01\x02", row_index=7) is None


@patch("nemo_curator.stages.interleaved.utils.image_utils.cv2.cvtColor")
@patch("nemo_curator.stages.interleaved.utils.image_utils.cv2.imdecode")
def test_image_bytes_to_array_cv2_cvtcolor_error_returns_none(mock_imdecode: MagicMock, mock_cvt: MagicMock) -> None:
    mock_imdecode.return_value = np.zeros((2, 2, 3), dtype=np.uint8)
    mock_cvt.side_effect = cv2.error("mock cvtColor failure")
    assert image_bytes_to_array(b"\x00\x01\x02", row_index=None) is None
