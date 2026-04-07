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

from nemo_curator.stages.interleaved.filter.qrcode_filter import (
    InterleavedQRCodeFilterStage,
    _qr_code_ratio,
)

from .conftest import interleaved_task, make_jpeg_bytes


def test_qr_code_ratio_no_qr_returns_zero() -> None:
    rng = np.random.default_rng()
    arr = rng.integers(0, 256, size=(50, 50, 3), dtype=np.uint8)
    ratio = _qr_code_ratio(arr)
    assert ratio == 0.0


def test_qr_code_ratio_zero_image_area_returns_zero() -> None:
    arr = np.zeros((0, 10, 3), dtype=np.uint8)
    assert _qr_code_ratio(arr) == 0.0


@patch("nemo_curator.stages.interleaved.filter.qrcode_filter.cv2.QRCodeDetector")
def test_qr_code_ratio_cv2_error_returns_zero(mock_detector_cls: MagicMock) -> None:
    detector = MagicMock()
    detector.detectAndDecodeMulti.side_effect = cv2.error("mock decode failure")
    mock_detector_cls.return_value = detector
    arr = np.ones((8, 8, 3), dtype=np.uint8)
    assert _qr_code_ratio(arr) == 0.0


def test_qrcode_filter_empty_task_unchanged() -> None:
    task = interleaved_task([])
    stage = InterleavedQRCodeFilterStage(score_threshold=0.05)
    out = stage.process(task)
    assert out.num_items == 0


def test_qrcode_filter_metadata_and_text_passthrough() -> None:
    rows = [
        {
            "sample_id": "s1",
            "position": -1,
            "modality": "metadata",
            "content_type": "application/json",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
        {
            "sample_id": "s1",
            "position": 0,
            "modality": "text",
            "content_type": "text/plain",
            "text_content": "hello",
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
    ]
    task = interleaved_task(rows)
    stage = InterleavedQRCodeFilterStage(score_threshold=0.05)
    out = stage.process(task)
    out_frame = out.to_pandas()
    assert len(out_frame) == 2


def test_qrcode_filter_text_only_passthrough() -> None:
    rows = [
        {
            "sample_id": "s1",
            "position": 0,
            "modality": "text",
            "content_type": "text/plain",
            "text_content": "hello",
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
    ]
    task = interleaved_task(rows)
    stage = InterleavedQRCodeFilterStage(score_threshold=0.05)
    out = stage.process(task)
    out_frame = out.to_pandas()
    assert len(out_frame) == 1


@patch("nemo_curator.stages.interleaved.filter.qrcode_filter.image_bytes_to_array")
def test_qrcode_filter_image_decode_error_drops_row(mock_to_array: MagicMock) -> None:
    mock_to_array.return_value = None
    rows = [
        {
            "sample_id": "s1",
            "position": 0,
            "modality": "image",
            "content_type": "image/jpeg",
            "text_content": None,
            "binary_content": b"garbage",
            "source_ref": None,
            "materialize_error": None,
        },
    ]
    task = interleaved_task(rows)
    stage = InterleavedQRCodeFilterStage(score_threshold=0.05)
    out = stage.process(task)
    out_frame = out.to_pandas()
    assert len(out_frame) == 0


def test_qrcode_filter_image_below_threshold_kept() -> None:
    jpeg = make_jpeg_bytes()
    rows = [
        {
            "sample_id": "s1",
            "position": 0,
            "modality": "image",
            "content_type": "image/jpeg",
            "text_content": None,
            "binary_content": jpeg,
            "source_ref": None,
            "materialize_error": None,
        },
    ]
    task = interleaved_task(rows)
    stage = InterleavedQRCodeFilterStage(score_threshold=1.0)
    out = stage.process(task)
    out_frame = out.to_pandas()
    assert len(out_frame) == 1
