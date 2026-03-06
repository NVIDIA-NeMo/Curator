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

import numpy as np

from nemo_curator.stages.interleaved.filter.qrcode_filter import (
    InterleavedQRCodeFilterStage,
    _process_one_qrcode,
    _qr_code_ratio,
)

from .conftest import interleaved_task, make_jpeg_bytes


def test_qr_code_ratio_no_qr_returns_zero() -> None:
    rng = np.random.default_rng()
    arr = rng.integers(0, 256, size=(50, 50, 3), dtype=np.uint8)
    ratio = _qr_code_ratio(arr)
    assert ratio == 0.0


def test_process_one_qrcode_none_bytes_dropped() -> None:
    idx, keep = _process_one_qrcode((0, None), score_threshold=1.0)
    assert idx == 0
    assert keep is False


def test_process_one_qrcode_image_kept_when_below_threshold() -> None:
    jpeg = make_jpeg_bytes()
    idx, keep = _process_one_qrcode((0, jpeg), score_threshold=1.0)
    assert idx == 0
    assert keep is True


def test_process_one_qrcode_image_dropped_when_above_threshold() -> None:
    jpeg = make_jpeg_bytes()
    idx, keep = _process_one_qrcode((0, jpeg), score_threshold=0.0)
    assert idx == 0
    assert keep is False


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
    stage = InterleavedQRCodeFilterStage(score_threshold=0.05, max_workers=None)
    out = stage.process(task)
    df = out.to_pandas()
    assert len(df) == 1


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
    stage = InterleavedQRCodeFilterStage(score_threshold=1.0, max_workers=None)
    out = stage.process(task)
    df = out.to_pandas()
    assert len(df) == 1
