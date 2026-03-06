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

from nemo_curator.stages.interleaved.filter.blur_filter import (
    InterleavedBlurFilterStage,
    _image_bytes_to_array,
    _process_one_blur,
    _sharpness_score,
)

from .conftest import interleaved_task, make_jpeg_bytes


def test_sharpness_score_solid_image_is_low() -> None:
    arr = np.full((10, 10, 3), 100, dtype=np.uint8)
    assert _sharpness_score(arr) == 0.0


def test_sharpness_score_high_frequency_is_high() -> None:
    rng = np.random.default_rng()
    arr = rng.integers(0, 256, size=(20, 20, 3), dtype=np.uint8)
    score = _sharpness_score(arr)
    assert score > 0.0


def test_image_bytes_to_array_valid_jpeg() -> None:
    jpeg = make_jpeg_bytes()
    arr = _image_bytes_to_array(jpeg)
    assert arr is not None
    assert arr.shape[-1] == 3


def test_image_bytes_to_array_invalid_returns_none() -> None:
    assert _image_bytes_to_array(b"not-an-image") is None


def test_process_one_blur_none_bytes_dropped() -> None:
    idx, keep = _process_one_blur((0, None), score_threshold=0.0)
    assert idx == 0
    assert keep is False


def test_process_one_blur_sharp_kept() -> None:
    jpeg = make_jpeg_bytes(sharp=True)
    idx, keep = _process_one_blur((0, jpeg), score_threshold=0.0)
    assert idx == 0
    assert keep is True


def test_process_one_blur_blurry_dropped_when_above_threshold() -> None:
    jpeg = make_jpeg_bytes(sharp=False)
    idx, keep = _process_one_blur((0, jpeg), score_threshold=1e6)
    assert idx == 0
    assert keep is False


def test_blur_filter_text_only_passthrough() -> None:
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
        {
            "sample_id": "s1",
            "position": 1,
            "modality": "text",
            "content_type": "text/plain",
            "text_content": "world",
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
    ]
    task = interleaved_task(rows)
    stage = InterleavedBlurFilterStage(score_threshold=100.0, max_workers=None)
    out = stage.process(task)
    df = out.to_pandas()
    assert len(df) == 2


def test_blur_filter_image_with_binary_content_sharp_kept() -> None:
    jpeg = make_jpeg_bytes(sharp=True)
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
    stage = InterleavedBlurFilterStage(score_threshold=0.0, max_workers=None)
    out = stage.process(task)
    df = out.to_pandas()
    assert len(df) == 1


def test_blur_filter_image_with_binary_content_blurry_dropped() -> None:
    jpeg = make_jpeg_bytes(sharp=False)
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
    stage = InterleavedBlurFilterStage(score_threshold=1e6, max_workers=None)
    out = stage.process(task)
    df = out.to_pandas()
    assert len(df) == 0
