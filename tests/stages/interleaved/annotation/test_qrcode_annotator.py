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

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pandas as pd

from nemo_curator.stages.interleaved.annotation.pass_mask import interleaved_score_pass_mask
from nemo_curator.stages.interleaved.annotation.qrcode_annotator import (
    InterleavedQRCodeAnnotatorStage,
    _qr_code_ratio,
)

from .conftest import interleaved_task, make_jpeg_bytes


def test_qr_code_ratio_no_qr_returns_zero() -> None:
    rng = np.random.default_rng()
    arr = rng.integers(0, 256, size=(50, 50, 3), dtype=np.uint8)
    assert _qr_code_ratio(arr) == 0.0


def test_qr_code_ratio_zero_image_area_returns_zero() -> None:
    arr = np.zeros((0, 10, 3), dtype=np.uint8)
    assert _qr_code_ratio(arr) == 0.0


@patch("nemo_curator.stages.interleaved.annotation.qrcode_annotator.cv2.QRCodeDetector")
def test_qr_code_ratio_cv2_error_returns_zero(mock_detector_cls: MagicMock) -> None:
    detector = MagicMock()
    detector.detectAndDecodeMulti.side_effect = cv2.error("mock decode failure")
    mock_detector_cls.return_value = detector
    arr = np.ones((8, 8, 3), dtype=np.uint8)
    assert _qr_code_ratio(arr) == 0.0


def test_qrcode_annotator_empty_task_unchanged() -> None:
    task = interleaved_task([])
    stage = InterleavedQRCodeAnnotatorStage()
    assert stage.process(task).num_items == 0


def test_qrcode_annotator_text_row_has_na_score() -> None:
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
    stage = InterleavedQRCodeAnnotatorStage()
    out_frame = stage.process(task).to_pandas()
    col = f"{stage.name}_qr_area_ratio"
    assert col in out_frame.columns
    assert pd.isna(out_frame.iloc[0][col])


def test_qrcode_annotator_does_not_drop_high_qr_image() -> None:
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
    with patch("nemo_curator.stages.interleaved.annotation.qrcode_annotator._qr_code_ratio") as mock_ratio:
        mock_ratio.return_value = 0.9
        stage = InterleavedQRCodeAnnotatorStage()
        out_frame = stage.process(task).to_pandas()
    assert len(out_frame) == 1


def test_qrcode_annotator_column_name_uses_stage_name() -> None:
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
    stage = InterleavedQRCodeAnnotatorStage()
    out_frame = stage.process(task).to_pandas()
    assert f"{stage.name}_qr_area_ratio" in out_frame.columns


def test_qrcode_annotator_image_gets_qr_ratio_score() -> None:
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
    stage = InterleavedQRCodeAnnotatorStage()
    out_frame = stage.process(task).to_pandas()
    col = f"{stage.name}_qr_area_ratio"
    assert not pd.isna(out_frame.iloc[0][col])
    assert 0.0 <= float(out_frame.iloc[0][col]) <= 1.0


def test_qrcode_annotator_none_bytes_gives_na_score() -> None:
    rows = [
        {
            "sample_id": "s1",
            "position": 0,
            "modality": "image",
            "content_type": "image/jpeg",
            "text_content": None,
            "binary_content": b"unused",
            "source_ref": None,
            "materialize_error": None,
        },
    ]
    task = interleaved_task(rows)

    def _none_bytes(self: object, task: object, df: pd.DataFrame, row_mask: pd.Series) -> Iterator[tuple[Any, None]]:
        del self, task
        for idx in df[row_mask].index:
            yield idx, None

    with patch.object(InterleavedQRCodeAnnotatorStage, "iter_materialized_bytes", _none_bytes):
        stage = InterleavedQRCodeAnnotatorStage()
        out_frame = stage.process(task).to_pandas()
    assert pd.isna(out_frame.iloc[0][f"{stage.name}_qr_area_ratio"])


def test_qrcode_annotator_pass_mask_low_qr_passes() -> None:
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
    stage = InterleavedQRCodeAnnotatorStage()
    mask = interleaved_score_pass_mask(stage, task, task.to_pandas(), score_threshold=1.0)
    assert mask.all()


@patch("nemo_curator.stages.interleaved.annotation.qrcode_annotator._qr_code_ratio")
def test_qrcode_annotator_pass_mask_high_qr_fails(mock_ratio: MagicMock) -> None:
    mock_ratio.return_value = 0.5
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
    stage = InterleavedQRCodeAnnotatorStage()
    mask = interleaved_score_pass_mask(stage, task, task.to_pandas(), score_threshold=0.05)
    assert not mask.any()
