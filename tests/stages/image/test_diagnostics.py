# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import io
from typing import Never

import cv2
import numpy as np
import pytest
from PIL import Image

from nemo_curator.stages.image.diagnostics import ImageDiagnosticsStage, decode_image_rgb, image_diagnostics
from nemo_curator.tasks import ImageBatch
from nemo_curator.tasks.image import ImageObject


def test_decode_orientation_and_transparency():
    image = Image.new("RGBA", (3, 2), (255, 0, 0, 0))
    exif = Image.Exif()
    exif[274] = 6
    raw = io.BytesIO()
    image.save(raw, format="PNG", exif=exif)
    rgb = decode_image_rgb(raw.getvalue())
    assert rgb.shape == (3, 2, 3)
    assert (rgb == 255).all()


def test_constant_and_real_qr():
    result = image_diagnostics(np.full((100, 100, 3), 255, dtype=np.uint8))
    assert result["image_sharpness_score"] == 0
    assert result["image_qr_area_ratio"] == 0
    qr = cv2.QRCodeEncoder_create().encode("curator diagnostics")
    qr = cv2.copyMakeBorder(qr, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    qr = cv2.resize(qr, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
    rgb = np.repeat(qr[:, :, None], 3, axis=2)
    result = image_diagnostics(rgb)
    assert result["image_sharpness_score"] > 0
    assert 0 < result["image_qr_area_ratio"] <= 1
    assert result["image_qr_status"] == "ok"


def test_independent_failure_and_row_preservation():
    class BrokenDetector:
        def detectMulti(self, image: np.ndarray) -> Never:  # noqa: N802
            message = "detector failure"
            raise cv2.error(message)

    rgb = np.full((20, 20, 3), 100, dtype=np.uint8)
    values = image_diagnostics(rgb, BrokenDetector())
    assert values["image_width"] == 20
    assert values["image_sharpness_status"] == "ok"
    assert values["image_qr_status"] == "compute_error"
    assert values["image_qr_area_ratio"] is None
    task = ImageBatch(
        dataset_name="test",
        data=[ImageObject(image_id="good", image_data=rgb, metadata={"source": 1}), ImageObject(image_id="bad")],
    )
    result = ImageDiagnosticsStage().process(task)
    assert [i.image_id for i in result.data] == ["good", "bad"]
    assert result.data[0].metadata["source"] == 1
    assert result.data[1].metadata["image_qr_status"] == "decode_error"


def test_nonfinite_qr_is_not_a_successful_full_image_detection():
    class NonfiniteDetector:
        def detectMulti(self, image: np.ndarray) -> tuple[bool, np.ndarray]:  # noqa: N802
            return True, np.full((1, 4, 2), np.nan, dtype=np.float32)

    result = image_diagnostics(np.zeros((20, 20, 3), dtype=np.uint8), NonfiniteDetector())
    assert result["image_qr_area_ratio"] is None
    assert result["image_qr_status"] == "compute_error"
    assert result["image_sharpness_status"] == "ok"


def test_selective_diagnostics_skip_qr_and_release_pixels(monkeypatch: pytest.MonkeyPatch):
    def forbidden() -> Never:
        message = "QR detector must not be constructed"
        raise AssertionError(message)

    monkeypatch.setattr(cv2, "QRCodeDetectorAruco", forbidden)
    stage = ImageDiagnosticsStage(metrics=("sharpness",), remove_image_data=True)
    stage.setup()
    task = ImageBatch(
        dataset_name="test",
        data=[
            ImageObject(image_data=np.zeros((20, 20, 3), dtype=np.uint8), metadata={"retained": True}),
            ImageObject(image_id="bad"),
        ],
    )
    result = stage.process(task)
    assert result.data[0].metadata["image_sharpness_score"] == 0
    assert result.data[0].metadata["retained"]
    assert result.data[1].metadata["image_sharpness_status"] == "decode_error"
    assert all("image_qr_status" not in image.metadata for image in result.data)
    assert all(image.image_data is None for image in result.data)
