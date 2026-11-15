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

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import ImageBatch


def decode_image_rgb(payload: bytes) -> np.ndarray:
    """Decode frame zero, apply EXIF orientation and composite alpha over white.

    Returns native-resolution, contiguous uint8 RGB pixels. Decode failures
    propagate to the caller; truncated images are not silently accepted.
    """
    with Image.open(io.BytesIO(payload)) as source:
        source.seek(0)
        image = ImageOps.exif_transpose(source)
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba)
        return np.array(image.convert("RGB"), dtype=np.uint8)


def image_diagnostics(
    rgb: np.ndarray, detector: cv2.QRCodeDetectorAruco | None = None, *, metrics: tuple[str, ...] = ("sharpness", "qr")
) -> dict[str, int | float | str | None]:
    """Annotate native RGB pixels without dropping an image on scorer failure.

    Sharpness is grayscale Laplacian variance (CV_64F, ksize=1).
    QR coverage is min(1, sum(detected quadrilateral areas) / image area).
    Detection does not require successfully decoding QR contents; overlapping
    quadrilaterals may double-count area before the cap.
    """
    if set(metrics) - {"sharpness", "qr"}:
        message = "Supported image diagnostics are sharpness and qr"
        raise ValueError(message)
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:  # noqa: PLR2004
        message = "Expected uint8 RGB pixels with shape (height, width, 3)"
        raise ValueError(message)
    height, width = rgb.shape[:2]
    if not height or not width:
        message = "Image dimensions must be positive"
        raise ValueError(message)
    result = {"image_width": width, "image_height": height}
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if metrics else None
    if "sharpness" in metrics:
        result.update(image_sharpness_score=None, image_sharpness_status="compute_error")
        try:
            value = float(cv2.Laplacian(gray, cv2.CV_64F, ksize=1).var())
            if math.isfinite(value):
                result.update(image_sharpness_score=value, image_sharpness_status="ok")
        except (cv2.error, ValueError, FloatingPointError):
            pass
    if "qr" in metrics:
        result.update(image_qr_area_ratio=None, image_qr_status="compute_error")
        try:
            found, points = (detector or cv2.QRCodeDetectorAruco()).detectMulti(gray)
            area = sum(abs(cv2.contourArea(np.asarray(p, dtype=np.float32))) for p in points) if found else 0.0
            if math.isfinite(area):
                result.update(image_qr_area_ratio=min(1.0, area / (height * width)), image_qr_status="ok")
        except (cv2.error, ValueError, TypeError, FloatingPointError):
            pass
    return result


@dataclass
class ImageDiagnosticsStage(ProcessingStage[ImageBatch, ImageBatch]):
    """Add dimensions, sharpness and QR coverage to each image's metadata.

    Accepts decoded native uint8 RGB ``image_data``. The decoder's frame,
    orientation and alpha policy belongs to the input recipe. Use
    :func:`decode_image_rgb` for frame-zero, EXIF-aware, white-alpha decoding.
    Every input image remains in the returned batch, in the original order.
    """

    metrics: tuple[str, ...] = ("sharpness", "qr")
    remove_image_data: bool = False
    name: str = "image_diagnostics"
    resources = Resources(cpus=1.0)

    def __post_init__(self) -> None:
        if set(self.metrics) - {"sharpness", "qr"}:
            message = "Supported image diagnostics are sharpness and qr"
            raise ValueError(message)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: ImageBatch) -> ImageBatch:
        for image in task.data:
            try:
                values = image_diagnostics(image.image_data, metrics=self.metrics)
            except (ValueError, cv2.error):
                values = {"image_width": None, "image_height": None}
                for metric, score in [("sharpness", "image_sharpness_score"), ("qr", "image_qr_area_ratio")]:
                    if metric in self.metrics:
                        values.update({score: None, f"image_{metric}_status": "decode_error"})
            image.metadata.update(values)
            if self.remove_image_data:
                image.image_data = None
        return task
