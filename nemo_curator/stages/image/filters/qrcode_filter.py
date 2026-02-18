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

from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.image.filters.base import BaseFilterStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import ImageBatch


def _qr_code_ratio(image: np.ndarray) -> float:
    """Return the ratio of image area covered by all detected QR code(s), in [0, 1]."""
    height, width = image.shape[:2]
    img_area = float(height * width)
    if img_area <= 0:
        return 0.0
    detector = cv2.QRCodeDetector()
    retval, _decoded_info, points, _ = detector.detectAndDecodeMulti(image)
    if not retval or points is None or points.size == 0:
        data, points, _ = detector.detectAndDecode(image)
        if not data or points is None or points.size == 0:
            return 0.0
        points = [np.asarray(points, dtype=np.float32)]
    points = np.asarray(points, dtype=np.float32)
    total_qr_area = 0.0
    for i in range(len(points)):
        pts = points[i].reshape(-1, 1, 2)
        total_qr_area += cv2.contourArea(pts)
    return total_qr_area / img_area


@dataclass
class ImageQRCodeFilterStage(BaseFilterStage):
    """Stage for filtering out images by QR code area ratio.

    Expects ImageObject.image_data to be set (e.g. by the image reader stage).
    Detects one or multiple QR codes per image via detectAndDecodeMulti. Sets
    qr_score to the ratio of image area covered by all detected
    QR codes in [0, 1]. Images with qr_score >= score_threshold are filtered out.
    """

    model_dir: str = None
    num_gpus_per_worker: float = 0.25
    model_inference_batch_size: int = 32
    score_threshold: float = 0.05
    verbose: bool = False
    name: str = "image_qrcode_filter"

    def __post_init__(self) -> None:
        self.resources = Resources()

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self.verbose:
            logger.info("ImageQRCodeFilterStage ready (CPU QR detection)")

    def process(self, task: ImageBatch) -> ImageBatch:
        for batch in self.yield_next_batch(task):
            for image_obj in batch:
                image_obj.qr_score = _qr_code_ratio(image_obj.image_data)

            if self.verbose:
                logger.info(f"Computed QR ratio for {len(batch)} images in batch")

        filtered_images = []
        filtered_count = 0
        for image_obj in task.data:
            if image_obj.qr_score < self.score_threshold:
                filtered_images.append(image_obj)
            else:
                filtered_count += 1
                if self.verbose:
                    logger.info(
                        f"Image {image_obj.image_id} (path: {image_obj.image_path}) has QR ratio "
                        f"{image_obj.qr_score:.4f} >= {self.score_threshold}, filtered out."
                    )

        if self.verbose:
            logger.info(
                f"QR code filtering: {len(filtered_images)}/{len(task.data)} images passed, "
                f"{filtered_count} filtered out"
            )

        return ImageBatch(
            data=filtered_images,
            dataset_name=task.dataset_name,
            task_id=f"{task.task_id}_{self.name}",
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


__all__ = ["ImageQRCodeFilterStage"]
