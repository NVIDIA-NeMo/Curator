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
from multiprocessing import Process, Queue

import cv2
import numpy as np
import pandas as pd

from nemo_curator.stages.interleaved.stages import BaseInterleavedFilterStage
from nemo_curator.tasks import InterleavedBatch


def _sharpness_score(image: np.ndarray) -> float:
    """Compute Laplacian variance as sharpness score; higher is sharper."""
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def _image_bytes_to_array(image_bytes: bytes) -> np.ndarray | None:
    """Decode image bytes to RGB numpy array for OpenCV, or None on failure."""
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _process_one_blur(item: tuple, score_threshold: float) -> tuple:
    """Process one (idx, image_bytes); return (idx, keep)."""
    idx, image_bytes = item
    if image_bytes is None:
        return (idx, False)
    image = _image_bytes_to_array(image_bytes)
    if image is None:
        return (idx, False)
    return (idx, _sharpness_score(image) >= score_threshold)


def _worker_blur(
    work_queue: Queue,
    result_queue: Queue,
    score_threshold: float,
) -> None:
    """Module-level worker for multiprocessing; runs until sentinel."""
    while True:
        item = work_queue.get()
        idx, image_bytes = item
        _, keep = _process_one_blur((idx, image_bytes), score_threshold)
        result_queue.put((idx, keep))


@dataclass
class InterleavedBlurFilterStage(BaseInterleavedFilterStage):
    """Filter interleaved image rows by sharpness (Laplacian variance); drop blurry images."""

    score_threshold: float = 100.0
    image_content_types: tuple[str, ...] = ("image/jpeg", "image/jpg", "image/png")
    max_workers: int | None = None
    name: str = "interleaved_blur_filter"

    def _run_workers(
        self,
        task: InterleavedBatch,
        df: pd.DataFrame,
        image_mask: pd.Series,
        keep_mask: pd.Series,
    ) -> None:
        """Run blur filter in worker processes; updates keep_mask in place."""
        num_items = int(image_mask.sum())
        work_queue = Queue()
        result_queue = Queue()

        processes = [
            Process(
                target=_worker_blur,
                args=(work_queue, result_queue, self.score_threshold),
            )
            for _ in range(self.max_workers)
        ]
        for p in processes:
            p.start()
        for idx, image_bytes in self.iter_materialized_bytes(task=task, df=df, row_mask=image_mask):
            work_queue.put((idx, image_bytes))
        for _ in range(num_items):
            i, keep = result_queue.get()
            keep_mask.loc[i] = keep
        for p in processes:
            p.terminate()
            p.join()

    def content_keep_mask(self, task: InterleavedBatch, df: pd.DataFrame) -> pd.Series:
        keep_mask = pd.Series(True, index=df.index, dtype=bool)
        image_mask = (df["modality"] == "image") & (df["content_type"].isin(self.image_content_types))
        if not image_mask.any():
            return keep_mask
        if self.max_workers is not None and self.max_workers > 1:
            self._run_workers(task, df, image_mask, keep_mask)
        else:
            for idx, image_bytes in self.iter_materialized_bytes(task=task, df=df, row_mask=image_mask):
                _, keep = _process_one_blur((idx, image_bytes), self.score_threshold)
                keep_mask.loc[idx] = keep
        return keep_mask
