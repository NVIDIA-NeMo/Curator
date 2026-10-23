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

"""NemotronOCR-v2 word-level dense OCR stage (English / no-text route)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from PIL import Image

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks.image import ImageSampleTask
from nemo_curator.tasks.ocr import OCRData, OCRDenseItem

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

_HF_REPO_ID = "nvidia/nemotron-ocr-v2"
_DEFAULT_SUBDIR = "v2_multilingual"


def _to_ocr_dense_item(pred: dict[str, Any]) -> OCRDenseItem:
    """Convert a NemotronOCR-v2 prediction dict to OCRDenseItem (0-1000 coords).

    NemotronOCR-v2 uses screen coordinates (y=0 at top) but with inverted naming:
    ``lower`` holds the *smaller* y value (top edge) and ``upper`` holds the
    *larger* y value (bottom edge).  We sort to ensure y1 <= y2.
    """
    x1 = int(pred["left"] * 1000)
    x2 = int(pred["right"] * 1000)
    y1 = int(min(pred["upper"], pred["lower"]) * 1000)
    y2 = int(max(pred["upper"], pred["lower"]) * 1000)
    return OCRDenseItem(
        bbox_2d=[x1, y1, x2, y2],
        text_content=str(pred["text"]),
    )


class OCRNemotronV2Stage(ProcessingStage[ImageSampleTask[OCRData], ImageSampleTask[OCRData]]):
    """Word-level dense OCR using NemotronOCR-v2 (multilingual).

    ``model_dir`` should point to the ``v2_multilingual`` directory of the
    NemotronOCR-v2 checkout; if None, it's downloaded from HuggingFace on
    ``setup()``. Populates ``ocr_dense`` with
    ``{"bbox_2d": [x1, y1, x2, y2], "text_content": "..."}`` entries
    (normalized 0-1000 coordinates, y=0 at top). Requires the ``nemotron-ocr``
    package in the runtime environment.
    """

    name = "ocr_nemotron_v2"
    resources = Resources(cpus=8.0, gpus=1)
    batch_size = 32  # NemotronOCRV2 supports detector batching

    def __init__(
        self,
        model_dir: str | Path | None = None,
        merge_level: str = "word",
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.merge_level = merge_level
        self._model: Any = None

    def _resolve_model_dir(self) -> str:
        if self.model_dir is not None:
            return str(self.model_dir)
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(repo_id=_HF_REPO_ID)
        return str(Path(snapshot) / _DEFAULT_SUBDIR)

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        """Pre-download weights once per node so concurrent workers don't race."""
        if self.model_dir is None:
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=_HF_REPO_ID)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2

        model_dir = self._resolve_model_dir()
        logger.info(f"{self.name}: loading model from {model_dir}")
        self._model = NemotronOCRV2(model_dir=model_dir)
        logger.info(f"{self.name}: model loaded")

    def process(self, task: ImageSampleTask[OCRData]) -> ImageSampleTask[OCRData]:
        msg = "OCRNemotronV2Stage does not support single-task processing; use process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[ImageSampleTask[OCRData]]) -> list[ImageSampleTask[OCRData]]:
        for task in tasks:
            if not task.data.is_valid:
                continue
            try:
                self._process_one(task)
            except Exception as e:  # noqa: BLE001
                logger.error(f"{self.name}: error on task {task.task_id}: {e}")
                task.data.is_valid = False
                task.data.error = f"{self.name}: {e}"

        return tasks

    def _process_one(self, task: ImageSampleTask[OCRData]) -> None:
        image = Image.open(task.data.image_path)

        # NemotronOCRV2 takes a file path.
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(tmp_path, format="JPEG")
            preds = self._model(tmp_path, merge_level=self.merge_level)
        finally:
            os.unlink(tmp_path)

        task.data.ocr_dense = [_to_ocr_dense_item(p) for p in preds]

    def teardown(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
