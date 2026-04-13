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

"""NemotronOCR-v2 word-level dense OCR stage (English / no-text route)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

from nemo_curator.stages.resources import Resources
from nemo_curator.stages.synthetic.omni.base import VLMProcessingStage
from nemo_curator.stages.synthetic.omni.io import load_image_from_task
from nemo_curator.tasks.image import SingleDataTask
from nemo_curator.tasks.ocr import OCRData, OCRDenseWord

_HF_REPO_ID = "nvidia/nemotron-ocr-v2"
_DEFAULT_SUBDIR = "v2_english"

# NemotronOCR-v2 uses a non-standard coordinate system: y=0 is at the *bottom*
# of the image (PDF convention), so (left, upper, right, lower) with upper > lower
# means the text is near the top of the image.  We flip the y-axis when converting
# to the standard OCRDenseWord [x1, y1, x2, y2] schema where y=0 is at the top.


def _to_ocr_dense_word(pred: dict[str, Any]) -> OCRDenseWord:
    """Convert a NemotronOCR-v2 prediction dict to OCRDenseWord (0-1000 coords)."""
    x1 = int(pred["left"] * 1000)
    x2 = int(pred["right"] * 1000)
    # Flip y-axis: nemotron gives y=0 at bottom; OCRDenseWord uses y=0 at top.
    y1 = int((1.0 - pred["upper"]) * 1000)
    y2 = int((1.0 - pred["lower"]) * 1000)
    return OCRDenseWord(
        bbox_2d=[x1, y1, x2, y2],
        text_content=str(pred["text"]),
    )


class OCRNemotronV2Stage(VLMProcessingStage[OCRData]):
    """Word-level dense OCR using NemotronOCR-v2.

    Only processes tasks routed to "rtx" by OCRLanguageRoutingStage.
    Tasks with any other route (qwen, skip, None) are passed through unchanged.

    The ``model_dir`` parameter should point to the ``v2_english`` (or
    ``v2_multilingual``) directory inside the NemotronOCR-v2 model checkout.
    If ``model_dir`` is None the stage downloads the model from HuggingFace on
    first use (``nvidia/nemotron-ocr-v2``).

    Output field ``ocr_qwen_dense`` is populated with a list of
    ``{"bbox_2d": [x1, y1, x2, y2], "text_content": "..."}`` entries
    (normalized 0-1000 coordinates, y=0 at top).

    Note: The ``nemotron-ocr`` package must be installed in the runtime
    environment.  See installation instructions in the project SKILL.md.
    """

    name = "ocr_nemotron_v2"
    resources = Resources(cpus=8.0, gpus=1)
    batch_size = 32  # NemotronOCRV2 supports detector batching

    def __init__(
        self,
        model_dir: str | Path | None = None,
        num_workers: int | None = None,
        merge_level: str = "word",
        **kwargs: Any,
    ) -> None:
        """Initialize the NemotronOCR-v2 stage.

        Args:
            model_dir: Path to the model directory (e.g. ``<repo>/v2_english``).
                If None, the model is downloaded from HuggingFace on ``setup()``.
            num_workers: If set, the number of Xenna workers for this stage.
                If None, the Xenna autoscaler decides.
            merge_level: NemotronOCR merge level — "word", "sentence", or
                "paragraph".  Defaults to "word" for compatibility with the
                Qwen OCR output schema.
            **kwargs: Additional arguments forwarded to VLMProcessingStage.
        """
        super().__init__(**kwargs)
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.num_workers = num_workers
        self.merge_level = merge_level
        self._model: Any = None  # NemotronOCRV2, loaded in setup()

    def _resolve_model_dir(self) -> str:
        if self.model_dir is not None:
            return str(self.model_dir)
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(repo_id=_HF_REPO_ID)
        return str(Path(snapshot) / _DEFAULT_SUBDIR)

    def xenna_stage_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if self.num_workers is not None:
            spec["num_workers"] = self.num_workers
        return spec

    def setup(self, worker_metadata: dict | None = None) -> None:
        """Load NemotronOCRV2 onto the GPU."""
        self._maybe_set_cuda_device()
        from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2

        model_dir = self._resolve_model_dir()
        logger.info(f"{self.name}: loading model from {model_dir}")
        self._model = NemotronOCRV2(model_dir=model_dir)
        logger.info(f"{self.name}: model loaded")

    def process_batch(self, tasks: list[SingleDataTask[OCRData]]) -> list[SingleDataTask[OCRData]]:
        """Run NemotronOCR-v2 on all "rtx"-routed tasks in the batch."""
        for task in tasks:
            if not task.data.is_valid:
                continue
            if task.data.ocr_language_route != "rtx":
                continue  # pass through qwen/skip tasks unchanged

            try:
                self._process_one(task)
            except Exception as e:
                logger.error(f"{self.name}: error on task {task.task_id}: {e}")
                task.data.is_valid = False
                task.data.error = f"{self.name}: {e}"

        return tasks

    def _process_one(self, task: SingleDataTask[OCRData]) -> None:
        """Run inference on a single image task (mutates task in-place)."""
        image = load_image_from_task(task)

        # NemotronOCRV2 takes a file path — write to a temp file.
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            image.save(tmp_path, format="JPEG")
            preds = self._model(tmp_path, merge_level=self.merge_level)
        finally:
            os.unlink(tmp_path)

        task.data.ocr_qwen_dense = [_to_ocr_dense_word(p) for p in preds]

    def teardown(self) -> None:
        """Release GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
