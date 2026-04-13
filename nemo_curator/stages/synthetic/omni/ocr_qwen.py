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

"""Qwen3-VL word-level dense OCR stage."""

from __future__ import annotations

import json
import re
from typing import Any

from nemo_curator.models.omni.base import InferenceConfig, VLLMModelConfig
from nemo_curator.models.omni.qwen3_vl import Qwen3VL
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.synthetic.omni.base import ModelProcessingStage, SkipSample
from nemo_curator.tasks.image import SingleDataTask
from nemo_curator.tasks.ocr import OCRData, OCRDenseWord


_OCR_PROMPT = (
    "Transcribe all text in the image at word-level, and output in JSON format "
    "as [{'bbox_2d': [x1, y1, x2, y2], 'text_content': 'text'}, ...]. Use "
    "normalized coordinates between 0 and 1000. Return [] if no words are present."
)

_JSON_LIST_PATTERN = re.compile(r"\[.*\]", re.DOTALL)


def _parse_json_list(response: str) -> list[dict[str, Any]] | None:
    """Return the first JSON list found in *response*, or None."""
    for match in _JSON_LIST_PATTERN.finditer(response):
        candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed
    return None


class OCRQwenStage(ModelProcessingStage[OCRData]):
    """Word-level dense OCR using Qwen3-VL.

    Only processes tasks routed to "qwen" by OCRLanguageRoutingStage.
    Tasks with any other route (rtx, skip, None) are passed through unchanged
    via SkipSample.

    Output field ``ocr_qwen_dense`` is set to a list of
    ``{"bbox_2d": [x1, y1, x2, y2], "text_content": "..."}`` dicts
    (normalized 0–1000 coordinates).
    """

    name = "ocr_qwen"
    resources = Resources(cpus=8.0, gpus=1)
    batch_size = 8

    def __init__(self, num_workers: int | None = None, model_id: str | None = None, **kwargs: Any) -> None:
        model_kwargs: dict[str, Any] = {}
        if model_id is not None:
            model_kwargs["model_id"] = model_id
        super().__init__(
            model=Qwen3VL(
                model_config=VLLMModelConfig(
                    gpu_memory_utilization=self._get_gpu_memory_utilization(),
                    tensor_parallel_size=self._get_tensor_parallel_size(),
                    max_tokens=8192,
                ),
                **model_kwargs,
            ),
            inference_config=InferenceConfig(
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
            ),
            batch_size=self.batch_size,
            **kwargs,
        )
        self.num_workers = num_workers

    def xenna_stage_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if self.num_workers is not None:
            spec["num_workers"] = self.num_workers
        return spec

    def build_prompt(self, task: SingleDataTask[OCRData]) -> str:
        if task.data.ocr_language_route != "qwen":
            raise SkipSample
        task.data.ocr_is_word_level = True
        task.data.ocr_qwen_dense_prompt = _OCR_PROMPT
        return _OCR_PROMPT

    def handle_response(
        self, task: SingleDataTask[OCRData], response: str
    ) -> SingleDataTask[OCRData]:
        raw = _parse_json_list(response)
        if raw is None:
            task.data.is_valid = False
            task.data.error = f"{self.name}: could not parse JSON list from response: {response!r}"
            return task

        task.data.ocr_qwen_dense = [
            OCRDenseWord.from_dict(item) if isinstance(item, dict) else item
            for item in raw
        ]
        return task
