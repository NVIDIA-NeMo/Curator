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

"""Language detection + routing stage for the OCR mixed dense pipeline.

Uses Qwen3-VL to decide which OCR backend should handle each image:
  "qwen"  -> Chinese-only text  (handled by OCRQwenStage)
  "rtx"   -> English-only or no text  (handled by OCRRTXStage, to be added)
  "skip"  -> Mixed or unsupported languages  (marked invalid, no OCR attempted)

Images routed to "rtx" are NOT marked invalid — they pass through the pipeline
with their label intact so a future RTX/nemotron-ocr stage can pick them up.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from nemo_curator.models.omni.base import InferenceConfig, VLLMModelConfig
from nemo_curator.models.omni.qwen3_vl import Qwen3VL
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.synthetic.omni.base import ModelProcessingStage
from nemo_curator.tasks.image import SingleDataTask
from nemo_curator.tasks.ocr import OCRData


_ROUTE_PROMPT = (
    "You are given an image. Determine which text languages are present.\n\n"
    "Return a JSON object with ONLY these boolean keys:\n"
    '  {"has_text": true|false, "has_chinese": true|false, "has_english": true|false, "has_other": true|false}\n\n'
    "Definitions:\n"
    "- has_text: there is any readable text.\n"
    "- has_chinese: any Chinese characters present.\n"
    "- has_english: any English (Latin alphabet) words present.\n"
    "- has_other: any other language/script present (e.g., Japanese, Korean, Arabic, Cyrillic, etc.).\n\n"
    "Important:\n"
    "- If there is no text, set has_text=false and all others false.\n"
    "- If both Chinese and English are present, set both true.\n"
    "- Output ONLY valid JSON."
)


def _first_json_object(text: str) -> dict[str, Any] | None:
    """Return the first valid JSON object found in *text*, if any."""
    in_string = False
    escape = False
    depth = 0
    start: int | None = None

    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth <= 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    start = None
                    continue
                if isinstance(parsed, dict):
                    return parsed
                start = None

    return None


@dataclass(frozen=True)
class _LangFlags:
    has_text: bool
    has_chinese: bool
    has_english: bool
    has_other: bool


def _coerce_flags(obj: dict[str, Any]) -> _LangFlags | None:
    """Parse and validate the boolean flags from the model's JSON response.

    Returns None if any required key is missing or not a plain bool.
    """
    def _to_bool(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        return None

    has_text = _to_bool(obj.get("has_text"))
    has_chinese = _to_bool(obj.get("has_chinese"))
    has_english = _to_bool(obj.get("has_english"))
    has_other = _to_bool(obj.get("has_other"))

    if any(v is None for v in (has_text, has_chinese, has_english, has_other)):
        return None

    # Any positive language flag implies text is present.
    if has_chinese or has_english or has_other:
        has_text = True

    return _LangFlags(
        has_text=has_text,
        has_chinese=has_chinese,
        has_english=has_english,
        has_other=has_other,
    )


def _route_for(flags: _LangFlags) -> tuple[str, str | None]:
    """Return (route, error_message_if_skip).

    Routes:
      "qwen"  -> Chinese-only
      "rtx"   -> English-only or no text
      "skip"  -> unsupported (other language, or mixed Chinese+English)
    """
    if flags.has_other:
        return "skip", "ocr_language_route: other/unsupported language present"
    if flags.has_chinese and flags.has_english:
        return "skip", "ocr_language_route: mixed Chinese+English not supported"
    if flags.has_chinese:
        return "qwen", None
    # English-only or no text: both go to RTX backend.
    return "rtx", None


class OCRLanguageRoutingStage(ModelProcessingStage[OCRData]):
    """Label each image with an OCR backend route using Qwen3-VL language detection.

    Output field ``ocr_language_route`` is set to one of:
      "qwen"  -> downstream OCRQwenStage will process this image
      "rtx"   -> downstream OCRRTXStage will process this image (future)
      "skip"  -> no OCR backend can handle this language combination

    Only "skip" tasks are marked ``is_valid=False``.  "rtx" tasks pass through
    unmodified so a future RTX stage can pick them up later.
    """

    name = "ocr_language_route"
    resources = Resources(cpus=8.0, gpus=1)
    batch_size = 16

    def __init__(self, num_workers: int | None = None, model_id: str | None = None, **kwargs: Any) -> None:
        """Initialize the language routing stage.

        Args:
            num_workers: If set, the number of Xenna workers for this stage.
                If None, the Xenna autoscaler decides.
            model_id: HuggingFace model ID override. Defaults to Qwen3-VL-32B.
            **kwargs: Additional arguments forwarded to ModelProcessingStage.
        """
        model_kwargs = {}
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
        task.data.ocr_language_route_prompt = _ROUTE_PROMPT
        return _ROUTE_PROMPT

    def handle_response(
        self, task: SingleDataTask[OCRData], response: str
    ) -> SingleDataTask[OCRData]:
        task.data.ocr_language_route_response_raw = response

        parsed_obj = _first_json_object(response)
        if parsed_obj is None:
            task.data.is_valid = False
            task.data.error = f"{self.name}: could not parse JSON from response: {response!r}"
            task.data.ocr_language_route = "skip"
            return task

        flags = _coerce_flags(parsed_obj)
        if flags is None:
            task.data.is_valid = False
            task.data.error = f"{self.name}: JSON missing or non-boolean keys: {parsed_obj!r}"
            task.data.ocr_language_route = "skip"
            return task

        task.data.ocr_has_text = flags.has_text
        task.data.ocr_has_chinese = flags.has_chinese
        task.data.ocr_has_english = flags.has_english
        task.data.ocr_has_other_language = flags.has_other

        route, skip_error = _route_for(flags)
        task.data.ocr_language_route = route

        if route == "skip":
            task.data.is_valid = False
            task.data.error = skip_error

        return task
