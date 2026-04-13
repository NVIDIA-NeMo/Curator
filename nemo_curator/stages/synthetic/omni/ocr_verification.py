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

"""OCR verification stage — validates dense OCR bounding boxes using Gemini 3 Pro."""

from __future__ import annotations

import json
import re
from typing import Any

from PIL import Image

from nemo_curator.models.omni.base import InferenceConfig, NVInferenceModelConfig
from nemo_curator.models.omni.gemini import Gemini3Pro
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.synthetic.omni.base import ModelProcessingStage, SkipSample
from nemo_curator.tasks.image import SingleDataTask
from nemo_curator.tasks.ocr import OCRData


_JSON_LIST_PATTERN = re.compile(r"\[.*\]", re.DOTALL)

_QUESTION_CHECKS_WORD: list[tuple[str, str]] = [
    ("Does each bounding box text have only one word?", "yes"),
    ("Does each bounding box correspond to the exact same word in the image?", "yes"),
    ("Are all bounding boxes placed accurately?", "yes"),
    ("Are there any words in the image that are not covered by a bounding box?", "no"),
]

_QUESTION_CHECKS_EMPTY: list[tuple[str, str]] = [
    ("Is there any text in the image?", "no"),
    ("Is it correct that there should be no text bounding boxes returned?", "yes"),
    ("Are there any doubts you have about the absence of text?", "no"),
]


def _parse_answers_list(response: str) -> list[dict[str, str]] | None:
    """Parse [{'question': str, 'answer': 'yes'|'no'}, ...] from response."""
    for match in _JSON_LIST_PATTERN.finditer(response):
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, list):
            continue
        if all(
            isinstance(item, dict)
            and isinstance(item.get("question"), str)
            and isinstance(item.get("answer"), str)
            for item in parsed
        ):
            return [{"question": item["question"], "answer": item["answer"]} for item in parsed]
    return None


def _x_first_to_y_first(bbox: list[int]) -> list[int]:
    """Convert [x1, y1, x2, y2] → [y1, x1, y2, x2] for Gemini prompt."""
    x1, y1, x2, y2 = bbox
    return [y1, x1, y2, x2]


class OCRVerificationStage(ModelProcessingStage[OCRData]):
    """Verify dense OCR bounding boxes using Gemini 3 Pro via NVIDIA Inference API.

    Skips tasks where ``ocr_qwen_dense`` is None (OCR stage has not run yet).
    Marks tasks invalid if Gemini's yes/no answers don't match expected values.

    Reads:  ``ocr_qwen_dense``, ``ocr_is_word_level``, ``ocr_has_text``
    Writes: ``ocr_verification_prompt``, ``ocr_verification_model``,
            ``ocr_verification_response_raw``, ``ocr_verification_answers``,
            ``is_valid``, ``error``
    """

    name = "ocr_verification"
    resources = Resources(cpus=1.0)
    batch_size = 16
    multimodal = True

    def __init__(
        self,
        model_id: str = "gcp/google/gemini-3-pro",
        temperature: float = 1.0,
        max_tokens: int = 2048,
        batch_size: int | None = None,
        priority_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        self._verification_model_id = model_id
        super().__init__(
            model=Gemini3Pro(
                model_config=NVInferenceModelConfig(max_tokens=max_tokens),
                model_id=model_id,
            ),
            inference_config=InferenceConfig(
                temperature=temperature,
                top_p=1.0,
                do_sample=False,
                priority_mode=priority_mode,
            ),
            batch_size=batch_size or self.batch_size,
            **kwargs,
        )

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {}

    def load_image(self, task: SingleDataTask[OCRData]) -> Image.Image:
        """Load image — NVInferenceModel handles base64 encoding internally."""
        from nemo_curator.stages.synthetic.omni.io import load_image_from_task
        return load_image_from_task(task)

    def build_prompt(self, task: SingleDataTask[OCRData]) -> str:
        ocr_items = task.data.ocr_qwen_dense

        # Skip tasks that haven't been through any OCR stage yet.
        if ocr_items is None:
            raise SkipSample

        is_empty = len(ocr_items) == 0

        # Sanity check: if routing said there's text but OCR returned nothing, invalidate.
        if task.data.ocr_has_text is not None:
            expected_empty = not task.data.ocr_has_text
            if is_empty != expected_empty:
                task.data.is_valid = False
                task.data.error = (
                    f"ocr_verification: mismatch ocr_has_text={task.data.ocr_has_text} "
                    f"but ocr_qwen_dense={'empty' if is_empty else 'non-empty'}"
                )
                raise SkipSample

        if is_empty:
            question_checks = _QUESTION_CHECKS_EMPTY
            bboxes_for_prompt: list[dict[str, Any]] = []
        else:
            question_checks = _QUESTION_CHECKS_WORD

            bboxes_for_prompt = []
            for item in ocr_items:
                bbox = item.bbox_2d if hasattr(item, "bbox_2d") else item.get("bbox_2d")
                text = item.text_content if hasattr(item, "text_content") else item.get("text_content", "")
                if bbox is None or len(bbox) != 4:
                    continue
                # Gemini prompt uses y-first order: [y1, x1, y2, x2]
                bboxes_for_prompt.append({
                    "bbox_2d": _x_first_to_y_first(list(bbox)),
                    "text": str(text or ""),
                })

            if not bboxes_for_prompt:
                task.data.is_valid = False
                task.data.error = "ocr_verification: no valid bbox entries to verify"
                raise SkipSample

        # Build pseudo-JSON example showing expected output format.
        lines = ["["]
        for idx, (question, _) in enumerate(question_checks):
            comma = "," if idx < len(question_checks) - 1 else ""
            lines += [
                "  {",
                f'    "question": "{question}",',
                '    "answer": "yes"  # or "no"',
                "  }" + comma,
            ]
        lines.append("]")
        example = "\n".join(lines)

        if is_empty:
            prompt = (
                "The OCR system returned an empty list (no bounding boxes). "
                "Please verify whether this is correct and answer each question "
                "in the following JSON format:\n"
                f"{example}\n\n"
                "Returned bounding boxes (expected empty list if no text):\n"
                f"{json.dumps(bboxes_for_prompt, ensure_ascii=False)}\n\n"
                "Only output JSON."
            )
        else:
            prompt = (
                "Please check if the following OCR bounding boxes are correct and answer "
                "each of the questions in the following JSON format:\n"
                f"{example}\n\n"
                "Bounding boxes to check (bbox_2d is [y1, x1, y2, x2] on a 0-1000 normalized grid):\n"
                f"{json.dumps(bboxes_for_prompt, ensure_ascii=False)}\n\n"
                "Only output JSON."
            )

        task.data.ocr_verification_prompt = prompt
        task.data.ocr_verification_model = self._verification_model_id
        return prompt

    def handle_response(
        self, task: SingleDataTask[OCRData], response: str
    ) -> SingleDataTask[OCRData]:
        if not response:
            task.data.is_valid = False
            task.data.error = "ocr_verification: empty response from model"
            return task

        task.data.ocr_verification_response_raw = response

        answers = _parse_answers_list(response)
        if answers is None:
            task.data.is_valid = False
            task.data.error = f"ocr_verification: could not parse JSON answers: {response!r}"
            return task

        task.data.ocr_verification_answers = answers

        is_empty = task.data.ocr_qwen_dense is not None and len(task.data.ocr_qwen_dense) == 0
        question_checks = _QUESTION_CHECKS_EMPTY if is_empty else _QUESTION_CHECKS_WORD

        by_question = {item["question"].strip(): item["answer"].strip().lower() for item in answers}

        for question, expected in question_checks:
            got = by_question.get(question)
            if got not in {"yes", "no"} or got != expected:
                task.data.is_valid = False
                task.data.error = f"ocr_verification: answers wrong: {answers!r}"
                break

        return task
