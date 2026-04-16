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

"""OCR scoring verification stage — per-bbox quality scoring via Gemini."""

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


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

_PROMPT = """\
Please check if the following OCR bounding boxes are correct and respond ONLY with JSON \
in this exact format:
{{
  "ocr_mode": "word" or "line",
  "text": [
    {{
      "idx": <integer matching input idx>,
      "is_word": <true if bbox covers a single word>,
      "is_line": <true if bbox covers a full line, phrase, or sentence>,
      "bbox_match": <0-10>,
      "text_errors": <integer>
    }}
  ],
  "missing_text": [
    {{
      "text": "<transcribed text>",
      "bbox_2d": [y1, x1, y2, x2]
    }}
  ]
}}

Scoring guide:
- ocr_mode: set to "word" if every bbox covers a single word; "line" if bboxes cover \
phrases, lines, or sentences
- bbox_match: 10 = bbox fits tightly around the text; 5 = bbox is ~1 character too \
large/small/shifted; 0 = completely wrong position or size
- text_errors: 0 = transcription matches the image exactly; count each substitution, \
insertion, or deletion as 1 error
- missing_text: list every legible text region visible in the image that is NOT covered \
by any of the provided bounding boxes, together with its estimated bbox_2d

Text and bounding boxes to check (bbox_2d is [y1, x1, y2, x2] on a 0-1000 normalised grid):
{bboxes_json}

Only output valid JSON."""


def _parse_json_object(text: str) -> dict | None:
    """Return the first JSON object found in *text*, stripping markdown fences."""
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    for match in _JSON_OBJECT_RE.finditer(cleaned):
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


class OCRScoringVerificationStage(ModelProcessingStage[OCRData]):
    """Per-bbox quality scoring using Gemini instead of global yes/no questions.

    Unlike ``OCRVerificationStage``, this stage:

    * Assigns each bbox a ``bbox_match`` score (0–10) and a ``text_errors`` count.
    * Sets ``word.valid = False`` on bboxes that fall below the quality thresholds,
      so ``OCRConversationalizeStage`` silently drops them without discarding the
      whole image.
    * Collects ``missing_text`` — text regions visible in the image but absent from
      the OCR output.
    * Infers whether the model ran in word-mode or line-mode and stores the result
      in ``ocr_scoring_mode`` / ``ocr_is_word_level``.

    Works for both ``v2_english`` (word-level bboxes) and ``v2_multilingual``
    (line-level bboxes) without separate question sets.

    Reads:  ``ocr_dense``
    Writes: ``ocr_scoring_prompt``, ``ocr_scoring_model``,
            ``ocr_scoring_response_raw``, ``ocr_scoring_mode``,
            ``ocr_scoring_missing``, ``ocr_is_word_level``,
            per-bbox: ``bbox_match``, ``text_errors``, ``valid``,
            ``is_valid``, ``error``
    """

    name = "ocr_scoring_verification"
    resources = Resources(cpus=1.0)
    batch_size = 16
    multimodal = True

    def __init__(
        self,
        model_id: str = "gcp/google/gemini-3-flash-preview",
        temperature: float = 1.0,
        max_tokens: int = 4096,
        min_bbox_match: int = 7,
        max_text_errors: int = 0,
        fail_on_missing_text: bool = True,
        batch_size: int | None = None,
        priority_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialise the scoring verification stage.

        Args:
            model_id: NVIDIA Inference API model to use.
            temperature: Sampling temperature (keep at 1.0 for Gemini).
            max_tokens: Max tokens for the model response.
            min_bbox_match: Minimum ``bbox_match`` score (0–10) for a bbox to be
                considered valid.  Bboxes below this threshold have ``valid``
                set to ``False``.
            max_text_errors: Maximum ``text_errors`` count for a valid bbox.
                Set to 0 to require exact transcription matches.
            fail_on_missing_text: If ``True``, mark the whole image invalid when
                Gemini reports any missing text regions.
            batch_size: Override the default batch size of 16.
            priority_mode: Use priority API queue (lower latency, higher cost).
        """
        self._scoring_model_id = model_id
        self.min_bbox_match = min_bbox_match
        self.max_text_errors = max_text_errors
        self.fail_on_missing_text = fail_on_missing_text
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
        from nemo_curator.stages.synthetic.omni.io import load_image_from_task
        return load_image_from_task(task)

    def build_prompt(self, task: SingleDataTask[OCRData]) -> str:
        ocr_items = task.data.ocr_dense
        if ocr_items is None:
            raise SkipSample

        # Build bbox list with idx; convert to Gemini's y-first [y1, x1, y2, x2]
        bboxes_for_prompt = []
        for idx, item in enumerate(ocr_items):
            bbox = item.bbox_2d if hasattr(item, "bbox_2d") else item.get("bbox_2d")
            text = item.text_content if hasattr(item, "text_content") else item.get("text_content", "")
            if bbox is None or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            bboxes_for_prompt.append({
                "idx": idx,
                "bbox_2d": [y1, x1, y2, x2],
                "text": str(text or ""),
            })

        prompt = _PROMPT.format(bboxes_json=json.dumps(bboxes_for_prompt, ensure_ascii=False))
        task.data.ocr_scoring_prompt = prompt
        task.data.ocr_scoring_model = self._scoring_model_id
        return prompt

    def handle_response(
        self, task: SingleDataTask[OCRData], response: str
    ) -> SingleDataTask[OCRData]:
        if not response:
            task.data.is_valid = False
            task.data.error = "ocr_scoring_verification: empty response from model"
            return task

        task.data.ocr_scoring_response_raw = response

        result = _parse_json_object(response)
        if result is None:
            task.data.is_valid = False
            task.data.error = (
                f"ocr_scoring_verification: could not parse JSON: {response[:200]!r}"
            )
            return task

        ocr_mode = result.get("ocr_mode", "unknown")
        text_results: list[dict] = result.get("text") or []
        missing_text: list[dict] = result.get("missing_text") or []

        task.data.ocr_scoring_mode = ocr_mode
        task.data.ocr_scoring_missing = missing_text

        # Let downstream stages (e.g. conversationalize) know the granularity
        if ocr_mode == "word":
            task.data.ocr_is_word_level = True
        elif ocr_mode == "line":
            task.data.ocr_is_word_level = False

        # Apply per-bbox scores, mark individual bboxes valid/invalid
        ocr_items = task.data.ocr_dense or []
        scores_by_idx: dict[int, dict] = {
            int(e["idx"]): e for e in text_results if "idx" in e
        }

        for i, word in enumerate(ocr_items):
            entry = scores_by_idx.get(i)
            if entry is None:
                word.valid = False
                continue
            raw_match = entry.get("bbox_match")
            raw_errors = entry.get("text_errors")
            try:
                word.bbox_match = int(raw_match)
                word.text_errors = int(raw_errors)
            except (TypeError, ValueError):
                word.valid = False
                continue
            word.valid = (
                word.bbox_match >= self.min_bbox_match
                and word.text_errors <= self.max_text_errors
            )

        # Image-level validity checks
        valid_words = [w for w in ocr_items if w.valid]

        if self.fail_on_missing_text and missing_text:
            task.data.is_valid = False
            task.data.error = (
                f"ocr_scoring_verification: {len(missing_text)} missing text region(s)"
            )
        elif ocr_items and not valid_words:
            # Had OCR output but every bbox failed quality threshold
            task.data.is_valid = False
            task.data.error = (
                f"ocr_scoring_verification: no bboxes passed quality threshold "
                f"(min_bbox_match={self.min_bbox_match}, max_text_errors={self.max_text_errors})"
            )

        return task
