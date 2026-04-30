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

"""Combined Gemini bbox scoring + dense QA generation stage.

Replaces running ``OCRScoringVerificationStage`` followed by
``OCRDenseQAStage`` as two separate pipeline stages.  In a single Gemini
call per image it:

  1. Scores every bbox (``bbox_match`` 0-10, ``text_errors`` count).
  2. Marks low-quality bboxes ``valid=False``.
  3. Detects missing text regions (``ocr_scoring_missing``).
  4. Generates up to 100 multi-turn QA pairs via ``build_qa_tagged`` /
     ``build_conversation`` from ``ocr_dense_qa``.
  5. Includes a dense-dump (list-all-bboxes) QA turn **only** when
     ``ocr_scoring_missing`` is empty — Gemini predictions cannot be used
     as training labels, only for filtering.

Output task data is ``OCRConversationData`` (``OCRData`` + ``conversation``).
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image

    from nemo_curator.tasks.image import SingleDataTask

from nemo_curator.models.omni.base import InferenceConfig, NVInferenceModelConfig
from nemo_curator.models.omni.gemini import Gemini3Pro
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.synthetic.omni.base import ModelProcessingStage, SkipSample
from nemo_curator.stages.synthetic.omni.ocr_conversationalize import OCRConversationData
from nemo_curator.stages.synthetic.omni.ocr_dense_qa import (
    build_conversation,
    build_dense_conversation,
    build_qa_tagged,
)
from nemo_curator.tasks.ocr import OCRData

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_BBOX_COORD_COUNT = 4

# Reuse exact same prompt as OCRScoringVerificationStage
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


def _try_parse_match(match: re.Match) -> dict | None:
    """Try to parse one regex match as a JSON object, returning None on failure."""
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _parse_json_object(text: str) -> dict | None:
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    for match in _JSON_OBJECT_RE.finditer(cleaned):
        obj = _try_parse_match(match)
        if obj is not None:
            return obj
    return None


def _copy_to_conversation_data(src: OCRData) -> OCRConversationData:
    return OCRConversationData(
        image_path=src.image_path,
        image_id=src.image_id,
        is_valid=src.is_valid,
        error=src.error,
        ocr_is_word_level=src.ocr_is_word_level,
        ocr_dense_prompt=src.ocr_dense_prompt,
        ocr_dense=src.ocr_dense,
        ocr_scoring_prompt=src.ocr_scoring_prompt,
        ocr_scoring_model=src.ocr_scoring_model,
        ocr_scoring_response_raw=src.ocr_scoring_response_raw,
        ocr_scoring_mode=src.ocr_scoring_mode,
        ocr_scoring_missing=src.ocr_scoring_missing,
    )


class OCRScoringQAStage(ModelProcessingStage[OCRData]):
    """Gemini bbox scoring + multi-turn QA generation in a single pipeline stage.

    * Calls Gemini once per image to score every bbox.
    * Marks low-quality bboxes ``valid=False`` (below ``min_bbox_match`` or
      above ``max_text_errors``).
    * Calls ``build_qa_tagged`` / ``build_conversation`` to produce up to 100
      multi-turn QA pairs.
    * Adds a dense-dump turn only when ``ocr_scoring_missing`` is empty (Gemini
      predictions are approved for filtering only, not for training labels).

    Unlike ``OCRScoringVerificationStage``, ``fail_on_missing_text`` defaults to
    ``False`` here: missing text regions disable the dense dump but do not
    invalidate the whole image — the partial QA pairs are still useful.

    Reads:  ``ocr_dense``
    Writes: all ``ocr_scoring_*`` fields, per-bbox ``bbox_match`` / ``text_errors``
            / ``valid``, ``ocr_is_word_level``, ``conversation``
    Output task data type: ``OCRConversationData``
    """

    name = "ocr_scoring_qa"
    resources = Resources(cpus=1.0)
    batch_size = 16
    multimodal = True

    def __init__(  # noqa: PLR0913
        self,
        model_id: str = "gcp/google/gemini-3-flash-preview",
        temperature: float = 1.0,
        max_tokens: int = 4096,
        min_bbox_match: int = 7,
        max_text_errors: int = 0,
        fail_on_missing_text: bool = False,
        dense_dump_prob: float = 0.10,
        batch_size: int | None = None,
        priority_mode: bool = False,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialise the combined scoring + QA stage.

        Args:
            model_id: NVIDIA Inference API model to use for Gemini scoring.
            temperature: Sampling temperature (keep at 1.0 for Gemini).
            max_tokens: Max tokens for the Gemini response.
            min_bbox_match: Minimum ``bbox_match`` score for a valid bbox.
            max_text_errors: Maximum ``text_errors`` count for a valid bbox.
            fail_on_missing_text: If ``True``, mark the whole image invalid
                when Gemini reports missing text.  Defaults to ``False`` —
                missing text only disables the dense-dump QA turn.
            dense_dump_prob: Probability (0-1) of generating a single-turn
                dense dump conversation instead of multi-turn QA, for images
                where OCR is provably complete (no missing text).
                Defaults to 0.10 (10%).
            batch_size: Override the default batch size of 16.
            priority_mode: Use priority API queue (lower latency, higher cost).
        """
        self._scoring_model_id = model_id
        self.min_bbox_match = min_bbox_match
        self.max_text_errors = max_text_errors
        self.fail_on_missing_text = fail_on_missing_text
        self.dense_dump_prob = dense_dump_prob
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
        if not ocr_items:
            raise SkipSample

        bboxes_for_prompt = []
        for idx, item in enumerate(ocr_items):
            bbox = item.bbox_2d if hasattr(item, "bbox_2d") else item.get("bbox_2d")
            text = item.text_content if hasattr(item, "text_content") else item.get("text_content", "")
            if bbox is None or len(bbox) != _BBOX_COORD_COUNT:
                continue
            x1, y1, x2, y2 = bbox
            bboxes_for_prompt.append(
                {
                    "idx": idx,
                    "bbox_2d": [y1, x1, y2, x2],
                    "text": str(text or ""),
                }
            )

        prompt = _PROMPT.format(bboxes_json=json.dumps(bboxes_for_prompt, ensure_ascii=False))
        task.data.ocr_scoring_prompt = prompt
        task.data.ocr_scoring_model = self._scoring_model_id
        return prompt

    def handle_response(  # noqa: C901
        self, task: SingleDataTask[OCRData], response: str
    ) -> SingleDataTask[OCRData]:
        # --- 1. Convert to OCRConversationData so we can set .conversation ---
        task.data = _copy_to_conversation_data(task.data)

        if not response:
            task.data.is_valid = False
            task.data.error = "ocr_scoring_qa: empty response from model"
            return task

        task.data.ocr_scoring_response_raw = response

        result = _parse_json_object(response)
        if result is None:
            task.data.is_valid = False
            task.data.error = f"ocr_scoring_qa: could not parse JSON: {response[:200]!r}"
            return task

        ocr_mode = result.get("ocr_mode", "unknown")
        text_results: list[dict] = result.get("text") or []
        missing_text: list[dict] = result.get("missing_text") or []

        task.data.ocr_scoring_mode = ocr_mode
        task.data.ocr_scoring_missing = missing_text

        if ocr_mode == "word":
            task.data.ocr_is_word_level = True
        elif ocr_mode == "line":
            task.data.ocr_is_word_level = False

        # --- 2. Apply per-bbox scores ---
        ocr_items = task.data.ocr_dense or []
        scores_by_idx: dict[int, dict] = {int(e["idx"]): e for e in text_results if "idx" in e}
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
            word.valid = word.bbox_match >= self.min_bbox_match and word.text_errors <= self.max_text_errors

        valid_words = [w for w in ocr_items if w.valid]

        # --- 3. Image-level validity checks ---
        if self.fail_on_missing_text and missing_text:
            task.data.is_valid = False
            task.data.error = f"ocr_scoring_qa: {len(missing_text)} missing text region(s)"
            return task

        if ocr_items and not valid_words:
            task.data.is_valid = False
            task.data.error = (
                f"ocr_scoring_qa: no bboxes passed quality threshold "
                f"(min_bbox_match={self.min_bbox_match}, max_text_errors={self.max_text_errors})"
            )
            return task

        # --- 4. Generate conversation ---
        # When OCR is provably complete (no missing text), 10% of images get a
        # single-turn dense dump; the other 90% get multi-turn QA.
        # When OCR is incomplete, always use multi-turn QA (dense dump would lie).
        image_name = Path(str(task.data.image_path)).name
        rng = random.Random(task.task_id)  # noqa: S311
        ocr_complete = not missing_text
        if ocr_complete and rng.random() < self.dense_dump_prob:
            task.data.conversation = build_dense_conversation(valid_words, rng, image_name)
        else:
            qa_tagged, rng = build_qa_tagged(task.data, task.task_id)
            task.data.conversation = build_conversation(qa_tagged, rng, image_name)

        return task
