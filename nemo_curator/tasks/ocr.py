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

"""Task data classes for the OCR mixed dense pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from nemo_curator.tasks.image import ImageTaskData


@dataclass(kw_only=True)
class OCRDenseWord:
    """Single word (or line/block) entry in dense OCR output.

    Coordinates are normalized 0–1000.
    """

    bbox_2d: list[int] | tuple[int, int, int, int]
    text_content: str
    quad: list[tuple[int, int]] | None = None
    valid: bool = True

    # Scoring verification fields (set by OCRScoringVerificationStage)
    bbox_match: int | None = None    # Gemini bbox fit score 0-10
    text_errors: int | None = None   # Gemini transcription error count

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_2d": list(self.bbox_2d),
            "text_content": self.text_content,
            "quad": self.quad,
            "valid": self.valid,
            "bbox_match": self.bbox_match,
            "text_errors": self.text_errors,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OCRDenseWord":
        bbox = data.get("bbox_2d")
        if bbox is not None and not isinstance(bbox, (list, tuple)):
            bbox = list(bbox)
        return cls(
            bbox_2d=bbox if isinstance(bbox, (list, tuple)) else [0, 0, 0, 0],
            text_content=str(data.get("text_content") or ""),
            quad=data.get("quad"),
            valid=data.get("valid", True),
            bbox_match=data.get("bbox_match"),
            text_errors=data.get("text_errors"),
        )

    @staticmethod
    def join(words: Iterable["OCRDenseWord"], separator: str = " ") -> "OCRDenseWord":
        """Merge multiple words into one by unioning their bboxes and joining text."""
        it = iter(words)
        try:
            first = next(it)
        except StopIteration:
            return OCRDenseWord(bbox_2d=[0, 0, 0, 0], text_content="", valid=False)
        texts = [first.text_content]
        x0, y0, x1, y1 = first.bbox_2d[0], first.bbox_2d[1], first.bbox_2d[2], first.bbox_2d[3]
        for w in it:
            texts.append(w.text_content)
            x0 = min(x0, w.bbox_2d[0])
            y0 = min(y0, w.bbox_2d[1])
            x1 = max(x1, w.bbox_2d[2])
            y1 = max(y1, w.bbox_2d[3])
        return OCRDenseWord(
            bbox_2d=(x0, y0, x1, y1),
            text_content=separator.join(texts),
            valid=True,
        )


@dataclass(kw_only=True)
class OCRData(ImageTaskData):
    """Task data for the OCR dense pipeline.

    Fields are populated incrementally as the task moves through pipeline stages:
    - OCR stage (NemotronOCR-v2): ocr_dense
    - Scoring QA stage (Gemini): ocr_scoring_*
    - Conversationalize stage: conversation
    """

    ocr_is_word_level: bool = True
    ocr_dense_prompt: str | None = None
    ocr_dense: list[OCRDenseWord] | None = None

    # --- Verification (OCRVerificationStage) ---
    ocr_verification_prompt: str | None = None
    ocr_verification_model: str | None = None
    ocr_verification_response_raw: str | None = None
    ocr_verification_answers: list[dict] | None = None

    # --- Scoring verification (OCRScoringVerificationStage) ---
    ocr_scoring_prompt: str | None = None
    ocr_scoring_model: str | None = None
    ocr_scoring_response_raw: str | None = None
    ocr_scoring_mode: str | None = None          # "word" or "line" as inferred by Gemini
    ocr_scoring_missing: list[dict] | None = None  # missing text regions with bbox_2d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OCRData":
        """Deserialize from a JSONL record (produced by ResultWriterStage)."""
        qwen_raw = data.get("ocr_dense")
        if isinstance(qwen_raw, list):
            qwen_items: list[OCRDenseWord] | None = [
                OCRDenseWord.from_dict(x) if isinstance(x, dict) else x
                for x in qwen_raw
            ]
        else:
            qwen_items = None

        if "ocr_is_word_level" in data:
            is_word_level = bool(data["ocr_is_word_level"])
        else:
            is_word_level = True

        return cls(
            image_path=Path(data["image_path"]) if data.get("image_path") else Path(""),
            image_id=data.get("image_id"),
            is_valid=data.get("is_valid", True),
            error=data.get("error"),
            ocr_is_word_level=is_word_level,
            ocr_dense_prompt=data.get("ocr_dense_prompt"),
            ocr_dense=qwen_items,
            ocr_verification_prompt=data.get("ocr_verification_prompt"),
            ocr_verification_model=data.get("ocr_verification_model"),
            ocr_verification_response_raw=data.get("ocr_verification_response_raw"),
            ocr_verification_answers=data.get("ocr_verification_answers"),
            ocr_scoring_prompt=data.get("ocr_scoring_prompt"),
            ocr_scoring_model=data.get("ocr_scoring_model"),
            ocr_scoring_response_raw=data.get("ocr_scoring_response_raw"),
            ocr_scoring_mode=data.get("ocr_scoring_mode"),
            ocr_scoring_missing=data.get("ocr_scoring_missing"),
        )
