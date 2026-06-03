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

"""Task data classes for the OCR mixed dense pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

from nemo_curator.tasks.image import ImageTaskData


@dataclass(kw_only=True)
class OCRDenseItem:
    """Single entry (word, line, or block) in dense OCR output.

    Coordinates are normalized 0-1000.
    """

    bbox_2d: list[int] | tuple[int, int, int, int]
    text_content: str
    quad: list[tuple[int, int]] | None = None
    valid: bool = True

    # Scoring verification fields (set by OCRScoringVerificationStage)
    bbox_match: int | None = None  # verifier bbox fit score 0-10
    text_errors: int | None = None  # verifier transcription error count

    def __post_init__(self) -> None:
        self.bbox_2d = list(self.bbox_2d)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OCRDenseItem:
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
    def join(items: Iterable[OCRDenseItem], separator: str = " ") -> OCRDenseItem:
        """Merge multiple items into one by unioning their bboxes and joining text."""
        it = iter(items)
        try:
            first = next(it)
        except StopIteration:
            return OCRDenseItem(bbox_2d=[0, 0, 0, 0], text_content="", valid=False)
        texts = [first.text_content]
        x0, y0, x1, y1 = first.bbox_2d[0], first.bbox_2d[1], first.bbox_2d[2], first.bbox_2d[3]
        for item in it:
            texts.append(item.text_content)
            x0 = min(x0, item.bbox_2d[0])
            y0 = min(y0, item.bbox_2d[1])
            x1 = max(x1, item.bbox_2d[2])
            y1 = max(y1, item.bbox_2d[3])
        return OCRDenseItem(
            bbox_2d=(x0, y0, x1, y1),
            text_content=separator.join(texts),
            valid=True,
        )


@dataclass(kw_only=True)
class OCRData(ImageTaskData):
    """Task data for the OCR dense pipeline.

    Fields are populated incrementally as the task moves through pipeline stages:
    - OCR stage (NemotronOCR-v2): ocr_dense
    - Scoring QA stage (Nemotron-Nano-Omni): ocr_scoring_*
    - Conversationalize stage: conversation
    """

    ocr_is_word_level: bool = True
    ocr_dense_prompt: str | None = None
    ocr_dense: list[OCRDenseItem] | None = None

    # --- Scoring QA (OCRScoringQAStage) ---
    ocr_scoring_prompt: str | None = None
    ocr_scoring_model: str | None = None
    ocr_scoring_response_raw: str | None = None
    ocr_scoring_mode: str | None = None  # "word" or "line" as inferred by the verifier
    # Each entry has {"text": str, "bbox_2d": [y0, x0, y1, x1]} — note y-first ordering,
    # matching the verifier prompt/response convention. This differs from OCRDenseItem.bbox_2d
    # which uses the standard [x0, y0, x1, y1] convention from NemotronOCR-v2.
    ocr_scoring_missing: list[dict] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OCRData:
        """Deserialize from a JSONL record (produced by JsonlSampleWriterStage)."""
        qwen_raw = data.get("ocr_dense")
        if isinstance(qwen_raw, list):
            ocr_items: list[OCRDenseItem] | None = [
                OCRDenseItem.from_dict(x) if isinstance(x, dict) else x for x in qwen_raw
            ]
        else:
            ocr_items = None

        is_word_level = bool(data["ocr_is_word_level"]) if "ocr_is_word_level" in data else True

        return cls(
            image_path=Path(data["image_path"]) if data.get("image_path") else Path(""),
            image_id=data.get("image_id"),
            is_valid=data.get("is_valid", True),
            error=data.get("error"),
            ocr_is_word_level=is_word_level,
            ocr_dense_prompt=data.get("ocr_dense_prompt"),
            ocr_dense=ocr_items,
            ocr_scoring_prompt=data.get("ocr_scoring_prompt"),
            ocr_scoring_model=data.get("ocr_scoring_model"),
            ocr_scoring_response_raw=data.get("ocr_scoring_response_raw"),
            ocr_scoring_mode=data.get("ocr_scoring_mode"),
            ocr_scoring_missing=data.get("ocr_scoring_missing"),
        )
