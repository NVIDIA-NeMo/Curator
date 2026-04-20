"""OCRConversationData dataclass and word-level prompt/format variation pools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from nemo_curator.tasks.ocr import OCRData, OCRDenseWord


# ---------------------------------------------------------------------------
# Prompt variations (question / instruction sent to the model)
# ---------------------------------------------------------------------------

SDG_PROMPT_VARIATIONS: list[str] = [
    "Perform a word-level transcription of the image.",
    "Detect every word in the image.",
    "Extract all text at the word level and map each to its specific region.",
    "Generate a word-level OCR output for this image.",
    "List all words found in the image along with their bounding boxes.",
    "Find all text regions at the word level. Each entry should have text and bbox.",
    "Map every word in the image to its bounding box.",
    "Provide a word-level transcription of this image.",
    "Capture every word and its corresponding bounding box within the image.",
    "For every word visible, provide the text and its bounding box.",
    "Identify the bounding boxes for all words in the image.",
    "Provide a detailed word-level transcription. Each word should have a bbox.",
    "Locate all words in the image and define their areas.",
    "Execute a word-level OCR. Each item should include text and bbox.",
    "Extract text word-by-word from the image. For each, specify the bounding box.",
    "Identify the bounding box for every individual word in the image. Output the text and coordinates.",
    "Identify all text at the word level.",
    "Scan the image for words and provide their bounding boxes.",
    "Transcribe every word found in the image.",
    "Detect individual word regions and their contents.",
    "Map each word in the image to its respective crop.",
    "Perform word-level OCR. Each item should have text and bbox.",
    "List every word found in the image with its bounding box.",
    "Capture the area of every word in the image.",
    "For all words in the image, provide the text and its bounding box.",
    "Provide a word-level breakdown of the text in this image.",
    "Find all word regions.",
    "List the bounding boxes for all words.",
    "Extract each word's text and its region.",
    "Transcribe every word in this image.",
    "For every word detected, provide text and its bbox.",
    "Transcribe all text at a word-specific level.",
    "Perform a word-level scan of the image.",
]


# ---------------------------------------------------------------------------
# Output format variations (how the answer is formatted)
# Each callable takes list[OCRDenseWord] and returns (format_suffix, answer).
# ---------------------------------------------------------------------------

def _fmt_json_plain(items: list[OCRDenseWord]) -> tuple[str, str]:
    return (
        "Output must be a JSON list only, no markdown. Output the text and bounding box.",
        json.dumps([{"bbox_2d": list(o.bbox_2d), "text_content": o.text_content} for o in items]),
    )

def _fmt_json_markdown(items: list[OCRDenseWord]) -> tuple[str, str]:
    return (
        "Wrap the JSON output containing each bounding box and text in a markdown code block: ```json ... ```",
        "```json\n" + json.dumps([{"bbox_2d": list(o.bbox_2d), "text_content": o.text_content} for o in items]) + "\n```",
    )

def _fmt_json_keys(items: list[OCRDenseWord]) -> tuple[str, str]:
    return (
        "Use keys bbox_2d (list [x1, y1, x2, y2]) and text_content (string). Return a JSON array.",
        json.dumps([{"bbox_2d": list(o.bbox_2d), "text_content": o.text_content} for o in items]),
    )

def _fmt_json_explicit(items: list[OCRDenseWord]) -> tuple[str, str]:
    return (
        'Format each item as {"bbox_2d": [x1, y1, x2, y2], "text_content": "..."}. One JSON list.',
        json.dumps([{"bbox_2d": list(o.bbox_2d), "text_content": o.text_content} for o in items]),
    )

def _fmt_json_no_extra(items: list[OCRDenseWord]) -> tuple[str, str]:
    return (
        "Reply with a JSON list of objects with keys bbox_2d and text_content. No extra text.",
        json.dumps([{"bbox_2d": list(o.bbox_2d), "text_content": o.text_content} for o in items]),
    )

def _fmt_json_xyxy(items: list[OCRDenseWord]) -> tuple[str, str]:
    return (
        "Output a JSON list. Each entry: bbox_2d as [x_min, y_min, x_max, y_max] and text_content.",
        json.dumps([{"bbox_2d": list(o.bbox_2d), "text_content": o.text_content} for o in items]),
    )

def _fmt_text_per_line(items: list[OCRDenseWord]) -> tuple[str, str]:
    return (
        "Output one word per line as: text followed by bbox [x1, y1, x2, y2].",
        "\n".join(f"{o.text_content} {list(o.bbox_2d)}" for o in items),
    )

def _fmt_text_bracket(items: list[OCRDenseWord]) -> tuple[str, str]:
    return (
        'List each word on its own line as: "[x1, y1, x2, y2]: text".',
        "\n".join(f"{list(o.bbox_2d)}: {o.text_content}" for o in items),
    )

def _fmt_text_tuple(items: list[OCRDenseWord]) -> tuple[str, str]:
    return (
        "Reply with plain text, one word per line as: text (x1, y1, x2, y2).",
        "\n".join(f"{o.text_content} {tuple(o.bbox_2d)}" for o in items),
    )

def _fmt_markdown_table(items: list[OCRDenseWord]) -> tuple[str, str]:
    return (
        "Output a markdown table with columns: text | bbox.",
        "| text | bbox |\n|------|------|\n" + "\n".join(f"| {o.text_content!r} | {list(o.bbox_2d)} |" for o in items),
    )

def _fmt_tsv(items: list[OCRDenseWord]) -> tuple[str, str]:
    return (
        "Give each word as a single line: tab-separated text and bbox coordinates.",
        "\n".join(f"{o.text_content}\t{o.bbox_2d[0]}\t{o.bbox_2d[1]}\t{o.bbox_2d[2]}\t{o.bbox_2d[3]}" for o in items),
    )


WORD_OUTPUT_FORMATS = [
    _fmt_json_plain,
    _fmt_json_markdown,
    _fmt_json_keys,
    _fmt_json_explicit,
    _fmt_json_no_extra,
    _fmt_json_xyxy,
    _fmt_text_per_line,
    _fmt_text_bracket,
    _fmt_text_tuple,
    _fmt_markdown_table,
    _fmt_tsv,
]


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class OCRConversationData(OCRData):
    """OCRData with a conversation field added by OCRConversationalizeStage."""

    conversation: ConversationSample | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OCRConversationData":
        base = OCRData.from_dict(data)
        return cls(
            image_path=base.image_path,
            image_id=base.image_id,
            is_valid=base.is_valid,
            error=base.error,
            ocr_is_word_level=base.ocr_is_word_level,
            ocr_dense_prompt=base.ocr_dense_prompt,
            ocr_dense=base.ocr_dense,
            ocr_verification_prompt=base.ocr_verification_prompt,
            ocr_verification_model=base.ocr_verification_model,
            ocr_verification_response_raw=base.ocr_verification_response_raw,
            ocr_verification_answers=base.ocr_verification_answers,
            ocr_scoring_prompt=base.ocr_scoring_prompt,
            ocr_scoring_model=base.ocr_scoring_model,
            ocr_scoring_response_raw=base.ocr_scoring_response_raw,
            ocr_scoring_mode=base.ocr_scoring_mode,
            ocr_scoring_missing=base.ocr_scoring_missing,
        )


