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

"""QA conversation helpers for dense OCR output.

Builds up to MAX_QA_PAIRS multi-turn QA pairs per image, balanced across
4 question types:

  1. bbox_to_text   — given a bbox, return the word/line text
  2. point_to_text  — given a center point, return the word/line text
  3. text_to_bbox   — given text, locate its bbox(es)
  4. text_to_point  — given text, locate its center point(s)

Types 3–4 are disabled when OCR quality is low (many invalid bboxes).

Public API: ``build_qa_tagged``, ``build_conversation``, ``build_dense_conversation``.
Used by ``OCRScoringQAStage``.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from typing import Any, Callable

from nemo_curator.stages.synthetic.omni.ocr_conversationalize import (
    OCRConversationData,
    SDG_PROMPT_VARIATIONS,
    WORD_OUTPUT_FORMATS,
)
from nemo_curator.stages.synthetic.omni.utils.conversation import ConversationSample, ImageMedia, Message
from nemo_curator.tasks.ocr import OCRData, OCRDenseWord


MAX_QA_PAIRS = 100

QA_TYPE_BBOX_TO_TEXT = "bbox_to_text"
QA_TYPE_POINT_TO_TEXT = "point_to_text"
QA_TYPE_TEXT_TO_BBOX = "text_to_bbox"
QA_TYPE_TEXT_TO_POINT = "text_to_point"
QA_TYPE_DENSE_DUMP = "dense_dump"  # list-all-bboxes turn; only included when OCR is complete


# ---------------------------------------------------------------------------
# Balanced sampler
# ---------------------------------------------------------------------------

def _balanced_sample_qa(
    tagged: list[tuple[str, str, str]],
    max_pairs: int,
    rng: random.Random,
) -> list[tuple[str, str]]:
    """Sample up to max_pairs (q, a) from tagged (type, q, a), balancing by type."""
    if len(tagged) <= max_pairs:
        return [(q, a) for _, q, a in tagged]
    by_type: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for typ, q, a in tagged:
        by_type[typ].append((q, a))
    types = sorted(by_type.keys())
    n_types = len(types)
    base_quota = max_pairs // n_types
    remainder = max_pairs % n_types
    selected: list[tuple[str, str]] = []
    leftover: list[tuple[str, str]] = []
    for i, typ in enumerate(types):
        bucket = by_type[typ]
        quota = base_quota + (1 if i < remainder else 0)
        take = min(quota, len(bucket))
        if take >= len(bucket):
            selected.extend(bucket)
        else:
            indices = set(rng.sample(range(len(bucket)), take))
            for j, p in enumerate(bucket):
                if j in indices:
                    selected.append(p)
                else:
                    leftover.append(p)
    need = max_pairs - len(selected)
    if need > 0 and leftover:
        selected.extend(rng.sample(leftover, min(need, len(leftover))))
    rng.shuffle(selected)
    return selected


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _fmt_box(bbox: list[int] | tuple[int, ...]) -> str:
    return f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"


def _bbox_center(bbox: list[int] | tuple[int, ...]) -> tuple[int, int]:
    return (
        (int(bbox[0]) + int(bbox[2])) // 2,
        (int(bbox[1]) + int(bbox[3])) // 2,
    )


def _bbox_center_x(b: list[int] | tuple[int, ...]) -> float:
    return (b[0] + b[2]) / 2


def _bbox_center_y(b: list[int] | tuple[int, ...]) -> float:
    return (b[1] + b[3]) / 2


def _bbox_dist_from_center(b: list[int] | tuple[int, ...]) -> float:
    cx, cy = _bbox_center_x(b), _bbox_center_y(b)
    return math.sqrt((cx - 500) ** 2 + (cy - 500) ** 2)


def _point_dist_from_center(p: tuple[int, int]) -> float:
    return math.sqrt((p[0] - 500) ** 2 + (p[1] - 500) ** 2)


# ---------------------------------------------------------------------------
# Text escaping
# ---------------------------------------------------------------------------

def _escape_text_for_prompt(text: str, rng: random.Random) -> str:
    """Quote text for safe insertion into prompts."""
    if text.isupper() and any(c.isalpha() for c in text) and rng.random() < 0.5:
        return text
    if '"' in text:
        escaped = text.replace("\\", "\\\\").replace("'", "\\'")
        return "'" + escaped + "'"
    if "'" in text:
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return '"' + escaped + '"'
    if rng.choice([True, False]):
        escaped = text.replace("\\", "\\\\").replace("'", "\\'")
        return "'" + escaped + "'"
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return '"' + escaped + '"'


# ---------------------------------------------------------------------------
# Question / format templates
# ---------------------------------------------------------------------------

_BBOX_TO_TEXT_TEMPLATES: list[str] = [
    "What text is in the bounding box {}?",
    "Read the text at bounding box {}.",
    "What does the text say in the region {}?",
    "Give me the text content inside the box {}.",
    "What is the text at coordinates {}?",
    "Write out the text in the region {}.",
    "Look at the bounding box {}. What does it say?",
    "Extract the text from the area {}.",
    "What word or text is located at {}?",
    "Describe the text content in the box {}.",
]

_BBOX_FORMAT_TEMPLATES: list[Callable[[tuple[int, ...]], tuple[str, str]]] = [
    lambda b: ("Answer with the bounding box as [x1, y1, x2, y2].", f"[{b[0]}, {b[1]}, {b[2]}, {b[3]}]"),
    lambda b: ("Give the bounding box coordinates as [x_min, y_min, x_max, y_max].", f"[{b[0]}, {b[1]}, {b[2]}, {b[3]}]"),
    lambda b: ("Provide the box as [x0, y0, x1, y1].", f"[{b[0]}, {b[1]}, {b[2]}, {b[3]}]"),
    lambda b: ("Just write down the box coordinates.", f"{b[0]}, {b[1]}, {b[2]}, {b[3]}"),
    lambda b: ("Reply with coordinates x1, y1, x2, y2.", f"{b[0]}, {b[1]}, {b[2]}, {b[3]}"),
    lambda b: ("Give me the bounding box coordinates as [x0, y0, x1, y1].", f"[{b[0]}, {b[1]}, {b[2]}, {b[3]}]"),
    lambda b: (
        "Would be great to get the bounding box as json {x0, y0, x1, y1}.",
        f'{{"x0": {b[0]}, "y0": {b[1]}, "x1": {b[2]}, "y1": {b[3]}}}',
    ),
    lambda b: (
        "Format the box as a dictionary with keys x0, y0, x1, y1.",
        f'{{"x0": {b[0]}, "y0": {b[1]}, "x1": {b[2]}, "y1": {b[3]}}}',
    ),
    lambda b: (
        "Give the bounding box as x_min, y_min, x_max, y_max.",
        f"{b[0]}, {b[1]}, {b[2]}, {b[3]}",
    ),
    lambda b: ("Provide the box as [x_min, y_min, x_max, y_max].", f"[{b[0]}, {b[1]}, {b[2]}, {b[3]}]"),
    lambda b: (
        "Answer with a dictionary with keys x_min, y_min, x_max, y_max.",
        f'{{"x_min": {b[0]}, "y_min": {b[1]}, "x_max": {b[2]}, "y_max": {b[3]}}}',
    ),
    lambda b: (
        "Format the box as json {x_min, y_min, x_max, y_max}.",
        f'{{"x_min": {b[0]}, "y_min": {b[1]}, "x_max": {b[2]}, "y_max": {b[3]}}}',
    ),
    lambda b: (
        "Wrap the bounding box in <box></box> tags as [x1, y1, x2, y2].",
        f"<box>[{b[0]}, {b[1]}, {b[2]}, {b[3]}]</box>",
    ),
    lambda b: (
        "Reply with a JSON object with key bbox_2d (list [x1, y1, x2, y2]).",
        json.dumps({"bbox_2d": list(b)}),
    ),
]

_TEXT_TO_POINT_BASES: list[str] = [
    "Point at the text {}.",
    "Indicate the center of the text {}.",
    "Where is the center of {}? Give the point.",
    "Click on the text {}. What are the coordinates of that point?",
    "Point to where the text {} is located.",
]

_TEXT_TO_POINT_MULTI_BASES: list[str] = [
    "Point at every occurrence of the text {}.",
    "Indicate the center of each instance of {} in the image.",
    "Where are all the centers of {}? List each point.",
    "Give the center point for every place where {} appears.",
    "Click on each occurrence of {}. What are the coordinates of those points?",
    "List the center coordinates for each time {} appears in the image.",
]

_POINT_FORMAT_TEMPLATES: list[Callable[[tuple[int, int]], tuple[str, str]]] = [
    lambda c: ("Give the point as x, y.", f"{c[0]}, {c[1]}"),
    lambda c: ("Answer with the center as [x, y].", f"[{c[0]}, {c[1]}]"),
    lambda c: ("Provide the point coordinates as [x, y].", f"[{c[0]}, {c[1]}]"),
    lambda c: ("Reply with the center point x, y.", f"{c[0]}, {c[1]}"),
    lambda c: ("Give the point as a dict with keys x and y.", f'{{"x": {c[0]}, "y": {c[1]}}}'),
    lambda c: ("Wrap the point in <point></point> tags as (x, y).", f"<point>({c[0]}, {c[1]})</point>"),
    lambda c: ("Reply with a JSON object with key point_2d (list [x, y]).", json.dumps({"point_2d": [c[0], c[1]]})),
]

_POINT_LIST_FORMAT_TEMPLATES: list[Callable[[list[tuple[int, int]]], tuple[str, str]]] = [
    lambda pts: ("Give each point as x, y, one per line.", "\n".join(f"{x}, {y}" for x, y in pts)),
    lambda pts: ("Provide each center as [x, y], comma-separated.", ", ".join(f"[{x}, {y}]" for x, y in pts)),
    lambda pts: ("List each point as [x, y] on its own line.", "\n".join(f"[{x}, {y}]" for x, y in pts)),
    lambda pts: (
        'Reply with each point as x, y separated by the word "and".',
        " and ".join(f"{x}, {y}" for x, y in pts),
    ),
    lambda pts: (
        "Wrap all points in <point></point> as a nested list of (x, y).",
        "<point>[" + ", ".join(f"({x}, {y})" for x, y in pts) + "]</point>",
    ),
    lambda pts: (
        "Output a JSON list of objects, each with key point_2d (list [x, y]).",
        json.dumps([{"point_2d": [x, y]} for x, y in pts]),
    ),
]

_POINT_TO_WORD_QUESTION_TEMPLATES: list[str] = [
    "Which word is at the point {}?",
    "What word is at the coordinates {}?",
    "What does the image say at point {}?",
    "Identify the word at location {}.",
    "What word is located at {}?",
    "Read the word at the point {}.",
    "Which word appears at coordinates {}?",
    "What is the word at {}?",
    "Tell me the text at point {}. Just give the single word.",
    "What character or word is at {}?",
]

_POINT_IN_QUESTION_FORMATS: list[Callable[[tuple[int, int]], str]] = [
    lambda c: f"{c[0]}, {c[1]}",
    lambda c: f"[{c[0]}, {c[1]}]",
    lambda c: f"({c[0]}, {c[1]})",
    lambda c: f"{c[0]} {c[1]}",
    lambda c: f'{{"x": {c[0]}, "y": {c[1]}}}',
]

_TEXT_TO_BBOX_SINGLE_BASES: list[str] = [
    "Where does the text {} appear?",
    "Locate the text {} in the image.",
    "Find the bounding box that contains the text {}.",
    "Where is the text {} in the image?",
    "Give the location of text {}.",
]

_TEXT_TO_BBOX_MULTI_BASES: list[str] = [
    "List all bounding boxes that contain the text {}.",
    "For the text {}, give every bounding box for it.",
    "Where does {} appear? List all locations as bounding boxes.",
    "Find every occurrence of {} and give each bounding box.",
]

_LIST_FORMAT_TEMPLATES: list[Callable[[list[list[int]]], tuple[str, str]]] = [
    lambda boxes: (
        "Give each bounding box as [x1, y1, x2, y2], one per line.",
        "\n".join(_fmt_box(b) for b in boxes),
    ),
    lambda boxes: (
        "Provide each box as [x1, y1, x2, y2], comma-separated.",
        ", ".join(_fmt_box(b) for b in boxes),
    ),
    lambda boxes: (
        'List each bounding box as [x1, y1, x2, y2] separated by "and".',
        " and ".join(_fmt_box(b) for b in boxes),
    ),
    lambda boxes: (
        "Output a JSON array of arrays, each [x0, y0, x1, y1].",
        json.dumps([list(b) for b in boxes]),
    ),
    lambda boxes: (
        "Format as a JSON list of objects with keys x0, y0, x1, y1.",
        json.dumps([{"x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3]} for b in boxes]),
    ),
    lambda boxes: (
        "Give each box as x_min, y_min, x_max, y_max, one per line.",
        "\n".join(f"{b[0]}, {b[1]}, {b[2]}, {b[3]}" for b in boxes),
    ),
    lambda boxes: (
        "Output a JSON list of objects with keys x_min, y_min, x_max, y_max.",
        json.dumps([{"x_min": b[0], "y_min": b[1], "x_max": b[2], "y_max": b[3]} for b in boxes]),
    ),
    lambda boxes: (
        "Wrap all bounding boxes in a single <box></box> span as a nested list of [x1, y1, x2, y2] per box.",
        "<box>[" + ", ".join("[" + ",".join(str(c) for c in b) + "]" for b in boxes) + "]</box>",
    ),
    lambda boxes: (
        "Output a JSON list of objects, each with key bbox_2d (list [x1, y1, x2, y2]).",
        json.dumps([{"bbox_2d": list(b)} for b in boxes]),
    ),
]

_BBOX_SORT_GENERATORS: list[Callable[[list[list[int]]], tuple[str, list[list[int]]]]] = [
    lambda boxes: ("", sorted(boxes, key=lambda b: (b[0], b[1]))),
    lambda boxes: ("List them sorted from left to right.", sorted(boxes, key=lambda b: (b[0], b[1]))),
    lambda boxes: ("List them from top to bottom.", sorted(boxes, key=lambda b: (b[1], b[0]))),
    lambda boxes: ("Sort by horizontal center, left to right.", sorted(boxes, key=_bbox_center_x)),
    lambda boxes: ("Sort by vertical center, top to bottom.", sorted(boxes, key=_bbox_center_y)),
    lambda boxes: (
        "List them starting from the center of the image outward.",
        sorted(boxes, key=_bbox_dist_from_center),
    ),
    lambda boxes: (
        "Sort by horizontal centrality (closest to middle column first).",
        sorted(boxes, key=lambda b: abs(_bbox_center_x(b) - 500)),
    ),
    lambda boxes: (
        "Sort by vertical centrality (closest to middle row first).",
        sorted(boxes, key=lambda b: abs(_bbox_center_y(b) - 500)),
    ),
]

_POINT_SORT_GENERATORS: list[Callable[[list[tuple[int, int]]], tuple[str, list[tuple[int, int]]]]] = [
    lambda pts: ("", sorted(pts, key=lambda p: (p[0], p[1]))),
    lambda pts: ("List them sorted from left to right.", sorted(pts, key=lambda p: (p[0], p[1]))),
    lambda pts: ("List them from right to left.", sorted(pts, key=lambda p: (p[0], p[1]), reverse=True)),
    lambda pts: ("List them from top to bottom.", sorted(pts, key=lambda p: (p[1], p[0]))),
    lambda pts: ("List them from bottom to top.", sorted(pts, key=lambda p: (p[1], p[0]), reverse=True)),
    lambda pts: (
        "List them starting from the center of the image outward.",
        sorted(pts, key=_point_dist_from_center),
    ),
    lambda pts: (
        "List them from the edges inward.",
        sorted(pts, key=_point_dist_from_center, reverse=True),
    ),
    lambda pts: (
        "Sort by horizontal centrality (closest to middle column first).",
        sorted(pts, key=lambda p: abs(p[0] - 500)),
    ),
    lambda pts: (
        "Sort by vertical centrality (closest to middle row first).",
        sorted(pts, key=lambda p: abs(p[1] - 500)),
    ),
]


# ---------------------------------------------------------------------------
# QA generators (module-level, reused by stage and combined scoring+QA stage)
# ---------------------------------------------------------------------------

def _gen_bbox_to_text(rng: random.Random, bbox: list[int] | tuple[int, ...], text: str) -> tuple[str, str]:
    return (rng.choice(_BBOX_TO_TEXT_TEMPLATES).format(_fmt_box(bbox)), text)


def _gen_point_to_text(rng: random.Random, point: tuple[int, int], text: str) -> tuple[str, str]:
    q_tpl = rng.choice(_POINT_TO_WORD_QUESTION_TEMPLATES)
    point_str = rng.choice(_POINT_IN_QUESTION_FORMATS)(point)
    return (q_tpl.format(point_str), text)


def _gen_text_to_bbox_single(rng: random.Random, text: str, bbox: list[int] | tuple[int, ...]) -> tuple[str, str]:
    base = rng.choice(_TEXT_TO_BBOX_SINGLE_BASES).format(_escape_text_for_prompt(text, rng))
    fmt_instruction, answer = rng.choice(_BBOX_FORMAT_TEMPLATES)(tuple(bbox))
    return (f"{base} {fmt_instruction}", answer)


def _gen_text_to_bbox_multi(rng: random.Random, text: str, bboxes: list[list[int]]) -> tuple[str, str]:
    base = rng.choice(_TEXT_TO_BBOX_MULTI_BASES).format(_escape_text_for_prompt(text, rng))
    sort_instruction, sorted_boxes = rng.choice(_BBOX_SORT_GENERATORS)(bboxes)
    fmt_instruction, answer = rng.choice(_LIST_FORMAT_TEMPLATES)(sorted_boxes)
    parts = [base, sort_instruction, fmt_instruction]
    return (" ".join(p for p in parts if p), answer)


def _gen_text_to_point_single(rng: random.Random, text: str, bbox: list[int] | tuple[int, ...]) -> tuple[str, str]:
    base = rng.choice(_TEXT_TO_POINT_BASES).format(_escape_text_for_prompt(text, rng))
    center = _bbox_center(bbox)
    fmt_instruction, answer = rng.choice(_POINT_FORMAT_TEMPLATES)(center)
    return (f"{base} {fmt_instruction}", answer)


def _gen_text_to_point_multi(rng: random.Random, text: str, bboxes: list[list[int]]) -> tuple[str, str]:
    base = rng.choice(_TEXT_TO_POINT_MULTI_BASES).format(_escape_text_for_prompt(text, rng))
    centers = [_bbox_center(b) for b in bboxes]
    sort_instruction, sorted_centers = rng.choice(_POINT_SORT_GENERATORS)(centers)
    fmt_instruction, answer = rng.choice(_POINT_LIST_FORMAT_TEMPLATES)(sorted_centers)
    parts = [base, sort_instruction, fmt_instruction]
    return (" ".join(p for p in parts if p), answer)


def _gen_dense_dump(rng: random.Random, words: list[OCRDenseWord]) -> tuple[str, str]:
    """Generate a 'list all bboxes' QA pair (dense dump format)."""
    question_base = rng.choice(SDG_PROMPT_VARIATIONS)
    format_fn = rng.choice(WORD_OUTPUT_FORMATS)
    format_suffix, answer = format_fn(words)
    return (f"{question_base} {format_suffix}", answer)


def build_qa_tagged(
    data: OCRData,
    task_id: str,
) -> tuple[list[tuple[str, str, str]], random.Random]:
    """Build the full list of tagged QA pairs for ``data``.

    Returns ``(qa_tagged, rng)`` so callers can continue using the same RNG
    for sampling (e.g. ``_balanced_sample_qa``).

    Args:
        data: OCRData with ocr_dense populated.
        task_id: Used to seed the RNG for reproducibility.
    """
    words = data.ocr_dense or []
    valid_words = [w for w in words if w.valid]

    num_invalid = sum(1 for w in words if not w.valid)
    allow_text_to_bbox = num_invalid < 5

    rng = random.Random(hash(task_id))
    qa_tagged: list[tuple[str, str, str]] = []

    # ------------------------------------------------------------------
    # Types 1-4: per-bbox ↔ text (no hierarchy needed)
    # ------------------------------------------------------------------
    text_to_bboxes: dict[str, list[Any]] = defaultdict(list)
    for raw in valid_words:
        bbox = raw.bbox_2d
        text = (raw.text_content or "").strip()
        if not bbox or len(bbox) != 4 or not text:
            continue
        text_to_bboxes[text].append(bbox)

    for text, bboxes in text_to_bboxes.items():
        mode = rng.choice((0, 1, 2, 3) if allow_text_to_bbox else (0, 1))
        if mode == 0:
            q, a = _gen_bbox_to_text(rng, bboxes[0], text)
            qa_tagged.append((QA_TYPE_BBOX_TO_TEXT, q, a))
        elif mode == 1:
            point = _bbox_center(bboxes[0])
            q, a = _gen_point_to_text(rng, point, text)
            qa_tagged.append((QA_TYPE_POINT_TO_TEXT, q, a))
        elif allow_text_to_bbox:
            loc_type = rng.choice([QA_TYPE_TEXT_TO_BBOX, QA_TYPE_TEXT_TO_POINT])
            if len(bboxes) == 1:
                if loc_type == QA_TYPE_TEXT_TO_BBOX:
                    q, a = rng.choice((
                        lambda t, b: _gen_text_to_bbox_single(rng, t, b),
                        lambda t, b: _gen_text_to_bbox_multi(rng, t, [b]),
                    ))(text, bboxes[0])
                else:
                    q, a = rng.choice((
                        lambda t, b: _gen_text_to_point_single(rng, t, b),
                        lambda t, b: _gen_text_to_point_multi(rng, t, [b]),
                    ))(text, bboxes[0])
                qa_tagged.append((loc_type, q, a))
            else:
                if loc_type == QA_TYPE_TEXT_TO_BBOX:
                    q, a = _gen_text_to_bbox_multi(rng, text, bboxes)
                else:
                    q, a = _gen_text_to_point_multi(rng, text, bboxes)
                qa_tagged.append((loc_type, q, a))

    return qa_tagged, rng


def build_conversation(
    qa_tagged: list[tuple[str, str, str]],
    rng: random.Random,
    image_name: str,
) -> ConversationSample | None:
    """Sample from qa_tagged and assemble a ConversationSample, or None if empty."""
    qa_pairs = _balanced_sample_qa(qa_tagged, MAX_QA_PAIRS, rng)
    if not qa_pairs:
        return None
    first_q, first_a = qa_pairs[0]
    messages: list[Message] = [
        Message(sender="user", fragments=[ImageMedia(value=image_name), first_q]),
        Message(sender="assistant", fragments=[first_a]),
    ]
    for q, a in qa_pairs[1:]:
        messages.append(Message(sender="user", fragments=[q]))
        messages.append(Message(sender="assistant", fragments=[a]))
    return ConversationSample(conversation=messages)


def build_dense_conversation(
    words: list[OCRDenseWord],
    rng: random.Random,
    image_name: str,
) -> ConversationSample:
    """Build a single-turn dense dump conversation listing all words with bboxes.

    Used for ~10% of images where OCR is provably complete (no missing text).
    """
    q, a = _gen_dense_dump(rng, words)
    return ConversationSample(
        conversation=[
            Message(sender="user", fragments=[ImageMedia(value=image_name), q]),
            Message(sender="assistant", fragments=[a]),
        ]
    )
