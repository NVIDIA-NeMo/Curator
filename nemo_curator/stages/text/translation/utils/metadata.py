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

"""Output helpers for translation metadata and message reconstruction."""

from __future__ import annotations

import copy
import json
from typing import Any


def build_translation_metadata(
    target_lang: str,
    translated_text: str | None = None,
    segment_pairs_json: str | None = None,
    translation_map: dict[str, Any] | None = None,
    segmented_translation_map: dict[str, Any] | None = None,
) -> str:
    """Build Speaker-style translation metadata as JSON."""
    if translation_map is None:
        meta_translation: dict[str, Any] = {"content": translated_text or ""}
    else:
        meta_translation = translation_map

    if segmented_translation_map is None:
        if segment_pairs_json is not None:
            try:
                meta_segmented: Any = json.loads(segment_pairs_json)
            except (json.JSONDecodeError, TypeError):
                meta_segmented = []
        else:
            meta_segmented = []
    else:
        meta_segmented = segmented_translation_map

    meta: dict[str, Any] = {
        "target_lang": target_lang,
        "translation": meta_translation,
        "segmented_translation": meta_segmented,
    }

    return json.dumps(meta, ensure_ascii=False)


def merge_faith_scores_into_metadata(
    metadata_json: str,
    faith_scores: dict[str, Any],
) -> str:
    """Merge FAITH scores into existing translation metadata."""
    try:
        meta = json.loads(metadata_json)
    except (json.JSONDecodeError, TypeError):
        meta = {}

    meta["faith_scores"] = faith_scores
    return json.dumps(meta, ensure_ascii=False)


def reconstruct_messages_with_translation(
    original_messages: list[dict[str, Any]],
    translated_text: Any,
    field_path: str = "content",
) -> list[dict[str, Any]]:
    """Return a copy of messages with translated content inserted."""
    if not original_messages:
        return []

    messages = copy.deepcopy(original_messages)

    structured_messages = _parse_structured_messages(translated_text)
    if structured_messages is not None:
        return structured_messages

    translated_text_str = "" if translated_text is None else str(translated_text)

    separator = "\n---\n"
    if separator in translated_text_str:
        parts = translated_text_str.split(separator)
    else:
        parts = [translated_text_str]

    path_keys = field_path.split(".")

    for idx, msg in enumerate(messages):
        if idx < len(parts):
            _set_nested(msg, path_keys, parts[idx])

    return messages


def _set_nested(obj: dict[str, Any], keys: list[str], value: Any) -> None:
    """Set a nested value when the full path already exists."""
    for key in keys[:-1]:
        if key in obj and isinstance(obj[key], dict):
            obj = obj[key]
        else:
            return
    if keys:
        obj[keys[-1]] = value


def _parse_structured_messages(translated_text: Any) -> list[dict[str, Any]] | None:
    """Return translated messages when they are already structured."""
    if isinstance(translated_text, list):
        if all(isinstance(item, dict) for item in translated_text):
            return copy.deepcopy(translated_text)
        return None

    if isinstance(translated_text, str):
        stripped = translated_text.strip()
        if not stripped.startswith("["):
            return None
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            return None
        if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
            return parsed

    return None
