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

"""Output utility functions for the translation pipeline.

Provides helpers for:
- Building segment-level source/target pairs (Gap 3.2)
- Constructing structured translation metadata (Gap 4.2)
- Merging FAITH scores into metadata (Gap 5.2)
- Reconstructing OpenAI-messages format with translations (Gap 5.4)
"""

from __future__ import annotations

import copy
import json
from typing import Any


def build_segment_pairs(segments: list[str], translations: list[str]) -> str:
    """Build JSON-serialized list of src/tgt pairs.

    Pairs each source segment with its corresponding translation.  If the
    lists have different lengths, pairing stops at the shorter list.

    Args:
        segments: Source-language segments.
        translations: Target-language translations (same order as *segments*).

    Returns:
        JSON string encoding a list of ``{"src": ..., "tgt": ...}`` dicts.

    Example::

        >>> build_segment_pairs(["Hello", "World"], ["Hola", "Mundo"])
        '[{"src": "Hello", "tgt": "Hola"}, {"src": "World", "tgt": "Mundo"}]'
    """
    pairs = [{"src": s, "tgt": t} for s, t in zip(segments, translations)]
    return json.dumps(pairs, ensure_ascii=False)


def build_translation_metadata(
    target_lang: str,
    translated_text: str,
    segment_pairs_json: str | None = None,
    faith_scores: dict[str, Any] | None = None,
) -> str:
    """Build a JSON-serialized translation metadata structure.

    Follows Speaker's output convention::

        {
            "target_lang": "hi",
            "translation": {"content": "reassembled translated text"},
            "segmented_translation": [{"src": "...", "tgt": "..."}, ...],
            "faith_scores": { ... }          # optional
        }

    Args:
        target_lang: ISO 639-1 target language code.
        translated_text: The fully reassembled translated text.
        segment_pairs_json: Optional JSON string of segment pairs (from
            :func:`build_segment_pairs`).  Decoded and embedded as a list.
        faith_scores: Optional dict of FAITH evaluation scores to include.

    Returns:
        JSON string encoding the metadata structure.
    """
    meta: dict[str, Any] = {
        "target_lang": target_lang,
        "translation": {"content": translated_text},
    }

    if segment_pairs_json is not None:
        try:
            meta["segmented_translation"] = json.loads(segment_pairs_json)
        except (json.JSONDecodeError, TypeError):
            meta["segmented_translation"] = []
    else:
        meta["segmented_translation"] = []

    if faith_scores is not None:
        meta["faith_scores"] = faith_scores

    return json.dumps(meta, ensure_ascii=False)


def merge_faith_scores_into_metadata(
    metadata_json: str,
    faith_scores: dict[str, Any],
) -> str:
    """Merge FAITH evaluation scores into an existing translation metadata JSON.

    Args:
        metadata_json: JSON string of existing translation metadata (from
            :func:`build_translation_metadata`).
        faith_scores: Dict of FAITH scores (e.g.
            ``{"Fluency": 4.0, "Accuracy": 3.5, ...}``).

    Returns:
        Updated JSON string with ``"faith_scores"`` key added/overwritten.
    """
    try:
        meta = json.loads(metadata_json)
    except (json.JSONDecodeError, TypeError):
        meta = {}

    meta["faith_scores"] = faith_scores
    return json.dumps(meta, ensure_ascii=False)


def reconstruct_messages_with_translation(
    original_messages: list[dict[str, Any]],
    translated_text: str,
    field_path: str = "content",
) -> list[dict[str, Any]]:
    """Replace content in messages with translated text, return new messages.

    For OpenAI-format message lists, this replaces the value at *field_path*
    in each message dict with consecutive portions of *translated_text*.

    When *translated_text* cannot be split into enough parts for all messages,
    the full translated text is placed into the first message's field and the
    remaining messages retain their original values.

    Args:
        original_messages: List of message dicts (e.g.
            ``[{"role": "user", "content": "Hello"}, ...]``).
        translated_text: The translated text to insert.  If it contains the
            separator ``"\\n---\\n"`` it will be split across messages;
            otherwise the entire string goes into the first message.
        field_path: Dot-separated path to the field to replace within each
            message dict (default ``"content"``).

    Returns:
        Deep copy of *original_messages* with the specified field replaced by
        translated content.
    """
    if not original_messages:
        return []

    messages = copy.deepcopy(original_messages)

    # Split translated text into per-message parts when a separator is present
    separator = "\n---\n"
    if separator in translated_text:
        parts = translated_text.split(separator)
    else:
        parts = [translated_text]

    # Resolve nested field_path (e.g. "content" or "nested.content")
    path_keys = field_path.split(".")

    for idx, msg in enumerate(messages):
        if idx < len(parts):
            _set_nested(msg, path_keys, parts[idx])
        # Messages beyond the available parts keep their original values

    return messages


def _set_nested(obj: dict[str, Any], keys: list[str], value: Any) -> None:
    """Set a value in a nested dict following the given key path.

    Args:
        obj: The dict to modify in place.
        keys: List of string keys forming the path (e.g. ``["a", "b"]``
            sets ``obj["a"]["b"]``).
        value: The value to set at the terminal key.
    """
    for key in keys[:-1]:
        if key in obj and isinstance(obj[key], dict):
            obj = obj[key]
        else:
            return  # path doesn't exist; skip silently
    if keys:
        obj[keys[-1]] = value
