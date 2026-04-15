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

"""Unit tests for output_utils.py -- translation output helpers."""

from __future__ import annotations

import json

import pytest

from nemo_curator.stages.text.translation.output_utils import (
    build_segment_pairs,
    build_translation_metadata,
    merge_faith_scores_into_metadata,
    reconstruct_messages_with_translation,
)


# ---------------------------------------------------------------------------
# build_segment_pairs tests
# ---------------------------------------------------------------------------


class TestBuildSegmentPairs:
    """Tests for build_segment_pairs() -- JSON segment pair construction."""

    def test_basic_pairs(self) -> None:
        """Matching-length lists produce correct src/tgt pairs."""
        result = build_segment_pairs(["Hello", "World"], ["Hola", "Mundo"])
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[0] == {"src": "Hello", "tgt": "Hola"}
        assert parsed[1] == {"src": "World", "tgt": "Mundo"}

    def test_empty_lists(self) -> None:
        """Empty input lists produce an empty JSON array."""
        result = build_segment_pairs([], [])
        parsed = json.loads(result)
        assert parsed == []

    def test_single_pair(self) -> None:
        """Single-element lists produce a single pair."""
        result = build_segment_pairs(["one"], ["uno"])
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0] == {"src": "one", "tgt": "uno"}

    def test_mismatched_lengths_uses_shorter(self) -> None:
        """When lists differ in length, pairing stops at the shorter one."""
        result = build_segment_pairs(["a", "b", "c"], ["x", "y"])
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[0] == {"src": "a", "tgt": "x"}
        assert parsed[1] == {"src": "b", "tgt": "y"}

    def test_unicode_content(self) -> None:
        """Unicode strings are preserved (ensure_ascii=False)."""
        result = build_segment_pairs(["Hello"], ["\u4f60\u597d"])
        parsed = json.loads(result)
        assert parsed[0]["tgt"] == "\u4f60\u597d"

    def test_output_is_valid_json(self) -> None:
        """Output is a valid JSON string that can be parsed."""
        result = build_segment_pairs(["a"], ["b"])
        assert isinstance(result, str)
        # Should not raise
        json.loads(result)


# ---------------------------------------------------------------------------
# build_translation_metadata tests
# ---------------------------------------------------------------------------


class TestBuildTranslationMetadata:
    """Tests for build_translation_metadata() -- structured metadata construction."""

    def test_basic_metadata(self) -> None:
        """Minimal metadata with just target_lang and translated_text."""
        result = build_translation_metadata(
            target_lang="hi",
            translated_text="Translated content here.",
        )
        parsed = json.loads(result)
        assert parsed["target_lang"] == "hi"
        assert parsed["translation"]["content"] == "Translated content here."
        # Default empty segmented_translation when not provided
        assert parsed["segmented_translation"] == []

    def test_with_segment_pairs(self) -> None:
        """Segment pairs JSON is decoded and embedded as a list."""
        pairs_json = json.dumps([{"src": "Hello", "tgt": "Hola"}])
        result = build_translation_metadata(
            target_lang="es",
            translated_text="Hola",
            segment_pairs_json=pairs_json,
        )
        parsed = json.loads(result)
        assert len(parsed["segmented_translation"]) == 1
        assert parsed["segmented_translation"][0]["src"] == "Hello"

    def test_with_faith_scores(self) -> None:
        """FAITH scores are included when provided."""
        scores = {"Fluency": 4.0, "Accuracy": 3.5}
        result = build_translation_metadata(
            target_lang="zh",
            translated_text="test",
            faith_scores=scores,
        )
        parsed = json.loads(result)
        assert parsed["faith_scores"]["Fluency"] == 4.0
        assert parsed["faith_scores"]["Accuracy"] == 3.5

    def test_without_faith_scores(self) -> None:
        """When faith_scores is None, the key is absent."""
        result = build_translation_metadata(
            target_lang="de",
            translated_text="test",
        )
        parsed = json.loads(result)
        assert "faith_scores" not in parsed

    def test_invalid_segment_pairs_json(self) -> None:
        """Invalid JSON for segment pairs falls back to empty list."""
        result = build_translation_metadata(
            target_lang="fr",
            translated_text="test",
            segment_pairs_json="not valid json",
        )
        parsed = json.loads(result)
        assert parsed["segmented_translation"] == []


# ---------------------------------------------------------------------------
# merge_faith_scores_into_metadata tests
# ---------------------------------------------------------------------------


class TestMergeFaithScoresIntoMetadata:
    """Tests for merge_faith_scores_into_metadata()."""

    def test_merge_into_existing_metadata(self) -> None:
        """Scores are added to existing metadata JSON."""
        meta_json = json.dumps({"target_lang": "hi", "translation": {"content": "test"}})
        scores = {"Fluency": 5.0, "Accuracy": 4.0}
        result = merge_faith_scores_into_metadata(meta_json, scores)
        parsed = json.loads(result)
        assert parsed["faith_scores"] == scores
        # Original keys are preserved
        assert parsed["target_lang"] == "hi"

    def test_merge_overwrites_existing_scores(self) -> None:
        """If faith_scores already exists, it gets overwritten."""
        meta_json = json.dumps({"faith_scores": {"Fluency": 1.0}})
        new_scores = {"Fluency": 5.0}
        result = merge_faith_scores_into_metadata(meta_json, new_scores)
        parsed = json.loads(result)
        assert parsed["faith_scores"]["Fluency"] == 5.0

    def test_merge_into_invalid_json(self) -> None:
        """Invalid metadata JSON is replaced with just the scores."""
        result = merge_faith_scores_into_metadata("not json", {"Fluency": 3.0})
        parsed = json.loads(result)
        assert parsed["faith_scores"]["Fluency"] == 3.0

    def test_merge_into_empty_json(self) -> None:
        """Empty object gets scores added."""
        result = merge_faith_scores_into_metadata("{}", {"Fluency": 4.0})
        parsed = json.loads(result)
        assert parsed["faith_scores"]["Fluency"] == 4.0


# ---------------------------------------------------------------------------
# reconstruct_messages_with_translation tests
# ---------------------------------------------------------------------------


class TestReconstructMessagesWithTranslation:
    """Tests for reconstruct_messages_with_translation() -- OpenAI message format reconstruction."""

    def test_single_message_replacement(self) -> None:
        """Single message gets its content replaced."""
        original = [{"role": "user", "content": "Hello"}]
        result = reconstruct_messages_with_translation(original, "Hola")
        assert result[0]["content"] == "Hola"
        assert result[0]["role"] == "user"
        # Original should not be mutated
        assert original[0]["content"] == "Hello"

    def test_multiple_messages_with_separator(self) -> None:
        """Multiple messages are split using the \\n---\\n separator."""
        original = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        translated_text = "Hola\n---\nHola, que tal"
        result = reconstruct_messages_with_translation(original, translated_text)
        assert result[0]["content"] == "Hola"
        assert result[1]["content"] == "Hola, que tal"

    def test_fewer_parts_than_messages(self) -> None:
        """When translated text has fewer parts than messages, remaining keep originals."""
        original = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]
        # Only one part (no separator), so only the first message gets replaced
        result = reconstruct_messages_with_translation(original, "Hola")
        assert result[0]["content"] == "Hola"
        assert result[1]["content"] == "Hi there"  # unchanged
        assert result[2]["content"] == "How are you?"  # unchanged

    def test_empty_messages_list(self) -> None:
        """Empty original messages returns empty list."""
        result = reconstruct_messages_with_translation([], "anything")
        assert result == []

    def test_original_not_mutated(self) -> None:
        """The original messages list is not mutated."""
        original = [{"role": "user", "content": "Original"}]
        result = reconstruct_messages_with_translation(original, "Changed")
        assert original[0]["content"] == "Original"
        assert result[0]["content"] == "Changed"

    def test_nested_field_path(self) -> None:
        """Custom field_path replaces a nested field within each message."""
        original = [
            {"role": "user", "data": {"text": "Hello"}},
        ]
        result = reconstruct_messages_with_translation(
            original, "Hola", field_path="data.text"
        )
        assert result[0]["data"]["text"] == "Hola"
        assert result[0]["role"] == "user"

    def test_field_path_missing_skips_silently(self) -> None:
        """When field_path does not exist in a message, that message is unchanged."""
        original = [
            {"role": "user", "content": "Hello"},
        ]
        # "nonexistent.path" won't match anything in the message
        result = reconstruct_messages_with_translation(
            original, "Hola", field_path="nonexistent.path"
        )
        # Message is unchanged because the path doesn't match
        assert result[0]["content"] == "Hello"

    def test_preserves_extra_fields(self) -> None:
        """Extra fields in messages (like tool_calls, metadata) are preserved."""
        original = [
            {"role": "user", "content": "Hello", "name": "Alice", "metadata": {"ts": 123}},
        ]
        result = reconstruct_messages_with_translation(original, "Hola")
        assert result[0]["content"] == "Hola"
        assert result[0]["name"] == "Alice"
        assert result[0]["metadata"]["ts"] == 123
