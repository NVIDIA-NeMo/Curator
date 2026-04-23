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

from nemo_curator.stages.text.translation.utils.metadata import (
    build_translation_metadata,
    merge_faith_scores_into_metadata,
    reconstruct_messages_with_translation,
)


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

    def test_with_segmented_translation_map(self) -> None:
        """Segmented translation mappings are embedded as provided."""
        result = build_translation_metadata(
            target_lang="es",
            translated_text="Hola",
            segmented_translation_map=[{"src": "Hello", "tgt": "Hola"}],
        )
        parsed = json.loads(result)
        assert len(parsed["segmented_translation"]) == 1
        assert parsed["segmented_translation"][0]["src"] == "Hello"

    def test_without_faith_scores(self) -> None:
        """build_translation_metadata never emits the faith_scores key.

        FAITH scores are attached separately via
        :func:`merge_faith_scores_into_metadata` once scoring has run.
        """
        result = build_translation_metadata(
            target_lang="de",
            translated_text="test",
        )
        parsed = json.loads(result)
        assert "faith_scores" not in parsed

    def test_without_segmented_translation_map(self) -> None:
        """Omitting segmented translation data falls back to an empty list."""
        result = build_translation_metadata(
            target_lang="fr",
            translated_text="test",
        )
        parsed = json.loads(result)
        assert parsed["segmented_translation"] == []

    def test_with_translation_maps(self) -> None:
        """Speaker-style translation maps are preserved verbatim."""
        result = build_translation_metadata(
            target_lang="hi",
            translation_map={"question": "Namaste"},
            segmented_translation_map={
                "question": [{"src": "Hello", "tgt": "Namaste"}],
            },
        )
        parsed = json.loads(result)
        assert parsed["translation"]["question"] == "Namaste"
        assert parsed["segmented_translation"]["question"][0]["src"] == "Hello"


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

    def test_structured_messages_passthrough(self) -> None:
        """Structured translated messages should be returned directly."""
        original = [{"role": "user", "content": "Hello"}]
        translated = [{"role": "user", "content": "Hola"}]
        result = reconstruct_messages_with_translation(original, translated)
        assert result == translated
        assert original[0]["content"] == "Hello"
