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

"""Unit tests for field_utils.py -- wildcard dot-path nested field utilities."""

from __future__ import annotations

import json

import pytest

from nemo_curator.stages.text.translation.field_utils import (
    extract_nested_fields,
    is_wildcard_path,
    normalize_text_field,
    parse_structured_value,
    set_nested_fields,
)


# ---------------------------------------------------------------------------
# extract_nested_fields tests
# ---------------------------------------------------------------------------


class TestExtractNestedFields:
    """Tests for extract_nested_fields() with various path patterns."""

    def test_simple_flat_field(self) -> None:
        """A flat field name (no wildcards) extracts a single value."""
        record = {"text": "Hello world", "id": 1}
        result = extract_nested_fields(record, "text")
        assert result == ["Hello world"]

    def test_wildcard_messages_content(self) -> None:
        """The canonical OpenAI-format path ``messages.*.content`` extracts all content strings."""
        record = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
        }
        result = extract_nested_fields(record, "messages.*.content")
        assert result == ["Hello", "Hi there"]

    def test_wildcard_conversations_value(self) -> None:
        """Alternate format: ``conversations.*.value``."""
        record = {
            "conversations": [
                {"from": "human", "value": "What is AI?"},
                {"from": "gpt", "value": "Artificial Intelligence..."},
            ]
        }
        result = extract_nested_fields(record, "conversations.*.value")
        assert result == ["What is AI?", "Artificial Intelligence..."]

    def test_missing_key_returns_empty(self) -> None:
        """When the top-level key does not exist, return an empty list."""
        record = {"other_field": "data"}
        result = extract_nested_fields(record, "messages.*.content")
        assert result == []

    def test_empty_list_returns_empty(self) -> None:
        """When the list at the wildcard level is empty, return an empty list."""
        record = {"messages": []}
        result = extract_nested_fields(record, "messages.*.content")
        assert result == []

    def test_missing_nested_key_skipped(self) -> None:
        """Items in the list that lack the target key are silently skipped."""
        record = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "system"},  # no "content" key
                {"role": "assistant", "content": "Hi"},
            ]
        }
        result = extract_nested_fields(record, "messages.*.content")
        assert result == ["Hello", "Hi"]

    def test_non_string_values_skipped(self) -> None:
        """Non-string leaf values are silently skipped."""
        record = {
            "messages": [
                {"role": "user", "content": 42},
                {"role": "assistant", "content": "Hi"},
            ]
        }
        result = extract_nested_fields(record, "messages.*.content")
        assert result == ["Hi"]

    def test_deeply_nested_path(self) -> None:
        """Paths with multiple levels of nesting work correctly."""
        record = {
            "data": {
                "items": [
                    {"meta": {"title": "First"}},
                    {"meta": {"title": "Second"}},
                ]
            }
        }
        result = extract_nested_fields(record, "data.items.*.meta.title")
        assert result == ["First", "Second"]

    def test_single_item_list(self) -> None:
        """A list with a single item returns a single-element list."""
        record = {
            "messages": [
                {"role": "user", "content": "Only one"},
            ]
        }
        result = extract_nested_fields(record, "messages.*.content")
        assert result == ["Only one"]

    def test_flat_field_non_string_returns_empty(self) -> None:
        """A flat field pointing to a non-string value returns empty list."""
        record = {"count": 42}
        result = extract_nested_fields(record, "count")
        assert result == []

    def test_empty_record(self) -> None:
        """An empty dict returns empty list."""
        result = extract_nested_fields({}, "text")
        assert result == []


# ---------------------------------------------------------------------------
# set_nested_fields tests
# ---------------------------------------------------------------------------


class TestSetNestedFields:
    """Tests for set_nested_fields() -- writing values back into nested dicts."""

    def test_simple_flat_field(self) -> None:
        """Replace a flat string field."""
        record = {"text": "original", "id": 1}
        result = set_nested_fields(record, "text", ["replaced"])
        assert result["text"] == "replaced"
        assert result["id"] == 1
        # Original should not be mutated
        assert record["text"] == "original"

    def test_wildcard_round_trip(self) -> None:
        """Extract, modify, and set back -- round trip should replace all values."""
        record = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
        }
        extracted = extract_nested_fields(record, "messages.*.content")
        assert extracted == ["Hello", "Hi there"]

        # Modify extracted values
        translated = [v.upper() for v in extracted]
        result = set_nested_fields(record, "messages.*.content", translated)

        assert result["messages"][0]["content"] == "HELLO"
        assert result["messages"][1]["content"] == "HI THERE"
        # Roles should be preserved
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"

    def test_original_not_mutated(self) -> None:
        """set_nested_fields makes a deep copy; original is unchanged."""
        record = {
            "messages": [
                {"role": "user", "content": "Original"},
            ]
        }
        result = set_nested_fields(record, "messages.*.content", ["Changed"])
        assert record["messages"][0]["content"] == "Original"
        assert result["messages"][0]["content"] == "Changed"

    def test_value_count_mismatch_warns(self) -> None:
        """When values list is longer than positions found, a warning is logged."""
        record = {"text": "only one"}
        # We provide 2 values but there's only 1 position
        result = set_nested_fields(record, "text", ["replacement", "extra"])
        # The single field gets replaced; the extra value triggers a warning
        assert result["text"] == "replacement"

    def test_missing_key_no_error(self) -> None:
        """When path doesn't match anything, no error is raised."""
        record = {"other": "value"}
        # No match for "text" -- values list has 1 element, 0 set => warning
        result = set_nested_fields(record, "text", ["noop"])
        assert result == {"other": "value"}

    def test_empty_values_no_change(self) -> None:
        """Empty values list with no matching positions is fine."""
        record = {"messages": []}
        result = set_nested_fields(record, "messages.*.content", [])
        assert result == {"messages": []}

    def test_deeply_nested_set(self) -> None:
        """Setting values in deeply nested structures works."""
        record = {
            "data": {
                "items": [
                    {"meta": {"title": "First"}},
                    {"meta": {"title": "Second"}},
                ]
            }
        }
        result = set_nested_fields(record, "data.items.*.meta.title", ["A", "B"])
        assert result["data"]["items"][0]["meta"]["title"] == "A"
        assert result["data"]["items"][1]["meta"]["title"] == "B"


# ---------------------------------------------------------------------------
# is_wildcard_path tests
# ---------------------------------------------------------------------------


class TestIsWildcardPath:
    """Tests for is_wildcard_path()."""

    def test_wildcard_present(self) -> None:
        assert is_wildcard_path("messages.*.content") is True

    def test_no_wildcard(self) -> None:
        assert is_wildcard_path("text") is False

    def test_multiple_wildcards(self) -> None:
        assert is_wildcard_path("a.*.b.*.c") is True

    def test_empty_string(self) -> None:
        assert is_wildcard_path("") is False


# ---------------------------------------------------------------------------
# normalize_text_field tests
# ---------------------------------------------------------------------------


class TestNormalizeTextField:
    """Tests for normalize_text_field()."""

    def test_single_string(self) -> None:
        result = normalize_text_field("text")
        assert result == ["text"]

    def test_list_of_strings(self) -> None:
        result = normalize_text_field(["text", "messages.*.content"])
        assert result == ["text", "messages.*.content"]

    def test_empty_list(self) -> None:
        result = normalize_text_field([])
        assert result == []

    def test_single_item_list(self) -> None:
        result = normalize_text_field(["only"])
        assert result == ["only"]


# ---------------------------------------------------------------------------
# parse_structured_value tests
# ---------------------------------------------------------------------------


class TestParseStructuredValue:
    """Tests for parse_structured_value()."""

    def test_dict_passthrough(self) -> None:
        d = {"key": "value"}
        result = parse_structured_value(d)
        assert result is d

    def test_valid_json_string(self) -> None:
        s = '{"key": "value"}'
        result = parse_structured_value(s)
        assert result == {"key": "value"}

    def test_invalid_json_string(self) -> None:
        s = "not json at all"
        result = parse_structured_value(s)
        assert result is None

    def test_json_array_returns_none(self) -> None:
        """A JSON array is not a dict, so returns None."""
        s = '[1, 2, 3]'
        result = parse_structured_value(s)
        assert result is None

    def test_integer_returns_none(self) -> None:
        result = parse_structured_value(42)
        assert result is None

    def test_none_returns_none(self) -> None:
        result = parse_structured_value(None)
        assert result is None

    def test_nested_json_string(self) -> None:
        s = json.dumps({"messages": [{"role": "user", "content": "Hi"}]})
        result = parse_structured_value(s)
        assert result is not None
        assert "messages" in result
        assert result["messages"][0]["content"] == "Hi"
