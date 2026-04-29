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

"""Tests for nemo_curator/stages/interleaved/utils/schema.py."""

import logging

import pyarrow as pa

from nemo_curator.stages.interleaved.utils.schema import align_table, reconcile_schema, resolve_schema
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA, RESERVED_COLUMNS

_BASE = [pa.field("sample_id", pa.string()), pa.field("position", pa.int32()), pa.field("modality", pa.string())]


# --- resolve_schema ---


def test_resolve_schema_none_returns_none() -> None:
    assert resolve_schema(None, None) is None


def test_resolve_schema_overrides_merge_onto_interleaved_schema() -> None:
    result = resolve_schema(None, {"url": pa.string(), "text_content": pa.large_string()})
    assert all(result.get_field_index(n) >= 0 for n in RESERVED_COLUMNS)
    assert result.field("url").type == pa.string()
    assert result.field("text_content").type == pa.large_string()  # reserved field overridden


def test_resolve_schema_explicit_schema_ignores_overrides(caplog) -> None:
    custom = pa.schema([pa.field("sample_id", pa.string())])
    with caplog.at_level(logging.WARNING, logger="nemo_curator"):
        result = resolve_schema(custom, {"extra": pa.float64()})
    assert result == custom
    assert any("schema_overrides ignored" in r.message for r in caplog.records)


# --- reconcile_schema ---


def test_reconcile_schema_large_string_not_downcast() -> None:
    result = reconcile_schema(pa.schema([*_BASE, pa.field("text_content", pa.large_string())]))
    assert result.field("text_content").type == pa.large_string()


def test_reconcile_schema_unwraps_dictionary_passthrough() -> None:
    result = reconcile_schema(pa.schema([*_BASE, pa.field("tag", pa.dictionary(pa.int8(), pa.string()))]))
    assert result.field("tag").type == pa.string()


def test_reconcile_schema_casts_reserved_to_canonical_type() -> None:
    result = reconcile_schema(
        pa.schema(
            [pa.field("sample_id", pa.string()), pa.field("position", pa.int64()), pa.field("modality", pa.string())]
        )
    )
    assert result.field("position").type == pa.int32()


# --- align_table ---


def test_align_table_null_fills_missing_column() -> None:
    # Minimal table — only sample_id present; binary_content and all others must be null-filled
    table = pa.table({"sample_id": pa.array(["s1"], type=pa.string())})
    result = align_table(table, INTERLEAVED_SCHEMA)
    assert result.schema == INTERLEAVED_SCHEMA
    assert result.column("binary_content").null_count == 1


def test_align_table_drops_extra_columns_and_casts_passthrough() -> None:
    target = pa.schema([pa.field("sample_id", pa.string()), pa.field("score", pa.float64())])
    table = pa.table({"sample_id": ["s1"], "score": pa.array([0.5], type=pa.float32()), "extra": ["drop"]})
    result = align_table(table, target)
    assert result.schema.names == ["sample_id", "score"]
    assert result.schema.field("score").type == pa.float64()
