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

"""Unit tests for ReassemblyStage (coarse and fine modes)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from nemo_curator.stages.text.translation.stages.reassembly import (
    _INTERNAL_COLUMNS,
    ReassemblyStage,
)
from nemo_curator.stages.text.translation.stages.segmentation import SegmentationStage
from nemo_curator.tasks import DocumentBatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(texts: list[str], **extra_columns: list) -> DocumentBatch:
    """Build a minimal DocumentBatch from a list of text strings."""
    data = {"text": texts}
    data.update(extra_columns)
    df = pd.DataFrame(data)
    return DocumentBatch(data=df, dataset_name="test", task_id="1")


def _segment_and_add_translations(
    texts: list[str],
    mode: str = "coarse",
    translate_fn=None,
    extra_columns: dict | None = None,
) -> DocumentBatch:
    """Segment texts, then add a _translated column (identity or custom).

    Returns a DocumentBatch ready for ReassemblyStage.process().
    """
    data = {"text": texts}
    if extra_columns:
        data.update(extra_columns)
    df = pd.DataFrame(data)
    batch = DocumentBatch(data=df, dataset_name="test", task_id="1")

    seg_stage = SegmentationStage(mode=mode)
    segmented = seg_stage.process(batch)
    seg_df = segmented.to_pandas()

    if translate_fn is None:
        # Identity translation: _translated == _seg_segments
        seg_df["_translated"] = seg_df["_seg_segments"]
    else:
        seg_df["_translated"] = seg_df["_seg_segments"].apply(translate_fn)

    return DocumentBatch(
        data=seg_df,
        dataset_name=segmented.dataset_name,
        task_id=segmented.task_id,
        _metadata=segmented._metadata,
        _stage_perf=segmented._stage_perf,
    )


# ---------------------------------------------------------------------------
# spaCy availability check for fine-mode tests
# ---------------------------------------------------------------------------

_SPACY_AVAILABLE = False
try:
    import spacy  # noqa: F401

    try:
        spacy.load("en_core_web_sm")
        _SPACY_AVAILABLE = True
    except OSError:
        pass
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Coarse reassembly tests
# ---------------------------------------------------------------------------


class TestCoarseReassembly:
    """Tests for coarse-mode reassembly."""

    def test_coarse_reassembly(self) -> None:
        """Reassemble coarse-segmented text with identity translations."""
        text = "Hello world\nThis is a test\nGoodbye"
        translated_batch = _segment_and_add_translations([text], mode="coarse")

        stage = ReassemblyStage()
        result = stage.process(translated_batch)
        df = result.to_pandas()

        assert len(df) == 1
        assert df.iloc[0]["translated_text"] == text

    def test_coarse_reassembly_with_real_translations(self) -> None:
        """Reassemble with actual different translations substituted."""
        text = "Hello world\n---\nGoodbye"
        translated_batch = _segment_and_add_translations(
            [text],
            mode="coarse",
            translate_fn=lambda s: f"[TR:{s}]",
        )

        stage = ReassemblyStage()
        result = stage.process(translated_batch)
        df = result.to_pandas()

        assert len(df) == 1
        reassembled = df.iloc[0]["translated_text"]
        # "---" is non-translatable and should be preserved verbatim
        assert "---" in reassembled
        # Translated segments should have the [TR:...] wrapper
        assert "[TR:Hello world]" in reassembled
        assert "[TR:Goodbye]" in reassembled
        # Full structure preserved
        assert reassembled == "[TR:Hello world]\n---\n[TR:Goodbye]"


# ---------------------------------------------------------------------------
# Fine reassembly tests
# ---------------------------------------------------------------------------


class TestFineReassembly:
    """Tests for fine-mode reassembly."""

    @pytest.mark.skipif(not _SPACY_AVAILABLE, reason="spaCy en_core_web_sm not available")
    def test_fine_reassembly(self) -> None:
        """Reassemble fine-segmented text with identity translations."""
        text = "Hello world. This is a test. Goodbye."
        translated_batch = _segment_and_add_translations([text], mode="fine")

        stage = ReassemblyStage()
        result = stage.process(translated_batch)
        df = result.to_pandas()

        assert len(df) == 1
        # Identity translation should reconstruct the original text
        assert df.iloc[0]["translated_text"] == text


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Segment then reassemble with identity translation should produce original text."""

    def test_round_trip_coarse(self) -> None:
        """Coarse segment -> identity translate -> reassemble == original."""
        text = "First line\n```python\ncode here\n```\n  Indented line\n\nLast line"
        translated_batch = _segment_and_add_translations([text], mode="coarse")

        stage = ReassemblyStage()
        result = stage.process(translated_batch)
        df = result.to_pandas()

        assert len(df) == 1
        assert df.iloc[0]["translated_text"] == text

    @pytest.mark.skipif(not _SPACY_AVAILABLE, reason="spaCy en_core_web_sm not available")
    def test_round_trip_fine(self) -> None:
        """Fine segment -> identity translate -> reassemble == original."""
        text = "This is sentence one. And here is sentence two! What about three?"
        translated_batch = _segment_and_add_translations([text], mode="fine")

        stage = ReassemblyStage()
        result = stage.process(translated_batch)
        df = result.to_pandas()

        assert len(df) == 1
        assert df.iloc[0]["translated_text"] == text


# ---------------------------------------------------------------------------
# Column management tests
# ---------------------------------------------------------------------------


class TestColumnManagement:
    """Tests for column preservation and cleanup."""

    def test_preserves_original_columns(self) -> None:
        """Columns from the input batch (e.g., 'id') are preserved in output."""
        texts = ["Hello\nWorld"]
        translated_batch = _segment_and_add_translations(
            texts,
            mode="coarse",
            extra_columns={"id": [42]},
        )

        stage = ReassemblyStage()
        result = stage.process(translated_batch)
        df = result.to_pandas()

        assert "id" in df.columns
        assert df.iloc[0]["id"] == 42

    def test_drops_internal_columns(self) -> None:
        """_seg_* and _translated columns are removed from output."""
        texts = ["Hello\nWorld"]
        translated_batch = _segment_and_add_translations(texts, mode="coarse")

        stage = ReassemblyStage()
        result = stage.process(translated_batch)
        df = result.to_pandas()

        for col in _INTERNAL_COLUMNS:
            assert col not in df.columns

    def test_inputs_outputs(self) -> None:
        """inputs() and outputs() return correct column declarations."""
        stage = ReassemblyStage(output_field="tgt_text")

        resource_kind, input_cols = stage.inputs()
        assert resource_kind == ["data"]
        assert set(input_cols) == {"_translated", "_seg_metadata", "_seg_doc_id"}

        resource_kind, output_cols = stage.outputs()
        assert resource_kind == ["data"]
        assert output_cols == ["tgt_text", "translation_time", "translation_errors"]


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and multi-document batches."""

    def test_empty_batch(self) -> None:
        """Empty DataFrame is handled gracefully."""
        df = pd.DataFrame({
            "_seg_segments": pd.Series(dtype="str"),
            "_seg_metadata": pd.Series(dtype="str"),
            "_seg_doc_id": pd.Series(dtype="int"),
            "_translated": pd.Series(dtype="str"),
            "text": pd.Series(dtype="str"),
        })
        batch = DocumentBatch(data=df, dataset_name="test", task_id="1")

        stage = ReassemblyStage()
        result = stage.process(batch)
        result_df = result.to_pandas()

        assert result_df.empty

    def test_multiple_documents(self) -> None:
        """Reassembly groups correctly by _seg_doc_id with multiple docs."""
        texts = [
            "Doc one line A\nDoc one line B",
            "Doc two single line",
            "Line X\n---\nLine Y",
        ]
        translated_batch = _segment_and_add_translations(
            texts,
            mode="coarse",
            extra_columns={"id": [10, 20, 30]},
        )

        stage = ReassemblyStage()
        result = stage.process(translated_batch)
        df = result.to_pandas()

        assert len(df) == 3

        # Each document should be reconstructed exactly (identity translation)
        assert df.iloc[0]["translated_text"] == texts[0]
        assert df.iloc[1]["translated_text"] == texts[1]
        assert df.iloc[2]["translated_text"] == texts[2]

        # Original 'id' column should be preserved
        assert list(df["id"]) == [10, 20, 30]

        # No internal columns remain
        for col in _INTERNAL_COLUMNS:
            assert col not in df.columns

    def test_wildcard_list_field_replaced_in_place(self) -> None:
        """Wildcard paths over list roots are reassembled back into the source schema."""
        df = pd.DataFrame(
            {
                "messages": [[
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ]]
            }
        )
        batch = DocumentBatch(data=df, dataset_name="test", task_id="1")

        segmented = SegmentationStage(
            text_field="messages.*.content",
            mode="coarse",
        ).process(batch)
        seg_df = segmented.to_pandas()
        seg_df["_translated"] = ["Hola", "Que tal"]
        translated_batch = DocumentBatch(
            data=seg_df,
            dataset_name=segmented.dataset_name,
            task_id=segmented.task_id,
            _metadata=segmented._metadata,
            _stage_perf=segmented._stage_perf,
        )

        result = ReassemblyStage(
            text_field="messages.*.content",
            replace_source_fields=True,
            emit_metadata_helpers=True,
            emit_faith_helpers=True,
        ).process(translated_batch)
        result_df = result.to_pandas()

        assert result_df.iloc[0]["messages"][0]["content"] == "Hola"
        assert result_df.iloc[0]["messages"][1]["content"] == "Que tal"
        assert result_df.iloc[0]["translated_text"][0]["content"] == "Hola"
        assert '"field_path": "messages.*.content"' in result_df.iloc[0]["_faith_source_text"]
        assert '"text": "Hola"' in result_df.iloc[0]["_faith_translated_text"]

        translation_map = json.loads(result_df.iloc[0]["_translation_map"])
        segmented_map = json.loads(result_df.iloc[0]["_segmented_translation_map"])
        assert translation_map["content"] == ["Hola", "Que tal"]
        assert segmented_map["content"][0]["src"] == "Hello"
