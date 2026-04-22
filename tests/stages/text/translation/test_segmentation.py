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

"""Unit tests for SegmentationStage (coarse and fine modes)."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from nemo_curator.stages.text.translation.stages import segmentation as segmentation_module
from nemo_curator.stages.text.translation.stages.segmentation import SegmentationStage
from nemo_curator.stages.text.translation.stages.translate import TranslateStage
from nemo_curator.tasks import DocumentBatch

from .conftest import MockAsyncLLMClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(texts: list[str], **extra_columns: list) -> DocumentBatch:
    """Build a minimal DocumentBatch from a list of text strings."""
    data = {"text": texts}
    data.update(extra_columns)
    df = pd.DataFrame(data)
    return DocumentBatch(data=df, dataset_name="test", task_id="1")


def _seg_metadata(batch: DocumentBatch, row: int = 0) -> dict:
    """Parse the _seg_metadata JSON from a result row."""
    metadata = json.loads(batch.to_pandas().iloc[row]["_seg_metadata"])
    field_metadatas = metadata.get("field_metadatas")
    if isinstance(field_metadatas, list) and len(field_metadatas) == 1:
        return field_metadatas[0]
    return metadata


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
# Coarse-mode tests
# ---------------------------------------------------------------------------


class TestCoarseSegmentation:
    """Tests for coarse (line-level) segmentation."""

    def test_coarse_basic(self) -> None:
        """Simple multi-line text produces correct segments and metadata."""
        text = "Hello world\nThis is a test\nGoodbye"
        stage = SegmentationStage(mode="coarse")
        result = stage.process(_make_batch([text]))
        df = result.to_pandas()

        # Three translatable lines => three output rows
        assert len(df) == 3
        assert list(df["_seg_segments"]) == ["Hello world", "This is a test", "Goodbye"]

        # All rows share the same doc id
        assert all(df["_seg_doc_id"] == 0)

        # Metadata template has three None entries (one per translatable line)
        meta = _seg_metadata(result, 0)
        assert meta["mode"] == "coarse"
        assert meta["template"] == [None, None, None]

    def test_coarse_code_blocks(self) -> None:
        """Lines inside code blocks are not treated as translatable segments."""
        text = "Before code\n```python\nprint('hi')\n```\nAfter code"
        stage = SegmentationStage(mode="coarse")
        result = stage.process(_make_batch([text]))
        df = result.to_pandas()

        # Only "Before code" and "After code" are translatable
        assert list(df["_seg_segments"]) == ["Before code", "After code"]

        meta = _seg_metadata(result)
        template = meta["template"]
        # Template should have: None, "```python", "print('hi')", "```", None
        assert template[0] is None
        assert template[1] == "```python"
        assert template[2] == "print('hi')"
        assert template[3] == "```"
        assert template[4] is None

    def test_coarse_empty_lines(self) -> None:
        """Empty lines and whitespace-only lines are not segments."""
        text = "Line one\n\n   \nLine two"
        stage = SegmentationStage(mode="coarse")
        result = stage.process(_make_batch([text]))
        df = result.to_pandas()

        assert list(df["_seg_segments"]) == ["Line one", "Line two"]

        meta = _seg_metadata(result)
        template = meta["template"]
        # Non-translatable lines (empty, whitespace) are stored verbatim
        assert template[0] is None  # "Line one" -> translatable
        assert template[1] == ""    # empty line
        assert template[2] == "   " # whitespace-only line
        assert template[3] is None  # "Line two" -> translatable

    def test_coarse_leading_whitespace(self) -> None:
        """Indented lines preserve whitespace in metadata leading_spaces."""
        text = "  Indented line\n    Double indented"
        stage = SegmentationStage(mode="coarse")
        result = stage.process(_make_batch([text]))
        df = result.to_pandas()

        # Segments are stripped of leading whitespace
        assert list(df["_seg_segments"]) == ["Indented line", "Double indented"]

        meta = _seg_metadata(result)
        assert meta["leading_spaces"] == ["  ", "    "]

    def test_coarse_non_translatable(self) -> None:
        """Lines with no alpha characters (e.g., '---', '***') are skipped."""
        text = "Title\n---\n***\nContent"
        stage = SegmentationStage(mode="coarse")
        result = stage.process(_make_batch([text]))
        df = result.to_pandas()

        assert list(df["_seg_segments"]) == ["Title", "Content"]

        meta = _seg_metadata(result)
        template = meta["template"]
        assert template[0] is None   # "Title" -> translatable
        assert template[1] == "---"  # non-translatable
        assert template[2] == "***"  # non-translatable
        assert template[3] is None   # "Content" -> translatable

    def test_coarse_json_blob_non_translatable(self) -> None:
        """Machine-readable JSON lines should be preserved, not translated."""
        text = 'Before\n{"tool":"lookup","payload":{"model":"DeepSeek V3"}}\nAfter'
        stage = SegmentationStage(mode="coarse")
        result = stage.process(_make_batch([text]))
        df = result.to_pandas()

        assert list(df["_seg_segments"]) == ["Before", "After"]

        meta = _seg_metadata(result)
        template = meta["template"]
        assert template[0] is None
        assert template[1] == '{"tool":"lookup","payload":{"model":"DeepSeek V3"}}'
        assert template[2] is None


# ---------------------------------------------------------------------------
# Fine-mode tests
# ---------------------------------------------------------------------------


class TestFineSegmentation:
    """Tests for fine (sentence-level) segmentation.

    Fine mode requires spaCy with the en_core_web_sm model.
    Tests are skipped if spaCy is not available.
    """

    @pytest.mark.skipif(not _SPACY_AVAILABLE, reason="spaCy en_core_web_sm not available")
    def test_fine_basic(self) -> None:
        """Multi-sentence paragraph splits into individual sentences."""
        text = "Hello world. This is a test. Goodbye."
        stage = SegmentationStage(mode="fine", source_lang="en")
        result = stage.process(_make_batch([text]))
        df = result.to_pandas()

        # Should produce at least 2 segments (spaCy sentence splitting)
        assert len(df) >= 2

        # All segments should be non-empty translatable text
        for seg in df["_seg_segments"]:
            assert isinstance(seg, str)
            assert len(seg.strip()) > 0

        meta = _seg_metadata(result)
        assert meta["mode"] == "fine"
        assert "units" in meta

    @pytest.mark.skipif(not _SPACY_AVAILABLE, reason="spaCy en_core_web_sm not available")
    def test_fine_preserves_separators(self) -> None:
        """Separators between sentences are stored in metadata."""
        text = "First sentence. Second sentence."
        stage = SegmentationStage(mode="fine", source_lang="en")
        result = stage.process(_make_batch([text]))

        meta = _seg_metadata(result)
        units = meta["units"]

        # At least one unit should have a non-empty separator
        separators = [u["separator"] for u in units if u["translatable"]]
        # The space between sentences should be captured somewhere
        all_separators = "".join(separators)
        # Reconstruction should work: join all original + separator
        reconstructed = "".join(u["original"] + u["separator"] for u in units)
        assert reconstructed == text


# ---------------------------------------------------------------------------
# General / edge-case tests
# ---------------------------------------------------------------------------


class TestSegmentationGeneral:
    """Tests for edge cases and general behaviour."""

    def test_single_line(self) -> None:
        """Single line produces one segment."""
        text = "Just one line"
        stage = SegmentationStage(mode="coarse")
        result = stage.process(_make_batch([text]))
        df = result.to_pandas()

        assert len(df) == 1
        assert df.iloc[0]["_seg_segments"] == "Just one line"

    def test_empty_text(self) -> None:
        """Empty string input is handled gracefully."""
        text = ""
        stage = SegmentationStage(mode="coarse")
        result = stage.process(_make_batch([text]))
        df = result.to_pandas()

        # Empty text has one empty line which is not translatable,
        # so there are no segments. The stage emits one row with empty segment.
        assert len(df) == 1
        assert df.iloc[0]["_seg_segments"] == ""

    def test_row_explosion(self) -> None:
        """N translatable segments produce N output rows with correct _seg_doc_id."""
        text = "Line A\nLine B\nLine C\nLine D"
        stage = SegmentationStage(mode="coarse")
        result = stage.process(_make_batch([text]))
        df = result.to_pandas()

        assert len(df) == 4
        assert list(df["_seg_segments"]) == ["Line A", "Line B", "Line C", "Line D"]
        # All rows belong to doc 0
        assert all(df["_seg_doc_id"] == 0)
        # Index is reset to 0-based
        assert list(df.index) == [0, 1, 2, 3]

    def test_inputs_outputs(self) -> None:
        """inputs() and outputs() return correct column declarations."""
        stage = SegmentationStage(text_field="body")

        resource_kind, input_cols = stage.inputs()
        assert resource_kind == ["data"]
        assert input_cols == ["body"]

        resource_kind, output_cols = stage.outputs()
        assert resource_kind == ["data"]
        assert set(output_cols) == {"_seg_segments", "_seg_metadata", "_seg_doc_id"}

    def test_long_text_uses_separate_spacy_cache_entry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Long-text handling should not mutate the default cached spaCy model."""

        class FakeNLP:
            def __init__(self) -> None:
                self.max_length = 5

            def __call__(self, text: str) -> SimpleNamespace:
                return SimpleNamespace(
                    sents=[SimpleNamespace(start_char=0, end_char=len(text))]
                )

        fake_spacy = SimpleNamespace(load=lambda _name: FakeNLP())
        monkeypatch.setitem(sys.modules, "spacy", fake_spacy)
        segmentation_module._nlp_cache.clear()

        default_nlp = segmentation_module._get_spacy_nlp("en")
        assert default_nlp.max_length == 5

        units = segmentation_module.split_into_sentences_with_structure(
            "This text is longer than five characters.",
            src_lang="en",
        )

        assert units == [("This text is longer than five characters.", "")]
        assert segmentation_module._get_spacy_nlp("en").max_length == 5
        assert ("en_core_web_sm", 10_000_000) in segmentation_module._nlp_cache

    def test_multiple_documents(self) -> None:
        """Batch with multiple rows assigns unique _seg_doc_id per document."""
        texts = [
            "Doc one line A\nDoc one line B",
            "Doc two single line",
            "Doc three line X\nDoc three line Y\nDoc three line Z",
        ]
        stage = SegmentationStage(mode="coarse")
        result = stage.process(_make_batch(texts))
        df = result.to_pandas()

        # Doc 0: 2 segments, Doc 1: 1 segment, Doc 2: 3 segments = 6 total
        assert len(df) == 6

        # Check doc_id assignment
        doc_ids = df["_seg_doc_id"].tolist()
        assert doc_ids == [0, 0, 1, 2, 2, 2]

        # Verify segments per document
        doc0 = df[df["_seg_doc_id"] == 0]["_seg_segments"].tolist()
        assert doc0 == ["Doc one line A", "Doc one line B"]

        doc1 = df[df["_seg_doc_id"] == 1]["_seg_segments"].tolist()
        assert doc1 == ["Doc two single line"]

        doc2 = df[df["_seg_doc_id"] == 2]["_seg_segments"].tolist()
        assert doc2 == ["Doc three line X", "Doc three line Y", "Doc three line Z"]


# ---------------------------------------------------------------------------
# F1 regression: passthrough + translatability filter in TranslateStage
# ---------------------------------------------------------------------------


class TestPassthroughTranslatabilityFilter:
    """Regression tests for F1.

    Segments emitted by the passthrough branch of SegmentationStage that
    contain no translatable content (pure code / numeric / XML-tag text)
    must flow through TranslateStage unchanged and without triggering an
    LLM call.
    """

    def test_passthrough_pure_code_skips_llm(self) -> None:
        """A code-only passthrough segment must NOT call the LLM."""
        class CountingMockClient(MockAsyncLLMClient):
            def __init__(self) -> None:
                super().__init__()
                self.call_count = 0

            async def _query_model_impl(self, **kwargs):  # type: ignore[override]
                self.call_count += 1
                return await super()._query_model_impl(**kwargs)

        client = CountingMockClient()
        stage = TranslateStage(
            client=client,
            model_name="test-model",
            source_lang="en",
            target_lang="zh",
            backend_type="llm",
        )
        stage._system_prompt = "You are a translator."
        stage._user_template = "Translate {source_lang} to {target_lang}: {src}"
        stage._initialized = True

        # Pure-numeric and tag-only content: no alpha chars / tag-shaped.
        # ``is_line_translatable_content`` returns False for both, so
        # TranslateStage should short-circuit without calling the LLM.
        code_block = "12345\n67890"
        tag_only = "<hr/>"
        json_blob = '{"tool":"lookup","payload":{"model":"DeepSeek V3"}}'
        df = pd.DataFrame({"_seg_segments": [code_block, tag_only, json_blob]})
        batch = DocumentBatch(data=df, dataset_name="test", task_id="1")

        result = stage.process(batch)
        result_df = result.to_pandas()

        # LLM must not have been called.
        assert client.call_count == 0
        # Segments pass through verbatim so reassembly offsets stay aligned.
        assert result_df["_translated"].tolist() == [code_block, tag_only, json_blob]

    def test_passthrough_real_text_still_calls_llm(self) -> None:
        """A passthrough segment with translatable content still calls the LLM."""
        class CountingMockClient(MockAsyncLLMClient):
            def __init__(self) -> None:
                super().__init__()
                self.call_count = 0

            async def _query_model_impl(self, **kwargs):  # type: ignore[override]
                self.call_count += 1
                return await super()._query_model_impl(**kwargs)

        client = CountingMockClient()
        stage = TranslateStage(
            client=client,
            model_name="test-model",
            source_lang="en",
            target_lang="zh",
            backend_type="llm",
        )
        stage._system_prompt = "You are a translator."
        stage._user_template = "Translate {source_lang} to {target_lang}: {src}"
        stage._initialized = True

        df = pd.DataFrame({"_seg_segments": ["Hello world"]})
        batch = DocumentBatch(data=df, dataset_name="test", task_id="1")
        result = stage.process(batch)
        result_df = result.to_pandas()

        assert client.call_count == 1
        assert "mock translation" in result_df["_translated"].iloc[0]
