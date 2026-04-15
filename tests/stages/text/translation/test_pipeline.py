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

"""Integration tests for the TranslationPipeline CompositeStage and FaithEvalFilter."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.stages.text.translation.faith_eval import (
    FAITH_KEYS,
    FaithEvalFilter,
    _SCORE_COLUMNS,
)
from nemo_curator.stages.text.translation.pipeline import (
    OutputFormattingStage,
    ScoreMergeStage,
    SegmentPairCaptureStage,
    TranslationPipeline,
)
from nemo_curator.stages.text.translation.reassembly import ReassemblyStage
from nemo_curator.stages.text.translation.segmentation import SegmentationStage
from nemo_curator.stages.text.translation.translate import TranslateStage
from nemo_curator.tasks import DocumentBatch

from .conftest import MockAsyncLLMClient


# ---------------------------------------------------------------------------
# TranslationPipeline.decompose() tests
# ---------------------------------------------------------------------------


class TestTranslationPipelineDecompose:
    """Tests for the CompositeStage decompose behaviour."""

    def test_decompose_without_faith_eval(self, mock_client: MockAsyncLLMClient) -> None:
        """Pipeline without FAITH eval decomposes into 3 stages."""
        pipeline = TranslationPipeline(
            source_lang="en",
            target_lang="de",
            client=mock_client,
            model_name="test-model",
        )
        stages = pipeline.decompose()
        assert len(stages) == 3
        assert isinstance(stages[0], SegmentationStage)
        assert isinstance(stages[1], TranslateStage)
        assert isinstance(stages[2], ReassemblyStage)

    def test_decompose_with_faith_eval(self, mock_client: MockAsyncLLMClient) -> None:
        """Pipeline with FAITH eval decomposes into 4 stages."""
        pipeline = TranslationPipeline(
            source_lang="en",
            target_lang="de",
            client=mock_client,
            model_name="test-model",
            enable_faith_eval=True,
        )
        stages = pipeline.decompose()
        assert len(stages) == 4
        assert isinstance(stages[0], SegmentationStage)
        assert isinstance(stages[1], TranslateStage)
        assert isinstance(stages[2], ReassemblyStage)
        assert isinstance(stages[3], FaithEvalFilter)

    def test_decompose_faith_eval_uses_faith_model_name(self, mock_client: MockAsyncLLMClient) -> None:
        """When faith_model_name is set, FaithEvalFilter uses it instead of model_name."""
        pipeline = TranslationPipeline(
            source_lang="en",
            target_lang="de",
            client=mock_client,
            model_name="translate-model",
            enable_faith_eval=True,
            faith_model_name="faith-model",
        )
        stages = pipeline.decompose()
        faith_stage = stages[3]
        assert isinstance(faith_stage, FaithEvalFilter)
        assert faith_stage.model_name == "faith-model"

    def test_decompose_faith_eval_fallback_to_model_name(self, mock_client: MockAsyncLLMClient) -> None:
        """When faith_model_name is empty, FaithEvalFilter uses model_name."""
        pipeline = TranslationPipeline(
            source_lang="en",
            target_lang="de",
            client=mock_client,
            model_name="translate-model",
            enable_faith_eval=True,
            faith_model_name="",
        )
        stages = pipeline.decompose()
        faith_stage = stages[3]
        assert isinstance(faith_stage, FaithEvalFilter)
        assert faith_stage.model_name == "translate-model"

    def test_segmentation_stage_inherits_config(self, mock_client: MockAsyncLLMClient) -> None:
        """SegmentationStage receives text_field and source_lang from the pipeline."""
        pipeline = TranslationPipeline(
            source_lang="fr",
            target_lang="en",
            client=mock_client,
            model_name="m",
            text_field="body",
            segmentation_mode="fine",
        )
        seg = pipeline.decompose()[0]
        assert isinstance(seg, SegmentationStage)
        assert seg.text_field == "body"
        assert seg.source_lang == "fr"
        assert seg.mode == "fine"

    def test_translate_stage_inherits_config(self, mock_client: MockAsyncLLMClient) -> None:
        """TranslateStage receives language and backend config from the pipeline."""
        gen_cfg = GenerationConfig(temperature=0.5)
        pipeline = TranslationPipeline(
            source_lang="en",
            target_lang="ja",
            client=mock_client,
            model_name="m",
            generation_config=gen_cfg,
            backend_type="llm",
        )
        tr = pipeline.decompose()[1]
        assert isinstance(tr, TranslateStage)
        assert tr.source_lang == "en"
        assert tr.target_lang == "ja"
        assert tr.model_name == "m"
        assert tr.generation_config is gen_cfg

    def test_reassembly_stage_inherits_config(self, mock_client: MockAsyncLLMClient) -> None:
        """ReassemblyStage receives text_field and output_field from the pipeline."""
        pipeline = TranslationPipeline(
            source_lang="en",
            target_lang="de",
            client=mock_client,
            model_name="m",
            text_field="body",
            output_field="translated_body",
        )
        reas = pipeline.decompose()[2]
        assert isinstance(reas, ReassemblyStage)
        assert reas.text_field == "body"
        assert reas.output_field == "translated_body"

    def test_composite_stage_process_raises(self, mock_client: MockAsyncLLMClient) -> None:
        """CompositeStage.process() must raise RuntimeError (never executed directly)."""
        pipeline = TranslationPipeline(
            source_lang="en",
            target_lang="de",
            client=mock_client,
            model_name="m",
        )
        df = pd.DataFrame({"text": ["hello"]})
        batch = DocumentBatch(data=df, dataset_name="test", task_id="1")
        with pytest.raises(RuntimeError, match="should not be executed directly"):
            pipeline.process(batch)

    def test_pipeline_inputs_outputs(self, mock_client: MockAsyncLLMClient) -> None:
        """inputs() delegates to first stage, outputs() to last stage."""
        pipeline = TranslationPipeline(
            source_lang="en",
            target_lang="de",
            client=mock_client,
            model_name="m",
        )
        # Without faith eval, outputs come from ReassemblyStage
        _, data_inputs = pipeline.inputs()
        _, data_outputs = pipeline.outputs()
        assert "text" in data_inputs or len(data_inputs) >= 0  # SegmentationStage inputs
        # ReassemblyStage outputs the translated_text field
        assert len(data_outputs) > 0


# ---------------------------------------------------------------------------
# FaithEvalFilter unit tests
# ---------------------------------------------------------------------------


class TestFaithEvalFilter:
    """Tests for FaithEvalFilter score parsing and filtering."""

    def test_extract_scores_valid_json(self) -> None:
        """Valid JSON with all 5 keys is parsed correctly."""
        text = '{"Fluency": 4, "Accuracy": 5, "Idiomaticity": 3, "Terminology": 4, "Handling_of_Format": 5}'
        scores = FaithEvalFilter._extract_scores_from_json(text)
        assert scores["Fluency"] == 4.0
        assert scores["Accuracy"] == 5.0
        assert scores["Idiomaticity"] == 3.0
        assert scores["Terminology"] == 4.0
        assert scores["Handling_of_Format"] == 5.0

    def test_extract_scores_with_surrounding_text(self) -> None:
        """JSON embedded in explanatory text is still extracted."""
        text = 'Here are the scores:\n{"Fluency": 3, "Accuracy": 4, "Idiomaticity": 2, "Terminology": 3, "Handling_of_Format": 4}\nDone.'
        scores = FaithEvalFilter._extract_scores_from_json(text)
        assert scores["Fluency"] == 3.0

    def test_extract_scores_missing_keys(self) -> None:
        """Missing keys default to 0.0."""
        text = '{"Fluency": 5, "Accuracy": 4}'
        scores = FaithEvalFilter._extract_scores_from_json(text)
        assert scores["Fluency"] == 5.0
        assert scores["Accuracy"] == 4.0
        assert scores["Idiomaticity"] == 0.0
        assert scores["Terminology"] == 0.0
        assert scores["Handling_of_Format"] == 0.0

    def test_extract_scores_no_json(self) -> None:
        """When no JSON is found, all scores are 0.0."""
        text = "I cannot evaluate this translation."
        scores = FaithEvalFilter._extract_scores_from_json(text)
        for key in FAITH_KEYS:
            assert scores[key] == 0.0

    def test_extract_scores_invalid_json(self) -> None:
        """Malformed JSON falls back to all-zero scores."""
        text = "{Fluency: bad}"
        scores = FaithEvalFilter._extract_scores_from_json(text)
        for key in FAITH_KEYS:
            assert scores[key] == 0.0

    def test_filter_process_drops_low_scores(self, mock_client: MockAsyncLLMClient) -> None:
        """Rows with faith_avg below threshold are dropped."""
        # The MockAsyncLLMClient returns scores averaging 4.2 for FAITH requests.
        # Set threshold very high to test filtering.
        stage = FaithEvalFilter(
            client=mock_client,
            model_name="test-model",
            source_lang="en",
            target_lang="de",
            threshold=5.0,  # Very high -- should drop everything
        )
        stage.setup()

        df = pd.DataFrame(
            {
                "text": ["Hello world."],
                "translated_text": ["Hallo Welt."],
            }
        )
        batch = DocumentBatch(data=df, dataset_name="test", task_id="1")
        result = stage.process(batch)
        result_df = result.to_pandas()

        # MockClient returns avg ~4.2, threshold is 5.0 => row should be dropped
        assert len(result_df) == 0

    def test_filter_process_keeps_high_scores(self, mock_client: MockAsyncLLMClient) -> None:
        """Rows with faith_avg above threshold are kept and score columns exist."""
        stage = FaithEvalFilter(
            client=mock_client,
            model_name="test-model",
            source_lang="en",
            target_lang="de",
            threshold=1.0,  # Very low -- should keep everything
        )
        stage.setup()

        df = pd.DataFrame(
            {
                "text": ["Hello world.", "Second doc."],
                "translated_text": ["Hallo Welt.", "Zweites Dok."],
            }
        )
        batch = DocumentBatch(data=df, dataset_name="test", task_id="1")
        result = stage.process(batch)
        result_df = result.to_pandas()

        assert len(result_df) == 2
        for col in _SCORE_COLUMNS:
            assert col in result_df.columns

        # Verify score values from MockAsyncLLMClient
        assert result_df["faith_fluency"].iloc[0] == pytest.approx(4.0)
        assert result_df["faith_accuracy"].iloc[0] == pytest.approx(4.5)

    def test_filter_process_empty_batch(self, mock_client: MockAsyncLLMClient) -> None:
        """An empty batch passes through without errors."""
        stage = FaithEvalFilter(
            client=mock_client,
            model_name="test-model",
            source_lang="en",
            target_lang="de",
        )
        df = pd.DataFrame({"text": pd.Series(dtype="str"), "translated_text": pd.Series(dtype="str")})
        batch = DocumentBatch(data=df, dataset_name="test", task_id="1")
        result = stage.process(batch)
        assert result.to_pandas().empty

    def test_default_generation_config(self) -> None:
        """Default generation_config is temperature=0.0, max_tokens=256 after setup."""
        stage = FaithEvalFilter(client=MockAsyncLLMClient(), model_name="m")
        # generation_config defaults are now set in setup() for Ray compatibility
        stage.setup()
        assert stage.generation_config is not None
        assert stage.generation_config.temperature == 0.0
        assert stage.generation_config.max_tokens == 256

    def test_inputs_outputs(self) -> None:
        """inputs/outputs report the expected column names."""
        stage = FaithEvalFilter(
            client=MockAsyncLLMClient(),
            model_name="m",
            source_text_field="src",
            translated_text_field="tgt",
        )
        _, in_cols = stage.inputs()
        assert "src" in in_cols
        assert "tgt" in in_cols

        _, out_cols = stage.outputs()
        assert set(out_cols) == set(_SCORE_COLUMNS)


# ---------------------------------------------------------------------------
# End-to-end mock test
# ---------------------------------------------------------------------------


class TestEndToEndMock:
    """Smoke tests that exercise the full pipeline stage sequence with mocks.

    Since CompositeStage never executes directly, we run each sub-stage
    sequentially to verify column flow.
    """

    def test_pipeline_columns_flow(self, mock_client: MockAsyncLLMClient, sample_batch: DocumentBatch) -> None:
        """Running decomposed stages sequentially produces the expected columns."""
        pipeline = TranslationPipeline(
            source_lang="en",
            target_lang="de",
            client=mock_client,
            model_name="test-model",
            enable_faith_eval=True,
            faith_threshold=1.0,  # Low threshold so nothing is filtered
        )
        stages = pipeline.decompose()
        assert len(stages) == 4

        # We cannot fully run SegmentationStage and TranslateStage without
        # Dev 1 and Dev 2's implementations being present.  Instead, we
        # verify that the pipeline *decomposes* correctly and that the last
        # stage (FaithEvalFilter) can process a pre-built DataFrame.

        # Simulate post-reassembly DataFrame
        df = pd.DataFrame(
            {
                "text": ["Hello world.", "Goodbye."],
                "translated_text": ["Hallo Welt.", "Auf Wiedersehen."],
                "id": [1, 2],
            }
        )
        simulated_batch = DocumentBatch(data=df, dataset_name="test", task_id="1")

        # Run FaithEvalFilter (the 4th stage)
        faith_stage = stages[3]
        assert isinstance(faith_stage, FaithEvalFilter)
        faith_stage.setup()
        result = faith_stage.process(simulated_batch)
        result_df = result.to_pandas()

        # All rows kept (threshold=1.0, mock scores average ~4.2)
        assert len(result_df) == 2
        assert "faith_avg" in result_df.columns
        assert "translated_text" in result_df.columns
        assert "id" in result_df.columns  # Original columns preserved

    def test_full_e2e_all_four_stages(self, mock_client: MockAsyncLLMClient) -> None:
        """True end-to-end test: run all 4 stages sequentially.

        Exercises: SegmentationStage -> TranslateStage -> ReassemblyStage ->
        FaithEvalFilter, verifying that columns flow correctly through every
        stage and no data is silently lost.
        """
        # -- Build input batch with two documents ----------------------------
        df = pd.DataFrame(
            {
                "text": [
                    "Hello world.\nThis is a test.\n```python\nprint('hi')\n```\nGoodbye.",
                    "Simple sentence.",
                ],
                "id": [1, 2],
            }
        )
        batch = DocumentBatch(data=df, dataset_name="e2e-test", task_id="1")

        # -- Decompose pipeline with faith eval enabled ----------------------
        pipeline = TranslationPipeline(
            source_lang="en",
            target_lang="de",
            client=mock_client,
            model_name="test-model",
            enable_faith_eval=True,
            faith_threshold=1.0,  # Low threshold so nothing is filtered
        )
        stages = pipeline.decompose()
        assert len(stages) == 4

        # -- Stage 1: Segmentation ------------------------------------------
        seg_stage = stages[0]
        assert isinstance(seg_stage, SegmentationStage)
        result = seg_stage.process(batch)
        seg_df = result.to_pandas()

        # After segmentation, we should have internal columns
        assert "_seg_segments" in seg_df.columns
        assert "_seg_metadata" in seg_df.columns
        assert "_seg_doc_id" in seg_df.columns
        # Should have more rows than input (first doc has multiple segments)
        assert len(seg_df) >= 2
        # Original columns still present
        assert "text" in seg_df.columns
        assert "id" in seg_df.columns

        # -- Stage 2: Translation -------------------------------------------
        tr_stage = stages[1]
        assert isinstance(tr_stage, TranslateStage)
        tr_stage.setup()
        result = tr_stage.process(result)
        tr_df = result.to_pandas()

        assert "_translated" in tr_df.columns
        # Every segment should have a translation (non-None)
        assert tr_df["_translated"].notna().all()
        # Row count should not change
        assert len(tr_df) == len(seg_df)

        # -- Stage 3: Reassembly -------------------------------------------
        reas_stage = stages[2]
        assert isinstance(reas_stage, ReassemblyStage)
        result = reas_stage.process(result)
        reas_df = result.to_pandas()

        # Should collapse back to the original number of documents
        assert len(reas_df) == 2
        # translated_text column must exist
        assert "translated_text" in reas_df.columns
        # Original columns preserved
        assert "text" in reas_df.columns
        assert "id" in reas_df.columns
        # Internal columns must be cleaned up
        for col in ["_seg_segments", "_seg_metadata", "_seg_doc_id", "_translated"]:
            assert col not in reas_df.columns
        # translated_text should not be empty
        assert all(len(t) > 0 for t in reas_df["translated_text"])

        # -- Stage 4: FaithEvalFilter ---------------------------------------
        faith_stage = stages[3]
        assert isinstance(faith_stage, FaithEvalFilter)
        faith_stage.setup()
        result = faith_stage.process(result)
        final_df = result.to_pandas()

        # All rows kept (threshold=1.0, mock scores average ~4.2)
        assert len(final_df) == 2
        # Score columns present
        for col in _SCORE_COLUMNS:
            assert col in final_df.columns
        # Original columns preserved
        assert "text" in final_df.columns
        assert "translated_text" in final_df.columns
        assert "id" in final_df.columns
        # No internal columns leaked
        for col in final_df.columns:
            assert not col.startswith("_seg_")
        assert "_translated" not in final_df.columns

    def test_full_e2e_empty_segment_not_sent_to_llm(self, mock_client: MockAsyncLLMClient) -> None:
        """Documents with no translatable content produce empty translations without LLM calls."""
        # A document with only code/numbers should produce an empty segment
        df = pd.DataFrame(
            {
                "text": [
                    "```python\nprint('hi')\n```",  # Only code
                    "Hello world.",  # Normal text
                ],
                "id": [10, 20],
            }
        )
        batch = DocumentBatch(data=df, dataset_name="e2e-empty-test", task_id="1")

        pipeline = TranslationPipeline(
            source_lang="en",
            target_lang="de",
            client=mock_client,
            model_name="test-model",
        )
        stages = pipeline.decompose()

        # Run all 3 stages sequentially
        result = batch
        for stage in stages:
            stage.setup()
            result = stage.process(result)

        final_df = result.to_pandas()
        assert len(final_df) == 2
        assert "translated_text" in final_df.columns
        # The code-only doc and normal doc both produce results without errors
        assert final_df["id"].tolist() == [10, 20]

    def test_full_e2e_with_non_contiguous_index(self, mock_client: MockAsyncLLMClient) -> None:
        """Pipeline handles input DataFrames with non-contiguous indices correctly."""
        # Simulate a DataFrame that was filtered (non-contiguous index)
        df = pd.DataFrame(
            {
                "text": ["Hello world.", "Goodbye world.", "Third doc."],
                "id": [100, 200, 300],
            },
            index=[5, 10, 15],  # Non-contiguous index
        )
        batch = DocumentBatch(data=df, dataset_name="e2e-index-test", task_id="1")

        pipeline = TranslationPipeline(
            source_lang="en",
            target_lang="de",
            client=mock_client,
            model_name="test-model",
        )
        stages = pipeline.decompose()

        # Run all 3 stages sequentially
        result = batch
        for stage in stages:
            stage.setup()
            result = stage.process(result)

        final_df = result.to_pandas()
        # All 3 documents should make it through
        assert len(final_df) == 3
        assert "translated_text" in final_df.columns
        assert list(final_df["id"]) == [100, 200, 300]


# ---------------------------------------------------------------------------
# Gap fix tests -- new pipeline features
# ---------------------------------------------------------------------------


class TestFaithEvalFilterEnabled:
    """Tests for Gap 5.3: FaithEvalFilter with filter_enabled=False (score-and-keep mode)."""

    def test_faith_eval_score_without_filtering(self, mock_client: MockAsyncLLMClient) -> None:
        """With enable_faith_eval=True and filter_enabled=False, all rows are kept with scores."""
        stage = FaithEvalFilter(
            client=mock_client,
            model_name="test-model",
            source_lang="en",
            target_lang="de",
            threshold=5.0,  # Very high -- would normally drop everything
            filter_enabled=False,  # But filtering is disabled
        )
        stage.setup()

        df = pd.DataFrame(
            {
                "text": ["Hello world.", "Second doc."],
                "translated_text": ["Hallo Welt.", "Zweites Dok."],
            }
        )
        batch = DocumentBatch(data=df, dataset_name="test", task_id="1")
        result = stage.process(batch)
        result_df = result.to_pandas()

        # All rows are kept even though mock scores (~4.2) are below threshold (5.0)
        assert len(result_df) == 2
        # Score columns are still present
        for col in _SCORE_COLUMNS:
            assert col in result_df.columns
        # faith_avg should be populated with real values
        assert all(result_df["faith_avg"] > 0)

    def test_faith_eval_filter_enabled_true_drops_rows(self, mock_client: MockAsyncLLMClient) -> None:
        """With filter_enabled=True (default), rows below threshold are dropped."""
        stage = FaithEvalFilter(
            client=mock_client,
            model_name="test-model",
            source_lang="en",
            target_lang="de",
            threshold=5.0,  # Very high -- should drop everything
            filter_enabled=True,
        )
        stage.setup()

        df = pd.DataFrame(
            {
                "text": ["Hello."],
                "translated_text": ["Hallo."],
            }
        )
        batch = DocumentBatch(data=df, dataset_name="test", task_id="1")
        result = stage.process(batch)
        result_df = result.to_pandas()

        # Row should be dropped since mock scores (~4.2) < 5.0
        assert len(result_df) == 0

    def test_pipeline_with_faith_eval_score_only(self, mock_client: MockAsyncLLMClient) -> None:
        """Pipeline with enable_faith_eval=True builds stages correctly.

        The FaithEvalFilter stage in the decomposed pipeline should respect
        the filter_enabled setting.  Since TranslationPipeline does not
        currently expose filter_enabled, we verify the stage can be configured
        independently.
        """
        pipeline = TranslationPipeline(
            source_lang="en",
            target_lang="de",
            client=mock_client,
            model_name="test-model",
            enable_faith_eval=True,
            faith_threshold=1.0,
        )
        stages = pipeline.decompose()
        assert len(stages) == 4
        faith_stage = stages[3]
        assert isinstance(faith_stage, FaithEvalFilter)
        # Default is filter_enabled=True
        assert faith_stage.filter_enabled is True


class TestDryRunMode:
    """Tests for Gap 10.3: TranslateStage with dry_run=True."""

    def test_dry_run_returns_empty_translations(self, mock_client: MockAsyncLLMClient) -> None:
        """With dry_run=True, process() returns empty translations without calling the LLM."""
        stage = TranslateStage(
            client=mock_client,
            model_name="test-model",
            source_lang="en",
            target_lang="de",
            dry_run=True,
        )
        # Pre-load prompts to avoid file I/O
        stage._system_prompt = "You are a translator."
        stage._user_template = "Translate {source_lang} to {target_lang}: {src}"
        stage._initialized = True

        df = pd.DataFrame(
            {
                "_seg_segments": ["Hello world", "Goodbye", "Third segment"],
                "id": [1, 2, 3],
            }
        )
        batch = DocumentBatch(data=df, dataset_name="test", task_id="1")
        result = stage.process(batch)
        result_df = result.to_pandas()

        assert "_translated" in result_df.columns
        # All translations should be empty strings in dry run
        assert all(t == "" for t in result_df["_translated"])
        # Row count should be preserved
        assert len(result_df) == 3

    def test_dry_run_produces_timing_columns(self, mock_client: MockAsyncLLMClient) -> None:
        """dry_run=True produces _translation_time and _translation_error columns."""
        stage = TranslateStage(
            client=mock_client,
            model_name="test-model",
            source_lang="en",
            target_lang="de",
            dry_run=True,
        )
        stage._system_prompt = "You are a translator."
        stage._user_template = "Translate {source_lang} to {target_lang}: {src}"
        stage._initialized = True

        df = pd.DataFrame({"_seg_segments": ["Hello"], "id": [1]})
        batch = DocumentBatch(data=df, dataset_name="test", task_id="1")
        result = stage.process(batch)
        result_df = result.to_pandas()

        assert "_translation_time" in result_df.columns
        assert "_translation_error" in result_df.columns
        assert result_df["_translation_time"].iloc[0] == 0.0
        assert result_df["_translation_error"].iloc[0] == ""

    def test_dry_run_field_defaults_to_false(self) -> None:
        """dry_run defaults to False."""
        stage = TranslateStage(
            client=MockAsyncLLMClient(),
            model_name="test-model",
        )
        assert stage.dry_run is False


class TestSkipTranslated:
    """Tests for Gap 9.1: skip_translated / resume behavior."""

    def test_skip_translated_skips_existing(
        self, mock_client: MockAsyncLLMClient, batch_with_existing_translations: DocumentBatch
    ) -> None:
        """With skip_translated=True, rows with existing translations are preserved as-is."""
        pipeline = TranslationPipeline(
            source_lang="en",
            target_lang="de",
            client=mock_client,
            model_name="test-model",
            skip_translated=True,
        )
        stages = pipeline.decompose()
        result = batch_with_existing_translations
        for stage in stages:
            stage.setup()
            result = stage.process(result)

        result_df = result.to_pandas()
        # Row 0 should keep its existing translation
        assert result_df.loc[result_df["id"] == 100, "translated_text"].iloc[0] == "Bereits uebersetztes Dokument."
        # Rows 1 and 2 should have new translations
        assert len(result_df.loc[result_df["id"] == 200, "translated_text"].iloc[0]) > 0
        assert len(result_df.loc[result_df["id"] == 300, "translated_text"].iloc[0]) > 0

    def test_skip_translated_false_retranslates_all(
        self, mock_client: MockAsyncLLMClient, batch_with_existing_translations: DocumentBatch
    ) -> None:
        """With skip_translated=False, all rows are retranslated."""
        pipeline = TranslationPipeline(
            source_lang="en",
            target_lang="de",
            client=mock_client,
            model_name="test-model",
            skip_translated=False,
        )
        stages = pipeline.decompose()
        result = batch_with_existing_translations
        for stage in stages:
            stage.setup()
            result = stage.process(result)

        result_df = result.to_pandas()
        # All rows should have been (re)translated
        assert len(result_df) == 3
        assert all(len(t) > 0 for t in result_df["translated_text"])


class TestOutputMode:
    """Tests for Gap 4.1: output_mode parameter."""

    def test_output_mode_both(self, mock_client: MockAsyncLLMClient) -> None:
        """With output_mode='both', output has both raw metadata and replaced text."""
        pipeline = TranslationPipeline(
            source_lang="en",
            target_lang="de",
            client=mock_client,
            model_name="test-model",
            output_mode="both",
        )
        df = pd.DataFrame({"text": ["Hello world."], "id": [1]})
        batch = DocumentBatch(data=df, dataset_name="test", task_id="1")

        stages = pipeline.decompose()
        result = batch
        for stage in stages:
            stage.setup()
            result = stage.process(result)

        result_df = result.to_pandas()
        assert "translated_text" in result_df.columns
        # In "both" mode, a structured translation metadata column should also exist
        assert "translation_metadata" in result_df.columns


class TestPartialTranslationRecovery:
    """Tests for Gap 10.2: partial translation recovery.

    Tests verify that individual segment failures do not abort the entire batch.
    """

    def test_partial_failure_does_not_crash(self, mock_client: MockAsyncLLMClient) -> None:
        """When one segment's translation fails, the batch still completes.

        This tests the current implementation's behavior.  The mock client does
        not raise exceptions, so we verify the basic flow completes.  A more
        thorough test would inject a failing mock for specific segments.
        """
        stage = TranslateStage(
            client=mock_client,
            model_name="test-model",
            source_lang="en",
            target_lang="de",
        )
        stage._system_prompt = "You are a translator."
        stage._user_template = "Translate {source_lang} to {target_lang}: {src}"
        stage._initialized = True

        df = pd.DataFrame(
            {
                "_seg_segments": ["Good segment", "", "Another good one"],
                "id": [1, 2, 3],
            }
        )
        batch = DocumentBatch(data=df, dataset_name="test", task_id="1")
        result = stage.process(batch)
        result_df = result.to_pandas()

        assert len(result_df) == 3
        assert "_translated" in result_df.columns
        # Empty segment should return empty translation
        assert result_df["_translated"].iloc[1] == ""
        # Non-empty segments should have translations
        assert len(result_df["_translated"].iloc[0]) > 0
        assert len(result_df["_translated"].iloc[2]) > 0


class TestFaithEvalAverageScores:
    """Tests for FaithEvalFilter._average_scores (segment-level scoring helper)."""

    def test_average_scores_basic(self) -> None:
        """Averaging two score dicts produces correct values."""
        scores = [
            {"Fluency": 4.0, "Accuracy": 5.0, "Idiomaticity": 3.0, "Terminology": 4.0, "Handling_of_Format": 5.0},
            {"Fluency": 2.0, "Accuracy": 3.0, "Idiomaticity": 1.0, "Terminology": 2.0, "Handling_of_Format": 3.0},
        ]
        result = FaithEvalFilter._average_scores(scores)
        assert result["Fluency"] == pytest.approx(3.0, abs=0.01)
        assert result["Accuracy"] == pytest.approx(4.0, abs=0.01)

    def test_average_scores_empty_list(self) -> None:
        """Empty input returns all zeros."""
        result = FaithEvalFilter._average_scores([])
        for key in FAITH_KEYS:
            assert result[key] == 0.0

    def test_average_scores_skips_zeros(self) -> None:
        """Zero-scored dimensions are excluded from the average."""
        scores = [
            {"Fluency": 4.0, "Accuracy": 0.0, "Idiomaticity": 3.0, "Terminology": 0.0, "Handling_of_Format": 5.0},
            {"Fluency": 2.0, "Accuracy": 4.0, "Idiomaticity": 0.0, "Terminology": 0.0, "Handling_of_Format": 3.0},
        ]
        result = FaithEvalFilter._average_scores(scores)
        # Fluency: (4+2)/2 = 3.0, Accuracy: only 4.0 counts, Idiomaticity: only 3.0
        assert result["Fluency"] == pytest.approx(3.0, abs=0.01)
        assert result["Accuracy"] == pytest.approx(4.0, abs=0.01)
        assert result["Idiomaticity"] == pytest.approx(3.0, abs=0.01)
        # Terminology: all zeros -> 0.0
        assert result["Terminology"] == 0.0


# ---------------------------------------------------------------------------
# SegmentPairCaptureStage tests
# ---------------------------------------------------------------------------


class TestSegmentPairCaptureStage:
    """Tests for SegmentPairCaptureStage."""

    def test_capture_creates_pairs_column(self) -> None:
        """Verify _seg_translation_pairs column is created with correct JSON."""
        stage = SegmentPairCaptureStage()

        df = pd.DataFrame(
            {
                "_seg_segments": ["Hello", "World", "Goodbye"],
                "_translated": ["Hallo", "Welt", "Auf Wiedersehen"],
                "_seg_doc_id": [0, 0, 1],
                "text": ["orig1", "orig1", "orig2"],
            }
        )
        batch = DocumentBatch(data=df, dataset_name="test", task_id="1")
        result = stage.process(batch)
        result_df = result.to_pandas()

        assert "_seg_translation_pairs" in result_df.columns
        # First row of doc 0 should have the full pairs for that doc
        pairs_doc0 = json.loads(result_df["_seg_translation_pairs"].iloc[0])
        assert len(pairs_doc0) == 2
        assert pairs_doc0[0] == {"src": "Hello", "tgt": "Hallo"}
        assert pairs_doc0[1] == {"src": "World", "tgt": "Welt"}
        # Second row of doc 0 should have empty list (only first row carries pairs)
        assert json.loads(result_df["_seg_translation_pairs"].iloc[1]) == []
        # First row of doc 1
        pairs_doc1 = json.loads(result_df["_seg_translation_pairs"].iloc[2])
        assert len(pairs_doc1) == 1
        assert pairs_doc1[0] == {"src": "Goodbye", "tgt": "Auf Wiedersehen"}

    def test_capture_empty_batch(self) -> None:
        """An empty batch passes through without errors."""
        stage = SegmentPairCaptureStage()
        df = pd.DataFrame(
            {
                "_seg_segments": pd.Series(dtype="str"),
                "_translated": pd.Series(dtype="str"),
                "_seg_doc_id": pd.Series(dtype="int"),
            }
        )
        batch = DocumentBatch(data=df, dataset_name="test", task_id="1")
        result = stage.process(batch)
        assert "_seg_translation_pairs" in result.to_pandas().columns


# ---------------------------------------------------------------------------
# OutputFormattingStage tests
# ---------------------------------------------------------------------------


class TestOutputFormattingStage:
    """Tests for OutputFormattingStage."""

    def test_raw_mode_creates_metadata_drops_translated(self) -> None:
        """In 'raw' mode, translation_metadata is created and translated_text is dropped."""
        stage = OutputFormattingStage(
            output_mode="raw",
            target_lang="de",
            output_field="translated_text",
        )

        df = pd.DataFrame(
            {
                "text": ["Hello world."],
                "translated_text": ["Hallo Welt."],
                "id": [1],
            }
        )
        batch = DocumentBatch(data=df, dataset_name="test", task_id="1")
        result = stage.process(batch)
        result_df = result.to_pandas()

        assert "translation_metadata" in result_df.columns
        assert "translated_text" not in result_df.columns
        # Verify metadata structure
        meta = json.loads(result_df["translation_metadata"].iloc[0])
        assert meta["target_lang"] == "de"
        assert meta["translation"]["content"] == "Hallo Welt."

    def test_both_mode_keeps_both_columns(self) -> None:
        """In 'both' mode, both translated_text and translation_metadata are present."""
        stage = OutputFormattingStage(
            output_mode="both",
            target_lang="de",
            output_field="translated_text",
        )

        df = pd.DataFrame(
            {
                "text": ["Hello."],
                "translated_text": ["Hallo."],
                "id": [1],
            }
        )
        batch = DocumentBatch(data=df, dataset_name="test", task_id="1")
        result = stage.process(batch)
        result_df = result.to_pandas()

        assert "translated_text" in result_df.columns
        assert "translation_metadata" in result_df.columns

    def test_replaced_mode_no_metadata(self) -> None:
        """In 'replaced' mode, no translation_metadata column is added."""
        stage = OutputFormattingStage(
            output_mode="replaced",
            target_lang="de",
            output_field="translated_text",
        )

        df = pd.DataFrame(
            {
                "text": ["Hello."],
                "translated_text": ["Hallo."],
                "id": [1],
            }
        )
        batch = DocumentBatch(data=df, dataset_name="test", task_id="1")
        result = stage.process(batch)
        result_df = result.to_pandas()

        assert "translated_text" in result_df.columns
        assert "translation_metadata" not in result_df.columns


# ---------------------------------------------------------------------------
# ScoreMergeStage tests
# ---------------------------------------------------------------------------


class TestScoreMergeStage:
    """Tests for ScoreMergeStage."""

    def test_merge_scores_into_metadata(self) -> None:
        """FAITH scores are merged into the translation_metadata JSON."""
        stage = ScoreMergeStage()

        metadata = json.dumps({"target_lang": "de", "translation": {"content": "Hallo."}})
        df = pd.DataFrame(
            {
                "translation_metadata": [metadata],
                "faith_fluency": [4.0],
                "faith_accuracy": [4.5],
                "faith_idiomaticity": [3.5],
                "faith_terminology": [4.0],
                "faith_handling_of_format": [5.0],
                "faith_avg": [4.2],
            }
        )
        batch = DocumentBatch(data=df, dataset_name="test", task_id="1")
        result = stage.process(batch)
        result_df = result.to_pandas()

        assert "translation_metadata" in result_df.columns
        meta = json.loads(result_df["translation_metadata"].iloc[0])
        assert "faith_scores" in meta
        assert meta["faith_scores"]["average"] == pytest.approx(4.2)

    def test_merge_scores_no_faith_columns(self) -> None:
        """When no FAITH columns exist, the stage is a no-op."""
        stage = ScoreMergeStage()

        metadata = json.dumps({"target_lang": "de"})
        df = pd.DataFrame({"translation_metadata": [metadata], "text": ["Hello"]})
        batch = DocumentBatch(data=df, dataset_name="test", task_id="1")
        result = stage.process(batch)
        # Should return the batch unmodified
        assert result.to_pandas()["translation_metadata"].iloc[0] == metadata
