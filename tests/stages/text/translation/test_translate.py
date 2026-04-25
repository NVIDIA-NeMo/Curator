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

"""Unit tests for SegmentTranslationStage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nemo_curator.stages.text.translation.stages.translate import (
    SegmentTranslationStage,
)
from nemo_curator.tasks import DocumentBatch

from .conftest import MockAsyncLLMClient


# ---------------------------------------------------------------------------
# _unwrap_translation tests
# ---------------------------------------------------------------------------


class TestUnwrapTranslation:
    """Tests for the static _unwrap_translation helper."""

    def test_unwrap_translation_both_brackets(self) -> None:
        """Both brackets present -- extract the inner text."""
        result = SegmentTranslationStage._unwrap_translation("text \u3018translated\u3019 more")
        assert result == "translated"

    def test_unwrap_translation_left_only(self) -> None:
        """Only the left bracket present -- return everything after it."""
        result = SegmentTranslationStage._unwrap_translation("text \u3018translated")
        assert result == "translated"

    def test_unwrap_translation_no_brackets(self) -> None:
        """No brackets at all -- return the text unchanged."""
        result = SegmentTranslationStage._unwrap_translation("plain text")
        assert result == "plain text"

    def test_unwrap_translation_empty(self) -> None:
        """Empty string -- return empty string."""
        result = SegmentTranslationStage._unwrap_translation("")
        assert result == ""

    def test_unwrap_nested_brackets(self) -> None:
        """Multiple bracket pairs -- rfind picks the last left bracket."""
        text = "prefix \u3018first\u3019 middle \u3018second\u3019 suffix"
        result = SegmentTranslationStage._unwrap_translation(text)
        # rfind finds the last \u3018, which is before "second", and the last
        # \u3019 is after "second". So we get "second".
        assert result == "second"


# ---------------------------------------------------------------------------
# _build_messages tests
# ---------------------------------------------------------------------------


class TestBuildMessages:
    """Tests for _build_messages prompt construction."""

    def test_build_messages(self) -> None:
        """Verify message list structure with system + user roles."""
        client = MockAsyncLLMClient()
        stage = SegmentTranslationStage(
            client=client,
            model_name="test-model",
            source_lang="en",
            target_lang="hi",
        )
        # Trigger prompt loading (normally done in setup on a worker).
        stage._system_prompt = "You are a translator."
        stage._user_template = "Translate {source_lang} to {target_lang}: {src}"

        messages = stage._build_messages("Hello world")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a translator."
        assert messages[1]["role"] == "user"
        assert "English" in messages[1]["content"]
        assert "Hindi" in messages[1]["content"]
        assert "Hello world" in messages[1]["content"]


# ---------------------------------------------------------------------------
# process() tests
# ---------------------------------------------------------------------------


class TestProcessLLMBackend:
    """Tests for process() with the LLM backend path."""

    def test_llm_backend_requires_model_name(self) -> None:
        """LLM translation should fail fast when model_name is unset."""
        with pytest.raises(ValueError, match="non-empty 'model_name'"):
            SegmentTranslationStage(
                client=MockAsyncLLMClient(),
                model_name="",
                backend_type="llm",
                source_lang="en",
                target_lang="hi",
            )

    def test_process_llm_backend(self) -> None:
        """Mock AsyncLLMClient -- verify _translated column is populated."""
        client = MockAsyncLLMClient()
        stage = SegmentTranslationStage(
            client=client,
            model_name="test-model",
            source_lang="en",
            target_lang="hi",
            backend_type="llm",
        )
        # Pre-load prompts to avoid file I/O in unit tests.
        stage._system_prompt = "You are a translator."
        stage._user_template = "Translate {source_lang} to {target_lang}: {src}"
        stage._initialized = True

        df = pd.DataFrame(
            {
                "_seg_segments": ["Hello world", "Goodbye"],
                "id": [1, 2],
            }
        )
        batch = DocumentBatch(data=df, dataset_name="test", task_id="1")

        result = stage.process(batch)
        result_df = result.to_pandas()

        assert "_translated" in result_df.columns
        assert len(result_df["_translated"]) == 2
        # MockAsyncLLMClient wraps in brackets, _unwrap_translation strips them.
        for val in result_df["_translated"]:
            assert isinstance(val, str)
            assert len(val) > 0
            # Should NOT contain the bracket characters after unwrapping.
            assert "\u3018" not in val
            assert "\u3019" not in val


class TestProcessNonLLMBackend:
    """Tests for process() with a non-LLM backend (delegation path)."""

    def test_process_non_llm_backend(self) -> None:
        """Mock backend translate_batch_async -- verify bulk delegation.

        The non-LLM backend path should batch all translatable segments into a
        single backend request, falling back to per-segment requests only on
        backend failure.
        """
        mock_backend = MagicMock()

        async def _fake_async(texts, src, tgt):
            assert texts == ["Hello world", "Goodbye"]
            assert src == "en"
            assert tgt == "es"
            return ["Hola mundo", "Adios"]

        mock_backend.translate_batch_async = MagicMock(side_effect=_fake_async)

        stage = SegmentTranslationStage(
            client=None,
            backend_type="google",
            source_lang="en",
            target_lang="es",
        )
        # Manually set the backend (normally done in setup).
        stage._backend = mock_backend
        stage._initialized = True

        df = pd.DataFrame(
            {
                "_seg_segments": ["Hello world", "Goodbye"],
                "id": [1, 2],
            }
        )
        batch = DocumentBatch(data=df, dataset_name="test", task_id="1")

        result = stage.process(batch)
        result_df = result.to_pandas()

        assert "_translated" in result_df.columns
        assert result_df["_translated"].tolist() == ["Hola mundo", "Adios"]
        assert mock_backend.translate_batch_async.call_count == 1
        mock_backend.translate_batch_async.assert_called_once_with(
            ["Hello world", "Goodbye"], "en", "es"
        )

    def test_fallback_preserves_non_translatable_segments(self) -> None:
        """Fallback path should keep passthrough segments instead of dropping them."""
        mock_backend = MagicMock()

        call_count = {"value": 0}

        async def _fake_async(texts, src, tgt):
            assert src == "en"
            assert tgt == "es"
            call_count["value"] += 1
            if call_count["value"] == 1:
                raise RuntimeError("bulk failure")
            return [f"TR:{texts[0]}"]

        mock_backend.translate_batch_async = MagicMock(side_effect=_fake_async)

        stage = SegmentTranslationStage(
            client=None,
            backend_type="google",
            source_lang="en",
            target_lang="es",
        )
        stage._backend = mock_backend
        stage._initialized = True

        json_blob = '{"tool":"lookup","payload":{"model":"DeepSeek V3"}}'
        df = pd.DataFrame(
            {
                "_seg_segments": ["Hello world", json_blob, "Goodbye"],
                "id": [1, 2, 3],
            }
        )
        batch = DocumentBatch(data=df, dataset_name="test", task_id="1")

        result = stage.process(batch)
        result_df = result.to_pandas()

        assert result_df["_translated"].tolist() == [
            "TR:Hello world",
            json_blob,
            "TR:Goodbye",
        ]
        assert mock_backend.translate_batch_async.call_count == 3


# ---------------------------------------------------------------------------
# inputs / outputs tests
# ---------------------------------------------------------------------------


class TestInputsOutputs:
    """Tests for column declarations."""

    def test_inputs_outputs(self) -> None:
        """Verify column declarations match the plan."""
        client = MockAsyncLLMClient()
        stage = SegmentTranslationStage(
            client=client,
            model_name="test-model",
            source_lang="en",
            target_lang="hi",
        )
        top_attrs, data_cols = stage.inputs()
        assert "data" in top_attrs
        assert "_seg_segments" in data_cols

        top_attrs_out, data_cols_out = stage.outputs()
        assert "data" in top_attrs_out
        assert "_translated" in data_cols_out


# ---------------------------------------------------------------------------
# setup() tests
# ---------------------------------------------------------------------------


class TestSetup:
    """Tests for setup() lifecycle."""

    @patch(
        "nemo_curator.stages.text.translation.stages.translate.load_prompt_template",
        return_value=("system prompt", "user {source_lang} {target_lang} {src}"),
    )
    def test_setup_initializes_client(self, mock_load: MagicMock) -> None:
        """Verify setup() calls client.setup()."""
        client = MockAsyncLLMClient()
        client.setup = MagicMock()
        stage = SegmentTranslationStage(
            client=client,
            model_name="test-model",
            source_lang="en",
            target_lang="hi",
        )
        stage.setup(worker_metadata=None)

        client.setup.assert_called_once()
        mock_load.assert_called_once()
        assert stage._system_prompt == "system prompt"


# ---------------------------------------------------------------------------
# Field defaults tests
# ---------------------------------------------------------------------------


class TestFieldDefaults:
    """Tests for dataclass field defaults."""

    def test_semaphore_field(self) -> None:
        """Verify max_concurrent_requests field exists and defaults to 64."""
        client = MockAsyncLLMClient()
        stage = SegmentTranslationStage(
            client=client,
            model_name="test-model",
            source_lang="en",
            target_lang="hi",
        )
        assert stage.max_concurrent_requests == 64


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    """Tests for constructor validation."""

    def test_client_none_for_llm_raises(self) -> None:
        """backend_type='llm' with client=None raises ValueError."""
        with pytest.raises(ValueError, match="requires a non-None 'client'"):
            SegmentTranslationStage(
                client=None,
                backend_type="llm",
                model_name="test-model",
                source_lang="en",
                target_lang="hi",
            )
