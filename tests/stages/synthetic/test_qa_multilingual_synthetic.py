# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Unit tests for nemo_curator.stages.synthetic.qa_multilingual_synthetic module.
"""

import asyncio
from collections.abc import Iterable
from unittest.mock import patch

import pandas as pd

from nemo_curator.models.client.llm_client import AsyncLLMClient, GenerationConfig, LLMClient
from nemo_curator.stages.synthetic.qa_multilingual_synthetic import QAMultilingualSyntheticStage
from nemo_curator.tasks import DocumentBatch, _EmptyTask


class MockSyncLLMClient(LLMClient):
    """Mock synchronous LLM client for testing."""

    def __init__(self, responses: list[list[str]] | None = None):
        self.responses = responses or [["test response"]]
        self.call_count = 0
        self.setup_called = False

    def setup(self) -> None:
        self.setup_called = True

    def query_model(
        self, *, messages: Iterable, model: str, generation_config: GenerationConfig | None = None, **kwargs: object
    ) -> list[str]:
        del messages, model, generation_config, kwargs  # Unused in mock
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


class MockAsyncLLMClient(AsyncLLMClient):
    """Mock asynchronous LLM client for testing."""

    def __init__(self, responses: list[list[str]] | None = None, delay: float = 0.0):
        super().__init__()
        self.responses = responses or [["test response"]]
        self.call_count = 0
        self.setup_called = False
        self.delay = delay

    def setup(self) -> None:
        self.setup_called = True

    async def _query_model_impl(
        self, *, messages: Iterable, model: str, generation_config: GenerationConfig | None = None, **kwargs: object
    ) -> list[str]:
        del messages, model, generation_config, kwargs  # Unused in mock
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


class TestQAMultilingualSyntheticStage:
    """Test cases for QAMultilingualSyntheticStage."""

    def test_setup(self) -> None:
        """Test setup() method calls client.setup()."""
        client = MockSyncLLMClient()
        stage = QAMultilingualSyntheticStage(
            prompt="Test {language}", languages=["English"], client=client, model_name="test-model", num_samples=1
        )

        assert client.setup_called is False
        stage.setup()
        assert client.setup_called is True

    def test_process_llm_response_simple(self) -> None:
        """Test _process_llm_response with simple text."""
        client = MockSyncLLMClient()
        stage = QAMultilingualSyntheticStage(
            prompt="Test {language}", languages=["English"], client=client, model_name="test-model", num_samples=1
        )

        result = stage._process_llm_response(["Simple response"])
        assert result == "Simple response"

    def test_process_llm_response_with_asterisks(self) -> None:
        """Test _process_llm_response removes asterisks."""
        client = MockSyncLLMClient()
        stage = QAMultilingualSyntheticStage(
            prompt="Test {language}", languages=["English"], client=client, model_name="test-model", num_samples=1
        )

        result = stage._process_llm_response(["**Bold text** with *emphasis*"])
        assert result == "Bold text with emphasis"

    def test_process_llm_response_empty(self) -> None:
        """Test _process_llm_response with empty response."""
        client = MockSyncLLMClient()
        stage = QAMultilingualSyntheticStage(
            prompt="Test {language}", languages=["English"], client=client, model_name="test-model", num_samples=1
        )

        result = stage._process_llm_response([])
        assert result == ""

    def test_process_sync_single_sample(self) -> None:
        """Test synchronous processing with single sample."""
        client = MockSyncLLMClient(responses=[["English response"]])
        stage = QAMultilingualSyntheticStage(
            prompt="Generate in {language}",
            languages=["English"],
            client=client,
            model_name="test-model",
            num_samples=1,
        )

        task = _EmptyTask(task_id="test", dataset_name="test", data=None)
        result = stage.process(task)

        assert isinstance(result, DocumentBatch)
        assert result.dataset_name == "simple_synthetic_data"
        assert result.task_id == 1
        assert isinstance(result.data, pd.DataFrame)
        assert len(result.data) == 1
        assert "text" in result.data.columns
        assert result.data["text"].iloc[0] == "English response"
        assert client.call_count == 1

    def test_process_sync_multiple_samples(self) -> None:
        """Test synchronous processing with multiple samples."""
        responses = [["Response 1"], ["Response 2"], ["Response 3"]]
        client = MockSyncLLMClient(responses=responses)
        stage = QAMultilingualSyntheticStage(
            prompt="Generate in {language}",
            languages=["English", "Spanish"],
            client=client,
            model_name="test-model",
            num_samples=3,
        )

        task = _EmptyTask(task_id="test", dataset_name="test", data=None)
        with patch("nemo_curator.models.client.llm_client.logger"):  # Suppress log statements
            result = stage.process(task)

        assert isinstance(result, DocumentBatch)
        assert len(result.data) == 3
        assert result.data["text"].tolist() == ["Response 1", "Response 2", "Response 3"]
        assert client.call_count == 3

    def test_process_async_single_sample(self) -> None:
        """Test asynchronous processing with single sample."""
        client = MockAsyncLLMClient(responses=[["Async response"]])
        stage = QAMultilingualSyntheticStage(
            prompt="Generate in {language}",
            languages=["English"],
            client=client,
            model_name="test-model",
            num_samples=1,
        )

        task = _EmptyTask(task_id="test", dataset_name="test", data=None)
        result = stage.process(task)

        assert isinstance(result, DocumentBatch)
        assert len(result.data) == 1
        assert result.data["text"].iloc[0] == "Async response"
        assert client.call_count == 1

    def test_process_async_multiple_samples(self) -> None:
        """Test asynchronous processing with multiple concurrent samples."""
        responses = [["Async 1"], ["Async 2"], ["Async 3"], ["Async 4"], ["Async 5"]]
        client = MockAsyncLLMClient(responses=responses, delay=0.01)
        stage = QAMultilingualSyntheticStage(
            prompt="Generate in {language}",
            languages=["English", "Spanish", "French"],
            client=client,
            model_name="test-model",
            num_samples=5,
        )

        task = _EmptyTask(task_id="test", dataset_name="test", data=None)
        result = stage.process(task)

        assert isinstance(result, DocumentBatch)
        assert len(result.data) == 5
        # Check all responses are present (order might vary due to async)
        assert set(result.data["text"].tolist()) == {"Async 1", "Async 2", "Async 3", "Async 4", "Async 5"}
        assert client.call_count == 5

    def test_process_sync_with_generation_config(self) -> None:
        """Test synchronous processing with generation config."""
        client = MockSyncLLMClient(responses=[["Config response"]])
        config = GenerationConfig(max_tokens=50, temperature=0.5, seed=42)
        stage = QAMultilingualSyntheticStage(
            prompt="Generate in {language}",
            languages=["English"],
            client=client,
            model_name="test-model",
            num_samples=1,
            generation_config=config,
        )

        task = _EmptyTask(task_id="test", dataset_name="test", data=None)
        with patch("nemo_curator.models.client.llm_client.logger"):
            result = stage.process(task)

        assert len(result.data) == 1
        assert result.data["text"].iloc[0] == "Config response"

    def test_process_sync_language_formatting(self) -> None:
        """Test that language is properly formatted into prompt."""
        captured_prompts = []

        class CapturePromptClient(LLMClient):
            def setup(self) -> None:
                pass

            def query_model(self, *, messages: Iterable, model: str, **kwargs: object) -> list[str]:
                captured_prompts.append(messages[0]["content"])
                return ["response"]

        client = CapturePromptClient()
        stage = QAMultilingualSyntheticStage(
            prompt="Please write in {language} language",
            languages=["Japanese"],
            client=client,
            model_name="test-model",
            num_samples=2,
        )

        with patch("nemo_curator.models.client.llm_client.logger"), patch("secrets.choice", return_value="Japanese"):
            task = _EmptyTask(task_id="test", dataset_name="test", data=None)
            stage.process(task)

        assert len(captured_prompts) == 2
        assert all(p == "Please write in Japanese language" for p in captured_prompts)

    def test_process_sync_with_response_asterisks(self) -> None:
        """Test that asterisks are removed from responses in sync mode."""
        client = MockSyncLLMClient(responses=[["**Bold** *italic* text"]])
        stage = QAMultilingualSyntheticStage(
            prompt="Test {language}", languages=["English"], client=client, model_name="test-model", num_samples=1
        )

        task = _EmptyTask(task_id="test", dataset_name="test", data=None)
        with patch("nemo_curator.models.client.llm_client.logger"):
            result = stage.process(task)

        assert result.data["text"].iloc[0] == "Bold italic text"

    def test_process_async_with_response_asterisks(self) -> None:
        """Test that asterisks are removed from responses in async mode."""
        client = MockAsyncLLMClient(responses=[["**Another** *styled* response"]])
        stage = QAMultilingualSyntheticStage(
            prompt="Test {language}", languages=["English"], client=client, model_name="test-model", num_samples=1
        )

        task = _EmptyTask(task_id="test", dataset_name="test", data=None)
        result = stage.process(task)

        assert result.data["text"].iloc[0] == "Another styled response"

    def test_language_selection_randomness(self) -> None:
        """Test that languages are selected from the provided list."""
        selected_languages = []

        class LanguageCaptureClient(LLMClient):
            def setup(self) -> None:
                pass

            def query_model(self, *, messages: Iterable, model: str, **kwargs: object) -> list[str]:
                content = messages[0]["content"]
                for lang in ["English", "Spanish", "French"]:
                    if lang in content:
                        selected_languages.append(lang)
                        break
                return ["response"]

        client = LanguageCaptureClient()
        stage = QAMultilingualSyntheticStage(
            prompt="Use {language}",
            languages=["English", "Spanish", "French"],
            client=client,
            model_name="test-model",
            num_samples=10,
        )

        with patch("nemo_curator.models.client.llm_client.logger"):
            task = _EmptyTask(task_id="test", dataset_name="test", data=None)
            stage.process(task)

        # Should have captured 10 languages
        assert len(selected_languages) == 10
        # All should be from the allowed set
        assert all(lang in ["English", "Spanish", "French"] for lang in selected_languages)
