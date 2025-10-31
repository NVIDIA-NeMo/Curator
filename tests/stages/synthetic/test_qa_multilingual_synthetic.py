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
from nemo_curator.stages.synthetic.qa_multilingual_synthetic import (
    LanguageFilter,
    QAMultilingualSyntheticStage,
)
from nemo_curator.tasks import DocumentBatch, _EmptyTask


class MockSyncLLMClient(LLMClient):
    """Mock synchronous LLM client for testing."""

    def __init__(self, responses: list[list[str]] | None = None):
        self.responses = responses or [["test response"]]
        self.call_count = 0
        self.setup_called = False

    def setup(self) -> None:
        self.setup_called = True

    def query_model(self, *, messages: Iterable, model: str, generation_config: GenerationConfig | None = None, **kwargs: object) -> list[str]:  # noqa: ARG002
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

    async def _query_model_impl(self, *, messages: Iterable, model: str, generation_config: GenerationConfig | None = None, **kwargs: object) -> list[str]:  # noqa: ARG002
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


class TestQAMultilingualSyntheticStage:
    """Test cases for QAMultilingualSyntheticStage."""

    def test_init_with_sync_client(self) -> None:
        """Test initialization with synchronous client."""
        client = MockSyncLLMClient()
        stage = QAMultilingualSyntheticStage(
            prompt="Generate text in {language}",
            languages=["English", "Spanish"],
            client=client,
            model_name="test-model",
            num_samples=5
        )

        assert stage.prompt == "Generate text in {language}"
        assert stage.languages == ["English", "Spanish"]
        assert stage.client is client
        assert stage.model_name == "test-model"
        assert stage.num_samples == 5
        assert stage.generation_config is None
        assert stage._name == "QAMultilingualSyntheticStage"
        assert stage.is_async_client is False

    def test_init_with_async_client(self) -> None:
        """Test initialization with asynchronous client."""
        client = MockAsyncLLMClient()
        stage = QAMultilingualSyntheticStage(
            prompt="Generate text in {language}",
            languages=["French", "German"],
            client=client,
            model_name="test-model",
            num_samples=3
        )

        assert stage.is_async_client is True

    def test_init_with_generation_config(self) -> None:
        """Test initialization with generation config."""
        client = MockSyncLLMClient()
        config = GenerationConfig(max_tokens=100, temperature=0.7)
        stage = QAMultilingualSyntheticStage(
            prompt="Test prompt {language}",
            languages=["English"],
            client=client,
            model_name="test-model",
            num_samples=1,
            generation_config=config
        )

        assert stage.generation_config is config
        assert stage.generation_config.max_tokens == 100
        assert stage.generation_config.temperature == 0.7

    def test_inputs_outputs(self) -> None:
        """Test inputs() and outputs() methods."""
        client = MockSyncLLMClient()
        stage = QAMultilingualSyntheticStage(
            prompt="Test {language}",
            languages=["English"],
            client=client,
            model_name="test-model",
            num_samples=1
        )

        inputs = stage.inputs()
        assert inputs == ([], [])

        outputs = stage.outputs()
        assert outputs == (["data"], ["text"])

    def test_setup(self) -> None:
        """Test setup() method calls client.setup()."""
        client = MockSyncLLMClient()
        stage = QAMultilingualSyntheticStage(
            prompt="Test {language}",
            languages=["English"],
            client=client,
            model_name="test-model",
            num_samples=1
        )

        assert client.setup_called is False
        stage.setup()
        assert client.setup_called is True

    def test_process_llm_response_simple(self) -> None:
        """Test _process_llm_response with simple text."""
        client = MockSyncLLMClient()
        stage = QAMultilingualSyntheticStage(
            prompt="Test {language}",
            languages=["English"],
            client=client,
            model_name="test-model",
            num_samples=1
        )

        result = stage._process_llm_response(["Simple response"])
        assert result == "Simple response"

    def test_process_llm_response_with_asterisks(self) -> None:
        """Test _process_llm_response removes asterisks."""
        client = MockSyncLLMClient()
        stage = QAMultilingualSyntheticStage(
            prompt="Test {language}",
            languages=["English"],
            client=client,
            model_name="test-model",
            num_samples=1
        )

        result = stage._process_llm_response(["**Bold text** with *emphasis*"])
        assert result == "Bold text with emphasis"

    def test_process_llm_response_empty(self) -> None:
        """Test _process_llm_response with empty response."""
        client = MockSyncLLMClient()
        stage = QAMultilingualSyntheticStage(
            prompt="Test {language}",
            languages=["English"],
            client=client,
            model_name="test-model",
            num_samples=1
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
            num_samples=1
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
        responses = [
            ["Response 1"],
            ["Response 2"],
            ["Response 3"]
        ]
        client = MockSyncLLMClient(responses=responses)
        stage = QAMultilingualSyntheticStage(
            prompt="Generate in {language}",
            languages=["English", "Spanish"],
            client=client,
            model_name="test-model",
            num_samples=3
        )

        task = _EmptyTask(task_id="test", dataset_name="test", data=None)
        with patch("builtins.print"):  # Suppress print statements
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
            num_samples=1
        )

        task = _EmptyTask(task_id="test", dataset_name="test", data=None)
        result = stage.process(task)

        assert isinstance(result, DocumentBatch)
        assert len(result.data) == 1
        assert result.data["text"].iloc[0] == "Async response"
        assert client.call_count == 1

    def test_process_async_multiple_samples(self) -> None:
        """Test asynchronous processing with multiple concurrent samples."""
        responses = [
            ["Async 1"],
            ["Async 2"],
            ["Async 3"],
            ["Async 4"],
            ["Async 5"]
        ]
        client = MockAsyncLLMClient(responses=responses, delay=0.01)
        stage = QAMultilingualSyntheticStage(
            prompt="Generate in {language}",
            languages=["English", "Spanish", "French"],
            client=client,
            model_name="test-model",
            num_samples=5
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
            generation_config=config
        )

        task = _EmptyTask(task_id="test", dataset_name="test", data=None)
        with patch("builtins.print"):
            result = stage.process(task)

        assert len(result.data) == 1
        assert result.data["text"].iloc[0] == "Config response"

    def test_process_sync_language_formatting(self) -> None:
        """Test that language is properly formatted into prompt."""
        captured_prompts = []

        class CapturePromptClient(LLMClient):
            def setup(self) -> None:
                pass

            def query_model(self, *, messages: Iterable, model: str, **kwargs: object) -> list[str]:  # noqa: ARG002
                captured_prompts.append(messages[0]["content"])
                return ["response"]

        client = CapturePromptClient()
        stage = QAMultilingualSyntheticStage(
            prompt="Please write in {language} language",
            languages=["Japanese"],
            client=client,
            model_name="test-model",
            num_samples=2
        )

        with patch("builtins.print"), patch("secrets.choice", return_value="Japanese"):
            task = _EmptyTask(task_id="test", dataset_name="test", data=None)
            stage.process(task)

        assert len(captured_prompts) == 2
        assert all(p == "Please write in Japanese language" for p in captured_prompts)

    def test_process_sync_with_response_asterisks(self) -> None:
        """Test that asterisks are removed from responses in sync mode."""
        client = MockSyncLLMClient(responses=[["**Bold** *italic* text"]])
        stage = QAMultilingualSyntheticStage(
            prompt="Test {language}",
            languages=["English"],
            client=client,
            model_name="test-model",
            num_samples=1
        )

        task = _EmptyTask(task_id="test", dataset_name="test", data=None)
        with patch("builtins.print"):
            result = stage.process(task)

        assert result.data["text"].iloc[0] == "Bold italic text"

    def test_process_async_with_response_asterisks(self) -> None:
        """Test that asterisks are removed from responses in async mode."""
        client = MockAsyncLLMClient(responses=[["**Another** *styled* response"]])
        stage = QAMultilingualSyntheticStage(
            prompt="Test {language}",
            languages=["English"],
            client=client,
            model_name="test-model",
            num_samples=1
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

            def query_model(self, *, messages: Iterable, model: str, **kwargs: object) -> list[str]:  # noqa: ARG002
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
            num_samples=10
        )

        with patch("builtins.print"):
            task = _EmptyTask(task_id="test", dataset_name="test", data=None)
            stage.process(task)

        # Should have captured 10 languages
        assert len(selected_languages) == 10
        # All should be from the allowed set
        assert all(lang in ["English", "Spanish", "French"] for lang in selected_languages)


class TestLanguageFilter:
    """Test cases for LanguageFilter."""

    def test_init(self) -> None:
        """Test LanguageFilter initialization."""
        languages = ["English", "Spanish", "French"]
        filter_instance = LanguageFilter(languages)

        assert filter_instance._name == "language_filter"
        assert filter_instance.languages == languages

    def test_score_document_matching(self) -> None:
        """Test score_document with text starting with language."""
        filter_instance = LanguageFilter(["English", "Spanish"])

        score = filter_instance.score_document("English text here")
        assert score == 1.0

        score = filter_instance.score_document("Spanish content")
        assert score == 1.0

    def test_score_document_not_matching(self) -> None:
        """Test score_document with text not starting with language."""
        filter_instance = LanguageFilter(["English", "Spanish"])

        score = filter_instance.score_document("French text here")
        assert score == 0.0

        score = filter_instance.score_document("Some other content")
        assert score == 0.0

    def test_score_document_partial_match(self) -> None:
        """Test score_document with language appearing later in text."""
        filter_instance = LanguageFilter(["English"])

        # Language not at start - should not match
        score = filter_instance.score_document("This is English text")
        assert score == 0.0

    def test_score_document_empty(self) -> None:
        """Test score_document with empty text."""
        filter_instance = LanguageFilter(["English"])

        score = filter_instance.score_document("")
        assert score == 0.0

    def test_keep_document_with_matching_score(self) -> None:
        """Test keep_document returns True for score 1.0."""
        filter_instance = LanguageFilter(["English"])

        assert filter_instance.keep_document(1.0) is True

    def test_keep_document_with_non_matching_score(self) -> None:
        """Test keep_document returns False for score 0.0."""
        filter_instance = LanguageFilter(["English"])

        assert filter_instance.keep_document(0.0) is False

    def test_keep_document_with_other_scores(self) -> None:
        """Test keep_document with various scores."""
        filter_instance = LanguageFilter(["English"])

        # Only exact 1.0 should pass
        assert filter_instance.keep_document(0.5) is False
        assert filter_instance.keep_document(0.99) is False

    def test_filter_multiple_languages(self) -> None:
        """Test filter with multiple languages."""
        filter_instance = LanguageFilter(["English", "Spanish", "French", "German"])

        assert filter_instance.score_document("English text") == 1.0
        assert filter_instance.score_document("Spanish text") == 1.0
        assert filter_instance.score_document("French text") == 1.0
        assert filter_instance.score_document("German text") == 1.0
        assert filter_instance.score_document("Italian text") == 0.0

    def test_filter_case_sensitive(self) -> None:
        """Test that filter is case-sensitive."""
        filter_instance = LanguageFilter(["English"])

        # Exact match
        assert filter_instance.score_document("English text") == 1.0
        # Different case - won't match with startswith(tuple)
        assert filter_instance.score_document("english text") == 0.0
        assert filter_instance.score_document("ENGLISH text") == 0.0

    def test_filter_empty_language_list(self) -> None:
        """Test filter with empty language list keeps all documents."""
        filter_instance = LanguageFilter([])

        # With empty language list, should keep all documents (return 1.0)
        assert filter_instance.score_document("English text") == 1.0
        assert filter_instance.score_document("Spanish text") == 1.0
        assert filter_instance.score_document("[EN] Some text") == 1.0
        assert filter_instance.score_document("[FR] Autre texte") == 1.0
        assert filter_instance.score_document("Any text at all") == 1.0
        assert filter_instance.score_document("") == 1.0  # Even empty text

        # Verify keep_document also returns True
        score = filter_instance.score_document("Any text")
        assert filter_instance.keep_document(score) is True

    def test_integration_score_and_keep(self) -> None:
        """Test integration of score_document and keep_document."""
        filter_instance = LanguageFilter(["Python", "Java", "JavaScript"])

        texts = [
            "Python is great",
            "Java programming",
            "JavaScript code",
            "Ruby language",
            "C++ development"
        ]

        results = []
        for text in texts:
            score = filter_instance.score_document(text)
            keep = filter_instance.keep_document(score)
            results.append((text, keep))

        assert results[0][1] is True  # Python is great
        assert results[1][1] is True  # Java programming
        assert results[2][1] is True  # JavaScript code
        assert results[3][1] is False  # Ruby language
        assert results[4][1] is False  # C++ development

