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

from unittest.mock import Mock, patch

import pandas as pd
import pytest

# Import after setting up patches to ensure mocks are in place
from nemo_curator.stages.math.modifiers.llm_cleanup import LLMCleanupStage
from nemo_curator.tasks import DocumentBatch


class MockLLMOutput:
    """Mock output from vLLM's generate method."""

    def __init__(self, text: str):
        self.outputs = [Mock(text=text)]


class MockLLM:
    """Mock vLLM LLM class."""

    def __init__(self, *args, **kwargs):  # noqa: ARG002
        self.model = kwargs.get("model", "test-model")
        self.max_model_len = kwargs.get("max_model_len", 32000)

    def generate(self, prompts: list[str], sampling_params=None, use_tqdm=False):  # noqa: ARG002, ANN001
        """Mock generate method that returns cleaned text."""
        results = []
        for prompt in prompts:
            # Extract text from prompt - look for "Text:" marker first
            if "Text:" in prompt:
                # Extract text after "Text:" marker
                text_start = prompt.find("Text:") + len("Text:")
                text_end = prompt.find("\n", text_start)
                if text_end == -1:
                    text_end = len(prompt)
                original_text = prompt[text_start:text_end].strip()
                cleaned_text = f"Cleaned: {original_text}"
            else:
                # For prompts like "Clean this text: Original text here"
                # Try to extract text after the last colon
                # Split by newlines first to handle multi-line prompts
                lines = prompt.split("\n")
                last_line = lines[-1] if lines else prompt
                if ":" in last_line:
                    parts = last_line.split(":")
                    if len(parts) > 1:
                        original_text = parts[-1].strip()
                        cleaned_text = f"Cleaned: {original_text}" if original_text else "Cleaned output"
                    else:
                        cleaned_text = "Cleaned output"
                else:
                    # Fallback: use a default cleaned output
                    cleaned_text = "Cleaned output"
            results.append(MockLLMOutput(cleaned_text))
        return results


class MockSamplingParams:
    """Mock SamplingParams class."""

    def __init__(self, **kwargs):
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 0.8)
        self.top_k = kwargs.get("top_k")
        self.min_p = kwargs.get("min_p")
        self.max_tokens = kwargs.get("max_tokens")


class MockVLLMModel:
    """Mock VLLMModel class that prevents real vLLM initialization."""

    def __init__(self, *args, **kwargs):  # noqa: ARG002
        # Store all kwargs as attributes
        self.model = kwargs.get("model", "test-model")
        self.max_model_len = kwargs.get("max_model_len")
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 0.8)
        self.top_k = kwargs.get("top_k", 20)
        self.min_p = kwargs.get("min_p", 0.0)
        self.max_tokens = kwargs.get("max_tokens")
        self.cache_dir = kwargs.get("cache_dir")
        self._llm = None
        self._sampling_params = None

    def model_id_names(self):
        return [self.model]

    def setup(self):
        """Mock setup that initializes mock LLM - never calls real vLLM."""
        # Initialize mocks directly without calling real vLLM
        self._llm = MockLLM(model=self.model, max_model_len=self.max_model_len)
        # Create sampling params with all parameters that might be set
        sampling_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens if self.max_tokens is not None else self.max_model_len,
        }
        is_qwen3 = "Qwen3" in self.model or "qwen3" in self.model.lower()
        if is_qwen3:
            sampling_kwargs.update({
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
            })
        else:
            sampling_kwargs["top_p"] = self.top_p
        self._sampling_params = MockSamplingParams(**sampling_kwargs)

    def generate(self, prompts: list[str]) -> list[str]:
        """Mock generate method that returns cleaned text."""
        if self._llm is None or self._sampling_params is None:
            msg = "Model not initialized. Call setup() first."
            raise RuntimeError(msg)
        outputs = self._llm.generate(prompts, self._sampling_params, use_tqdm=False)
        return [out.outputs[0].text for out in outputs]


@pytest.fixture(autouse=True)
def setup_mocks():
    """Automatically setup mocks for VLLMModel."""
    # Patch VLLMModel where it's imported in llm_cleanup module
    # Also patch vLLM classes to prevent any real initialization attempts
    with (
        patch("nemo_curator.stages.math.modifiers.llm_cleanup.VLLMModel", MockVLLMModel),
        patch("nemo_curator.models.vllm_model.LLM", MockLLM),
        patch("nemo_curator.models.vllm_model.SamplingParams", MockSamplingParams),
        patch("nemo_curator.models.vllm_model.VLLM_AVAILABLE", True),
    ):
        yield


class TestLLMCleanupStage:
    """Test the LLMCleanupStage class."""

    def test_init_default_values(self):
        """Test LLMCleanupStage initialization with default values."""
        stage = LLMCleanupStage(model="test-model", system_prompt="Clean: {text}")

        assert stage._model.model == "test-model"
        assert stage.system_prompt == "Clean: {text}"
        assert stage.text_field == "text"
        assert stage.output_field == "cleaned_text"
        assert stage.max_model_len is None
        assert stage.classification is False
        assert stage.filter_by_n_tokens is False
        assert stage.n_tokens_field == "n_tokens"

    def test_init_custom_values(self):
        """Test LLMCleanupStage initialization with custom values."""
        stage = LLMCleanupStage(
            model="custom-model",
            system_prompt="Custom: {text}",
            text_field="content",
            output_field="cleaned_content",
            max_model_len=16000,
            classification=True,
            temperature=0.5,
            top_p=0.9,
            top_k=10,
            min_p=0.1,
            max_tokens=1000,
            cache_dir="/custom/cache",
            filter_by_n_tokens=True,
            n_tokens_field="custom_n_tokens",
        )

        assert stage._model.model == "custom-model"
        assert stage.system_prompt == "Custom: {text}"
        assert stage.text_field == "content"
        assert stage.output_field == "cleaned_content"
        assert stage.max_model_len == 16000
        assert stage.classification is True
        assert stage._model.temperature == 0.5
        assert stage._model.top_p == 0.9
        assert stage._model.top_k == 10
        assert stage._model.min_p == 0.1
        assert stage._model.max_tokens == 1000
        assert stage._model.cache_dir == "/custom/cache"
        assert stage.filter_by_n_tokens is True
        assert stage.n_tokens_field == "custom_n_tokens"

    def test_inputs_outputs_cleanup_mode(self):
        """Test inputs and outputs methods in cleanup mode."""
        stage = LLMCleanupStage(model="test-model", system_prompt="Clean: {text}")

        inputs = stage.inputs()
        outputs = stage.outputs()

        assert inputs == (["data"], ["text"])
        assert outputs == (["data"], ["cleaned_text"])

    def test_inputs_outputs_classification_mode(self):
        """Test inputs and outputs methods in classification mode."""
        stage = LLMCleanupStage(
            model="test-model", system_prompt="Classify: {text}", classification=True
        )

        inputs = stage.inputs()
        outputs = stage.outputs()

        assert inputs == (["data"], ["text"])
        assert outputs == (["data"], ["label"])

    def test_inputs_with_filter_by_n_tokens(self):
        """Test inputs method when filter_by_n_tokens is enabled."""
        stage = LLMCleanupStage(
            model="test-model", system_prompt="Clean: {text}", filter_by_n_tokens=True
        )

        inputs = stage.inputs()

        assert inputs == (["data"], ["text", "n_tokens"])

    def test_setup_initializes_llm(self):
        """Test that setup method initializes LLM and sampling params."""
        stage = LLMCleanupStage(model="test-model", system_prompt="Clean: {text}")

        stage.setup()

        assert stage._model._llm is not None
        assert stage._model._sampling_params is not None
        assert isinstance(stage._model._llm, MockLLM)
        assert isinstance(stage._model._sampling_params, MockSamplingParams)

    def test_setup_qwen3_model(self):
        """Test setup method with Qwen3 model (special sampling params)."""
        stage = LLMCleanupStage(
            model="Qwen/Qwen3-30B-A3B", system_prompt="Clean: {text}", top_k=10, min_p=0.1
        )

        stage.setup()

        assert stage._model._sampling_params is not None
        # Qwen3 models should have top_k and min_p set
        assert hasattr(stage._model._sampling_params, "top_p")
        assert hasattr(stage._model._sampling_params, "top_k")
        assert hasattr(stage._model._sampling_params, "min_p")

    def test_setup_non_qwen3_model(self):
        """Test setup method with non-Qwen3 model."""
        stage = LLMCleanupStage(model="microsoft/phi-4", system_prompt="Clean: {text}")

        stage.setup()

        assert stage._model._sampling_params is not None
        # Non-Qwen3 models should only have top_p
        assert hasattr(stage._model._sampling_params, "top_p")

    def test_setup_with_cache_dir(self):
        """Test setup method with cache_dir specified."""
        stage = LLMCleanupStage(
            model="test-model", system_prompt="Clean: {text}", cache_dir="/test/cache"
        )

        stage.setup()

        assert stage._model._llm is not None
        assert stage._model.cache_dir == "/test/cache"

    def test_setup_with_max_tokens(self):
        """Test setup method with max_tokens specified."""
        stage = LLMCleanupStage(
            model="test-model", system_prompt="Clean: {text}", max_tokens=500
        )

        stage.setup()

        assert stage._model._sampling_params is not None
        assert stage._model._sampling_params.max_tokens == 500

    def test_setup_max_tokens_fallback(self):
        """Test setup method falls back to max_model_len when max_tokens is None."""
        stage = LLMCleanupStage(
            model="test-model", system_prompt="Clean: {text}", max_model_len=16000, max_tokens=None
        )

        stage.setup()

        assert stage._model._sampling_params is not None
        assert stage._model._sampling_params.max_tokens == 16000

    def test_process_basic_cleanup(self):
        """Test process method for basic text cleanup."""
        stage = LLMCleanupStage(model="test-model", system_prompt="Clean this text: {text}")
        stage.setup()

        df = pd.DataFrame({"text": ["Original text here"]})
        batch = DocumentBatch(data=df, task_id="test", dataset_name="test")

        result = stage.process(batch)

        assert len(result.data) == 1
        assert "cleaned_text" in result.data.columns
        assert "text" in result.data.columns  # Original text preserved
        assert result.data["cleaned_text"].iloc[0].startswith("Cleaned:")

    def test_process_classification_mode(self):
        """Test process method in classification mode."""
        stage = LLMCleanupStage(
            model="test-model", system_prompt="Classify: {text}", classification=True
        )
        stage.setup()

        df = pd.DataFrame({"text": ["Some text to classify"]})
        batch = DocumentBatch(data=df, task_id="test", dataset_name="test")

        result = stage.process(batch)

        assert "label" in result.data.columns
        assert "text" not in result.data.columns  # Text removed in classification mode
        assert len(result.data) == 1

    def test_process_multiple_texts(self):
        """Test process method with multiple texts."""
        stage = LLMCleanupStage(model="test-model", system_prompt="Clean: {text}")
        stage.setup()

        df = pd.DataFrame({"text": ["First text", "Second text", "Third text"]})
        batch = DocumentBatch(data=df, task_id="test", dataset_name="test")

        result = stage.process(batch)

        assert len(result.data) == 3
        assert all("cleaned_text" in result.data.columns for _ in range(3))
        assert all(result.data["cleaned_text"].iloc[i].startswith("Cleaned:") for i in range(3))

    def test_process_preserves_metadata(self):
        """Test that process preserves original metadata fields."""
        stage = LLMCleanupStage(model="test-model", system_prompt="Clean: {text}")
        stage.setup()

        df = pd.DataFrame({
            "text": ["Some text"],
            "url": ["https://example.com"],
            "title": ["Test Title"],
            "metadata": ["extra info"],
        })
        batch = DocumentBatch(data=df, task_id="test", dataset_name="test")

        result = stage.process(batch)

        assert "url" in result.data.columns
        assert "title" in result.data.columns
        assert "metadata" in result.data.columns
        assert result.data["url"].iloc[0] == "https://example.com"

    def test_process_null_text(self):
        """Test process method with null/NaN text values."""
        stage = LLMCleanupStage(model="test-model", system_prompt="Clean: {text}")
        stage.setup()

        df = pd.DataFrame({"text": [None, pd.NA, "Valid text"]})
        batch = DocumentBatch(data=df, task_id="test", dataset_name="test")

        result = stage.process(batch)

        assert len(result.data) == 3
        # Null values should be converted to empty strings
        assert result.data["cleaned_text"].iloc[0] == "Cleaned output"
        assert result.data["cleaned_text"].iloc[1] == "Cleaned output"

    def test_process_filter_by_n_tokens(self):
        """Test process method with filter_by_n_tokens enabled."""
        stage = LLMCleanupStage(
            model="test-model",
            system_prompt="Clean: {text}",
            max_model_len=1000,
            filter_by_n_tokens=True,
        )
        stage.setup()

        # Create data with n_tokens field
        df = pd.DataFrame({
            "text": ["Short text", "Very long text that exceeds threshold"],
            "n_tokens": [100, 900],  # Second exceeds 80% of 1000 = 800
        })
        batch = DocumentBatch(data=df, task_id="test", dataset_name="test")

        result = stage.process(batch)

        # Should filter out chunks exceeding threshold
        assert len(result.data) == 1  # Only the short text should remain
        assert result.data["text"].iloc[0] == "Short text"

    def test_process_filter_by_n_tokens_all_filtered(self):
        """Test process method when all texts are filtered out."""
        stage = LLMCleanupStage(
            model="test-model",
            system_prompt="Clean: {text}",
            max_model_len=1000,
            filter_by_n_tokens=True,
        )
        stage.setup()

        df = pd.DataFrame({
            "text": ["Very long text 1", "Very long text 2"],
            "n_tokens": [900, 950],  # Both exceed threshold
        })
        batch = DocumentBatch(data=df, task_id="test", dataset_name="test")

        result = stage.process(batch)

        # Should return empty DataFrame
        assert len(result.data) == 0
        assert result.task_id == batch.task_id
        assert result.dataset_name == batch.dataset_name

    def test_process_sort_by_n_tokens(self):
        """Test that process sorts by n_tokens when available."""
        stage = LLMCleanupStage(model="test-model", system_prompt="Clean: {text}")
        stage.setup()

        df = pd.DataFrame({
            "text": ["Text 1", "Text 2", "Text 3"],
            "n_tokens": [300, 100, 200],
        })
        batch = DocumentBatch(data=df, task_id="test", dataset_name="test")

        result = stage.process(batch)

        # Should be sorted by n_tokens (ascending)
        n_tokens_order = result.data["n_tokens"].tolist()
        assert n_tokens_order == [100, 200, 300]

    def test_process_prompt_formatting(self):
        """Test that prompts are correctly formatted with text."""
        stage = LLMCleanupStage(model="test-model", system_prompt="Process this: {text}")
        stage.setup()

        df = pd.DataFrame({"text": ["Input text"]})
        batch = DocumentBatch(data=df, task_id="test", dataset_name="test")

        # Mock the generate method to capture prompts
        original_generate = stage._model.generate
        captured_prompts = []

        def capture_prompts(prompts: list[str]) -> list[str]:
            captured_prompts.extend(prompts)
            return original_generate(prompts)

        stage._model.generate = capture_prompts

        stage.process(batch)

        # Verify prompt was formatted correctly
        assert len(captured_prompts) == 1
        assert "Process this:" in captured_prompts[0]
        assert "Input text" in captured_prompts[0]

    def test_process_error_handling(self):
        """Test error handling in process method."""
        stage = LLMCleanupStage(model="test-model", system_prompt="Clean: {text}")
        stage.setup()

        # Make generate raise an exception
        error_msg = "LLM generation failed"

        def failing_generate(prompts: list[str]) -> None:  # noqa: ARG001
            raise RuntimeError(error_msg)

        stage._model.generate = failing_generate

        df = pd.DataFrame({"text": ["Some text"]})
        batch = DocumentBatch(data=df, task_id="test", dataset_name="test")

        # The error is raised directly from the mocked generate method
        # Since we're mocking stage._model.generate directly, it bypasses VLLMModel.generate()
        # which would wrap the error. So we just check for the original error message.
        with pytest.raises(RuntimeError, match="LLM generation failed"):
            stage.process(batch)

    def test_process_not_initialized(self):
        """Test process method when LLM is not initialized."""
        stage = LLMCleanupStage(model="test-model", system_prompt="Clean: {text}")
        # Don't call setup()

        df = pd.DataFrame({"text": ["Some text"]})
        batch = DocumentBatch(data=df, task_id="test", dataset_name="test")

        with pytest.raises(RuntimeError, match="Model not initialized"):
            stage.process(batch)

    def test_process_empty_output(self):
        """Test process method when LLM returns empty output."""
        stage = LLMCleanupStage(model="test-model", system_prompt="Clean: {text}")
        stage.setup()

        # Mock generate to return empty outputs (as strings, not MockLLMOutput objects)
        def empty_generate(prompts: list[str]) -> list[str]:  # noqa: ARG001
            return [""]

        stage._model.generate = empty_generate

        df = pd.DataFrame({"text": ["Some text"]})
        batch = DocumentBatch(data=df, task_id="test", dataset_name="test")

        result = stage.process(batch)

        assert len(result.data) == 1
        assert result.data["cleaned_text"].iloc[0] == ""
