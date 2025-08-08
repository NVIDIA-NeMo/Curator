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

import json
import tempfile
from collections.abc import Iterable
from unittest.mock import patch

import pandas as pd
import pytest

from ray_curator.stages.reasoning.correctness_filter import (
    LLMBasedCorrectnessFilter,
    LLMBasedGrader,
)
from ray_curator.stages.reasoning.difficulty_filter import (
    LLMBasedDifficultyFilter,
    LLMBasedDifficultyFilterFunction,
    ReasoningLengthDifficultyFilter,
)
from ray_curator.stages.reasoning.diversity_filter import (
    DiversitySampler,
    LLMBasedDomainClassifier,
)
from ray_curator.stages.reasoning.prompts import (
    DEFAULT_DOMAIN_CLASSIFICATION_PROMPT_TEMPLATE,
    DEFAULT_GRADING_PROMPT_TEMPLATE,
    DEFAULT_REASONING_TRACE_PROMPT_TEMPLATE,
)
from ray_curator.stages.reasoning.reasoning_traces_synthetic import (
    ReasoningTracesSyntheticStage,
)
from ray_curator.stages.services.conversation_formatter import ConversationFormatter
from ray_curator.stages.services.model_client import GenerationConfig, LLMClient
from ray_curator.tasks import DocumentBatch


class MockLLMClient(LLMClient):
    """Mock LLM client for testing."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or ["Mock response"]
        self.current_response = 0
        self.query_calls = []

    def setup(self) -> None:
        pass

    def query_model(
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: ConversationFormatter | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> list[str]:
        """Mock query_model method."""
        self.query_calls.append({
            "messages": messages,
            "model": model,
            "conversation_formatter": conversation_formatter,
            "generation_config": generation_config,
        })

        if self.current_response < len(self.responses):
            response = self.responses[self.current_response]
            self.current_response += 1
            return [response]
        return ["Mock response"]


def create_test_batch(data: dict) -> DocumentBatch:
    """Create a test DocumentBatch."""
    df = pd.DataFrame(data)
    return DocumentBatch(
        data=df,
        dataset_name="test_dataset",
        task_id="test_task",
    )


def create_domains_file() -> str:
    """Create a temporary domains file for testing."""
    domains_data = [
        {"code": "01", "prompt": "Mathematical problems and equations"},
        {"code": "02", "prompt": "Scientific concepts and phenomena"},
        {"code": "03", "prompt": "Historical events and figures"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(domains_data, f)
        return f.name


@pytest.fixture
def sample_reasoning_data() -> DocumentBatch:
    """Create sample reasoning data for testing."""
    data = {
        "question": [
            "What is 2+2?",
            "What is the capital of France?",
            "What is photosynthesis?",
        ],
        "answer": [
            "4",
            "Paris",
            "The process by which plants make food using sunlight",
        ],
    }
    return create_test_batch(data)


@pytest.fixture
def sample_grading_data() -> DocumentBatch:
    """Create sample grading data for testing."""
    data = {
        "question": [
            "What is 2+2?",
            "What is the capital of France?",
            "What is photosynthesis?",
        ],
        "answer": [
            "4",
            "Paris",
            "The process by which plants make food using sunlight",
        ],
        "reasoning_trace_attempt": [
            "Let me think: 2+2 = 4",
            "The capital of France is Paris",
            "Photosynthesis is how plants make food",
        ],
    }
    return create_test_batch(data)


@pytest.fixture
def sample_difficulty_data() -> DocumentBatch:
    """Create sample difficulty data for testing."""
    data = {
        "question": [
            "What is 2+2?",
            "What is the capital of France?",
            "What is photosynthesis?",
        ],
        "llm_difficulty_1_correctness": ["Yes", "No", "Yes"],
        "llm_difficulty_2_correctness": ["Yes", "Yes", "No"],
    }
    return create_test_batch(data)


@pytest.fixture
def sample_diversity_data() -> DocumentBatch:
    """Create sample diversity data for testing."""
    data = {
        "question": [
            "What is 2+2?",
            "What is the capital of France?",
            "What is photosynthesis?",
            "What is 3+3?",
            "What is the capital of Germany?",
            "What is mitosis?",
        ],
        "domain": ["01", "02", "03", "01", "02", "03"],
    }
    return create_test_batch(data)


class TestReasoningTracesSyntheticStage:
    """Tests for ReasoningTracesSyntheticStage"""

    def test_stage_initialization(self):
        """Test stage initialization"""
        mock_client = MockLLMClient()
        stage = ReasoningTracesSyntheticStage(
            prompt="Test prompt",
            client=mock_client,
            model_name="test-model",
            input_problem_field="question",
            output_field="reasoning_trace",
        )
        assert stage.prompt == "Test prompt"
        assert stage.client == mock_client
        assert stage.model_name == "test-model"
        assert stage.input_problem_field == "question"
        assert stage.output_field == "reasoning_trace"

    def test_stage_properties(self):
        """Test stage properties"""
        mock_client = MockLLMClient()
        stage = ReasoningTracesSyntheticStage(
            prompt="Test prompt",
            client=mock_client,
            model_name="test-model",
            input_problem_field="question",
            output_field="reasoning_trace",
        )

        assert stage.name == "ReasoningTracesSyntheticStage"
        assert stage.inputs() == ([], [])
        assert stage.outputs() == (["data"], ["reasoning_trace"])

    def test_setup(self):
        """Test setup method"""
        mock_client = MockLLMClient()
        stage = ReasoningTracesSyntheticStage(
            prompt="Test prompt",
            client=mock_client,
            model_name="test-model",
            input_problem_field="question",
            output_field="reasoning_trace",
        )
        # Setup should not raise any exceptions
        stage.setup()

    def test_process(self, sample_reasoning_data: DocumentBatch):
        """Test process method"""
        mock_client = MockLLMClient(responses=["Reasoning trace 1", "Reasoning trace 2", "Reasoning trace 3"])
        stage = ReasoningTracesSyntheticStage(
            prompt="Test prompt: {problem}",
            client=mock_client,
            model_name="test-model",
            input_problem_field="question",
            output_field="reasoning_trace",
        )

        result = stage.process(sample_reasoning_data)

        # Check that result has correct structure
        assert isinstance(result, DocumentBatch)
        assert len(result.data) == 3
        assert "reasoning_trace" in result.data.columns
        assert result.data["reasoning_trace"].tolist() == ["Reasoning trace 1", "Reasoning trace 2", "Reasoning trace 3"]

        # Check that LLM was called correctly
        assert len(mock_client.query_calls) == 3
        assert mock_client.query_calls[0]["model"] == "test-model"


class TestLLMBasedGrader:
    """Tests for LLMBasedGrader"""

    def test_stage_initialization(self):
        """Test stage initialization"""
        mock_client = MockLLMClient()
        stage = LLMBasedGrader(
            prompt="Test prompt",
            client=mock_client,
            model_name="test-model",
            input_problem_field="question",
            input_attempt_field="reasoning_trace_attempt",
            input_solution_field="answer",
            output_field="correctness",
        )
        assert stage.prompt == "Test prompt"
        assert stage.client == mock_client
        assert stage.model_name == "test-model"
        assert stage.input_problem_field == "question"
        assert stage.input_attempt_field == "reasoning_trace_attempt"
        assert stage.input_solution_field == "answer"
        assert stage.output_field == "correctness"

    def test_stage_properties(self):
        """Test stage properties"""
        mock_client = MockLLMClient()
        stage = LLMBasedGrader(
            prompt="Test prompt",
            client=mock_client,
            model_name="test-model",
            input_problem_field="question",
            input_attempt_field="reasoning_trace_attempt",
            input_solution_field="answer",
            output_field="correctness",
        )

        assert stage.name == "LLMBasedCorrectnessFilter"
        assert stage.inputs() == ([], [])
        assert stage.outputs() == (["data"], ["correctness"])

    def test_process(self, sample_grading_data: DocumentBatch):
        """Test process method"""
        mock_client = MockLLMClient(responses=["Analysis...\nYes", "Analysis...\nNo", "Analysis...\nYes"])
        stage = LLMBasedGrader(
            prompt="Test prompt: {problem} {attempt} {solution}",
            client=mock_client,
            model_name="test-model",
            input_problem_field="question",
            input_attempt_field="reasoning_trace_attempt",
            input_solution_field="answer",
            output_field="correctness",
        )

        result = stage.process(sample_grading_data)

        # Check that result has correct structure
        assert isinstance(result, DocumentBatch)
        assert len(result.data) == 3
        assert "correctness" in result.data.columns
        assert result.data["correctness"].tolist() == ["Yes", "No", "Yes"]

        # Check that LLM was called correctly
        assert len(mock_client.query_calls) == 3
        assert mock_client.query_calls[0]["model"] == "test-model"


class TestLLMBasedCorrectnessFilter:
    """Tests for LLMBasedCorrectnessFilter"""

    def test_initialization(self):
        """Test filter initialization"""
        filter_obj = LLMBasedCorrectnessFilter()
        assert filter_obj._name == "llm_based_correctness_filter"

    def test_score_document(self):
        """Test score_document method"""
        filter_obj = LLMBasedCorrectnessFilter()
        assert filter_obj.score_document("Yes") == 1.0
        assert filter_obj.score_document("No") == 0.0
        assert filter_obj.score_document("Maybe") == 0.0

    def test_keep_document(self):
        """Test keep_document method"""
        filter_obj = LLMBasedCorrectnessFilter()
        assert filter_obj.keep_document(1.0) is True
        assert filter_obj.keep_document(0.0) is False
        assert filter_obj.keep_document(0.5) is False


class TestReasoningLengthDifficultyFilter:
    """Tests for ReasoningLengthDifficultyFilter"""

    def test_initialization(self):
        """Test filter initialization"""
        filter_obj = ReasoningLengthDifficultyFilter(min_length=100)
        assert filter_obj._min_length == 100
        assert filter_obj._name == "reasoning_length_difficulty_filter"

    def test_score_document(self):
        """Test score_document method"""
        filter_obj = ReasoningLengthDifficultyFilter(min_length=5)

        # Test with short text (should score 0.0)
        short_text = "Short text"
        assert filter_obj.score_document(short_text) == 0.0

        # Test with long text (should score 1.0)
        long_text = "This is a much longer text that should definitely exceed the minimum length requirement"
        assert filter_obj.score_document(long_text) == 1.0

    def test_keep_document(self):
        """Test keep_document method"""
        filter_obj = ReasoningLengthDifficultyFilter(min_length=100)
        assert filter_obj.keep_document(1.0) is True
        assert filter_obj.keep_document(0.0) is False


class TestLLMBasedDifficultyFilterFunction:
    """Tests for LLMBasedDifficultyFilterFunction"""

    def test_initialization(self):
        """Test filter initialization"""
        filter_obj = LLMBasedDifficultyFilterFunction(
            llm_correctness_fields=["field1", "field2"]
        )
        assert filter_obj._name == "llm_based_difficulty_filter"
        assert filter_obj.llm_correctness_fields == ["field1", "field2"]

    def test_score_document(self):
        """Test score_document method"""
        filter_obj = LLMBasedDifficultyFilterFunction(
            llm_correctness_fields=["field1", "field2"]
        )

        # Test when all fields are "Yes" (should be difficult - score 0.0)
        sample_easy = {"field1": "Yes", "field2": "Yes"}
        assert filter_obj.score_document(sample_easy) == 0.0

        # Test when not all fields are "Yes" (should be easy - score 1.0)
        sample_difficult = {"field1": "Yes", "field2": "No"}
        assert filter_obj.score_document(sample_difficult) == 1.0

    def test_keep_document(self):
        """Test keep_document method"""
        filter_obj = LLMBasedDifficultyFilterFunction(
            llm_correctness_fields=["field1", "field2"]
        )
        assert filter_obj.keep_document(1.0) is True
        assert filter_obj.keep_document(0.0) is False


class TestLLMBasedDifficultyFilter:
    """Tests for LLMBasedDifficultyFilter"""

    def test_initialization(self):
        """Test filter initialization"""
        filter_func = LLMBasedDifficultyFilterFunction(
            llm_correctness_fields=["field1", "field2"]
        )
        filter_obj = LLMBasedDifficultyFilter(
            filter_obj=filter_func,
            llm_correctness_fields=["field1", "field2"],
        )
        assert filter_obj.llm_correctness_fields == ["field1", "field2"]

    def test_inputs(self):
        """Test inputs method"""
        filter_func = LLMBasedDifficultyFilterFunction(
            llm_correctness_fields=["field1", "field2"]
        )
        filter_obj = LLMBasedDifficultyFilter(
            filter_obj=filter_func,
            llm_correctness_fields=["field1", "field2"],
        )
        assert filter_obj.inputs() == (["data"], ["field1", "field2"])

    def test_compute_filter_mask(self, sample_difficulty_data: DocumentBatch):
        """Test compute_filter_mask method"""
        filter_func = LLMBasedDifficultyFilterFunction(
            llm_correctness_fields=["llm_difficulty_1_correctness", "llm_difficulty_2_correctness"]
        )
        filter_obj = LLMBasedDifficultyFilter(
            filter_obj=filter_func,
            llm_correctness_fields=["llm_difficulty_1_correctness", "llm_difficulty_2_correctness"],
        )

        df = sample_difficulty_data.to_pandas()
        mask = filter_obj.compute_filter_mask(df)

        # Check that mask is a boolean Series
        assert isinstance(mask, pd.Series)
        assert mask.dtype == bool
        assert len(mask) == len(df)

        # Check the expected results based on the sample data
        # Row 0: Yes, Yes -> easy -> keep (True)
        # Row 1: No, Yes -> difficult -> keep (True)
        # Row 2: Yes, No -> difficult -> keep (True)
        expected_mask = [False, True, True]  # Inverted because easy problems get score 0.0
        assert mask.tolist() == expected_mask


class TestLLMBasedDomainClassifier:
    """Tests for LLMBasedDomainClassifier"""

    def test_stage_initialization(self):
        """Test stage initialization"""
        domains_file = create_domains_file()
        mock_client = MockLLMClient()

        try:
            stage = LLMBasedDomainClassifier(
                prompt="Test prompt",
                client=mock_client,
                model_name="test-model",
                domains_file_path=domains_file,
                input_problem_field="question",
                output_field="domain",
            )
            assert stage.prompt == "Test prompt"
            assert stage.client == mock_client
            assert stage.model_name == "test-model"
            assert stage.input_problem_field == "question"
            assert stage.output_field == "domain"
            assert isinstance(stage.domains, pd.DataFrame)
            assert len(stage.domains) == 3
            assert "code" in stage.domains.columns
            assert "prompt" in stage.domains.columns
        finally:
            import os
            os.unlink(domains_file)

    def test_stage_properties(self):
        """Test stage properties"""
        domains_file = create_domains_file()
        mock_client = MockLLMClient()

        try:
            stage = LLMBasedDomainClassifier(
                prompt="Test prompt",
                client=mock_client,
                model_name="test-model",
                domains_file_path=domains_file,
                input_problem_field="question",
                output_field="domain",
            )

            assert stage.name == "LLMBasedDomainClassifier"
            assert stage.inputs() == ([], [])
            assert stage.outputs() == (["data"], ["domain"])
        finally:
            import os
            os.unlink(domains_file)

    def test_process(self, sample_reasoning_data: DocumentBatch):
        """Test process method"""
        domains_file = create_domains_file()
        mock_client = MockLLMClient(responses=["Analysis...\n01", "Analysis...\n02", "Analysis...\n03"])

        try:
            stage = LLMBasedDomainClassifier(
                prompt=None,
                client=mock_client,
                model_name="test-model",
                domains_file_path=domains_file,
                input_problem_field="question",
                output_field="domain",
            )

            result = stage.process(sample_reasoning_data)

            # Check that result has correct structure
            assert isinstance(result, DocumentBatch)
            assert len(result.data) == 3
            assert "domain" in result.data.columns
            assert result.data["domain"].tolist() == ["01", "02", "03"]

            # Check that LLM was called correctly
            assert len(mock_client.query_calls) == 3
            assert mock_client.query_calls[0]["model"] == "test-model"
        finally:
            import os
            os.unlink(domains_file)


class TestDiversitySampler:
    """Tests for DiversitySampler"""

    def test_stage_initialization(self):
        """Test stage initialization"""
        stage = DiversitySampler(
            sampling_size=5,
            input_problem_field="question",
            input_domain_field="domain",
        )
        assert stage.sampling_size == 5
        assert stage.input_problem_field == "question"
        assert stage.input_domain_field == "domain"

    def test_stage_properties(self):
        """Test stage properties"""
        stage = DiversitySampler(
            sampling_size=5,
            input_problem_field="question",
            input_domain_field="domain",
        )

        assert stage.name == "DiversitySampler"
        assert stage.inputs() == ([], [])
        assert stage.outputs() == (["data"], [])

    def test_setup(self):
        """Test setup method"""
        stage = DiversitySampler(
            sampling_size=5,
            input_problem_field="question",
            input_domain_field="domain",
        )
        # Setup should not raise any exceptions
        stage.setup()

    @patch("numpy.random.choice")
    def test_sample_uniformly(self, mock_choice: object, sample_diversity_data: DocumentBatch):
        """Test uniform sampling method"""
        stage = DiversitySampler(
            sampling_size=4,
            input_problem_field="question",
            input_domain_field="domain",
        )

        # Create a counter to cycle through choices
        call_count = 0

        def mock_choice_func(*args, **_kwargs) -> object:
            nonlocal call_count
            call_count += 1

            # If it's selecting a domain (args[0] is a list of domain strings)
            if isinstance(args[0], list) and len(args[0]) > 0 and isinstance(args[0][0], str):
                # Cycle through domains
                domains = args[0]
                return domains[call_count % len(domains)]
            # If it's selecting an index (args[0] is pandas Index or numpy array)
            else:
                # Return the first available index
                return args[0][0]

        mock_choice.side_effect = mock_choice_func

        df = sample_diversity_data.data
        result = stage._sample_uniformly(df)

        # Should return 4 samples
        assert len(result) == 4

        # Should have samples from different domains
        domains = result["domain"].unique()
        assert len(domains) >= 1  # At least 1 domain

    def test_process(self, sample_diversity_data: DocumentBatch):
        """Test process method"""
        stage = DiversitySampler(
            sampling_size=3,
            input_problem_field="question",
            input_domain_field="domain",
        )

        result = stage.process(sample_diversity_data)

        # Check that result has correct structure
        assert isinstance(result, DocumentBatch)
        assert len(result.data) == 3

        # Check that all required columns are present
        assert "question" in result.data.columns
        assert "domain" in result.data.columns

    def test_process_sampling_size_larger_than_data(self, sample_diversity_data: DocumentBatch):
        """Test process method when sampling size is larger than available data"""
        stage = DiversitySampler(
            sampling_size=10,  # Larger than the 6 samples in test data
            input_problem_field="question",
            input_domain_field="domain",
        )

        result = stage.process(sample_diversity_data)

        # Should return all available data
        assert len(result.data) == 6

    def test_process_with_single_domain(self):
        """Test process method with data from single domain"""
        single_domain_data = create_test_batch({
            "question": ["Q1", "Q2", "Q3"],
            "domain": ["01", "01", "01"],
        })

        stage = DiversitySampler(
            sampling_size=2,
            input_problem_field="question",
            input_domain_field="domain",
        )

        result = stage.process(single_domain_data)

        # Should return 2 samples from the single domain
        assert len(result.data) == 2
        assert all(result.data["domain"] == "01")


class TestPromptTemplates:
    """Tests for prompt templates"""

    def test_default_grading_prompt_template(self):
        """Test that default grading prompt template has required placeholders"""
        assert "{problem}" in DEFAULT_GRADING_PROMPT_TEMPLATE
        assert "{attempt}" in DEFAULT_GRADING_PROMPT_TEMPLATE
        assert "{solution}" in DEFAULT_GRADING_PROMPT_TEMPLATE

    def test_default_reasoning_trace_prompt_template(self):
        """Test that default reasoning trace prompt template has required placeholders"""
        assert "{problem}" in DEFAULT_REASONING_TRACE_PROMPT_TEMPLATE

    def test_default_domain_classification_prompt_template(self):
        """Test that default domain classification prompt template has required placeholders"""
        assert isinstance(DEFAULT_DOMAIN_CLASSIFICATION_PROMPT_TEMPLATE, str)
        assert len(DEFAULT_DOMAIN_CLASSIFICATION_PROMPT_TEMPLATE) > 0
        assert "{question}" in DEFAULT_DOMAIN_CLASSIFICATION_PROMPT_TEMPLATE
        assert "{domains_prompt}" in DEFAULT_DOMAIN_CLASSIFICATION_PROMPT_TEMPLATE

    def test_prompt_formatting(self):
        """Test that prompts can be formatted correctly"""
        # Test grading prompt
        grading_prompt = DEFAULT_GRADING_PROMPT_TEMPLATE.format(
            problem="What is 2+2?",
            attempt="4",
            solution="2+2=4"
        )
        assert "What is 2+2?" in grading_prompt
        assert "4" in grading_prompt
        assert "2+2=4" in grading_prompt

        # Test reasoning trace prompt
        reasoning_prompt = DEFAULT_REASONING_TRACE_PROMPT_TEMPLATE.format(
            problem="What is 2+2?"
        )
        assert "What is 2+2?" in reasoning_prompt

        # Test domain classification prompt
        domain_prompt = DEFAULT_DOMAIN_CLASSIFICATION_PROMPT_TEMPLATE.format(
            question="What is 2+2?",
            domains_prompt="01: Mathematics\n02: Science"
        )
        assert "What is 2+2?" in domain_prompt
        assert "Mathematics" in domain_prompt
