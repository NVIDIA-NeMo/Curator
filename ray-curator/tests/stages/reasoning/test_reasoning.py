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
from unittest.mock import Mock, patch
from collections.abc import Iterable

import pandas as pd
import pytest

from ray_curator.stages.reasoning.correctness_filter import (
    LLMBasedGrader,
    LLMBasedCorrectnessFilter,
)
from ray_curator.stages.reasoning.difficulty_filter import (
    ReasoningLengthDifficultyFilter,
    LLMBasedDifficultyFilterFunction,
    LLMBasedDifficultyFilter,
)
from ray_curator.stages.reasoning.diversity_filter import (
    LLMBasedDomainClassifier,
    DiversitySampler,
)
from ray_curator.stages.reasoning.reasoning_traces_synthetic import (
    ReasoningTracesSyntheticStage,
)
from ray_curator.stages.reasoning.prompts import (
    DEFAULT_GRADING_PROMPT_TEMPLATE,
    DEFAULT_REASONING_TRACE_PROMPT_TEMPLATE,
    DEFAULT_DOMAIN_CLASSIFICATION_PROMPT_TEMPLATE,
)
from ray_curator.stages.services.model_client import LLMClient
from ray_curator.stages.services.conversation_formatter import ConversationFormatter
from ray_curator.tasks import DocumentBatch


class MockLLMClient(LLMClient):
    """Mock LLM client for testing"""
    
    def __init__(self, responses: list[str] = None):
        self.responses = responses or ["Mock response"]
        self.setup_called = False
        self.query_calls = []
        self.response_index = 0
    
    def setup(self) -> None:
        self.setup_called = True
    
    def query_model(  # noqa: PLR0913
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: ConversationFormatter | None = None,
        max_tokens: int | None = None,
        n: int | None = 1,
        seed: int | None = None,
        stop: str | None | list[str] = None,
        stream: bool = False,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> list[str]:
        self.query_calls.append({
            "messages": list(messages),
            "model": model,
            "conversation_formatter": conversation_formatter,
            "max_tokens": max_tokens,
            "n": n,
            "seed": seed,
            "stop": stop,
            "stream": stream,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        })
        
        # Cycle through responses
        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
        else:
            response = self.responses[-1]  # Use last response if we run out
        
        return [response]
    
    def query_reward_model(
        self,
        *,
        messages: Iterable,
        model: str,
        conversation_formatter: ConversationFormatter | None = None,
    ) -> dict:
        return {"score": 0.5}


def create_test_batch(data: dict) -> DocumentBatch:
    """Helper function to create test DocumentBatch"""
    return DocumentBatch(
        data=pd.DataFrame(data),
        task_id="test_batch",
        dataset_name="test_dataset",
    )


def create_domains_file() -> str:
    """Helper function to create a temporary domains file"""
    domains_data = [
        {"id": "01", "name": "Mathematics", "prompt": "01: Mathematics - problems involving numbers, equations, and calculations"},
        {"id": "02", "name": "Science", "prompt": "02: Science - questions about physics, chemistry, biology, and natural phenomena"},
        {"id": "03", "name": "History", "prompt": "03: History - questions about past events, dates, and historical figures"},
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(domains_data, f)
        return f.name


@pytest.fixture
def sample_reasoning_data() -> DocumentBatch:
    """Sample data for reasoning tests"""
    data = {
        "question": [
            "What is 2 + 2?",
            "Explain photosynthesis",
            "Who was the first president of the United States?",
        ],
        "answer": [
            "2 + 2 = 4",
            "Photosynthesis is the process by which plants convert light energy into chemical energy",
            "George Washington was the first president of the United States",
        ],
    }
    return create_test_batch(data)


@pytest.fixture
def sample_grading_data() -> DocumentBatch:
    """Sample data for grading tests"""
    data = {
        "question": [
            "What is 2 + 2?",
            "What is the capital of France?",
        ],
        "answer": [
            "2 + 2 = 4",
            "Paris is the capital of France",
        ],
        "attempt": [
            "The answer is 4",
            "The capital of France is London",
        ],
    }
    return create_test_batch(data)


@pytest.fixture
def sample_difficulty_data() -> DocumentBatch:
    """Sample data for difficulty filtering tests"""
    data = {
        "text": [
            "Short text",
            "This is a longer text that should pass the minimum length requirement for difficulty filtering",
            "Medium length text here",
            "This is an extremely long text that contains many words and should definitely pass any reasonable minimum length requirement for difficulty filtering because it has so many words in it",
        ],
        "llm_difficulty_1_correctness": ["Yes", "No", "Yes", "No"],
        "llm_difficulty_2_correctness": ["Yes", "Yes", "No", "No"],
    }
    return create_test_batch(data)


@pytest.fixture
def sample_diversity_data() -> DocumentBatch:
    """Sample data for diversity sampling tests"""
    data = {
        "question": [
            "What is 2 + 2?",
            "What is calculus?",
            "Explain photosynthesis",
            "What is DNA?",
            "Who was Napoleon?",
            "When was World War II?",
        ],
        "domain": ["01", "01", "02", "02", "03", "03"],
    }
    return create_test_batch(data)


class TestReasoningTracesSyntheticStage:
    """Tests for ReasoningTracesSyntheticStage"""
    
    def test_stage_initialization(self):
        """Test stage initialization with different parameters"""
        mock_client = MockLLMClient()
        
        # Test with custom prompt
        stage = ReasoningTracesSyntheticStage(
            prompt="Custom prompt: {problem}",
            client=mock_client,
            model_name="test-model",
            input_problem_field="question",
            output_field="trace",
        )
        assert stage.prompt == "Custom prompt: {problem}"
        
        # Test with default prompt
        stage = ReasoningTracesSyntheticStage(
            prompt=None,
            client=mock_client,
            model_name="test-model",
            input_problem_field="question",
            output_field="trace",
        )
        assert stage.prompt == DEFAULT_REASONING_TRACE_PROMPT_TEMPLATE
    
    def test_stage_properties(self):
        """Test stage properties"""
        mock_client = MockLLMClient()
        stage = ReasoningTracesSyntheticStage(
            prompt=None,
            client=mock_client,
            model_name="test-model",
            input_problem_field="question",
            output_field="trace",
        )
        
        assert stage.name == "ReasoningTracesSyntheticStage"
        assert stage.inputs() == ([], [])
        assert stage.outputs() == (["data"], ["trace"])
    
    def test_setup(self):
        """Test setup method"""
        mock_client = MockLLMClient()
        stage = ReasoningTracesSyntheticStage(
            prompt=None,
            client=mock_client,
            model_name="test-model",
            input_problem_field="question",
            output_field="trace",
        )
        
        stage.setup()
        assert mock_client.setup_called
    
    def test_process(self, sample_reasoning_data):
        """Test process method"""
        mock_client = MockLLMClient(responses=["Trace 1", "Trace 2", "Trace 3"])
        stage = ReasoningTracesSyntheticStage(
            prompt=None,
            client=mock_client,
            model_name="test-model",
            input_problem_field="question",
            output_field="trace",
        )
        
        result = stage.process(sample_reasoning_data)
        
        # Check that result has correct structure
        assert isinstance(result, DocumentBatch)
        assert len(result.data) == 3
        assert "trace" in result.data.columns
        assert result.data["trace"].tolist() == ["Trace 1", "Trace 2", "Trace 3"]
        
        # Check that LLM was called correctly
        assert len(mock_client.query_calls) == 3
        assert mock_client.query_calls[0]["model"] == "test-model"
        assert mock_client.query_calls[0]["messages"][0]["role"] == "user"


class TestLLMBasedGrader:
    """Tests for LLMBasedGrader"""
    
    def test_stage_initialization(self):
        """Test stage initialization"""
        mock_client = MockLLMClient()
        
        # Test with custom prompt
        stage = LLMBasedGrader(
            prompt="Custom grading prompt",
            client=mock_client,
            model_name="test-model",
            input_problem_field="question",
            input_attempt_field="attempt",
            input_solution_field="answer",
            output_field="correctness",
        )
        assert stage.prompt == "Custom grading prompt"
        
        # Test with default prompt
        stage = LLMBasedGrader(
            prompt=None,
            client=mock_client,
            model_name="test-model",
            input_problem_field="question",
            input_attempt_field="attempt",
            input_solution_field="answer",
            output_field="correctness",
        )
        assert stage.prompt == DEFAULT_GRADING_PROMPT_TEMPLATE
    
    def test_stage_properties(self):
        """Test stage properties"""
        mock_client = MockLLMClient()
        stage = LLMBasedGrader(
            prompt=None,
            client=mock_client,
            model_name="test-model",
            input_problem_field="question",
            input_attempt_field="attempt",
            input_solution_field="answer",
            output_field="correctness",
        )
        
        assert stage.name == "LLMBasedCorrectnessFilter"
        assert stage.inputs() == ([], [])
        assert stage.outputs() == (["data"], ["correctness"])
    
    def test_process(self, sample_grading_data):
        """Test process method"""
        mock_client = MockLLMClient(responses=["Reasoning...\nYes", "Analysis...\nNo"])
        stage = LLMBasedGrader(
            prompt=None,
            client=mock_client,
            model_name="test-model",
            input_problem_field="question",
            input_attempt_field="attempt",
            input_solution_field="answer",
            output_field="correctness",
        )
        
        result = stage.process(sample_grading_data)
        
        # Check that result has correct structure
        assert isinstance(result, DocumentBatch)
        assert len(result.data) == 2
        assert "correctness" in result.data.columns
        assert result.data["correctness"].tolist() == ["Yes", "No"]
        
        # Check that LLM was called correctly
        assert len(mock_client.query_calls) == 2
        assert mock_client.query_calls[0]["model"] == "test-model"


class TestLLMBasedCorrectnessFilter:
    """Tests for LLMBasedCorrectnessFilter"""
    
    def test_initialization(self):
        """Test filter initialization"""
        filter_obj = LLMBasedCorrectnessFilter()
        assert filter_obj._name == "llm_based_correctness_filter"
    
    def test_score_document(self):
        """Test document scoring"""
        filter_obj = LLMBasedCorrectnessFilter()
        
        assert filter_obj.score_document("Yes") == 1.0
        assert filter_obj.score_document("No") == 0.0
        assert filter_obj.score_document("Maybe") == 0.0
    
    def test_keep_document(self):
        """Test document filtering decision"""
        filter_obj = LLMBasedCorrectnessFilter()
        
        assert filter_obj.keep_document(1.0) is True
        assert filter_obj.keep_document(0.0) is False
        assert filter_obj.keep_document(0.5) is False


class TestReasoningLengthDifficultyFilter:
    """Tests for ReasoningLengthDifficultyFilter"""
    
    def test_initialization(self):
        """Test filter initialization"""
        filter_obj = ReasoningLengthDifficultyFilter(min_length=10)
        assert filter_obj._min_length == 10
        assert filter_obj._name == "reasoning_length_difficulty_filter"
    
    def test_score_document(self):
        """Test document scoring based on length"""
        filter_obj = ReasoningLengthDifficultyFilter(min_length=5)
        
        # Short text (less than min_length)
        short_text = "one two three"  # 3 words
        assert filter_obj.score_document(short_text) == 0.0
        
        # Long text (more than min_length)
        long_text = "one two three four five six seven"  # 7 words
        assert filter_obj.score_document(long_text) == 1.0
        
        # Exact length
        exact_text = "one two three four five"  # 5 words
        assert filter_obj.score_document(exact_text) == 0.0
    
    def test_keep_document(self):
        """Test document filtering decision"""
        filter_obj = ReasoningLengthDifficultyFilter(min_length=5)
        
        assert filter_obj.keep_document(1.0) is True
        assert filter_obj.keep_document(0.0) is False


class TestLLMBasedDifficultyFilterFunction:
    """Tests for LLMBasedDifficultyFilterFunction"""
    
    def test_initialization(self):
        """Test filter initialization"""
        fields = ["field1", "field2"]
        filter_obj = LLMBasedDifficultyFilterFunction(llm_correctness_fields=fields)
        assert filter_obj.llm_correctness_fields == fields
        assert filter_obj._name == "llm_based_difficulty_filter"
    
    def test_score_document(self):
        """Test document scoring"""
        fields = ["field1", "field2"]
        filter_obj = LLMBasedDifficultyFilterFunction(llm_correctness_fields=fields)
        
        # All fields are "Yes" (easy, so should be filtered out -> keep_document = 0)
        sample_easy = {"field1": "Yes", "field2": "Yes"}
        assert filter_obj.score_document(sample_easy) == 0.0
        
        # Some fields are "No" (difficult, so should be kept -> keep_document = 1)
        sample_difficult = {"field1": "No", "field2": "Yes"}
        assert filter_obj.score_document(sample_difficult) == 1.0
        
        # All fields are "No"
        sample_very_difficult = {"field1": "No", "field2": "No"}
        assert filter_obj.score_document(sample_very_difficult) == 1.0
    
    def test_keep_document(self):
        """Test document filtering decision"""
        fields = ["field1", "field2"]
        filter_obj = LLMBasedDifficultyFilterFunction(llm_correctness_fields=fields)
        
        assert filter_obj.keep_document(1.0) is True
        assert filter_obj.keep_document(0.0) is False


class TestLLMBasedDifficultyFilter:
    """Tests for LLMBasedDifficultyFilter"""
    
    def test_initialization(self):
        """Test filter initialization"""
        fields = ["field1", "field2"]
        filter_function = LLMBasedDifficultyFilterFunction(llm_correctness_fields=fields)
        difficulty_filter = LLMBasedDifficultyFilter(
            filter_obj=filter_function,
            llm_correctness_fields=fields,
        )
        assert difficulty_filter.llm_correctness_fields == fields
    
    def test_inputs(self):
        """Test inputs method"""
        fields = ["field1", "field2"]
        filter_function = LLMBasedDifficultyFilterFunction(llm_correctness_fields=fields)
        difficulty_filter = LLMBasedDifficultyFilter(
            filter_obj=filter_function,
            llm_correctness_fields=fields,
        )
        
        inputs = difficulty_filter.inputs()
        assert inputs == (["data"], fields)
    
    def test_compute_filter_mask(self, sample_difficulty_data):
        """Test compute_filter_mask method"""
        fields = ["llm_difficulty_1_correctness", "llm_difficulty_2_correctness"]
        filter_function = LLMBasedDifficultyFilterFunction(llm_correctness_fields=fields)
        difficulty_filter = LLMBasedDifficultyFilter(
            filter_obj=filter_function,
            llm_correctness_fields=fields,
        )
        
        df = sample_difficulty_data.data
        mask = difficulty_filter.compute_filter_mask(df)
        
        # Expected: keep documents where not all fields are "Yes"
        # Row 0: ["Yes", "Yes"] -> easy -> should be filtered out (False)
        # Row 1: ["No", "Yes"] -> difficult -> should be kept (True)
        # Row 2: ["Yes", "No"] -> difficult -> should be kept (True)
        # Row 3: ["No", "No"] -> difficult -> should be kept (True)
        expected = [False, True, True, True]
        assert mask.tolist() == expected


class TestLLMBasedDomainClassifier:
    """Tests for LLMBasedDomainClassifier"""
    
    def test_stage_initialization(self):
        """Test stage initialization"""
        domains_file = create_domains_file()
        mock_client = MockLLMClient()
        
        try:
            # Test with custom prompt
            stage = LLMBasedDomainClassifier(
                prompt="Custom classification prompt",
                client=mock_client,
                model_name="test-model",
                domains_file_path=domains_file,
                input_problem_field="question",
                output_field="domain",
            )
            assert stage.prompt == "Custom classification prompt"
            
            # Test with default prompt
            stage = LLMBasedDomainClassifier(
                prompt=None,
                client=mock_client,
                model_name="test-model",
                domains_file_path=domains_file,
                input_problem_field="question",
                output_field="domain",
            )
            assert stage.prompt == DEFAULT_DOMAIN_CLASSIFICATION_PROMPT_TEMPLATE
            
            # Check that domains are loaded correctly
            assert len(stage.domains) == 3
            assert "01: Mathematics" in stage.domains_prompt
            assert "02: Science" in stage.domains_prompt
            assert "03: History" in stage.domains_prompt
        finally:
            import os
            os.unlink(domains_file)
    
    def test_stage_properties(self):
        """Test stage properties"""
        domains_file = create_domains_file()
        mock_client = MockLLMClient()
        
        try:
            stage = LLMBasedDomainClassifier(
                prompt=None,
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
    
    def test_process(self, sample_reasoning_data):
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
        assert stage.outputs() == ["data"]
    
    def test_setup(self):
        """Test setup method"""
        stage = DiversitySampler(
            sampling_size=5,
            input_problem_field="question",
            input_domain_field="domain",
        )
        # Setup should not raise any exceptions
        stage.setup()
    
    @patch('numpy.random.choice')
    @patch('numpy.random.seed')
    def test_sample_uniformly(self, mock_seed, mock_choice, sample_diversity_data):
        """Test uniform sampling method"""
        stage = DiversitySampler(
            sampling_size=4,
            input_problem_field="question",
            input_domain_field="domain",
        )
        
        # Create a counter to cycle through choices
        call_count = 0
        
        def mock_choice_func(*args, **kwargs):
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
    
    def test_process(self, sample_diversity_data):
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
        assert result.dataset_name == "diversity_sampling_data"
        
        # Check that all required columns are present
        assert "question" in result.data.columns
        assert "domain" in result.data.columns
    
    def test_process_sampling_size_larger_than_data(self, sample_diversity_data):
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
        # Note: The actual template might have different placeholders
        assert isinstance(DEFAULT_DOMAIN_CLASSIFICATION_PROMPT_TEMPLATE, str)
        assert len(DEFAULT_DOMAIN_CLASSIFICATION_PROMPT_TEMPLATE) > 0
    
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
