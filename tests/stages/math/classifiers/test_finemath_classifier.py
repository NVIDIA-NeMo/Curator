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

from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch

from nemo_curator.stages.math.classifiers.finemath import (
    FINEMATH_MODEL_ID,
    MAX_SEQ_LENGTH,
    FineMathClassifier,
    FineMathModelStage,
)
from nemo_curator.tasks import DocumentBatch


class TestFineMathModelStage:
    """Test the FineMathModelStage class."""

    def test_init_default_values(self) -> None:
        """Test FineMathModelStage initialization with default values."""
        stage = FineMathModelStage(model_identifier="test-model")

        assert stage.model_identifier == "test-model"
        assert stage.float_score_column == "finemath_scores"
        assert stage.int_score_column == "finemath_int_scores"
        assert stage.model_inference_batch_size == 256
        assert stage.has_seq_order is True
        assert stage.autocast is True

    def test_init_custom_values(self) -> None:
        """Test FineMathModelStage initialization with custom values."""
        stage = FineMathModelStage(
            model_identifier="custom-model",
            cache_dir="/custom/cache",
            float_score_column="custom_float_scores",
            int_score_column="custom_int_scores",
            model_inference_batch_size=128,
            has_seq_order=False,
            autocast=False,
        )

        assert stage.model_identifier == "custom-model"
        assert stage.cache_dir == "/custom/cache"
        assert stage.float_score_column == "custom_float_scores"
        assert stage.int_score_column == "custom_int_scores"
        assert stage.model_inference_batch_size == 128
        assert stage.has_seq_order is False
        assert stage.autocast is False

    def test_outputs(self) -> None:
        """Test outputs method returns correct column names."""
        stage = FineMathModelStage(model_identifier="test-model")
        outputs = stage.outputs()

        assert outputs == (["data"], ["finemath_scores", "finemath_int_scores"])

    def test_outputs_custom_columns(self) -> None:
        """Test outputs method with custom column names."""
        stage = FineMathModelStage(
            model_identifier="test-model", float_score_column="custom_float", int_score_column="custom_int"
        )
        outputs = stage.outputs()

        assert outputs == (["data"], ["custom_float", "custom_int"])

    def test_configure_forward(self) -> None:
        """Test _configure_forward method modifies model forward function."""
        # Create a mock model
        mock_model = mock.Mock()
        mock_logits = mock.Mock()
        mock_logits.squeeze.return_value.float.return_value = torch.tensor([1.5, 2.5, 3.5])
        mock_output = mock.Mock()
        mock_output.logits = mock_logits
        mock_model.forward.return_value = mock_output

        # Configure the forward function
        configured_model = FineMathModelStage._configure_forward(mock_model, autocast=False)

        # Test that the forward function was modified
        assert configured_model is mock_model

        # Test calling the modified forward function
        with mock.patch("torch.no_grad"):
            configured_model.forward(input_ids=torch.tensor([1, 2, 3]))

        # Verify the result is processed correctly
        mock_logits.squeeze.assert_called_once_with(-1)
        mock_logits.squeeze.return_value.float.assert_called_once()

    @mock.patch("torch.autocast")
    def test_configure_forward_with_autocast(self, mock_autocast: mock.Mock) -> None:
        """Test _configure_forward method with autocast enabled."""
        mock_model = mock.Mock()
        mock_logits = mock.Mock()
        mock_logits.squeeze.return_value.float.return_value = torch.tensor([1.5])
        mock_output = mock.Mock()
        mock_output.logits = mock_logits
        mock_model.forward.return_value = mock_output

        # Configure with autocast enabled
        configured_model = FineMathModelStage._configure_forward(mock_model, autocast=True)

        # Test calling the modified forward function
        with mock.patch("torch.no_grad"):
            configured_model.forward(input_ids=torch.tensor([1]))

        # Verify autocast was used
        mock_autocast.assert_called_once_with(device_type="cuda")

    def test_process_model_output(self) -> None:
        """Test process_model_output method."""
        stage = FineMathModelStage(model_identifier="test-model")

        # Create mock tensor output
        mock_tensor = mock.Mock()
        mock_tensor.cpu.return_value.numpy.return_value = np.array([1.2, 3.8, 5.5, -0.5, 2.0])

        result = stage.process_model_output(mock_tensor)

        # Check that scores are clamped to [0, 5] range
        expected_float_scores = [1.2, 3.8, 5.0, 0.0, 2.0]  # Clamped to [0, 5]
        expected_int_scores = [1, 4, 5, 0, 2]  # round(max(0, min(score, 5)))

        assert result["finemath_scores"] == expected_float_scores
        assert result["finemath_int_scores"] == expected_int_scores

    def test_process_model_output_custom_columns(self) -> None:
        """Test process_model_output with custom column names."""
        stage = FineMathModelStage(
            model_identifier="test-model", float_score_column="custom_float", int_score_column="custom_int"
        )

        mock_tensor = mock.Mock()
        mock_tensor.cpu.return_value.numpy.return_value = np.array([2.5])

        result = stage.process_model_output(mock_tensor)

        assert "custom_float" in result
        assert "custom_int" in result
        assert result["custom_float"] == [2.5]
        assert result["custom_int"] == [2]  # round(max(0, min(2.5, 5))) = round(2.5) = 2

    def test_create_output_dataframe(self) -> None:
        """Test create_output_dataframe method."""
        stage = FineMathModelStage(model_identifier="test-model")

        # Create input DataFrame with tokenizer columns
        input_df = pd.DataFrame(
            {
                "text": ["Sample text 1", "Sample text 2"],
                "input_ids": [[1, 2, 3], [4, 5, 6]],
                "attention_mask": [[1, 1, 1], [1, 1, 0]],
                "other_column": ["value1", "value2"],
            }
        )

        # Create collected output
        collected_output = {"finemath_scores": [2.5, 3.8], "finemath_int_scores": [3, 4]}

        result_df = stage.create_output_dataframe(input_df, collected_output)

        # Check that tokenizer columns are dropped
        assert "input_ids" not in result_df.columns
        assert "attention_mask" not in result_df.columns

        # Check that other columns are preserved
        assert "text" in result_df.columns
        assert "other_column" in result_df.columns

        # Check that score columns are added
        assert "finemath_scores" in result_df.columns
        assert "finemath_int_scores" in result_df.columns

        # Verify values
        assert result_df["finemath_scores"].tolist() == [2.5, 3.8]
        assert result_df["finemath_int_scores"].tolist() == [3, 4]


class TestFineMathClassifier:
    """Test the FineMathClassifier composite stage."""

    def test_init_default_values(self) -> None:
        """Test FineMathClassifier initialization with default values."""
        classifier = FineMathClassifier()

        assert classifier.cache_dir is None
        assert classifier.float_score_column == "finemath_scores"
        assert classifier.int_score_column == "finemath_int_scores"
        assert classifier.text_field == "text"
        assert classifier.max_chars is None
        assert classifier.max_seq_length == MAX_SEQ_LENGTH
        assert classifier.sort_by_length is True
        assert classifier.model_inference_batch_size == 256
        assert classifier.autocast is True

    def test_init_custom_values(self) -> None:
        """Test FineMathClassifier initialization with custom values."""
        classifier = FineMathClassifier(
            cache_dir="/custom/cache",
            float_score_column="custom_float_scores",
            int_score_column="custom_int_scores",
            text_field="content",
            max_chars=1000,
            max_seq_length=256,
            sort_by_length=False,
            model_inference_batch_size=128,
            autocast=False,
        )

        assert classifier.cache_dir == "/custom/cache"
        assert classifier.float_score_column == "custom_float_scores"
        assert classifier.int_score_column == "custom_int_scores"
        assert classifier.text_field == "content"
        assert classifier.max_chars == 1000
        assert classifier.max_seq_length == 256
        assert classifier.sort_by_length is False
        assert classifier.model_inference_batch_size == 128
        assert classifier.autocast is False

    def test_post_init_creates_stages(self) -> None:
        """Test that __post_init__ creates the correct stages."""
        classifier = FineMathClassifier()

        # Should have 2 stages: TokenizerStage and FineMathModelStage
        assert len(classifier.stages) == 2

        # Check tokenizer stage
        tokenizer_stage = classifier.stages[0]
        assert tokenizer_stage.model_identifier == FINEMATH_MODEL_ID
        assert tokenizer_stage.text_field == "text"
        assert tokenizer_stage.max_seq_length == MAX_SEQ_LENGTH

        # Check model stage
        model_stage = classifier.stages[1]
        assert isinstance(model_stage, FineMathModelStage)
        assert model_stage.model_identifier == FINEMATH_MODEL_ID
        assert model_stage.float_score_column == "finemath_scores"
        assert model_stage.int_score_column == "finemath_int_scores"

    def test_post_init_with_custom_parameters(self) -> None:
        """Test __post_init__ with custom parameters."""
        classifier = FineMathClassifier(
            cache_dir="/test/cache",
            text_field="content",
            max_chars=500,
            max_seq_length=256,
            float_score_column="custom_float",
            int_score_column="custom_int",
            model_inference_batch_size=64,
            sort_by_length=False,
            autocast=False,
        )

        # Check tokenizer stage configuration
        tokenizer_stage = classifier.stages[0]
        assert tokenizer_stage.cache_dir == "/test/cache"
        assert tokenizer_stage.text_field == "content"
        assert tokenizer_stage.max_chars == 500
        assert tokenizer_stage.max_seq_length == 256
        assert tokenizer_stage.sort_by_length is False

        # Check model stage configuration
        model_stage = classifier.stages[1]
        assert model_stage.cache_dir == "/test/cache"
        assert model_stage.float_score_column == "custom_float"
        assert model_stage.int_score_column == "custom_int"
        assert model_stage.model_inference_batch_size == 64
        assert model_stage.has_seq_order is False
        assert model_stage.autocast is False

    def test_inputs(self) -> None:
        """Test inputs method returns tokenizer stage inputs."""
        classifier = FineMathClassifier()

        # Should return the inputs from the first stage (tokenizer)
        inputs = classifier.inputs()
        tokenizer_inputs = classifier.stages[0].inputs()

        assert inputs == tokenizer_inputs

    def test_outputs(self) -> None:
        """Test outputs method returns model stage outputs."""
        classifier = FineMathClassifier()

        # Should return the outputs from the second stage (model)
        outputs = classifier.outputs()
        model_outputs = classifier.stages[1].outputs()

        assert outputs == model_outputs

    def test_outputs_custom_columns(self) -> None:
        """Test outputs method with custom column names."""
        classifier = FineMathClassifier(float_score_column="custom_float", int_score_column="custom_int")

        outputs = classifier.outputs()
        expected_outputs = (["data"], ["custom_float", "custom_int"])

        assert outputs == expected_outputs

    def test_decompose(self) -> None:
        """Test decompose method returns the stages list."""
        classifier = FineMathClassifier()

        decomposed_stages = classifier.decompose()

        assert decomposed_stages == classifier.stages
        assert len(decomposed_stages) == 2

    def test_name_generation(self) -> None:
        """Test that the classifier name is generated correctly."""
        classifier = FineMathClassifier()

        # Name should be based on the model identifier with format_name_with_suffix
        # "HuggingFaceTB/finemath-classifier" -> "finemath_classifier_classifier"
        expected_name = "finemath_classifier_classifier"
        assert classifier.name == expected_name

    @pytest.fixture
    def math_dataset(self) -> DocumentBatch:
        """Create a sample dataset with mathematical content."""
        text = [
            "The quadratic formula is x = (-b ± √(b² - 4ac)) / 2a",
            "In calculus, the derivative of x² is 2x",
            "The Pythagorean theorem states that a² + b² = c²",
            "Linear algebra deals with vector spaces and matrices",
            "This is just regular text without mathematical content",
        ]
        df = pd.DataFrame({"text": text})
        return DocumentBatch(
            data=df,
            task_id="math_batch_1",
            dataset_name="math_test_1",
        )

    def test_classifier_structure_with_math_dataset(self, math_dataset: DocumentBatch) -> None:
        """Test classifier structure with mathematical dataset."""
        classifier = FineMathClassifier()

        # Check that input columns match dataset
        input_columns = classifier.inputs()[1]
        assert all(col in math_dataset.data.columns for col in input_columns)

        # Check decomposition
        stages = classifier.decompose()
        assert len(stages) == 2

        # Verify stage types
        from nemo_curator.stages.text.models.tokenizer import TokenizerStage

        assert isinstance(stages[0], TokenizerStage)
        assert isinstance(stages[1], FineMathModelStage)

    def test_classifier_with_different_text_field(self) -> None:
        """Test classifier with different text field name."""
        classifier = FineMathClassifier(text_field="content")

        # Create dataset with different text field
        df = pd.DataFrame({"content": ["Mathematical equation: E = mc²"]})
        dataset = DocumentBatch(data=df, task_id="test", dataset_name="test")

        # Check that input columns match
        input_columns = classifier.inputs()[1]
        assert "content" in input_columns
        assert all(col in dataset.data.columns for col in input_columns)

    def test_edge_case_empty_dataset(self) -> None:
        """Test classifier behavior with empty dataset."""
        classifier = FineMathClassifier()

        # Create empty dataset
        df = pd.DataFrame({"text": []})
        empty_dataset = DocumentBatch(data=df, task_id="empty", dataset_name="empty")

        # Should still have correct input/output structure
        input_columns = classifier.inputs()[1]
        assert all(col in empty_dataset.data.columns for col in input_columns)

        output_columns = classifier.outputs()[1]
        expected_outputs = ["finemath_scores", "finemath_int_scores"]
        assert output_columns == expected_outputs

    def test_score_clamping_edge_cases(self) -> None:
        """Test score processing with edge case values."""
        stage = FineMathModelStage(model_identifier="test-model")

        # Test extreme values
        mock_tensor = mock.Mock()
        extreme_values = np.array([10.0, -5.0, 0.0, 5.0, 2.5, 4.9, 5.1])
        mock_tensor.cpu.return_value.numpy.return_value = extreme_values

        result = stage.process_model_output(mock_tensor)

        # Float scores should be clamped to [0, 5]
        expected_float = [5.0, 0.0, 0.0, 5.0, 2.5, 4.9, 5.0]
        assert result["finemath_scores"] == expected_float

        # Int scores should be clamped then rounded: round(max(0, min(score, 5)))
        # [10.0, -5.0, 0.0, 5.0, 2.5, 4.9, 5.1] -> [5, 0, 0, 5, 2, 5, 5]
        expected_int = [5, 0, 0, 5, 2, 5, 5]
        assert result["finemath_int_scores"] == expected_int
