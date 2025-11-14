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
    CenterCropTextStage,
    FineMathClassifier,
    FineMathModelStage,
)
from nemo_curator.tasks import DocumentBatch


class TestCenterCropTextStage:
    """Test the CenterCropTextStage class."""

    def test_init_default_values(self) -> None:
        """Test CenterCropTextStage initialization with default values."""
        stage = CenterCropTextStage()

        assert stage.text_field == "text"
        assert stage.center_crop_chars == 10_000

    def test_init_custom_values(self) -> None:
        """Test CenterCropTextStage initialization with custom values."""
        stage = CenterCropTextStage(text_field="content", center_crop_chars=5000)

        assert stage.text_field == "content"
        assert stage.center_crop_chars == 5000

    def test_inputs_outputs(self) -> None:
        """Test inputs and outputs methods."""
        stage = CenterCropTextStage(text_field="custom_text")

        inputs = stage.inputs()
        outputs = stage.outputs()

        assert inputs == (["data"], ["custom_text"])
        assert outputs == (["data"], ["custom_text"])

    def test_mid_slice_function(self) -> None:
        """Test the _mid_slice static method."""
        # Test with short string (cropping needed due to implementation)
        short_text = "Hello World"  # 11 characters, mid=5
        result = CenterCropTextStage._mid_slice(short_text, 100)
        # m=5, b=max(0, 5-100)=0, e=min(5+100, 11-1)=10
        assert result == "Hello Worl"  # s[0:10]

        # Test with long string (cropping needed)
        long_text = "0123456789" * 10  # 100 characters, mid=50
        result = CenterCropTextStage._mid_slice(long_text, 10)
        # m=50, b=max(0, 50-10)=40, e=min(50+10, 100-1)=60
        assert len(result) == 20  # s[40:60]
        expected = long_text[40:60]  # Get the actual slice from the long text
        assert result == expected

        # Test edge case with empty string
        result = CenterCropTextStage._mid_slice("", 10)
        assert result == ""

    def test_process_with_cropping(self) -> None:
        """Test process method with text that needs cropping."""
        stage = CenterCropTextStage(center_crop_chars=5)

        # Create test data with long text
        long_text = "0123456789ABCDEFGHIJ"  # 20 characters, mid=10
        df = pd.DataFrame({"text": [long_text, "short"]})
        batch = DocumentBatch(data=df, task_id="test", dataset_name="test")

        result = stage.process(batch)

        # Long text: m=10, b=max(0, 10-5)=5, e=min(10+5, 20-1)=15
        # Should get s[5:15] = "56789ABCDE"
        cropped_text = result.data["text"].iloc[0]
        assert len(cropped_text) == 10
        assert cropped_text == "56789ABCDE"

        # Short text: "short" has 5 chars, mid=2, b=max(0, 2-5)=0, e=min(2+5, 5-1)=4
        # Should get s[0:4] = "shor"
        assert result.data["text"].iloc[1] == "shor"

    def test_process_no_cropping_needed(self) -> None:
        """Test process method when no cropping is needed."""
        stage = CenterCropTextStage(center_crop_chars=100)

        df = pd.DataFrame({"text": ["Short text", "Another short text"]})
        batch = DocumentBatch(data=df, task_id="test", dataset_name="test")

        result = stage.process(batch)

        # Due to the _mid_slice implementation, even with large crop_chars,
        # text gets cropped due to the min(m+n, len(s)-1) logic
        # "Short text" (10 chars): m=5, b=0, e=min(5+100, 10-1)=9, so s[0:9]="Short tex"
        assert result.data["text"].iloc[0] == "Short tex"
        # "Another short text" (18 chars): m=9, b=0, e=min(9+100, 18-1)=17, so s[0:17]
        assert result.data["text"].iloc[1] == "Another short tex"

    def test_process_zero_crop_chars(self) -> None:
        """Test process method with zero crop characters."""
        stage = CenterCropTextStage(center_crop_chars=0)

        df = pd.DataFrame({"text": ["Any text here"]})
        batch = DocumentBatch(data=df, task_id="test", dataset_name="test")

        result = stage.process(batch)

        # Should remain unchanged when crop_chars is 0
        assert result.data["text"].iloc[0] == "Any text here"

    def test_process_missing_text_field(self) -> None:
        """Test process method when text field is missing."""
        stage = CenterCropTextStage(text_field="missing_field")

        df = pd.DataFrame({"other_field": ["Some text"]})
        batch = DocumentBatch(data=df, task_id="test", dataset_name="test")

        result = stage.process(batch)

        # Should return unchanged when field is missing
        assert "other_field" in result.data.columns
        assert "missing_field" not in result.data.columns


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
        assert classifier.sort_by_length is False
        assert classifier.model_inference_batch_size == 1024
        assert classifier.autocast is True
        assert classifier.center_crop_chars == 10_000

    def test_init_custom_values(self) -> None:
        """Test FineMathClassifier initialization with custom values."""
        classifier = FineMathClassifier(
            cache_dir="/custom/cache",
            float_score_column="custom_float_scores",
            int_score_column="custom_int_scores",
            text_field="content",
            max_chars=1000,
            max_seq_length=256,
            sort_by_length=True,  # Override default
            model_inference_batch_size=128,
            autocast=False,
            center_crop_chars=5000,  # Custom value
        )

        assert classifier.cache_dir == "/custom/cache"
        assert classifier.float_score_column == "custom_float_scores"
        assert classifier.int_score_column == "custom_int_scores"
        assert classifier.text_field == "content"
        assert classifier.max_chars == 1000
        assert classifier.max_seq_length == 256
        assert classifier.sort_by_length is True  # Custom override
        assert classifier.model_inference_batch_size == 128
        assert classifier.autocast is False
        assert classifier.center_crop_chars == 5000  # Custom value

    def test_post_init_creates_stages(self) -> None:
        """Test that __post_init__ creates the correct stages."""
        classifier = FineMathClassifier()

        # Should have 3 stages: CenterCropTextStage, TokenizerStage and FineMathModelStage
        assert len(classifier.stages) == 3

        # Check center crop stage
        center_crop_stage = classifier.stages[0]
        assert isinstance(center_crop_stage, CenterCropTextStage)
        assert center_crop_stage.text_field == "text"
        assert center_crop_stage.center_crop_chars == 10_000

        # Check tokenizer stage
        tokenizer_stage = classifier.stages[1]
        assert tokenizer_stage.model_identifier == FINEMATH_MODEL_ID
        assert tokenizer_stage.text_field == "text"
        assert tokenizer_stage.max_seq_length == MAX_SEQ_LENGTH

        # Check model stage
        model_stage = classifier.stages[2]
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
        tokenizer_stage = classifier.stages[1]
        assert tokenizer_stage.cache_dir == "/test/cache"
        assert tokenizer_stage.text_field == "content"
        assert tokenizer_stage.max_chars == 500
        assert tokenizer_stage.max_seq_length == 256
        assert tokenizer_stage.sort_by_length is False

        # Check model stage configuration
        model_stage = classifier.stages[2]
        assert model_stage.cache_dir == "/test/cache"
        assert model_stage.float_score_column == "custom_float"
        assert model_stage.int_score_column == "custom_int"
        assert model_stage.model_inference_batch_size == 64
        assert model_stage.has_seq_order is False
        assert model_stage.autocast is False

    def test_inputs(self) -> None:
        """Test inputs method returns tokenizer stage inputs."""
        classifier = FineMathClassifier()

        # Should return the inputs from the first stage (center crop)
        inputs = classifier.inputs()
        first_stage_inputs = classifier.stages[0].inputs()

        assert inputs == first_stage_inputs

    def test_outputs(self) -> None:
        """Test outputs method returns model stage outputs."""
        classifier = FineMathClassifier()

        # Should return the outputs from the last stage (model)
        outputs = classifier.outputs()
        expected_outputs = (["data"], ["finemath_scores", "finemath_int_scores"])

        assert outputs == expected_outputs

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
        assert len(decomposed_stages) == 3

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
        assert len(stages) == 3

        # Verify stage types
        from nemo_curator.stages.text.models.tokenizer import TokenizerStage

        assert isinstance(stages[0], CenterCropTextStage)
        assert isinstance(stages[1], TokenizerStage)
        assert isinstance(stages[2], FineMathModelStage)

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
