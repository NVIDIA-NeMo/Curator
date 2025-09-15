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

import pytest

# ruff: noqa: E402
cudf = pytest.importorskip("cudf", reason="EmbeddingCreatorStage tests require cudf")

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from transformers import AutoConfig, AutoModel, AutoTokenizer

from nemo_curator.stages.text.embedders.base import EmbeddingCreatorStage, EmbeddingModelStage
from nemo_curator.stages.text.models.tokenizer import TokenizerStage
from nemo_curator.stages.text.models.utils import ATTENTION_MASK_COLUMN, INPUT_ID_COLUMN
from nemo_curator.tasks import DocumentBatch


class TestEmbeddingModelStage:
    """Test EmbeddingModelStage class."""

    @pytest.fixture
    def pooling_test_data(self) -> dict[str, torch.Tensor]:
        """Unified test data for both pooling strategies."""
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 0],  # First seq: 3 valid tokens
                [1, 1, 1, 1],  # Second seq: 4 valid tokens
            ]
        )

        token_embeddings = torch.tensor(
            [
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [999.0, 999.0]],  # Batch 1
                [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0], [7.0, 7.0]],  # Batch 2
            ],
            dtype=torch.float32,
        )

        # Expected results for different pooling strategies
        expected_results = {
            "mean_pooling": [
                torch.tensor([2.0, 2.0]),  # Batch 1: mean of [1,1], [2,2], [3,3] = [2,2]
                torch.tensor([5.5, 5.5]),  # Batch 2: mean of [4,4], [5,5], [6,6], [7,7] = [5.5,5.5]
            ],
            "last_token": [
                torch.tensor([3.0, 3.0]),  # Batch 1: last valid token at index 2 = [3,3]
                torch.tensor([7.0, 7.0]),  # Batch 2: last valid token at index 3 = [7,7]
            ],
        }

        return {
            "attention_mask": attention_mask,
            "token_embeddings": token_embeddings,
            "expected_results": expected_results,
        }

    def test_embedding_model_stage_initialization(self) -> None:
        """Test EmbeddingModelStage initialization."""
        stage = EmbeddingModelStage(
            model_identifier="test-model",
            embedding_field="embeddings",
            pooling="mean_pooling",
        )

        assert stage.model_identifier == "test-model"
        assert stage.embedding_field == "embeddings"
        assert stage.pooling == "mean_pooling"
        assert stage.unpack_inference_batch is True

        inputs = stage.inputs()
        outputs = stage.outputs()

        # Check that the required columns are present in inputs
        assert inputs[0] == ["data"]
        assert INPUT_ID_COLUMN in inputs[1]
        assert ATTENTION_MASK_COLUMN in inputs[1]
        assert outputs == (["data"], ["embeddings"])

    @pytest.mark.parametrize("pooling_strategy", ["mean_pooling", "last_token"])
    def test_pooling_methods(self, pooling_strategy: str, pooling_test_data: dict[str, torch.Tensor]) -> None:
        """Test both pooling methods with unified test data."""
        stage = EmbeddingModelStage(model_identifier="test-model", pooling=pooling_strategy)

        # Extract test data
        attention_mask = pooling_test_data["attention_mask"]
        token_embeddings = pooling_test_data["token_embeddings"]
        expected_results = pooling_test_data["expected_results"][pooling_strategy]

        mock_output = (token_embeddings,)

        # Call the appropriate pooling method
        if pooling_strategy == "mean_pooling":
            result = stage._mean_pooling(mock_output, attention_mask)
        else:  # last_token
            result = stage._get_last_token(mock_output, attention_mask)

        # Check output properties
        batch_size, hidden_size = token_embeddings.shape[0], token_embeddings.shape[2]
        assert result.shape == (batch_size, hidden_size)

        # Check that embeddings are normalized
        norms = torch.norm(result, dim=1)
        assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5)

        # Verify the actual pooling results match expected normalized values
        for i, expected_value in enumerate(expected_results):
            expected_normalized = torch.nn.functional.normalize(expected_value.unsqueeze(0), dim=1).squeeze(0)
            assert torch.allclose(result[i], expected_normalized, atol=1e-5)

    @pytest.mark.parametrize("pooling_strategy", ["mean_pooling", "last_token"])
    @patch("nemo_curator.stages.text.embedders.base.AutoModel")
    def test_process_end_to_end(self, mock_auto_model: Mock, pooling_strategy: str) -> None:
        """Test end-to-end process() with both pooling strategies."""
        # Create a mock model that returns deterministic embeddings
        mock_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_model

        # Create sample data with tokenized inputs
        sample_data = DocumentBatch(
            task_id="test_batch",
            dataset_name="test_dataset",
            data=pd.DataFrame(
                {
                    "text": ["Hello world", "Test text"],
                    INPUT_ID_COLUMN: [[1, 2, 3, 0], [4, 5, 0, 0]],  # Second sequence shorter
                    ATTENTION_MASK_COLUMN: [[1, 1, 1, 0], [1, 1, 0, 0]],  # Corresponding masks
                }
            ),
        )

        # Test the specified pooling strategy
        stage = EmbeddingModelStage(
            model_identifier="test-model",
            pooling=pooling_strategy,
            model_inference_batch_size=4,  # Process all at once
            has_seq_order=False,  # Disable sequence ordering for simpler testing
        )

        # Mock model output - the model should return a tuple-like object where [0] is last_hidden_state
        # The attention mask will be clipped to match the shortest sequence, so we need to match that
        # After clipping: attention masks will be [[1,1,1], [1,1,0]] -> 3 tokens max
        last_hidden_state = torch.tensor(
            [
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],  # Batch 1: tokens [1,2,3]
                [[4.0, 4.0], [5.0, 5.0], [999.0, 999.0]],  # Batch 2: tokens [4,5], last masked
            ],
            dtype=torch.float32,
        )

        # The model returns a tuple where first element is last_hidden_state
        mock_model.return_value = (last_hidden_state,)
        mock_model.device = "cpu"  # Simplify for testing

        # Setup and process
        stage.setup()
        result = stage.process(sample_data)

        # Verify result structure
        assert isinstance(result, DocumentBatch)
        assert result.task_id == sample_data.task_id
        assert result.dataset_name == sample_data.dataset_name

        result_df = result.to_pandas()
        assert "embeddings" in result_df.columns
        assert len(result_df) == 2

        # Check embeddings are normalized
        embeddings = result_df["embeddings"].tolist()
        embeddings_array = torch.tensor(embeddings)
        norms = torch.norm(embeddings_array, dim=1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)

        # Define expected results based on pooling strategy
        if pooling_strategy == "mean_pooling":
            # Batch 1: mean of [1,1], [2,2], [3,3] = [2,2]
            # Batch 2: mean of [4,4], [5,5] = [4.5,4.5] (last token masked)
            expected_1 = torch.nn.functional.normalize(torch.tensor([2.0, 2.0]).unsqueeze(0), dim=1).squeeze(0)
            expected_2 = torch.nn.functional.normalize(torch.tensor([4.5, 4.5]).unsqueeze(0), dim=1).squeeze(0)
        else:  # last_token
            # Batch 1: last valid token at index 2 -> [3,3]
            # Batch 2: last valid token at index 1 -> [5,5]
            expected_1 = torch.nn.functional.normalize(torch.tensor([3.0, 3.0]).unsqueeze(0), dim=1).squeeze(0)
            expected_2 = torch.nn.functional.normalize(torch.tensor([5.0, 5.0]).unsqueeze(0), dim=1).squeeze(0)

        assert torch.allclose(embeddings_array[0], expected_1, atol=1e-5)
        assert torch.allclose(embeddings_array[1], expected_2, atol=1e-5)


class TestEmbeddingCreatorStage:
    """Test EmbeddingCreatorStage class."""

    @pytest.fixture
    def sample_data(self) -> DocumentBatch:
        """Create sample text data for testing."""
        texts = ["Hello world", "Test text"]
        data = pd.DataFrame({"text": texts})
        return DocumentBatch(task_id="test_batch", dataset_name="test_dataset", data=data)

    def test_embedding_creator_stage_initialization_and_decomposition(self) -> None:
        """Test initialization, decomposition, and parameter passing to decomposed stages."""
        # Test with custom parameters including hf_token and unk_token
        stage = EmbeddingCreatorStage(
            model_identifier="test-model",
            text_field="content",
            embedding_field="embeddings",
            max_chars=1000,
            max_seq_length=256,
            padding_side="left",
            embedding_pooling="last_token",
            model_inference_batch_size=128,
            sort_by_length=False,
            hf_token="test-token",  # noqa:S106
        )

        # Test decomposition and stage types
        stages = stage.decompose()
        tokenizer_stage, embedding_stage = stages[0], stages[1]
        assert len(stages) == 2
        assert isinstance(tokenizer_stage, TokenizerStage)
        assert isinstance(embedding_stage, EmbeddingModelStage)

        # Verify all TokenizerStage parameters
        assert tokenizer_stage.model_identifier == stage.model_identifier == "test-model"
        assert tokenizer_stage.hf_token == stage.hf_token == "test-token"  # noqa:S105
        assert tokenizer_stage.text_field == stage.text_field == "content"
        assert tokenizer_stage.max_chars == stage.max_chars == 1000
        assert tokenizer_stage.max_seq_length == stage.max_seq_length == 256
        assert tokenizer_stage.padding_side == stage.padding_side == "left"
        assert tokenizer_stage.sort_by_length == stage.sort_by_length is False
        assert tokenizer_stage.unk_token is False

        # Verify all EmbeddingModelStage parameters
        assert embedding_stage.model_identifier == stage.model_identifier == "test-model"
        assert embedding_stage.hf_token == stage.hf_token == "test-token"  # noqa:S105
        assert embedding_stage.embedding_field == stage.embedding_field == "embeddings"
        assert embedding_stage.pooling == stage.embedding_pooling == "last_token"
        assert embedding_stage.model_inference_batch_size == stage.model_inference_batch_size == 128
        assert embedding_stage.has_seq_order == stage.sort_by_length is False
        assert embedding_stage.padding_side == stage.padding_side == "left"
        assert embedding_stage.unpack_inference_batch is True

    def test_embedding_creator_stage_inputs_outputs(self) -> None:
        """Test inputs and outputs specification."""
        stage = EmbeddingCreatorStage(
            model_identifier="test-model", text_field="content", embedding_field="embeddings"
        )

        inputs = stage.inputs()
        outputs = stage.outputs()

        # Should inherit from CompositeStage
        assert inputs == (["data"], ["content"])
        assert outputs == (["data"], ["embeddings"])

    def test_embedding_creator_stage_process_integration(self) -> None:
        """Test that decomposed stages can be run in sequence."""
        stage = EmbeddingCreatorStage(model_identifier="test-model")

        # Get the actual decomposed stages
        stages = stage.decompose()

        # Verify we have the expected number of stages
        assert len(stages) == 2  # TokenizerStage + EmbeddingModelStage

        # Verify stage types
        from nemo_curator.stages.text.models.tokenizer import TokenizerStage

        assert isinstance(stages[0], TokenizerStage)
        assert isinstance(stages[1], EmbeddingModelStage)

        # Verify stage configuration is passed correctly
        tokenizer_stage, embedding_stage = stages
        assert tokenizer_stage.model_identifier == stage.model_identifier
        assert embedding_stage.model_identifier == stage.model_identifier
        assert embedding_stage.embedding_field == stage.embedding_field
        assert embedding_stage.pooling == stage.embedding_pooling

    @pytest.mark.parametrize("pooling_strategy", ["mean_pooling", "last_token"])
    @pytest.mark.parametrize("autocast", [True, False])
    @pytest.mark.gpu
    def test_embedding_creator_stage_with_reference_embeddings(
        self, pooling_strategy: str, sample_data: DocumentBatch, autocast: bool
    ) -> None:
        """Test embeddings match reference implementation (requires GPU and model download)."""
        stage = EmbeddingCreatorStage(
            model_identifier="sentence-transformers/all-MiniLM-L6-v2",
            embedding_pooling=pooling_strategy,
            model_inference_batch_size=32,
            autocast=autocast,
        )

        # Decompose and setup stages
        stages = stage.decompose()
        for sub_stage in stages:
            try:
                sub_stage.setup_on_node()
            except RuntimeError:
                pytest.skip("Skipping test due to flaky Hugging Face download")
            sub_stage.setup()

        # Run stages sequentially
        result = sample_data
        for sub_stage in stages:
            result = sub_stage.process(result)
        result_df = result.to_pandas()

        # Get embeddings and compare with reference
        embeddings = result_df["embeddings"].tolist()
        embeddings_array = np.array(embeddings)

        # Get reference embeddings
        texts = sample_data.to_pandas()["text"].tolist()
        reference_embeddings = self._get_reference_embeddings(texts, pooling_strategy=pooling_strategy)

        assert np.allclose(embeddings_array, reference_embeddings, atol=1e-3), (
            "Embeddings should match reference embeddings"
        )

    def _get_reference_embeddings(
        self,
        texts: list[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        pooling_strategy: str = "mean_pooling",
    ) -> np.ndarray:
        """
        Args:
            texts: List of input texts
            model_name: Name or path of the model to use
            pooling_strategy: Either "last_token" or "mean_pooling"
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model = model.to("cuda")
        model.eval()
        max_len_to_use = tokenizer.model_max_length
        if max_len_to_use > 1e5:
            max_len_to_use = AutoConfig.from_pretrained(model_name).max_position_embeddings
        max_seq_length: int = max_len_to_use

        embs = []
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_length,
            )
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            if pooling_strategy == "last_token":
                embeddings = outputs.last_hidden_state[:, -1, :]
            elif pooling_strategy == "mean_pooling":
                token_embeddings = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
                sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
            else:
                msg = "pooling_strategy must be either 'last_token' or 'mean_pooling'"
                raise ValueError(msg)

            normed_emb = F.normalize(embeddings, dim=1).cpu()
            normed_emb = normed_emb.squeeze(0)
            embs.append(normed_emb)

        return np.array(embs)
