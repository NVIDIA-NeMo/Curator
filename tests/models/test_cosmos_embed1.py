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

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from nemo_curator.models.cosmos_embed1 import CosmosEmbed1

if TYPE_CHECKING:
    from unittest.mock import MagicMock


class TestCosmosEmbed1:
    """Test cases for CosmosEmbed1 model class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.model = CosmosEmbed1(variant="336p", utils_only=True, model_dir="/test/model/dir")

    def test_model_initialization_defaults(self) -> None:
        """Test model initialization with default parameters."""
        model = CosmosEmbed1(model_dir="/test/model/dir")
        assert model.variant == "336p"
        assert model._utils_only is False
        assert model._weights_name == "nvidia/Cosmos-Embed1-336p"
        assert model._model is None

    def test_model_initialization_custom_params(self) -> None:
        """Test model initialization with custom parameters."""
        model = CosmosEmbed1(variant="448p", utils_only=True, model_dir="/custom/path")
        assert model.variant == "448p"
        assert model._utils_only is True
        assert model._weights_name == "nvidia/Cosmos-Embed1-448p"
        assert "/custom/path/nvidia/Cosmos-Embed1-448p" in model._weights_dir

    def test_model_initialization_all_variants(self) -> None:
        """Test model initialization with all supported variants."""
        variants = ["224p", "336p", "448p"]
        expected_weights = ["nvidia/Cosmos-Embed1-224p", "nvidia/Cosmos-Embed1-336p", "nvidia/Cosmos-Embed1-448p"]

        for variant, expected_weight in zip(variants, expected_weights, strict=False):
            model = CosmosEmbed1(variant=variant, model_dir="/test/model/dir")
            assert model.variant == variant
            assert model._weights_name == expected_weight

    def test_model_id_names(self) -> None:
        """Test model ID names property."""
        model_336p = CosmosEmbed1(variant="336p", model_dir="/test/model/dir")
        assert model_336p.model_id_names == ["nvidia/Cosmos-Embed1-336p"]

        model_448p = CosmosEmbed1(variant="448p", model_dir="/test/model/dir")
        assert model_448p.model_id_names == ["nvidia/Cosmos-Embed1-448p"]

    @patch("nemo_curator.models.cosmos_embed1.Path")
    @patch("nemo_curator.models.cosmos_embed1.AutoProcessor")
    def test_setup_utils_only(self, mock_processor: "MagicMock", mock_path: "MagicMock") -> None:
        """Test setup method with utils_only=True."""
        # Mock path exists
        mock_path.return_value.exists.return_value = True
        mock_processor_instance = Mock()
        mock_processor.from_pretrained.return_value = mock_processor_instance

        self.model.setup()

        # Verify path check
        mock_path.assert_called_once_with(self.model._weights_dir)
        mock_path.return_value.exists.assert_called_once()

        # Verify processor setup
        mock_processor.from_pretrained.assert_called_once_with(
            self.model._weights_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        assert self.model._processor == mock_processor_instance

    @patch("nemo_curator.models.cosmos_embed1.Path")
    @patch("nemo_curator.models.cosmos_embed1.AutoProcessor")
    @patch("nemo_curator.models.cosmos_embed1.AutoModel")
    def test_setup_with_model(
        self, mock_model: "MagicMock", mock_processor: "MagicMock", mock_path: "MagicMock"
    ) -> None:
        """Test setup method with full model loading."""
        # Setup model with utils_only=False
        model = CosmosEmbed1(variant="336p", utils_only=False, model_dir="/test/model/dir")

        # Mock path exists
        mock_path.return_value.exists.return_value = True
        mock_model_instance = Mock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_processor_instance = Mock()
        mock_processor.from_pretrained.return_value = mock_processor_instance

        model.setup()

        # Verify model setup
        mock_model.from_pretrained.assert_called_once_with(
            model._weights_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        mock_model_instance.to.assert_called_once_with("cuda", dtype=torch.bfloat16)
        mock_model_instance.eval.assert_called_once()

        # Verify processor setup
        mock_processor.from_pretrained.assert_called_once_with(
            model._weights_dir,
            trust_remote_code=True,
            local_files_only=True,
        )

    @patch("nemo_curator.models.cosmos_embed1.Path")
    def test_setup_missing_weights_dir(self, mock_path: "MagicMock") -> None:
        """Test setup method with missing weights directory."""
        mock_path.return_value.exists.return_value = False

        with pytest.raises(FileNotFoundError, match="Weights directory .* not found!"):
            self.model.setup()

    @patch("nemo_curator.models.cosmos_embed1.Path")
    @patch("nemo_curator.models.cosmos_embed1.AutoModel")
    def test_setup_model_load_failure(self, mock_model: "MagicMock", mock_path: "MagicMock") -> None:
        """Test setup method with model loading failure."""
        model = CosmosEmbed1(variant="336p", utils_only=False, model_dir="/test/model/dir")

        # Mock path exists but model loading fails
        mock_path.return_value.exists.return_value = True
        mock_model_instance = Mock()
        mock_model_instance.to.return_value = None  # Return None after .to() call
        mock_model.from_pretrained.return_value = mock_model_instance

        with pytest.raises(RuntimeError, match="Model failed to load"):
            model.setup()

    def test_get_target_num_frames(self) -> None:
        """Test get_target_num_frames method."""
        # Mock processor
        mock_processor = Mock()
        mock_processor.num_video_frames = 16
        self.model._processor = mock_processor

        result = self.model.get_target_num_frames()
        assert result == 16

    def test_formulate_input_frames_success(self) -> None:
        """Test formulate_input_frames method with successful processing."""
        # Mock processor
        mock_processor = Mock()
        mock_processor.num_video_frames = 8
        mock_processor.return_value = {"videos": Mock(numpy=Mock(return_value=np.array([[1, 2, 3]])))}
        self.model._processor = mock_processor

        # Create test frames
        rng = np.random.default_rng(42)
        frames = [rng.integers(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(16)]

        result = self.model.formulate_input_frames(frames)

        # Verify processor was called
        mock_processor.assert_called_once()
        assert result is not None

    def test_formulate_input_frames_insufficient_frames(self) -> None:
        """Test formulate_input_frames method with insufficient frames."""
        # Mock processor
        mock_processor = Mock()
        mock_processor.num_video_frames = 16
        self.model._processor = mock_processor

        # Create insufficient frames
        rng = np.random.default_rng(42)
        frames = [rng.integers(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(8)]

        with patch("nemo_curator.models.cosmos_embed1.logger") as mock_logger:
            result = self.model.formulate_input_frames(frames)

            # Verify error was logged and None returned
            mock_logger.error.assert_called_once()
            assert result is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_encode_video_frames_success(self) -> None:
        """Test encode_video_frames method with successful encoding."""
        # Mock model
        mock_model = Mock()
        mock_model.config.embed_dim = 512
        mock_output = Mock()
        mock_output.visual_proj.to.return_value = torch.randn(1, 512, dtype=torch.float16)
        mock_model.get_video_embeddings.return_value = mock_output
        self.model._model = mock_model

        # Create test frames
        rng = np.random.default_rng(42)
        frames = rng.random((1, 8, 3, 224, 224)).astype(np.float32)

        result = self.model.encode_video_frames(frames)

        # Verify model was called
        mock_model.get_video_embeddings.assert_called_once()
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 512)

    def test_encode_video_frames_no_model(self) -> None:
        """Test encode_video_frames method without loaded model."""
        rng = np.random.default_rng(42)
        frames = rng.random((1, 8, 3, 224, 224)).astype(np.float32)

        with pytest.raises(RuntimeError, match="Model is not loaded"):
            self.model.encode_video_frames(frames)

    def test_encode_video_frames_empty_input(self) -> None:
        """Test encode_video_frames method with empty input."""
        # Mock model
        mock_model = Mock()
        mock_model.config.embed_dim = 512
        self.model._model = mock_model

        # Create empty frames
        frames = np.array([]).astype(np.float32)

        result = self.model.encode_video_frames(frames)

        # Verify empty tensor is returned
        assert isinstance(result, torch.Tensor)
        assert result.shape == (0, 512)

    def test_get_text_embedding_success(self) -> None:
        """Test get_text_embedding method with successful encoding."""
        # Mock model and processor
        mock_model = Mock()
        mock_output = Mock()
        mock_output.text_proj.to.return_value = torch.randn(1, 512, dtype=torch.float16)
        mock_model.get_text_embeddings.return_value = mock_output
        self.model._model = mock_model

        mock_processor = Mock()
        mock_batch = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
        mock_batch_obj = Mock()
        mock_batch_obj.to.return_value = mock_batch
        mock_processor.return_value = mock_batch_obj
        self.model._processor = mock_processor

        result = self.model.get_text_embedding("test text")

        # Verify processor and model were called
        mock_processor.assert_called_once_with(text=["test text"], return_tensors="pt")
        mock_batch_obj.to.assert_called_once_with("cuda", dtype=torch.bfloat16)
        mock_model.get_text_embeddings.assert_called_once_with(**mock_batch)
        assert isinstance(result, torch.Tensor)

    def test_get_text_embedding_no_model(self) -> None:
        """Test get_text_embedding method without loaded model."""
        with pytest.raises(RuntimeError, match="Model is not loaded"):
            self.model.get_text_embedding("test text")

    def test_evaluate_success(self) -> None:
        """Test evaluate method with successful evaluation."""
        # Mock model
        mock_model = Mock()
        self.model._model = mock_model

        # Create test embeddings
        video_embd = torch.randn(1, 512, dtype=torch.float16)
        text_embds = [torch.randn(1, 512, dtype=torch.float16) for _ in range(3)]

        # Mock softmax result
        mock_probs = torch.tensor([[0.8, 0.15, 0.05]], dtype=torch.float32)
        mock_topk_result = Mock()
        mock_topk_result.cpu.return_value.numpy.return_value = np.array([[0.8, 0.15, 0.05]])
        mock_indices = Mock()
        mock_indices.cpu.return_value.long.return_value.numpy.return_value = np.array([[0, 1, 2]])

        with patch("torch.cat") as mock_cat:
            mock_cat.return_value = torch.randn(3, 512, dtype=torch.float16)
            with patch.object(torch.Tensor, "softmax") as mock_softmax:
                mock_softmax.return_value = mock_probs
                with patch.object(torch.Tensor, "topk") as mock_topk:
                    mock_topk.return_value = (mock_topk_result, mock_indices)

                    probs, indices = self.model.evaluate(video_embd, text_embds)

                    # Verify results
                    assert len(probs) == 3
                    assert len(indices) == 3
                    assert isinstance(probs, list)
                    assert isinstance(indices, list)

    def test_evaluate_no_model(self) -> None:
        """Test evaluate method without loaded model."""
        video_embd = torch.randn(1, 512, dtype=torch.float16)
        text_embds = [torch.randn(1, 512, dtype=torch.float16)]

        with pytest.raises(RuntimeError, match="Model is not loaded"):
            self.model.evaluate(video_embd, text_embds)

    def test_evaluate_empty_text_embeddings(self) -> None:
        """Test evaluate method with empty text embeddings."""
        # Mock model
        mock_model = Mock()
        self.model._model = mock_model

        video_embd = torch.randn(1, 512, dtype=torch.float16)
        text_embds = []

        # Mock empty tensor concatenation
        with patch("torch.cat") as mock_cat:
            mock_cat.return_value = torch.empty(0, 512, dtype=torch.float16)
            mock_probs = torch.empty(1, 0, dtype=torch.float32)

            with patch.object(torch.Tensor, "softmax") as mock_softmax:
                mock_softmax.return_value = mock_probs
                with patch.object(torch.Tensor, "topk") as mock_topk:
                    mock_topk_result = Mock()
                    mock_topk_result.cpu.return_value.numpy.return_value = np.array([[]])
                    mock_indices = Mock()
                    mock_indices.cpu.return_value.long.return_value.numpy.return_value = np.array([[]])
                    mock_topk.return_value = (mock_topk_result, mock_indices)

                    probs, indices = self.model.evaluate(video_embd, text_embds)

                    # Verify empty results
                    assert len(probs) == 0
                    assert len(indices) == 0

    def test_model_interface_inheritance(self) -> None:
        """Test that CosmosEmbed1 properly inherits from ModelInterface."""
        from nemo_curator.models.base import ModelInterface

        model = CosmosEmbed1(model_dir="/test/model/dir")
        assert isinstance(model, ModelInterface)

    def test_weights_dir_path_construction(self) -> None:
        """Test that weights directory path is constructed correctly."""
        model = CosmosEmbed1(variant="224p", model_dir="/custom/path")
        expected_path = "/custom/path/nvidia/Cosmos-Embed1-224p"
        assert model._weights_dir == expected_path

    def test_weights_dir_none_model_dir(self) -> None:
        """Test weights directory construction with None model_dir."""
        # This test demonstrates the current behavior - None model_dir causes an error
        # The model expects a valid model_dir to be provided
        with pytest.raises(TypeError):
            CosmosEmbed1(variant="336p", model_dir=None)
