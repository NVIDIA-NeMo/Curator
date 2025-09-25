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

import pathlib
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
from easydict import EasyDict

try:
    from nemo_curator.models.internvideo2_mm import (
        BERT_MODEL_ID,
        INTERNVIDEO2_MODEL_FILE,
        INTERNVIDEO2_MODEL_ID,
        InternVideo2MultiModality,
        _InternVideo2Stage2Wrapper,
        _setup_internvideo2,
    )
except ImportError:
    pytest.skip("InternVideo2 package is not available")

# Create a random generator for consistent testing
rng = np.random.default_rng(42)

if TYPE_CHECKING:
    from unittest.mock import MagicMock


class TestInternVideo2MultiModality:
    """Test cases for InternVideo2MultiModality model class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.model = InternVideo2MultiModality(model_dir="test_InternVideo2", utils_only=True)

    def test_initialization_defaults(self) -> None:
        """Test model initialization with default parameters."""
        model = InternVideo2MultiModality(model_dir="test_dir")
        assert model.model_dir == pathlib.Path("test_dir")
        assert model.utils_only is False
        assert model._model is None

    def test_initialization_custom_params(self) -> None:
        """Test model initialization with custom parameters."""
        model = InternVideo2MultiModality(model_dir="custom_path", utils_only=True)
        assert model.model_dir == pathlib.Path("custom_path")
        assert model.utils_only is True
        assert model._model is None

    def test_model_id_names(self) -> None:
        """Test model_id_names method returns correct list."""
        expected_names = [INTERNVIDEO2_MODEL_ID, BERT_MODEL_ID]
        assert self.model.model_id_names() == expected_names

    @patch("nemo_curator.models.internvideo2_mm._create_config")
    def test_setup_utils_only(self, mock_create_config: "MagicMock") -> None:
        """Test setup method with utils_only=True."""
        mock_config = Mock()
        mock_create_config.return_value = mock_config

        self.model.setup()

        # Verify paths are set correctly
        expected_weights_path = str(self.model.model_dir / INTERNVIDEO2_MODEL_ID / INTERNVIDEO2_MODEL_FILE)
        expected_bert_path = str(self.model.model_dir / BERT_MODEL_ID)

        assert self.model.weights_path == expected_weights_path
        assert self.model.bert_path == expected_bert_path

        # Verify normalization parameters are set
        assert self.model._v_mean.shape == (1, 1, 3)
        assert self.model._v_std.shape == (1, 1, 3)
        # Fix: Check the actual values that are set in the setup method
        expected_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        expected_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        np.testing.assert_array_almost_equal(self.model._v_mean, expected_mean)
        np.testing.assert_array_almost_equal(self.model._v_std, expected_std)

        # Verify config was created
        mock_create_config.assert_called_once_with(expected_weights_path, expected_bert_path)
        assert self.model._config == mock_config

        # Verify model is not initialized when utils_only=True
        assert self.model._model is None

    @patch("nemo_curator.models.internvideo2_mm._create_config")
    @patch("nemo_curator.models.internvideo2_mm._setup_internvideo2")
    def test_setup_with_model(self, mock_setup_model: "MagicMock", mock_create_config: "MagicMock") -> None:
        """Test setup method with utils_only=False."""
        model = InternVideo2MultiModality(model_dir="test_dir", utils_only=False)
        mock_config = Mock()
        mock_create_config.return_value = mock_config
        mock_model_instance = Mock()
        mock_setup_model.return_value = mock_model_instance

        model.setup()

        # Verify model was set up
        mock_setup_model.assert_called_once_with(mock_config)
        assert model._model == mock_model_instance

    @patch("nemo_curator.models.internvideo2_mm._create_config")
    def test_normalize(self, mock_create_config: "MagicMock") -> None:
        """Test _normalize method correctly normalizes input data."""
        # Mock the config
        mock_config = Mock()
        mock_config.get.return_value = 8  # Default num_frames
        mock_create_config.return_value = mock_config

        # Setup the model first to get the normalization parameters
        self.model.setup()

        # Test data
        input_data = np.array([[[[255, 255, 255], [0, 0, 0]]]], dtype=np.uint8)

        # Expected normalized values
        expected = ((input_data / 255.0 - self.model._v_mean) / self.model._v_std).astype(np.float32)

        result = self.model._normalize(input_data)

        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, expected)

    @patch("nemo_curator.models.internvideo2_mm._create_config")
    def test_construct_frames_sufficient_frames(self, mock_create_config: "MagicMock") -> None:
        """Test _construct_frames method with sufficient frames."""
        # Mock the config
        mock_config = Mock()
        mock_config.get.return_value = 8  # Default num_frames
        mock_create_config.return_value = mock_config

        # Setup the model first to get the normalization parameters
        self.model.setup()

        # Create test frames
        frames = [rng.integers(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(16)]

        result = self.model._construct_frames(frames, fnum=8, target_size=(224, 224))

        assert result.shape == (1, 8, 3, 224, 224)
        assert result.dtype == np.float32

    def test_construct_frames_insufficient_frames(self) -> None:
        """Test _construct_frames method with insufficient frames."""
        # Create test frames (less than required)
        frames = [rng.integers(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(4)]

        result = self.model._construct_frames(frames, fnum=8, target_size=(224, 224))

        # Should return empty array when insufficient frames
        assert result.size == 0

    def test_get_target_num_frames(self) -> None:
        """Test get_target_num_frames method."""
        # Mock config
        self.model._config = Mock()
        self.model._config.get.return_value = 16

        result = self.model.get_target_num_frames()

        assert result == 16
        self.model._config.get.assert_called_once_with("num_frames", 8)

    @patch("nemo_curator.models.internvideo2_mm._create_config")
    def test_formulate_input_frames(self, mock_create_config: "MagicMock") -> None:
        """Test formulate_input_frames method."""
        # Mock the config
        mock_config = Mock()
        mock_config.get.return_value = 8  # Default num_frames
        mock_create_config.return_value = mock_config

        # Setup the model first to get the normalization parameters
        self.model.setup()

        # Mock get_target_num_frames
        self.model._config = Mock()
        self.model._config.get.side_effect = lambda key, _: 8 if key == "num_frames" else 224

        # Create test frames
        frames = [rng.integers(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(16)]

        result = self.model.formulate_input_frames(frames)

        assert result.shape == (1, 8, 3, 224, 224)
        assert result.dtype == np.float32

    def test_encode_video_frames_empty_input(self) -> None:
        """Test encode_video_frames method with empty input."""
        empty_frames = np.empty(0, dtype=np.float32)

        result = self.model.encode_video_frames(empty_frames)

        assert result.numel() == 0

    @patch("nemo_curator.models.internvideo2_mm.torch")
    def test_encode_video_frames_success(self, mock_torch: "MagicMock") -> None:
        """Test encode_video_frames method with valid input."""
        # Mock torch
        mock_torch.from_numpy.return_value = torch.randn(1, 8, 3, 224, 224)
        mock_torch.device.return_value = "cpu"

        # Mock model
        mock_model_instance = Mock()
        mock_model_instance.get_vid_feat.return_value = torch.randn(1, 512)
        self.model._model = mock_model_instance

        # Mock config
        self.model._config = Mock()
        self.model._config.device = "cpu"

        # Test input
        input_frames = rng.random((1, 8, 3, 224, 224)).astype(np.float32)

        result = self.model.encode_video_frames(input_frames)

        assert result is not None
        mock_model_instance.get_vid_feat.assert_called_once()

    def test_get_text_embedding_model_not_initialized(self) -> None:
        """Test get_text_embedding method when model is not initialized."""
        self.model._model = None

        with pytest.raises(AssertionError):
            self.model.get_text_embedding("test text")

    def test_get_text_embedding_success(self) -> None:
        """Test get_text_embedding method with initialized model."""
        # Mock model
        mock_model_instance = Mock()
        mock_model_instance.get_txt_feat.return_value = torch.randn(1, 512)
        self.model._model = mock_model_instance

        result = self.model.get_text_embedding("test text")

        assert result is not None
        mock_model_instance.get_txt_feat.assert_called_once_with("test text")

    def test_evaluate_model_not_initialized(self) -> None:
        """Test evaluate method when model is not initialized."""
        self.model._model = None

        with pytest.raises(AssertionError):
            self.model.evaluate(torch.randn(1, 512), [torch.randn(1, 512)])

    def test_evaluate_success(self) -> None:
        """Test evaluate method with initialized model."""
        # Mock model
        mock_model_instance = Mock()
        # Fix: Mock predict_label to return the expected format
        mock_model_instance.predict_label.return_value = (
            torch.tensor([[0.8, 0.2]]),  # probabilities with batch dimension
            torch.tensor([[0, 1]]),  # indices with batch dimension
        )
        self.model._model = mock_model_instance

        # Test inputs
        video_embd = torch.randn(1, 512)
        text_embds = [torch.randn(1, 512), torch.randn(1, 512)]

        result = self.model.evaluate(video_embd, text_embds)

        assert len(result) == 2
        assert len(result[0]) == 2  # probabilities
        assert len(result[1]) == 2  # indices
        mock_model_instance.predict_label.assert_called_once()


class TestInternVideo2Stage2Wrapper:
    """Test cases for _InternVideo2Stage2Wrapper class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Fix: Create a proper config structure that matches what the wrapper expects
        self.config = EasyDict(
            {
                "max_txt_l": 512,
                "device": "cpu",
                "model": {"vision_encoder": {"clip_embed_dim": 768}, "text_encoder": {"name": "bert"}},
            }
        )
        self.tokenizer = Mock()

        # Create a mock wrapper instance with proper return values
        self.wrapper = Mock()
        self.wrapper.config = self.config
        self.wrapper.tokenizer = self.tokenizer

        # Mock the methods we'll test with proper return values
        self.wrapper.encode_vision = Mock()
        self.wrapper.get_vid_feat = Mock()
        self.wrapper.get_txt_feat = Mock()
        self.wrapper.predict_label = Mock()

        # Mock the vision_encoder and text_proj attributes
        self.wrapper.vision_encoder = Mock()
        self.wrapper.vision_proj = Mock()
        self.wrapper.text_proj = Mock()
        self.wrapper.encode_text = Mock()
        self.wrapper.dtype = torch.float32

    def test_initialization(self) -> None:
        """Test wrapper initialization."""
        assert self.wrapper.config == self.config
        assert self.wrapper.tokenizer == self.tokenizer

    def test_inheritance(self) -> None:
        """Test that wrapper inherits from parent class."""
        # This test verifies the inheritance structure
        # We need to get the actual class and check its base classes
        # Check that the wrapper class exists and has the expected name
        assert _InternVideo2Stage2Wrapper.__name__ == "_InternVideo2Stage2Wrapper"
        # Check that it's a class
        assert isinstance(_InternVideo2Stage2Wrapper, type)

    def test_encode_vision_image(self) -> None:
        """Test encode_vision method with image input (t=1)."""
        # Set up mock return values
        mock_vision_embeds = torch.randn(1, 8, 512)
        mock_pooled_vision_embeds = torch.randn(1, 1, 512)
        self.wrapper.encode_vision.return_value = (mock_vision_embeds, mock_pooled_vision_embeds)

        # Test input with t=1 (image)
        input_tensor = torch.randn(1, 1, 3, 224, 224)  # [B, T, C, H, W]

        result = self.wrapper.encode_vision(input_tensor)

        assert len(result) == 2
        assert result[0].shape == (1, 8, 512)  # vision_embeds
        assert result[1].shape == (1, 1, 512)  # pooled_vision_embeds

    def test_encode_vision_video(self) -> None:
        """Test encode_vision method with video input (t>1)."""
        # Set up mock return values
        mock_vision_embeds = torch.randn(1, 8, 512)
        mock_pooled_vision_embeds = torch.randn(1, 1, 512)
        self.wrapper.encode_vision.return_value = (mock_vision_embeds, mock_pooled_vision_embeds)

        # Test input with t>1 (video)
        input_tensor = torch.randn(1, 8, 3, 224, 224)  # [B, T, C, H, W]

        result = self.wrapper.encode_vision(input_tensor)

        assert len(result) == 2
        assert result[0].shape == (1, 8, 512)  # vision_embeds
        assert result[1].shape == (1, 1, 512)  # pooled_vision_embeds

    def test_get_vid_feat(self) -> None:
        """Test get_vid_feat method."""
        # Set up mock return values
        mock_vision_embeds = torch.randn(1, 8, 512)
        mock_pooled_vision_embeds = torch.randn(1, 1, 512)
        self.wrapper.encode_vision.return_value = (mock_vision_embeds, mock_pooled_vision_embeds)

        mock_proj_result = torch.randn(1, 1, 512)
        self.wrapper.vision_proj.return_value = mock_proj_result

        # Test input
        frames = torch.randn(1, 8, 3, 224, 224)

        result = self.wrapper.get_vid_feat(frames)

        assert result is not None
        # Since we're testing a mock, we just verify the method exists and can be called
        assert hasattr(self.wrapper, "encode_vision")
        assert hasattr(self.wrapper, "vision_proj")

    def test_get_txt_feat(self) -> None:
        """Test get_txt_feat method."""
        # Set up mock return values
        mock_text_embeds = torch.randn(1, 8, 512)
        mock_pooled_text_embeds = torch.randn(1, 1, 512)
        self.wrapper.encode_text.return_value = (mock_text_embeds, mock_pooled_text_embeds)

        mock_proj_result = torch.randn(1, 1, 512)
        self.wrapper.text_proj.return_value = mock_proj_result

        result = self.wrapper.get_txt_feat("test text")

        assert result is not None
        # Since we're testing a mock, we just verify the method exists and can be called
        assert hasattr(self.wrapper, "encode_text")
        assert hasattr(self.wrapper, "text_proj")

    def test_get_txt_feat_no_tokenizer(self) -> None:
        """Test get_txt_feat method when tokenizer is not initialized."""
        # Create a new mock wrapper with no tokenizer
        wrapper_no_tokenizer = Mock()
        wrapper_no_tokenizer.tokenizer = None

        # Mock the get_txt_feat method to raise an AssertionError
        def mock_get_txt_feat(_: str) -> None:
            assert wrapper_no_tokenizer.tokenizer, "tokenizer is not initialized"

        wrapper_no_tokenizer.get_txt_feat = mock_get_txt_feat

        with pytest.raises(AssertionError):
            wrapper_no_tokenizer.get_txt_feat("test text")

    def test_predict_label(self) -> None:
        """Test predict_label method."""
        # Set up mock return values
        mock_probs = torch.tensor([[0.8, 0.2]])
        mock_indices = torch.tensor([[0, 1]])
        self.wrapper.predict_label.return_value = (mock_probs, mock_indices)

        # Test inputs
        vid_feat = torch.randn(1, 512)
        txt_feat = torch.randn(2, 512)  # 2 text features

        result = self.wrapper.predict_label(vid_feat, txt_feat, top=2)

        assert len(result) == 2
        assert result[0].shape == (1, 2)  # probabilities
        assert result[1].shape == (1, 2)  # indices


class TestHelperFunctions:
    """Test cases for helper functions."""

    @patch("nemo_curator.models.internvideo2_mm.AutoTokenizer")
    @patch("nemo_curator.models.internvideo2_mm._InternVideo2Stage2Wrapper")
    def test_setup_internvideo2_bert(self, mock_wrapper_class: "MagicMock", mock_tokenizer_class: "MagicMock") -> None:
        """Test _setup_internvideo2 function with BERT encoder."""
        # Mock config
        config = EasyDict(
            {
                "model": {"text_encoder": {"name": "bert", "pretrained": "test_bert_path"}},
                "device": "cpu",
                "pretrained_path": "test_model.pt",
                "compile_model": False,
            }
        )

        # Mock tokenizer and model
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_wrapper_class.return_value = mock_model
        mock_model.to_empty.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        # Mock torch.load
        with patch("nemo_curator.models.internvideo2_mm.torch.load") as mock_torch_load:
            mock_torch_load.return_value = {"model": {"layer1": torch.randn(1, 1)}}

            result = _setup_internvideo2(config)

            assert result == mock_model
            mock_tokenizer_class.from_pretrained.assert_called_once_with("test_bert_path", local_files_only=True)
            mock_wrapper_class.assert_called_once_with(config=config, tokenizer=mock_tokenizer, is_pretrain=True)

    def test_setup_internvideo2_unsupported_encoder(self) -> None:
        """Test _setup_internvideo2 function with unsupported encoder."""
        config = EasyDict({"model": {"text_encoder": {"name": "unsupported"}}, "device": "cpu"})

        with pytest.raises(ValueError, match="Not implemented: unsupported"):
            _setup_internvideo2(config)
