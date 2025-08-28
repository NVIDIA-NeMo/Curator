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
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from nemo_curator.models.clip import CLIPAestheticScorer, CLIPImageEmbeddings


class TestCLIPImageEmbeddings:
    """Test cases for CLIPImageEmbeddings model class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.model = CLIPImageEmbeddings(model_dir="test_models/clip")

    def test_model_initialization(self) -> None:
        """Test model initialization."""
        assert self.model.model_dir == "test_models/clip"
        assert self.model.clip is None
        assert self.model.processor is None
        assert self.model.device in ["cuda", "cpu"]
        assert self.model.dtype == torch.float32

    def test_model_id_names_property(self) -> None:
        """Test model ID names property."""
        model_ids = self.model.model_id_names
        assert isinstance(model_ids, list)
        assert len(model_ids) == 1
        assert model_ids[0] == "openai/clip-vit-large-patch14"

    @patch("nemo_curator.models.clip.CLIPModel")
    @patch("nemo_curator.models.clip.CLIPProcessor")
    def test_setup_configures_expected_transforms(self, mock_processor: Mock, mock_clip_model: Mock) -> None:
        """Ensure torchvision transforms pipeline matches CLIP preprocessor settings."""
        # Arrange basic mocks
        mock_model_instance = Mock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.eval.return_value = mock_model_instance
        mock_clip_model.from_pretrained.return_value = mock_model_instance
        mock_processor.from_pretrained.return_value = Mock()

        # Act
        self.model.setup()

        # Assert transforms configured
        from torchvision import transforms

        assert self.model.transforms is not None
        assert isinstance(self.model.transforms, transforms.Compose)
        tfs = list(self.model.transforms.transforms)  # type: ignore[attr-defined]
        # Expected order and key attributes
        assert isinstance(tfs[0], transforms.Resize)
        assert tfs[0].size == 224
        assert tfs[0].interpolation.name == "BICUBIC"
        assert tfs[0].antialias is True

        assert isinstance(tfs[1], transforms.CenterCrop)
        # torchvision may represent size as int or (h, w)
        cc_size = tfs[1].size
        if isinstance(cc_size, tuple):
            assert cc_size == (224, 224)
        else:
            assert cc_size == 224

        assert type(tfs[2]).__name__ == "ConvertImageDtype"

        assert isinstance(tfs[3], transforms.Normalize)
        assert pytest.approx(tfs[3].mean, rel=1e-5) == (0.48145466, 0.4578275, 0.40821073)
        assert pytest.approx(tfs[3].std, rel=1e-5) == (0.26862954, 0.26130258, 0.27577711)

    @patch("nemo_curator.models.clip.CLIPModel")
    @patch("nemo_curator.models.clip.CLIPProcessor")
    def test_setup_success(self, mock_processor: Mock, mock_clip_model: Mock) -> None:
        """Test successful model setup."""
        # Mock the model loading
        mock_model_instance = Mock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.eval.return_value = mock_model_instance
        mock_clip_model.from_pretrained.return_value = mock_model_instance

        # Mock processor
        mock_processor_instance = Mock()
        mock_processor.from_pretrained.return_value = mock_processor_instance

        self.model.setup()

        # Verify model loading
        weight_file = str(pathlib.Path(self.model.model_dir) / self.model.model_id_names[0])
        mock_clip_model.from_pretrained.assert_called_once_with(weight_file)
        mock_model_instance.to.assert_called_once_with(self.model.device)
        mock_model_instance.eval.assert_called_once()

        # Verify processor setup
        mock_processor.from_pretrained.assert_called_once_with(weight_file)

        assert self.model.clip == mock_model_instance
        assert self.model.processor == mock_processor_instance

    @patch("nemo_curator.models.clip.torch.cuda.is_available")
    def test_device_selection_with_cuda(self, mock_cuda_available: Mock) -> None:
        """Test device selection when CUDA is available."""
        mock_cuda_available.return_value = True
        model = CLIPImageEmbeddings(model_dir="test_models/clip")
        assert model.device == "cuda"

    @patch("nemo_curator.models.clip.torch.cuda.is_available")
    def test_device_selection_without_cuda(self, mock_cuda_available: Mock) -> None:
        """Test device selection when CUDA is not available."""
        mock_cuda_available.return_value = False
        model = CLIPImageEmbeddings(model_dir="test_models/clip")
        assert model.device == "cpu"

    def test_call_with_numpy_array_uses_transforms_and_normalizes(self) -> None:
        """Calling with numpy array should use torchvision transforms path and return unit-norm embeddings."""
        # Arrange
        with (
            patch("nemo_curator.models.clip.torch.cuda.is_available", return_value=False),
            patch("nemo_curator.models.clip.CLIPModel") as mock_clip_model,
            patch("nemo_curator.models.clip.CLIPProcessor") as mock_processor,
        ):
            mock_model_instance = Mock()
            mock_model_instance.to.return_value = mock_model_instance
            mock_model_instance.eval.return_value = mock_model_instance
            mock_clip_model.from_pretrained.return_value = mock_model_instance
            mock_processor.from_pretrained.return_value = Mock()
            self.model.setup()
        assert self.model.transforms is not None
        # Mock CLIP forward
        mock_clip = Mock()
        # Deterministic embeddings to verify normalization easily
        embed = torch.tensor([[3.0, 4.0], [1.0, 2.0]], dtype=torch.float32)
        mock_clip.get_image_features.return_value = embed
        self.model.clip = mock_clip
        # Ensure processor is not consulted for numpy path
        self.model.processor = Mock()

        # 2 RGB images 224x224
        rng = np.random.default_rng(0)
        images = rng.integers(0, 256, size=(2, 224, 224, 3), dtype=np.uint8)

        # Act
        out = self.model(images)

        # Assert processor not used, transforms path used
        self.model.processor.assert_not_called()
        # Output is normalized row-wise
        norms = torch.linalg.vector_norm(out, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        # Expected normalized values from embed above
        expected = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_call_with_list_images_uses_processor_and_normalizes(self) -> None:
        """Calling with a list should use processor path and return unit-norm embeddings."""
        # Arrange
        mock_clip = Mock()
        embed = torch.tensor([[0.0, 2.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32)
        mock_clip.get_image_features.return_value = embed
        mock_processor = Mock()
        pixel_values = torch.randn(2, 3, 224, 224)
        mock_processor.return_value = {"pixel_values": pixel_values}

        self.model.clip = mock_clip
        self.model.processor = mock_processor
        self.model.transforms = Mock()  # should not be called for list input

        # Two images of potentially different sizes
        rng = np.random.default_rng(1)
        img1 = rng.integers(0, 256, size=(240, 240, 3), dtype=np.uint8)
        img2 = rng.integers(0, 256, size=(200, 300, 3), dtype=np.uint8)
        images = [img1, img2]

        # Act
        out = self.model(images)

        # Assert processor path used
        mock_processor.assert_called_once_with(images=images, return_tensors="pt")
        self.model.transforms.assert_not_called()
        mock_clip.get_image_features.assert_called_once()
        _, kwargs = mock_clip.get_image_features.call_args
        assert "pixel_values" in kwargs
        pv = kwargs["pixel_values"]
        assert isinstance(pv, torch.Tensor)
        assert tuple(pv.shape) == tuple(pixel_values.shape)
        # Device may be CPU or CUDA depending on env; ensure match to model device type
        assert pv.device.type == torch.device(self.model.device).type

        norms = torch.linalg.vector_norm(out, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestCLIPAestheticScorer:
    """Test cases for CLIPAestheticScorer model class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.model = CLIPAestheticScorer(model_dir="test_models/clip_aesthetic")

    def test_model_initialization(self) -> None:
        """Test model initialization."""
        assert self.model.model_dir == "test_models/clip_aesthetic"
        assert self.model._clip_model is None
        assert self.model._aesthetic_model is None

    def test_model_id_names_property(self) -> None:
        """Test model ID names property."""
        model_ids = self.model.model_id_names
        assert isinstance(model_ids, list)
        assert len(model_ids) == 1
        assert model_ids[0] == "openai/clip-vit-large-patch14"

    @patch("nemo_curator.models.clip.CLIPImageEmbeddings")
    @patch("nemo_curator.models.clip.AestheticScorer")
    def test_setup_success(self, mock_aesthetic_scorer: Mock, mock_clip_embeddings: Mock) -> None:
        """Test successful model setup."""
        # Mock the models
        mock_clip_instance = Mock()
        mock_aesthetic_instance = Mock()
        mock_clip_embeddings.return_value = mock_clip_instance
        mock_aesthetic_scorer.return_value = mock_aesthetic_instance

        self.model.setup()

        # Verify model creation
        mock_clip_embeddings.assert_called_once_with(model_dir=self.model.model_dir)
        mock_aesthetic_scorer.assert_called_once_with(model_dir=self.model.model_dir)

        # Verify setup calls
        mock_clip_instance.setup.assert_called_once()
        mock_aesthetic_instance.setup.assert_called_once()

        assert self.model._clip_model == mock_clip_instance
        assert self.model._aesthetic_model == mock_aesthetic_instance

    def test_call_success(self) -> None:
        """Test successful model call."""
        # Setup mock models
        mock_clip = Mock()
        mock_aesthetic = Mock()
        mock_embeddings = torch.randn(2, 768)
        mock_scores = torch.randn(2)

        mock_clip.return_value = mock_embeddings
        mock_aesthetic.return_value = mock_scores

        self.model._clip_model = mock_clip
        self.model._aesthetic_model = mock_aesthetic

        # Test input - use numpy.random.default_rng for modern API
        rng = np.random.default_rng(42)
        images = rng.integers(0, 255, size=(2, 224, 224, 3), dtype=np.uint8)

        result = self.model(images)

        # Verify pipeline
        mock_clip.assert_called_once_with(images)
        mock_aesthetic.assert_called_once_with(mock_embeddings)
        assert torch.equal(result, mock_scores)

    def test_call_without_setup_raises_error(self) -> None:
        """Test that calling model without setup raises error."""
        rng = np.random.default_rng(42)
        images = rng.integers(0, 255, size=(2, 224, 224, 3), dtype=np.uint8)

        with pytest.raises(RuntimeError, match="CLIPAestheticScorer model not initialized"):
            self.model(images)

    def test_call_with_torch_tensor(self) -> None:
        """Test calling model with torch tensor input."""
        # Setup mock models
        mock_clip = Mock()
        mock_aesthetic = Mock()
        mock_embeddings = torch.randn(2, 768)
        mock_scores = torch.randn(2)

        mock_clip.return_value = mock_embeddings
        mock_aesthetic.return_value = mock_scores

        self.model._clip_model = mock_clip
        self.model._aesthetic_model = mock_aesthetic

        # Test input
        images = torch.randint(0, 255, (2, 3, 224, 224), dtype=torch.uint8)

        result = self.model(images)

        # Verify pipeline
        mock_clip.assert_called_once_with(images)
        mock_aesthetic.assert_called_once_with(mock_embeddings)
        assert torch.equal(result, mock_scores)

    def test_call_with_none_clip_model_raises_error(self) -> None:
        """Test that calling with None clip model raises error."""
        self.model._clip_model = None
        self.model._aesthetic_model = Mock()

        rng = np.random.default_rng(42)
        images = rng.integers(0, 255, size=(2, 224, 224, 3), dtype=np.uint8)

        with pytest.raises(RuntimeError, match="CLIPAestheticScorer model not initialized"):
            self.model(images)

    def test_call_with_none_aesthetic_model_raises_error(self) -> None:
        """Test that calling with None aesthetic model raises error."""
        self.model._clip_model = Mock()
        self.model._aesthetic_model = None

        rng = np.random.default_rng(42)
        images = rng.integers(0, 255, size=(2, 224, 224, 3), dtype=np.uint8)

        with pytest.raises(RuntimeError, match="CLIPAestheticScorer model not initialized"):
            self.model(images)


class TestModelIntegration:
    """Integration tests for CLIP model components."""

    @patch("nemo_curator.models.clip.torch.cuda.is_available")
    def test_models_can_be_instantiated(self, mock_cuda_available: Mock) -> None:
        """Test that models can be instantiated without errors."""
        mock_cuda_available.return_value = False  # Use CPU for testing

        clip_model = CLIPImageEmbeddings(model_dir="test_models/clip")
        aesthetic_scorer = CLIPAestheticScorer(model_dir="test_models/clip_aesthetic")

        assert clip_model is not None
        assert aesthetic_scorer is not None
        assert clip_model.device == "cpu"

    def test_model_properties_consistency(self) -> None:
        """Test that model properties are consistent."""
        clip_model = CLIPImageEmbeddings(model_dir="test_models/clip")
        aesthetic_scorer = CLIPAestheticScorer(model_dir="test_models/clip_aesthetic")

        # Both should use same CLIP model
        assert clip_model.model_id_names == aesthetic_scorer.model_id_names
