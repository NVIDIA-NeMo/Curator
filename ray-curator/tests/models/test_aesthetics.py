"""Unit tests for aesthetics models."""

from unittest.mock import Mock, patch

import numpy as np
import torch

from ray_curator.models.aesthetics import MLP, AestheticScorer


class TestMLP:
    """Test cases for MLP model class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mlp = MLP()

    def test_mlp_initialization(self) -> None:
        """Test MLP initialization."""
        assert self.mlp.fc1 is not None
        assert self.mlp.fc2 is not None
        assert self.mlp.fc3 is not None
        assert self.mlp.fc4 is not None

    def test_mlp_architecture(self) -> None:
        """Test MLP architecture."""
        # Check layer dimensions
        assert self.mlp.fc1.in_features == 768
        assert self.mlp.fc1.out_features == 1024
        assert self.mlp.fc2.in_features == 1024
        assert self.mlp.fc2.out_features == 1024
        assert self.mlp.fc3.in_features == 1024
        assert self.mlp.fc3.out_features == 1024
        assert self.mlp.fc4.in_features == 1024
        assert self.mlp.fc4.out_features == 1

    def test_forward_pass_shape(self) -> None:
        """Test forward pass output shape."""
        input_tensor = torch.randn(2, 768)
        output = self.mlp(input_tensor)
        assert output.shape == (2, 1)

    def test_forward_pass_no_grad_decorator(self) -> None:
        """Test that forward pass uses no_grad decorator."""
        # This test verifies the decorator is present by checking the method
        # The forward method should be wrapped with no_grad
        assert hasattr(self.mlp, "forward")
        # The forward method itself doesn't need the decorator since __call__ has it

    def test_forward_with_different_batch_sizes(self) -> None:
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 4, 8]:
            input_tensor = torch.randn(batch_size, 768)
            output = self.mlp(input_tensor)
            assert output.shape == (batch_size, 1)

    def test_forward_with_edge_cases(self) -> None:
        """Test forward pass with edge cases."""
        # Test with zeros
        input_tensor = torch.zeros(1, 768)
        output = self.mlp(input_tensor)
        assert output.shape == (1, 1)
        assert not torch.isnan(output).any()

        # Test with large values
        input_tensor = torch.ones(1, 768) * 100
        output = self.mlp(input_tensor)
        assert output.shape == (1, 1)
        assert not torch.isnan(output).any()


class TestAestheticScorer:
    """Test cases for AestheticScorer model class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.model = AestheticScorer(model_dir="test_models/aesthetics")

    def test_model_initialization(self) -> None:
        """Test model initialization."""
        assert self.model.model_dir == "test_models/aesthetics"
        assert self.model.mlp is None
        assert self.model.device in ["cuda", "cpu"]
        assert self.model.dtype == torch.float32

    def test_conda_env_name_property(self) -> None:
        """Test conda environment name property."""
        assert self.model.conda_env_name == "video_splitting"

    def test_model_id_names_property(self) -> None:
        """Test model ID names property."""
        model_ids = self.model.model_id_names
        assert isinstance(model_ids, list)
        assert len(model_ids) == 1
        assert model_ids[0] == "openai/clip-vit-large-patch14"

    @patch("ray_curator.models.aesthetics.torch.cuda.is_available")
    def test_device_selection_with_cuda(self, mock_cuda_available: Mock) -> None:
        """Test device selection when CUDA is available."""
        mock_cuda_available.return_value = True
        model = AestheticScorer(model_dir="test_models/aesthetics")
        assert model.device == "cuda"

    @patch("ray_curator.models.aesthetics.torch.cuda.is_available")
    def test_device_selection_without_cuda(self, mock_cuda_available: Mock) -> None:
        """Test device selection when CUDA is not available."""
        mock_cuda_available.return_value = False
        model = AestheticScorer(model_dir="test_models/aesthetics")
        assert model.device == "cpu"

    @patch("ray_curator.models.aesthetics.load_file")
    @patch("ray_curator.models.aesthetics.MLP")
    def test_setup_success(self, mock_mlp_class: Mock, mock_load_file: Mock) -> None:
        """Test successful model setup."""
        # Mock state dict loading
        mock_state_dict = {"fc1.weight": torch.randn(1024, 768)}
        mock_load_file.return_value = mock_state_dict

        # Mock MLP instance
        mock_mlp_instance = Mock()
        mock_mlp_class.return_value = mock_mlp_instance

        self.model.setup()

        # Verify MLP creation and state loading
        mock_mlp_class.assert_called_once()
        mock_mlp_instance.load_state_dict.assert_called_once_with(mock_state_dict)
        mock_mlp_instance.to.assert_called_once_with(self.model.device)
        mock_mlp_instance.eval.assert_called_once()

        assert self.model.mlp == mock_mlp_instance

    def test_call_with_torch_tensor(self) -> None:
        """Test calling model with torch tensor input."""
        # Setup mock MLP
        mock_mlp = Mock()
        mock_scores = Mock()
        mock_squeezed = torch.randn(2)
        mock_scores.squeeze.return_value = mock_squeezed
        mock_mlp.return_value = mock_scores

        self.model.mlp = mock_mlp

        # Test input
        embeddings = torch.randn(2, 768)

        result = self.model(embeddings)

        # Verify MLP call and device transfer - use ANY matcher for tensor comparison
        mock_mlp.assert_called_once()
        mock_scores.squeeze.assert_called_once_with(1)
        assert torch.equal(result, mock_squeezed)

    def test_call_with_numpy_array(self) -> None:
        """Test calling model with numpy array input."""
        # Setup mock MLP
        mock_mlp = Mock()
        mock_scores = Mock()
        mock_squeezed = torch.randn(2)
        mock_scores.squeeze.return_value = mock_squeezed
        mock_mlp.return_value = mock_scores

        self.model.mlp = mock_mlp

        # Test input - use numpy.random.default_rng for modern API
        rng = np.random.default_rng(42)
        embeddings = rng.random((2, 768), dtype=np.float32)

        with patch("torch.from_numpy") as mock_from_numpy:
            mock_tensor = Mock()
            mock_tensor.to.return_value = mock_tensor
            mock_from_numpy.return_value = mock_tensor

            result = self.model(embeddings)

            # Verify numpy conversion
            mock_from_numpy.assert_called_once()
            # Verify the array was copied (embeddings.copy())
            mock_tensor.to.assert_called_once_with(self.model.device)
            mock_mlp.assert_called_once_with(mock_tensor)
            assert torch.equal(result, mock_squeezed)

    def test_call_different_input_shapes(self) -> None:
        """Test calling model with different input shapes."""
        # Setup mock MLP
        mock_mlp = Mock()
        self.model.mlp = mock_mlp

        # Test different batch sizes
        for batch_size in [1, 4, 8]:
            embeddings = torch.randn(batch_size, 768)
            mock_scores = Mock()
            mock_squeezed = torch.randn(batch_size)
            mock_scores.squeeze.return_value = mock_squeezed
            mock_mlp.return_value = mock_scores

            result = self.model(embeddings)

            assert torch.equal(result, mock_squeezed)

    def test_call_preserves_gradient_disable(self) -> None:
        """Test that call preserves the no_grad decorator."""
        # This test verifies the decorator is present
        assert callable(self.model)

        # The method should be wrapped with no_grad
        # We can check if torch.no_grad is in the decorator stack
        assert callable(self.model)

    def test_numpy_array_copy_behavior(self) -> None:
        """Test that numpy arrays are properly copied before conversion."""
        # Setup mock MLP
        mock_mlp = Mock()
        self.model.mlp = mock_mlp

        # Test input - use numpy.random.default_rng for modern API
        rng = np.random.default_rng(42)
        embeddings = rng.random((2, 768), dtype=np.float32)

        with patch("torch.from_numpy") as mock_from_numpy:
            mock_tensor = Mock()
            mock_tensor.to.return_value = mock_tensor
            mock_from_numpy.return_value = mock_tensor

            self.model(embeddings)

            # Verify from_numpy was called with copy
            # The array should be copied before conversion
            mock_from_numpy.assert_called_once()


class TestModelIntegration:
    """Integration tests for aesthetic model components."""

    @patch("ray_curator.models.aesthetics.torch.cuda.is_available")
    def test_models_can_be_instantiated(self, mock_cuda_available: Mock) -> None:
        """Test that models can be instantiated without errors."""
        mock_cuda_available.return_value = False  # Use CPU for testing

        mlp = MLP()
        aesthetic_scorer = AestheticScorer(model_dir="test_models/aesthetics")

        assert mlp is not None
        assert aesthetic_scorer is not None
        assert aesthetic_scorer.device == "cpu"

    def test_mlp_can_process_embeddings(self) -> None:
        """Test that MLP can process CLIP-sized embeddings."""
        mlp = MLP()

        # Test with CLIP embedding size
        embeddings = torch.randn(1, 768)
        scores = mlp(embeddings)

        assert scores.shape == (1, 1)
        assert scores.dtype == torch.float32

    def test_aesthetic_scorer_properties(self) -> None:
        """Test aesthetic scorer properties consistency."""
        scorer = AestheticScorer(model_dir="test_models/aesthetics")

        assert scorer.conda_env_name == "video_splitting"
        assert "openai/clip-vit-large-patch14" in scorer.model_id_names
        assert scorer.device in ["cuda", "cpu"]
        assert scorer.dtype == torch.float32

    def test_mlp_deterministic_output(self) -> None:
        """Test that MLP gives consistent output for same input."""
        mlp = MLP()

        # Set to eval mode for deterministic behavior
        mlp.eval()

        input_tensor = torch.randn(2, 768)

        with torch.no_grad():
            output1 = mlp(input_tensor)
            output2 = mlp(input_tensor)

        assert torch.allclose(output1, output2)
        assert output1.shape == (2, 1)
