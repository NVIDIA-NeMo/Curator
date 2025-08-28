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

import pytest
import torch

from nemo_curator.models.transnetv2 import (
    ColorHistograms,
    Conv3DConfigurable,
    DilatedDCNNV2,
    FrameSimilarity,
    StackedDDCNNV2,
    TransNetV2,
    _TransNetV2,
)


class TestTransNetV2:
    """Test cases for _TransNetV2 main model class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = _TransNetV2(
            rf=16,
            rl=3,
            rs=2,
            rd=1024,
            use_many_hot_targets=True,
            use_frame_similarity=True,
            use_color_histograms=True,
            use_mean_pooling=False,
            dropout_rate=0.5,
        )

    def test_model_initialization(self):
        """Test model initialization with default parameters."""
        model = _TransNetV2()
        assert model is not None
        assert isinstance(model, torch.nn.Module)
        assert model.training is False  # Model should be in eval mode by default

    def test_model_initialization_with_custom_params(self):
        """Test model initialization with custom parameters."""
        model = _TransNetV2(
            rf=32,
            rl=4,
            rs=3,
            rd=512,
            use_many_hot_targets=False,
            use_frame_similarity=False,
            use_color_histograms=False,
            use_mean_pooling=True,
            dropout_rate=0.3,
        )
        assert model is not None
        assert model.cls_layer2 is None  # Should be None when use_many_hot_targets=False
        assert model.frame_sim_layer is None  # Should be None when use_frame_similarity=False
        assert model.color_hist_layer is None  # Should be None when use_color_histograms=False
        assert model.use_mean_pooling is True

    def test_forward_valid_input(self):
        """Test forward pass with valid input."""
        batch_size = 2
        time_steps = 100
        height = 27
        width = 48
        channels = 3

        # Create valid input tensor
        inputs = torch.randint(0, 255, (batch_size, time_steps, height, width, channels), dtype=torch.uint8)

        with torch.no_grad():
            output = self.model(inputs)

        assert output.shape == (batch_size, time_steps, 1)
        assert output.dtype == torch.float32
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)

    def test_forward_invalid_input_type(self):
        """Test forward pass with invalid input type."""
        inputs = torch.randn(2, 100, 27, 48, 3)  # Wrong dtype (float32 instead of uint8)

        with pytest.raises(ValueError, match="incorrect dtype"):
            self.model(inputs)

    def test_forward_invalid_input_shape(self):
        """Test forward pass with invalid input shape."""
        inputs = torch.randint(0, 255, (2, 100, 28, 48, 3), dtype=torch.uint8)  # Wrong height (28 instead of 27)

        with pytest.raises(ValueError, match="incorrect shape"):
            self.model(inputs)

    def test_forward_invalid_input_dtype(self):
        """Test forward pass with invalid input dtype."""
        inputs = torch.randint(0, 255, (2, 100, 27, 48, 3), dtype=torch.int32)  # Wrong dtype

        with pytest.raises(ValueError, match="incorrect dtype"):
            self.model(inputs)

    def test_forward_without_optional_layers(self):
        """Test forward pass without optional layers."""
        model = _TransNetV2(
            use_frame_similarity=False,
            use_color_histograms=False,
            dropout_rate=None,
        )

        inputs = torch.randint(0, 255, (1, 50, 27, 48, 3), dtype=torch.uint8)

        with torch.no_grad():
            output = model(inputs)

        assert output.shape == (1, 50, 1)


class TestStackedDDCNNV2:
    """Test cases for StackedDDCNNV2 class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = StackedDDCNNV2(
            in_filters=3,
            n_blocks=2,
            filters=16,
            shortcut=True,
            pool_type="avg",
            stochastic_depth_drop_prob=0.1,
        )

    def test_initialization(self):
        """Test model initialization."""
        assert self.model is not None
        assert len(self.model.DDCNN) == 2
        assert self.model.shortcut is True
        assert self.model.stochastic_depth_drop_prob == 0.1

    def test_initialization_with_max_pool(self):
        """Test model initialization with max pooling."""
        model = StackedDDCNNV2(
            in_filters=3,
            n_blocks=2,
            filters=16,
            pool_type="max",
        )
        assert isinstance(model.pool, torch.nn.MaxPool3d)

    def test_initialization_invalid_pool_type(self):
        """Test model initialization with invalid pool type."""
        with pytest.raises(ValueError, match="pool_type must be 'max' or 'avg'"):
            StackedDDCNNV2(
                in_filters=3,
                n_blocks=2,
                filters=16,
                pool_type="invalid",
            )

    def test_forward_pass(self):
        """Test forward pass."""
        batch_size = 1
        time_steps = 10
        height = 27
        width = 48
        channels = 3

        inputs = torch.randn(batch_size, channels, time_steps, height, width)

        with torch.no_grad():
            output = self.model(inputs)

        # Output should be downsampled by factor of 2 in spatial dimensions
        expected_height = height // 2
        expected_width = width // 2
        expected_channels = 16 * 4  # filters * 4 from DilatedDCNNV2

        assert output.shape == (batch_size, expected_channels, time_steps, expected_height, expected_width)


class TestDilatedDCNNV2:
    """Test cases for DilatedDCNNV2 class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = DilatedDCNNV2(
            in_filters=3,
            filters=16,
            batch_norm=True,
            activation=torch.nn.functional.relu,
        )

    def test_initialization(self):
        """Test model initialization."""
        assert self.model is not None
        assert self.model.bn is not None
        assert self.model.activation is not None

    def test_initialization_without_batch_norm(self):
        """Test model initialization without batch normalization."""
        model = DilatedDCNNV2(
            in_filters=3,
            filters=16,
            batch_norm=False,
            activation=None,
        )
        assert model.bn is None
        assert model.activation is None

    def test_forward_pass(self):
        """Test forward pass."""
        batch_size = 1
        time_steps = 10
        height = 27
        width = 48
        channels = 3

        inputs = torch.randn(batch_size, channels, time_steps, height, width)

        with torch.no_grad():
            output = self.model(inputs)

        # Output should have 4 times the filters (concatenated from 4 dilated convs)
        expected_channels = 16 * 4
        assert output.shape == (batch_size, expected_channels, time_steps, height, width)


class TestConv3DConfigurable:
    """Test cases for Conv3DConfigurable class."""

    def test_separable_convolution(self):
        """Test separable convolution configuration."""
        model = Conv3DConfigurable(
            in_filters=3,
            filters=16,
            dilation_rate=2,
            separable=True,
            use_bias=True,
        )

        assert len(model.layers) == 2  # Two layers for separable convolution

        # Test forward pass
        inputs = torch.randn(1, 3, 10, 27, 48)

        with torch.no_grad():
            output = model(inputs)

        assert output.shape == (1, 16, 10, 27, 48)

    def test_non_separable_convolution(self):
        """Test non-separable convolution configuration."""
        model = Conv3DConfigurable(
            in_filters=3,
            filters=16,
            dilation_rate=2,
            separable=False,
            use_bias=True,
        )

        assert len(model.layers) == 1  # Single layer for non-separable convolution

        # Test forward pass
        inputs = torch.randn(1, 3, 10, 27, 48)

        with torch.no_grad():
            output = model(inputs)

        assert output.shape == (1, 16, 10, 27, 48)


class TestFrameSimilarity:
    """Test cases for FrameSimilarity class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = FrameSimilarity(
            in_filters=256,
            similarity_dim=128,
            lookup_window=101,
            output_dim=128,
            use_bias=False,
        )

    def test_initialization(self):
        """Test model initialization."""
        assert self.model is not None
        assert self.model.lookup_window == 101

    def test_initialization_invalid_lookup_window(self):
        """Test model initialization with invalid lookup window."""
        with pytest.raises(ValueError, match="`lookup_window` must be odd integer"):
            FrameSimilarity(
                in_filters=256,
                lookup_window=100,  # Even number
            )

    def test_forward_pass(self):
        """Test forward pass."""
        batch_size = 2
        time_steps = 50
        height = 7
        width = 12

        # Create mock input list (simulating block features from StackedDDCNNV2)
        # The total concatenated features should match the in_filters parameter (256)
        inputs = [
            torch.randn(batch_size, 64, time_steps, height, width),
            torch.randn(batch_size, 128, time_steps, height, width),
            torch.randn(batch_size, 64, time_steps, height, width),  # Total = 64+128+64=256
        ]

        with torch.no_grad():
            output = self.model(inputs)

        assert output.shape == (batch_size, time_steps, 128)
        assert output.dtype == torch.float32


class TestColorHistograms:
    """Test cases for ColorHistograms class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = ColorHistograms(
            lookup_window=101,
            output_dim=128,
        )

    def test_initialization(self):
        """Test model initialization."""
        assert self.model is not None
        assert self.model.lookup_window == 101
        assert self.model.fc is not None

    def test_initialization_without_output_dim(self):
        """Test model initialization without output dimension."""
        model = ColorHistograms(lookup_window=101)
        assert model.fc is None

    def test_initialization_invalid_lookup_window(self):
        """Test model initialization with invalid lookup window."""
        with pytest.raises(ValueError, match="`lookup_window` must be odd integer"):
            ColorHistograms(lookup_window=100)

    def test_compute_color_histograms(self):
        """Test static method for computing color histograms."""
        batch_size = 2
        time_steps = 10
        height = 27
        width = 48

        frames = torch.randint(0, 255, (batch_size, time_steps, height, width, 3), dtype=torch.uint8)

        histograms = ColorHistograms.compute_color_histograms(frames)

        assert histograms.shape == (batch_size, time_steps, 512)
        assert histograms.dtype == torch.float32

    def test_compute_color_histograms_wrong_channels(self):
        """Test color histogram computation with wrong number of channels."""
        frames = torch.randint(0, 255, (2, 10, 27, 48, 4), dtype=torch.uint8)  # 4 channels instead of 3

        with pytest.raises(ValueError, match="Expected 3 channels"):
            ColorHistograms.compute_color_histograms(frames)

    def test_forward_pass(self):
        """Test forward pass."""
        batch_size = 2
        time_steps = 50
        height = 27
        width = 48

        inputs = torch.randint(0, 255, (batch_size, time_steps, height, width, 3), dtype=torch.uint8)

        with torch.no_grad():
            output = self.model(inputs)

        assert output.shape == (batch_size, time_steps, 128)


class TestTransNetV2Interface:
    """Test cases for TransNetV2 interface class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.interface = TransNetV2(model_dir="test_model_dir")

    def test_initialization(self):
        """Test interface initialization."""
        assert self.interface is not None

    def test_model_id_names(self):
        """Test model ID names property."""
        model_ids = self.interface.model_id_names
        assert len(model_ids) == 1
        assert model_ids[0] == "Sn4kehead/TransNetV2"

    @patch("pathlib.Path.exists")
    @patch("torch.load")
    @patch("nemo_curator.models.transnetv2._TransNetV2")
    def test_setup_success(self, mock_transnetv2_class: Mock, mock_torch_load: Mock, mock_exists: Mock):
        """Test successful setup with mock model file."""
        mock_exists.return_value = True
        mock_torch_load.return_value = {}

        # Mock the model instance
        mock_model = Mock()
        mock_model.load_state_dict = Mock()
        mock_model.eval.return_value = mock_model
        mock_model.cuda.return_value = mock_model
        mock_transnetv2_class.return_value = mock_model

        self.interface.setup()

        assert mock_torch_load.called
        assert mock_model.load_state_dict.called
        assert mock_model.eval.called

    @patch("pathlib.Path.exists")
    def test_setup_file_not_found(self, mock_exists: Mock):
        """Test setup with missing model file."""
        mock_exists.return_value = False

        with pytest.raises(FileNotFoundError):
            self.interface.setup()

    def test_call_without_setup(self):
        """Test calling interface without setup."""
        inputs = torch.randint(0, 255, (1, 50, 27, 48, 3), dtype=torch.uint8)

        with pytest.raises(AttributeError):
            self.interface(inputs)

    @patch("pathlib.Path.exists")
    @patch("torch.load")
    @patch("nemo_curator.models.transnetv2._TransNetV2")
    def test_call_after_setup(self, mock_transnetv2_class: Mock, mock_torch_load: Mock, mock_exists: Mock):
        """Test calling interface after setup."""
        mock_exists.return_value = True
        mock_torch_load.return_value = {}

        # Mock the model
        mock_model = Mock()
        mock_model.load_state_dict = Mock()
        mock_model.eval.return_value = mock_model
        mock_model.cuda.return_value = mock_model
        mock_transnetv2_class.return_value = mock_model

        # Expected output
        expected_output = torch.sigmoid(torch.randn(1, 50, 1))
        mock_model.return_value = expected_output

        self.interface.setup()

        inputs = torch.randint(0, 255, (1, 50, 27, 48, 3), dtype=torch.uint8)
        output = self.interface(inputs)

        assert torch.equal(output, expected_output)
        mock_model.assert_called_once_with(inputs)


class TestModelIntegration:
    """Integration tests for model components."""

    def test_model_instantiation(self):
        """Test that model can be instantiated without errors."""
        model = _TransNetV2()
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_model_with_default_parameters(self):
        """Test model with default parameters."""
        model = _TransNetV2()
        assert model.training is False  # Should be in eval mode by default

        # Check that model has the expected components
        assert hasattr(model, "SDDCNN")
        assert hasattr(model, "fc1")
        assert hasattr(model, "cls_layer1")

    def test_model_components_exist(self):
        """Test that all model components are properly initialized."""
        model = _TransNetV2(
            use_frame_similarity=True,
            use_color_histograms=True,
            use_many_hot_targets=True,
        )

        assert model.frame_sim_layer is not None
        assert model.color_hist_layer is not None
        assert model.cls_layer2 is not None
        assert model.dropout is not None
