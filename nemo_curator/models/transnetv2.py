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

r"""Model for fast shot transition detection.

@article{soucek2020transnetv2,
    title={TransNet V2: An effective deep network architecture for fast shot transition detection},
    author={Sou{\v{c}}ek, Tom{\'a}{\v{s}} and Loko{\v{c}}, Jakub},
    year={2020},
    journal={arXiv preprint arXiv:2008.04838},
}
"""

import pathlib
from collections.abc import Callable
from typing import Final

import torch
from loguru import logger
from torch import nn
from torch.nn import functional

from nemo_curator.utils.hf_download_utils import download_model_from_hf

from .base import ModelInterface

_TRANSNETV2_MODEL_ID: Final = "Sn4kehead/TransNetV2"
_TRANSNETV2_MODEL_WEIGHTS: Final = "transnetv2-pytorch-weights.pth"
_TRANSNETV2_MODEL_REVISION: Final = "db6ceab"

class _TransNetV2(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        rf: int = 16,
        rl: int = 3,
        rs: int = 2,
        rd: int = 1024,
        *,
        use_many_hot_targets: bool = True,
        use_frame_similarity: bool = True,
        use_color_histograms: bool = True,
        use_mean_pooling: bool = False,
        dropout_rate: float = 0.5,
    ) -> None:
        """Initialize the TransNetV2 model.

        Args:
            rf: Number of filters in the first layer.
            rl: Number of layers in the network.
            rs: Number of blocks in the network.
            rd: Number of output features.
            use_many_hot_targets: Whether to use many-hot targets.
            use_frame_similarity: Whether to use frame similarity.
            use_color_histograms: Whether to use color histograms.
            use_mean_pooling: Whether to use mean pooling.
            dropout_rate: Dropout rate.

        """
        super().__init__()
        self.SDDCNN = nn.ModuleList(
            [StackedDDCNNV2(in_filters=3, n_blocks=rs, filters=rf, stochastic_depth_drop_prob=0.0)]
            + [
                StackedDDCNNV2(in_filters=(rf * 2 ** (i - 1)) * 4, n_blocks=rs, filters=rf * 2**i)
                for i in range(1, rl)
            ],
        )

        self.frame_sim_layer = (
            FrameSimilarity(
                sum([(rf * 2**i) * 4 for i in range(rl)]),
                lookup_window=101,
                output_dim=128,
                similarity_dim=128,
                use_bias=True,
            )
            if use_frame_similarity
            else None
        )
        self.color_hist_layer = ColorHistograms(lookup_window=101, output_dim=128) if use_color_histograms else None

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else None

        output_dim = ((rf * 2 ** (rl - 1)) * 4) * 3 * 6  # 3x6 for spatial dimensions
        if use_frame_similarity:
            output_dim += 128
        if use_color_histograms:
            output_dim += 128

        self.fc1 = nn.Linear(output_dim, rd)
        self.cls_layer1 = nn.Linear(rd, 1)
        self.cls_layer2 = nn.Linear(rd, 1) if use_many_hot_targets else None

        self.use_mean_pooling = use_mean_pooling
        self.eval()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Process input through the TransNetV2 model.

        Args:
            inputs: Input tensor of video frames.

        Returns:
            Model predictions for shot transitions.

        """
        if not isinstance(inputs, torch.Tensor):
            msg = "inputs must be a torch.Tensor"
            raise TypeError(msg)
        if list(inputs.shape[2:]) != [27, 48, 3]:
            msg = f"incorrect shape: expected [*, *, 27, 48, 3], got {inputs.shape}"
            raise ValueError(msg)
        if inputs.dtype != torch.uint8:
            msg = f"incorrect dtype: expected torch.uint8, got {inputs.dtype}"
            raise ValueError(msg)
        # uint8 of shape [B, T, H, W, 3] to float of shape [B, 3, T, H, W]
        x = inputs.permute([0, 4, 1, 2, 3]).float()
        x = x.div_(255.0)

        block_features = []
        for block in self.SDDCNN:
            x = block(x)
            block_features.append(x)

        if self.use_mean_pooling:
            x = torch.mean(x, dim=[3, 4])
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 3, 4, 1)
            x = x.reshape(x.shape[0], x.shape[1], -1)

        if self.frame_sim_layer is not None:
            x = torch.cat([self.frame_sim_layer(block_features), x], 2)

        if self.color_hist_layer is not None:
            x = torch.cat([self.color_hist_layer(inputs), x], 2)

        x = self.fc1(x)
        x = functional.relu(x)

        if self.dropout is not None:
            x = self.dropout(x)

        one_hot = self.cls_layer1(x)

        # scale from 0 to 1
        # one_hot
        return torch.sigmoid(one_hot)


class StackedDDCNNV2(nn.Module):
    """Stacked dilated dense convolutional neural network for video feature extraction."""

    def __init__(  # noqa: PLR0913
        self,
        in_filters: int,
        n_blocks: int,
        filters: int,
        *,
        shortcut: bool = True,
        pool_type: str = "avg",
        stochastic_depth_drop_prob: float = 0.0,
    ) -> None:
        """Initialize the stacked dilated dense convolutional network.

        Args:
            in_filters: Number of input filters.
            n_blocks: Number of blocks in the network.
            filters: Number of output filters.
            shortcut: Whether to use a shortcut connection.
            pool_type: Type of pooling to use.
            stochastic_depth_drop_prob: Dropout probability for stochastic depth.

        """
        super().__init__()
        if pool_type not in ("max", "avg"):
            msg = f"pool_type must be 'max' or 'avg', got {pool_type}"
            raise ValueError(msg)
        self.shortcut = shortcut
        self.DDCNN = nn.ModuleList(
            [
                DilatedDCNNV2(
                    in_filters if i == 1 else filters * 4,
                    filters,
                    activation=functional.relu if i != n_blocks else None,
                )
                for i in range(1, n_blocks + 1)
            ],
        )
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2)) if pool_type == "max" else nn.AvgPool3d(kernel_size=(1, 2, 2))
        self.stochastic_depth_drop_prob = stochastic_depth_drop_prob

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Process input through the stacked dilated dense convolutional network.

        Args:
            inputs: Input tensor.

        Returns:
            Processed tensor.

        """
        x = inputs
        shortcut: torch.Tensor | None = None

        for block in self.DDCNN:
            x = block(x)
            if shortcut is None:
                shortcut = x
        if shortcut is None:
            msg = "shortcut should not be None"
            raise RuntimeError(msg)

        x = functional.relu(x)

        if self.shortcut is not None:
            if self.stochastic_depth_drop_prob != 0.0:
                if self.training:
                    x = shortcut if torch.rand(1).item() < self.stochastic_depth_drop_prob else x + shortcut
                else:
                    x = (1 - self.stochastic_depth_drop_prob) * x + shortcut
            else:
                x += shortcut

        return self.pool(x)  # type: ignore[no-any-return]


class DilatedDCNNV2(nn.Module):
    """Dilated dense convolutional model with multiple dilation rates."""

    def __init__(
        self,
        in_filters: int,
        filters: int,
        *,
        batch_norm: bool = True,
        activation: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """Initialize the dilated dense convolutional model.

        Args:
            in_filters: Number of input filters.
            filters: Number of output filters.
            batch_norm: Whether to use batch normalization.
            activation: Activation function.

        """
        super().__init__()
        self.Conv3D_1 = Conv3DConfigurable(in_filters, filters, 1, use_bias=not batch_norm)
        self.Conv3D_2 = Conv3DConfigurable(in_filters, filters, 2, use_bias=not batch_norm)
        self.Conv3D_4 = Conv3DConfigurable(in_filters, filters, 4, use_bias=not batch_norm)
        self.Conv3D_8 = Conv3DConfigurable(in_filters, filters, 8, use_bias=not batch_norm)

        self.bn = nn.BatchNorm3d(filters * 4, eps=1e-3) if batch_norm else None
        self.activation = activation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Process input through the dilated dense convolutional network.

        Args:
            inputs: Input tensor.

        Returns:
            Processed tensor.

        """
        conv1 = self.Conv3D_1(inputs)
        conv2 = self.Conv3D_2(inputs)
        conv3 = self.Conv3D_4(inputs)
        conv4 = self.Conv3D_8(inputs)

        x = torch.cat([conv1, conv2, conv3, conv4], dim=1)

        if self.bn is not None:
            x = self.bn(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class Conv3DConfigurable(nn.Module):
    """Configurable 3D convolution layer with support for separable convolutions."""

    def __init__(
        self,
        in_filters: int,
        filters: int,
        dilation_rate: int,
        *,
        separable: bool = True,
        use_bias: bool = True,
    ) -> None:
        """Initialize the 3D convolutional layer.

        Args:
            in_filters: Number of input filters.
            filters: Number of output filters.
            dilation_rate: Dilation rate for the convolution.
            separable: Whether to use separable convolution.
            use_bias: Whether to use bias in the convolution.

        """
        super().__init__()

        if separable:
            # (2+1)D convolution https://arxiv.org/pdf/1711.11248.pdf
            conv1 = nn.Conv3d(
                in_filters,
                2 * filters,
                kernel_size=(1, 3, 3),
                dilation=(1, 1, 1),
                padding=(0, 1, 1),
                bias=False,
            )
            conv2 = nn.Conv3d(
                2 * filters,
                filters,
                kernel_size=(3, 1, 1),
                dilation=(dilation_rate, 1, 1),
                padding=(dilation_rate, 0, 0),
                bias=use_bias,
            )
            self.layers = nn.ModuleList([conv1, conv2])
        else:
            conv = nn.Conv3d(
                in_filters,
                filters,
                kernel_size=3,
                dilation=(dilation_rate, 1, 1),
                padding=(dilation_rate, 1, 1),
                bias=use_bias,
            )
            self.layers = nn.ModuleList([conv])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Process input through the 3D convolutional layers.

        Args:
            inputs: Input tensor.

        Returns:
            Processed tensor.

        """
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class FrameSimilarity(nn.Module):
    """Model for computing frame similarity features in video sequences."""

    def __init__(
        self,
        in_filters: int,
        similarity_dim: int = 128,
        lookup_window: int = 101,
        output_dim: int = 128,
        *,
        use_bias: bool = False,
    ) -> None:
        """Initialize the frame similarity model.

        Args:
            in_filters: Number of input filters.
            similarity_dim: Dimension of similarity features.
            lookup_window: Size of the window for similarity computation.
            output_dim: Dimension of the output features.
            use_bias: Whether to use bias in linear layers.

        """
        super().__init__()
        self.projection = nn.Linear(in_filters, similarity_dim, bias=use_bias)
        self.fc = nn.Linear(lookup_window, output_dim)

        self.lookup_window = lookup_window
        if lookup_window % 2 != 1:
            msg = "`lookup_window` must be odd integer"
            raise ValueError(msg)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Process input frames through the model.

        Args:
            inputs: Input tensor of video frames.

        Returns:
            Frame similarity features.

        """
        x = torch.cat([torch.mean(x, dim=[3, 4]) for x in inputs], dim=1)
        x = torch.transpose(x, 1, 2)

        x = self.projection(x)
        x = functional.normalize(x, p=2, dim=2)

        batch_size, time_window = x.shape[0], x.shape[1]
        similarities = torch.bmm(x, x.transpose(1, 2))  # [batch_size, time_window, time_window]
        similarities_padded = functional.pad(
            similarities,
            [(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2],
        )

        batch_indices = (
            torch.arange(0, batch_size, device=x.device)
            .view([batch_size, 1, 1])
            .repeat([1, time_window, self.lookup_window])
        )
        time_indices = (
            torch.arange(0, time_window, device=x.device)
            .view([1, time_window, 1])
            .repeat([batch_size, 1, self.lookup_window])
        )
        lookup_indices = (
            torch.arange(0, self.lookup_window, device=x.device)
            .view([1, 1, self.lookup_window])
            .repeat([batch_size, time_window, 1])
            + time_indices
        )

        similarities = similarities_padded[batch_indices, time_indices, lookup_indices]
        return functional.relu(self.fc(similarities))


class ColorHistograms(nn.Module):
    """Model for computing and comparing color histograms across video frames."""

    def __init__(self, lookup_window: int = 101, output_dim: int | None = None) -> None:
        """Initialize the color histogram model.

        Args:
            lookup_window: Size of the window for histogram computation.
            output_dim: Optional dimension for the output features.

        """
        super().__init__()

        self.fc = nn.Linear(lookup_window, output_dim) if output_dim is not None else None
        self.lookup_window = lookup_window
        if lookup_window % 2 != 1:
            msg = "`lookup_window` must be odd integer"
            raise ValueError(msg)

    @staticmethod
    def compute_color_histograms(frames: torch.Tensor) -> torch.Tensor:
        """Compute color histograms for video frames.

        Args:
            frames: Input tensor of video frames.

        Returns:
            Color histogram tensor.

        """
        frames = frames.int()

        num_chans: int = 3

        def get_bin(frames: torch.Tensor) -> torch.Tensor:
            """Get color bin indices for frames.

            Args:
                frames: Input tensor of video frames with RGB channels.

            Returns:
                Tensor of color bin indices, values are 0 .. 511

            """
            r, g, b = frames[:, :, 0], frames[:, :, 1], frames[:, :, 2]
            r, g, b = r >> 5, g >> 5, b >> 5
            return (r << 6) + (g << 3) + b

        batch_size, time_window, height, width, no_channels = frames.shape
        if no_channels != num_chans:
            msg = f"Expected {num_chans} channels, got {no_channels}"
            raise ValueError(msg)
        frames_flatten = frames.view(batch_size * time_window, height * width, 3)

        binned_values = get_bin(frames_flatten)
        frame_bin_prefix = (torch.arange(0, batch_size * time_window, device=frames.device) << 9).view(-1, 1)
        binned_values = (binned_values + frame_bin_prefix).view(-1)

        histograms = torch.zeros(batch_size * time_window * 512, dtype=torch.int32, device=frames.device)
        histograms.scatter_add_(
            0,
            binned_values,
            torch.ones(len(binned_values), dtype=torch.int32, device=frames.device),
        )

        histograms = histograms.view(batch_size, time_window, 512).float()
        # histograms_normalized
        return functional.normalize(histograms, p=2, dim=2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Process input frames through the model.

        Args:
            inputs: Input tensor of video frames.

        Returns:
            Model predictions for shot transitions.

        """
        x = self.compute_color_histograms(inputs)

        batch_size, time_window = x.shape[0], x.shape[1]
        similarities = torch.bmm(x, x.transpose(1, 2))  # [batch_size, time_window, time_window]
        similarities_padded = functional.pad(
            similarities,
            [(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2],
        )

        batch_indices = (
            torch.arange(0, batch_size, device=x.device)
            .view([batch_size, 1, 1])
            .repeat([1, time_window, self.lookup_window])
        )
        time_indices = (
            torch.arange(0, time_window, device=x.device)
            .view([1, time_window, 1])
            .repeat([batch_size, 1, self.lookup_window])
        )
        lookup_indices = (
            torch.arange(0, self.lookup_window, device=x.device)
            .view([1, 1, self.lookup_window])
            .repeat([batch_size, time_window, 1])
            + time_indices
        )

        similarities = similarities_padded[batch_indices, time_indices, lookup_indices]

        if self.fc is not None:
            return functional.relu(self.fc(similarities))
        return similarities


class TransNetV2(ModelInterface):
    """Interface for TransNetV2 shot transition detection model."""

    def __init__(self, model_dir: str | None = None) -> None:
        """Initialize the TransNetV2 model interface."""
        super().__init__()
        self.model_dir = model_dir

    @property
    def model_id_names(self) -> list[str]:
        """Get the model ID names.

        Returns:
            A list of model ID names.

        """
        return [_TRANSNETV2_MODEL_ID]

    def setup(self) -> None:
        """Set up the TransNetV2 model interface."""
        self._model = _TransNetV2()
        model_dir = pathlib.Path(self.model_dir) / _TRANSNETV2_MODEL_ID
        model_file = model_dir / _TRANSNETV2_MODEL_WEIGHTS
        if not model_file.exists():
            error_msg = f"{model_file} not found!"
            raise FileNotFoundError(error_msg)
        state_dict = torch.load(model_file.as_posix())
        self._model.load_state_dict(state_dict)
        self._model.eval().cuda()

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """TransNetV2 model call.

        Args:
            inputs: tensor of shape [# batch, # frames, height, width, RGB].

        Returns:
            tensor of shape [# batch, # frames, 1] of probabilities for each frame being a shot transition.

        """
        with torch.no_grad():
            return self._model(inputs)  # type: ignore[no-any-return]

    @classmethod
    def download_weights_on_node(cls, model_dir: str) -> None:
        """Download TransNetV2 weights on the node.

        Args:
            model_dir: Directory to save the model weights. If None, uses self.model_dir.
        """

        # Create the model directory if it doesn't exist
        model_dir_path = pathlib.Path(model_dir) / _TRANSNETV2_MODEL_ID
        model_dir_path.mkdir(parents=True, exist_ok=True)
        model_file = model_dir_path / _TRANSNETV2_MODEL_WEIGHTS
        if model_file.exists():
            return

        # Download the weights file from Hugging Face
        download_model_from_hf(
            model_id=_TRANSNETV2_MODEL_ID,
            local_dir=model_dir_path,
            filename=_TRANSNETV2_MODEL_WEIGHTS,
            revision=_TRANSNETV2_MODEL_REVISION,
        )

        logger.info(f"TransNetV2 weights downloaded to: {model_file}")
