# The MIT License
#
# Copyright (c) 2018-2020 Qiuqiang Kong
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""CNN14 decision-level model implementations for Sound Event Detection.

Vendored from the PANNs reference implementation at immutable commit
``d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4`` (MIT, copyright 2018-2020
Qiuqiang Kong), whose license is reproduced above. The model blocks and three
decision-level CNN14 variants come from:
https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4/pytorch/models.py
The interpolation and frame-padding helpers come from:
https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4/pytorch/pytorch_utils.py
See Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks for
Audio Pattern Recognition" (2020).

Only the three decision-level variants are included because SED needs
framewise output; the base ``Cnn14`` emits clip-level output only. The code
adds typing and factors shared helpers while retaining the reference forward
call contract; the architecture and tensor semantics are unchanged, so
published checkpoints load as-is.

Requires ``torchlibrosa``, which arrives with the ``audio_cuda12`` extra.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

try:
    from torchlibrosa.augmentation import SpecAugmentation
    from torchlibrosa.stft import LogmelFilterBank, Spectrogram
except ModuleNotFoundError as exc:
    msg = "SED models require torchlibrosa. Install it with: pip install nemo-curator[audio_cuda12]"
    raise ImportError(msg) from exc


def init_layer(layer: nn.Module) -> None:
    """Initialize a Linear or Conv layer."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.data.fill_(0.0)


def init_bn(bn: nn.BatchNorm1d | nn.BatchNorm2d) -> None:
    """Initialize a BatchNorm layer."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def interpolate(x: torch.Tensor, ratio: int) -> torch.Tensor:
    """Interpolate in time to compensate CNN downsampling.

    Args:
        x: (batch, time_steps, classes_num)
        ratio: upsample factor
    Returns:
        (batch, time_steps * ratio, classes_num)
    """
    batch_size, time_steps, classes_num = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    return upsampled.reshape(batch_size, time_steps * ratio, classes_num)


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int) -> torch.Tensor:
    """Pad framewise output to match input frame count."""
    pad = framewise_output[:, -1:, :].repeat(1, frames_num - framewise_output.shape[1], 1)
    return torch.cat((framewise_output, pad), dim=1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x: torch.Tensor, pool_size: tuple[int, int] = (2, 2), pool_type: str = "avg") -> torch.Tensor:
        x = functional.relu_(self.bn1(self.conv1(x)))
        x = functional.relu_(self.bn2(self.conv2(x)))
        if pool_type == "avg":
            return functional.avg_pool2d(x, kernel_size=pool_size)
        if pool_type == "max":
            return functional.max_pool2d(x, kernel_size=pool_size)
        x1 = functional.avg_pool2d(x, kernel_size=pool_size)
        x2 = functional.max_pool2d(x, kernel_size=pool_size)
        return x1 + x2


class AttBlock(nn.Module):
    def __init__(self, n_in: int, n_out: int, activation: str = "sigmoid") -> None:
        super().__init__()
        self.activation = activation
        self.att = nn.Conv1d(n_in, n_out, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(n_in, n_out, kernel_size=1, bias=True)
        self.bn_att = nn.BatchNorm1d(n_out)
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        cla = self._nonlinear(self.cla(x))
        clip_out = torch.sum(norm_att * cla, dim=2)
        return clip_out, norm_att, cla

    def _nonlinear(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "sigmoid":
            return torch.sigmoid(x)
        return x


def _cnn14_backbone(  # noqa: PLR0913 - parameters define the checkpoint-compatible audio frontend
    sample_rate: int,
    window_size: int,
    hop_size: int,
    mel_bins: int,
    fmin: int,
    fmax: int,
) -> tuple[Spectrogram, LogmelFilterBank, SpecAugmentation, nn.BatchNorm2d, list[ConvBlock]]:
    """Build the shared front-end layers for all CNN14 variants."""
    spec = Spectrogram(
        n_fft=window_size,
        hop_length=hop_size,
        win_length=window_size,
        window="hann",
        center=True,
        pad_mode="reflect",
        freeze_parameters=True,
    )
    logmel = LogmelFilterBank(
        sr=sample_rate,
        n_fft=window_size,
        n_mels=mel_bins,
        fmin=fmin,
        fmax=fmax,
        ref=1.0,
        amin=1e-10,
        top_db=None,
        freeze_parameters=True,
    )
    aug = SpecAugmentation(time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)
    bn0 = nn.BatchNorm2d(mel_bins)
    init_bn(bn0)
    blocks = [
        ConvBlock(1, 64),
        ConvBlock(64, 128),
        ConvBlock(128, 256),
        ConvBlock(256, 512),
        ConvBlock(512, 1024),
        ConvBlock(1024, 2048),
    ]
    return spec, logmel, aug, bn0, blocks


def _cnn14_encode(  # noqa: PLR0913 - shared encoder dependencies are explicit for checkpoint compatibility
    x: torch.Tensor,
    spec: Spectrogram,
    logmel: LogmelFilterBank,
    aug: SpecAugmentation,
    bn0: nn.BatchNorm2d,
    blocks: list[ConvBlock],
    training: bool,
) -> tuple[torch.Tensor, int]:
    """Run shared CNN14 encoding to get feature maps. Returns (features, frames_num)."""
    x = spec(x)
    x = logmel(x)
    frames_num = x.shape[2]
    x = x.transpose(1, 3)
    x = bn0(x)
    x = x.transpose(1, 3)
    if training:
        x = aug(x)
    pool_sizes = [(2, 2)] * 5 + [(1, 1)]
    for blk, ps in zip(blocks, pool_sizes):  # noqa: B905 - both internal sequences have six fixed entries
        x = blk(x, pool_size=ps, pool_type="avg")
        x = functional.dropout(x, p=0.2, training=training)
    return torch.mean(x, dim=3), frames_num


class Cnn14DecisionLevelMax(nn.Module):
    """CNN14 with decision-level max-pooling for SED."""

    interpolate_ratio = 32

    def __init__(  # noqa: PLR0913 - mirrors published PANNs checkpoint parameters
        self,
        sample_rate: int = 16000,
        window_size: int = 1024,
        hop_size: int = 320,
        mel_bins: int = 64,
        fmin: int = 50,
        fmax: int = 14000,
        classes_num: int = 527,
    ) -> None:
        super().__init__()
        self.spectrogram_extractor, self.logmel_extractor, self.spec_augmenter, self.bn0, blocks = _cnn14_backbone(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax
        )
        self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4, self.conv_block5, self.conv_block6 = (
            blocks
        )
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def _conv_blocks_list(self) -> list[ConvBlock]:
        return [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
            self.conv_block6,
        ]

    def forward(
        self,
        input: torch.Tensor,  # noqa: A002 - exact published PANNs API name
        mixup_lambda: torch.Tensor | None = None,  # noqa: ARG002 - retained for PANNs API compatibility
    ) -> dict[str, torch.Tensor]:
        x, frames_num = _cnn14_encode(
            input,
            self.spectrogram_extractor,
            self.logmel_extractor,
            self.spec_augmenter,
            self.bn0,
            self._conv_blocks_list(),
            self.training,
        )
        x1 = functional.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = functional.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = functional.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = functional.relu_(self.fc1(x))
        x = functional.dropout(x, p=0.5, training=self.training)
        segmentwise_output = torch.sigmoid(self.fc_audioset(x))
        clipwise_output, _ = torch.max(segmentwise_output, dim=1)
        framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)
        return {"framewise_output": framewise_output, "clipwise_output": clipwise_output}


class Cnn14DecisionLevelAvg(nn.Module):
    """CNN14 with decision-level average-pooling for SED."""

    interpolate_ratio = 32

    def __init__(  # noqa: PLR0913 - mirrors published PANNs checkpoint parameters
        self,
        sample_rate: int = 16000,
        window_size: int = 1024,
        hop_size: int = 320,
        mel_bins: int = 64,
        fmin: int = 50,
        fmax: int = 14000,
        classes_num: int = 527,
    ) -> None:
        super().__init__()
        self.spectrogram_extractor, self.logmel_extractor, self.spec_augmenter, self.bn0, blocks = _cnn14_backbone(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax
        )
        self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4, self.conv_block5, self.conv_block6 = (
            blocks
        )
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def _conv_blocks_list(self) -> list[ConvBlock]:
        return [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
            self.conv_block6,
        ]

    def forward(
        self,
        input: torch.Tensor,  # noqa: A002 - exact published PANNs API name
        mixup_lambda: torch.Tensor | None = None,  # noqa: ARG002 - retained for PANNs API compatibility
    ) -> dict[str, torch.Tensor]:
        x, frames_num = _cnn14_encode(
            input,
            self.spectrogram_extractor,
            self.logmel_extractor,
            self.spec_augmenter,
            self.bn0,
            self._conv_blocks_list(),
            self.training,
        )
        x1 = functional.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = functional.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = functional.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = functional.relu_(self.fc1(x))
        x = functional.dropout(x, p=0.5, training=self.training)
        segmentwise_output = torch.sigmoid(self.fc_audioset(x))
        clipwise_output = torch.mean(segmentwise_output, dim=1)
        framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)
        return {"framewise_output": framewise_output, "clipwise_output": clipwise_output}


class Cnn14DecisionLevelAtt(nn.Module):
    """CNN14 with decision-level attention for SED."""

    interpolate_ratio = 32

    def __init__(  # noqa: PLR0913 - mirrors published PANNs checkpoint parameters
        self,
        sample_rate: int = 16000,
        window_size: int = 1024,
        hop_size: int = 320,
        mel_bins: int = 64,
        fmin: int = 50,
        fmax: int = 14000,
        classes_num: int = 527,
    ) -> None:
        super().__init__()
        self.spectrogram_extractor, self.logmel_extractor, self.spec_augmenter, self.bn0, blocks = _cnn14_backbone(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax
        )
        self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4, self.conv_block5, self.conv_block6 = (
            blocks
        )
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.att_block = AttBlock(2048, classes_num, activation="sigmoid")
        init_layer(self.fc1)

    def _conv_blocks_list(self) -> list[ConvBlock]:
        return [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
            self.conv_block6,
        ]

    def forward(
        self,
        input: torch.Tensor,  # noqa: A002 - exact published PANNs API name
        mixup_lambda: torch.Tensor | None = None,  # noqa: ARG002 - retained for PANNs API compatibility
    ) -> dict[str, torch.Tensor]:
        x, frames_num = _cnn14_encode(
            input,
            self.spectrogram_extractor,
            self.logmel_extractor,
            self.spec_augmenter,
            self.bn0,
            self._conv_blocks_list(),
            self.training,
        )
        x1 = functional.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = functional.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = functional.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = functional.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = functional.dropout(x, p=0.5, training=self.training)
        clipwise_output, _, segmentwise_output = self.att_block(x)
        segmentwise_output = segmentwise_output.transpose(1, 2)
        framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)
        return {"framewise_output": framewise_output, "clipwise_output": clipwise_output}
