# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Shared waveform helpers for audio I/O stages."""

import hashlib
import os
from urllib.parse import urlparse

import torch

_WAVEFORM_2D_NDIM = 2


def audio_item_id_from_path(audio_path: str) -> str:
    parsed = urlparse(str(audio_path))
    basename = os.path.basename(parsed.path if parsed.scheme else str(audio_path))
    stem = os.path.splitext(basename)[0] or "audio"
    path_hash = hashlib.sha256(str(audio_path).encode()).hexdigest()[:8]
    return f"{stem}_{path_hash}"


def as_waveform_tensor(waveform: object) -> torch.Tensor:
    if waveform is None:
        msg = "waveform is required"
        raise ValueError(msg)
    if torch.is_tensor(waveform):
        tensor = waveform.detach().to(dtype=torch.float32)
    else:
        tensor = torch.as_tensor(waveform, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != _WAVEFORM_2D_NDIM:
        msg = f"waveform must be 1-D or 2-D, got shape {tuple(tensor.shape)}"
        raise ValueError(msg)
    return tensor.contiguous()


def convert_channels(waveform: torch.Tensor, target_nchannels: int) -> torch.Tensor:
    if target_nchannels <= 0:
        msg = f"target_nchannels must be > 0, got {target_nchannels}"
        raise ValueError(msg)
    if waveform.shape[0] == target_nchannels:
        return waveform
    if target_nchannels == 1:
        return waveform.mean(dim=0, keepdim=True)
    if waveform.shape[0] == 1:
        return waveform.repeat(target_nchannels, 1)
    msg = f"Cannot convert {waveform.shape[0]} channels to {target_nchannels} without a mixing policy"
    raise ValueError(msg)


def resample_waveform(waveform: torch.Tensor, sample_rate: int, target_sample_rate: int) -> torch.Tensor:
    if sample_rate <= 0:
        msg = f"sample_rate must be > 0, got {sample_rate}"
        raise ValueError(msg)
    if target_sample_rate <= 0:
        msg = f"target_sample_rate must be > 0, got {target_sample_rate}"
        raise ValueError(msg)
    if sample_rate == target_sample_rate:
        return waveform
    try:
        from torchaudio.functional import resample
    except ImportError as exc:
        msg = "Resampling an in-memory waveform requires torchaudio"
        raise RuntimeError(msg) from exc
    return resample(waveform, orig_freq=sample_rate, new_freq=target_sample_rate)


def prepare_waveform(
    waveform: object,
    sample_rate: int,
    *,
    target_sample_rate: int,
    target_nchannels: int,
) -> torch.Tensor:
    tensor = as_waveform_tensor(waveform)
    tensor = convert_channels(tensor, target_nchannels)
    tensor = resample_waveform(tensor, int(sample_rate), int(target_sample_rate))
    return tensor.contiguous()
