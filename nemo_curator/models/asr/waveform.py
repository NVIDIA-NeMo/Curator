# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Waveform normalization shared by ASR model implementations."""

from __future__ import annotations

import numpy as np

_WAVEFORM_2D_NDIM = 2


def to_mono_numpy_1d(waveform: object) -> np.ndarray:
    """Convert a Curator waveform to contiguous mono float32 samples."""
    if waveform is None:
        return np.asarray([], dtype=np.float32)
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().numpy()
    samples = np.asarray(waveform, dtype=np.float32)
    if samples.size == 0:
        return samples.reshape(0)
    if samples.ndim == 0:
        return samples.reshape(1)
    if samples.ndim == 1:
        return np.ascontiguousarray(samples)

    squeezed = np.squeeze(samples)
    if squeezed.ndim == 1:
        return np.ascontiguousarray(squeezed.astype(np.float32, copy=False))
    if squeezed.ndim == _WAVEFORM_2D_NDIM:
        # Curator is channels-first. Accept channel-last inputs by treating the
        # smaller dimension as channels, then average channels to mono.
        channel_axis = 0 if squeezed.shape[0] <= squeezed.shape[1] else 1
        mono = squeezed.mean(axis=channel_axis)
        return np.ascontiguousarray(mono.astype(np.float32, copy=False))

    msg = f"Expected 1-D or 2-D waveform, got shape {samples.shape}"
    raise ValueError(msg)


def resample_waveform(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample mono samples only when source and model rates differ."""
    if source_rate <= 0 or target_rate <= 0:
        msg = f"Sample rates must be positive, got source={source_rate}, target={target_rate}"
        raise ValueError(msg)
    if source_rate == target_rate:
        return waveform

    # Keep optional/heavy audio dependency lazy for base-package imports.
    import librosa

    resampled = librosa.resample(waveform, orig_sr=source_rate, target_sr=target_rate)
    return np.ascontiguousarray(resampled, dtype=np.float32)
