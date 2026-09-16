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

"""Private validation and resident-audio helpers for audio metrics stages."""

from __future__ import annotations

import math
from collections.abc import MutableMapping
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from nemo_curator.stages.audio._agent._residency import InputResidency

_SUPPORTED_TORCH_PCM_DTYPES = {
    torch.int16: 32768.0,
    torch.int32: 2147483648.0,
}
_SUPPORTED_PCM_WIDTHS = frozenset({2, 4})
_CHANNEL_FIRST_NDIM = 2


def validate_metric_keys(
    stage_name: str,
    *,
    keys: dict[str, Any],
    strict_fields: tuple[str, ...],
) -> None:
    """Validate keys introduced with agent-ready metrics configuration.

    Legacy key fields deliberately remain permissive: empty literals and
    in-place/cross-scope aliases were accepted before these stages became
    agent-ready. Same-scope incompatibilities are rejected by
    :func:`metrics_mapping` on the runtime branch that actually writes them.
    """
    for field_name in strict_fields:
        key = keys[field_name]
        if not isinstance(key, str) or not key.strip():
            msg = f"[{stage_name}] '{field_name}' must be a non-empty string"
            raise ValueError(msg)



def resident_sample_rate(value: Any, *, sample_rate_key: str, stage_name: str) -> int:  # noqa: ANN401
    """Return a positive integral resident sample rate without lossy coercion."""
    if torch.is_tensor(value) and value.ndim == 0:
        value = value.item()

    rate: int | None = None
    if isinstance(value, (bool, np.bool_)):
        rate = None
    elif isinstance(value, str):
        try:
            rate = int(value)
        except ValueError:
            rate = None
    elif isinstance(value, Integral):
        rate = int(value)
    elif isinstance(value, Real):
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            rate = int(numeric)

    if rate is None or rate <= 0:
        msg = (
            f"[{stage_name}] Resident sample rate '{sample_rate_key}' must be a positive, "
            f"losslessly integral, non-boolean value; got {value!r}"
        )
        raise ValueError(msg)
    return rate


def resident_pair_is_complete(
    item: dict[str, Any],
    *,
    residency: InputResidency,
    waveform_key: str,
    sample_rate_key: str,
    stage_name: str,
) -> bool:
    """Return whether both resident values exist, rejecting a partial usable pair.

    File mode deliberately ignores resident values to preserve its historical
    file-only behavior. Waveform and auto modes reject a waveform without its
    sample rate (or vice versa) instead of falling back to a potentially
    different file while stale resident state remains on the row.
    """
    if residency == "file":
        return False

    has_waveform = item.get(waveform_key) is not None
    has_sample_rate = item.get(sample_rate_key) is not None
    if has_waveform != has_sample_rate:
        present = waveform_key if has_waveform else sample_rate_key
        missing = sample_rate_key if has_waveform else waveform_key
        msg = (
            f"[{stage_name}] Incomplete resident audio for input_residency={residency!r}: "
            f"'{present}' is present but '{missing}' is missing"
        )
        raise ValueError(msg)
    if has_sample_rate:
        resident_sample_rate(item[sample_rate_key], sample_rate_key=sample_rate_key, stage_name=stage_name)
    return has_waveform


def resident_pcm_to_mono_float32(waveform: Any, *, stage_name: str) -> np.ndarray:  # noqa: ANN401
    """Convert channel-first resident audio to file-loader-equivalent mono float32.

    Floating NumPy arrays and Torch tensors are cast to float32. Signed PCM
    int16/int32 inputs are divided by 2**(bits-1), matching the amplitude
    convention used when librosa loads integer PCM files. Mono reduction happens
    only after that conversion. Other dtypes and dimensions are rejected.
    """
    if torch.is_tensor(waveform):
        if waveform.ndim not in {1, 2}:
            msg = f"[{stage_name}] Resident waveform must be 1-D or 2-D (channels, samples), got {waveform.ndim}-D"
            raise ValueError(msg)
        detached = waveform.detach().cpu()
        if detached.is_floating_point():
            audio = detached.to(dtype=torch.float32).numpy()
        elif detached.dtype in _SUPPORTED_TORCH_PCM_DTYPES:
            audio = detached.to(dtype=torch.float32).numpy() / _SUPPORTED_TORCH_PCM_DTYPES[detached.dtype]
        else:
            msg = (
                f"[{stage_name}] Unsupported resident waveform dtype {detached.dtype}; "
                "expected a floating dtype or signed PCM int16/int32"
            )
            raise TypeError(msg)
    else:
        try:
            array = np.asarray(waveform)
        except Exception as ex:
            msg = f"[{stage_name}] Resident waveform must be convertible to a NumPy array"
            raise TypeError(msg) from ex
        if array.ndim not in {1, 2}:
            msg = f"[{stage_name}] Resident waveform must be 1-D or 2-D (channels, samples), got {array.ndim}-D"
            raise ValueError(msg)
        if np.issubdtype(array.dtype, np.floating):
            audio = array.astype(np.float32, copy=False)
        elif np.issubdtype(array.dtype, np.signedinteger) and array.dtype.itemsize in _SUPPORTED_PCM_WIDTHS:
            scale = float(-(np.iinfo(array.dtype).min))
            audio = array.astype(np.float32) / scale
        else:
            msg = (
                f"[{stage_name}] Unsupported resident waveform dtype {array.dtype}; "
                "expected a floating dtype or signed PCM int16/int32"
            )
            raise TypeError(msg)

    if audio.ndim == _CHANNEL_FIRST_NDIM:
        audio = audio.mean(axis=0, dtype=np.float32)
    return np.asarray(audio, dtype=np.float32)


def metrics_mapping(item: dict[str, Any], *, metrics_key: str, stage_name: str) -> MutableMapping[str, Any]:
    """Return a mutable metrics mapping without inserting a missing container."""
    if metrics_key not in item:
        return {}
    metrics = item[metrics_key]
    if not isinstance(metrics, MutableMapping):
        msg = (
            f"[{stage_name}] Existing metrics container '{metrics_key}' must be a MutableMapping, "
            f"got {type(metrics).__name__}"
        )
        raise TypeError(msg)
    return metrics
