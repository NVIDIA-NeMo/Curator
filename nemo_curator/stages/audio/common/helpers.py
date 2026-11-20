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

import os
from typing import Any

import fsspec
import soundfile
import torch

# Longest audio any stage decodes; longer files are refused before their samples are read.
MAX_AUDIO_DURATION_SEC = 12 * 60 * 60


class AudioTooLongError(RuntimeError):
    """The file is longer than the decode limit. A RuntimeError, so stages skip the row like any unreadable file."""


def exceeds_max_duration(duration: Any, max_duration_sec: float | None = MAX_AUDIO_DURATION_SEC) -> bool:  # noqa: ANN401
    """Whether ``duration`` (seconds) is over the limit. ``None`` or a nonpositive limit disables it."""
    if max_duration_sec is None or max_duration_sec <= 0:
        return False
    try:
        return float(duration) > max_duration_sec
    except (TypeError, ValueError):
        return False


def load_audio_file(  # noqa: PLR0913
    audio_path: str,
    mono: bool = True,
    *,
    start: int = 0,
    stop: int | None = None,
    dtype: str = "float32",
    storage_options: dict[str, Any] | None = None,
    max_duration_sec: float | None = MAX_AUDIO_DURATION_SEC,
) -> tuple[torch.Tensor, int]:
    """Decode audio into ``(waveform [channels, samples], sample_rate)``. The one audio decoder.

    Opens local and cloud (``s3://``, ``gs://``) paths through fsspec. ``start``/``stop`` are
    frame indices for reading a slice. Raises :class:`AudioTooLongError`, from the header and
    before decoding, when the span to decode is longer than ``max_duration_sec``.
    """
    with (
        fsspec.open(str(audio_path), "rb", **(storage_options or {})) as f,
        soundfile.SoundFile(f) as audio,
    ):
        end = audio.frames if stop is None else min(stop, audio.frames)
        frames = max(end - start, 0)
        if exceeds_max_duration(frames / audio.samplerate, max_duration_sec):
            msg = f"{audio_path}: {frames / audio.samplerate:.0f}s to decode, over the {max_duration_sec}s limit"
            raise AudioTooLongError(msg)
        audio.seek(start)
        data = audio.read(frames, dtype=dtype, always_2d=True)
        sample_rate = audio.samplerate
    waveform = torch.from_numpy(data).T
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sample_rate


def resample_waveform(waveform: Any, orig_sr: int, target_sr: int) -> torch.Tensor:  # noqa: ANN401
    """Resample ``waveform`` (tensor or array, time last) to ``target_sr``. The one resampler."""
    waveform = torch.as_tensor(waveform)
    if int(orig_sr) == int(target_sr):
        return waveform
    import torchaudio

    return torchaudio.functional.resample(waveform, int(orig_sr), int(target_sr))


def ensure_waveform_2d(waveform: Any) -> torch.Tensor:  # noqa: ANN401
    """Ensure waveform is a torch.Tensor in 2D (channels, samples) format."""
    if not torch.is_tensor(waveform):
        waveform = torch.as_tensor(waveform, dtype=torch.float32)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    return waveform


def ensure_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Convert multi-channel waveform to mono. Assumes 2D (channels, samples) input."""
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def resolve_model_path(model_path: str, reference_file: str, module_subdir: str) -> str:
    """Resolve a relative model path using the reference file's directory and module subdirectory."""
    if os.path.isabs(model_path):
        return model_path
    current_dir = os.path.dirname(os.path.abspath(reference_file))
    module_dir = os.path.join(current_dir, module_subdir)
    for base in (module_dir, current_dir):
        resolved = os.path.join(base, model_path)
        if os.path.exists(resolved):
            return resolved
    return os.path.join(module_dir, model_path)
