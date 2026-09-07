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

"""Real Faster-Whisper dependency and single-GPU inference contracts."""

from __future__ import annotations

import inspect
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from nemo_curator.models.asr.faster_whisper import FasterWhisperASR, _faster_whisper_stack

pytestmark = pytest.mark.gpu

_MODEL_ID = "tiny.en"
_SAMPLE_RATE = 16000
_FIXTURE_PATH = Path(__file__).parents[2] / "fixtures/audio/qwen_omni/audio_1_5s_16khz_mono.wav"


def _load_short_fixture() -> np.ndarray:
    """Decode Curator's bundled five-second, 16 kHz mono WAV."""
    with wave.open(str(_FIXTURE_PATH), "rb") as wav_file:
        assert wav_file.getframerate() == _SAMPLE_RATE
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnframes() == 5 * _SAMPLE_RATE
        pcm = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")
    return np.ascontiguousarray(pcm.astype(np.float32) / 32768.0)


def _require_real_faster_whisper_stack() -> tuple[type, object]:
    try:
        __import__("ctranslate2")
        from faster_whisper import WhisperModel
        from faster_whisper.utils import download_model
    except ImportError as exc:
        pytest.fail(f"The Faster-Whisper GPU test environment is missing a required dependency: {exc}")

    resolved_model_class, resolved_download_model = _faster_whisper_stack()
    if resolved_model_class is not WhisperModel or resolved_download_model is not download_model:
        pytest.fail("Curator did not resolve the installed Faster-Whisper API")
    return WhisperModel, download_model


def test_faster_whisper_real_package_api_contract() -> None:
    """Exercise the external API surface that mocked CPU tests cannot validate."""
    model_class, download_model = _require_real_faster_whisper_stack()

    load_parameters = inspect.signature(model_class).parameters
    download_parameters = inspect.signature(download_model).parameters
    transcribe_parameters = inspect.signature(model_class.transcribe).parameters

    assert {"model_size_or_path", "device", "compute_type", "revision"} <= load_parameters.keys()
    assert {"size_or_id", "revision"} <= download_parameters.keys()
    assert {"audio", "language", "beam_size", "vad_filter", "without_timestamps"} <= transcribe_parameters.keys()


def test_faster_whisper_real_single_gpu_smoke() -> None:
    """Load the real model through Curator and transcribe one bundled sample."""
    _require_real_faster_whisper_stack()
    if torch.cuda.device_count() < 1:
        pytest.fail("Faster-Whisper smoke test requires one visible GPU")

    adapter = FasterWhisperASR(model_id=_MODEL_ID)
    adapter.load_model(num_gpus=1)
    try:
        results = adapter.transcribe_batch(
            [
                {
                    "waveform": _load_short_fixture(),
                    "sample_rate": _SAMPLE_RATE,
                    "language": "English",
                    "language_code": "en",
                    "task_id": "faster-whisper-gpu-smoke",
                }
            ]
        )
    finally:
        adapter.unload_model()

    assert len(results) == 1
    assert results[0].text.strip()
    assert results[0].skipped is False
