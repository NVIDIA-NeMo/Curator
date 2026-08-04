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

"""Real NeMo FastConformer adapter smoke test on a bundled audio fixture."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from nemo_curator.models.asr.nemo_asr import NeMoASRAdapter

pytestmark = pytest.mark.gpu

_MODEL_ID = "nvidia/stt_en_fastconformer_ctc_large"
_SAMPLE_RATE = 16_000
_FIXTURE_PATH = Path(__file__).parents[2] / "fixtures/audio/qwen_omni/audio_1_5s_16khz_mono.wav"


def _load_fixture() -> np.ndarray:
    with wave.open(str(_FIXTURE_PATH), "rb") as wav_file:
        assert wav_file.getframerate() == _SAMPLE_RATE
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        pcm = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")
    return np.ascontiguousarray(pcm.astype(np.float32) / 32768.0)


def test_nemo_fastconformer_real_one_gpu_smoke() -> None:
    """Load the default model and transcribe one existing five-second WAV."""
    if torch.cuda.device_count() < 1:
        pytest.fail("NeMo FastConformer smoke test requires one visible GPU")

    adapter = NeMoASRAdapter(model_id=_MODEL_ID)
    adapter.load_model(num_gpus=1)
    try:
        results = adapter.transcribe_batch(
            [
                {
                    "waveform": _load_fixture(),
                    "sample_rate": _SAMPLE_RATE,
                    "language": "English",
                    "language_code": "en",
                    "task_id": "nemo-fastconformer-gpu-smoke",
                }
            ]
        )
    finally:
        adapter.unload_model()

    assert len(results) == 1
    assert results[0].text.strip()
    assert results[0].skipped is False
