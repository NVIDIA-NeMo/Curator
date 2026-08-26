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

"""Real Qwen-Omni dependency and two-GPU inference contracts."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from nemo_curator.models.asr.qwen_omni import (
    Qwen3OmniMoeProcessor,
    QwenOmniASRAdapter,
    SamplingParams,
    process_mm_info,
)

pytestmark = pytest.mark.gpu

_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
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


def _require_real_qwen_stack() -> None:
    try:
        __import__("qwen_omni_utils")
        __import__("vllm")
    except ImportError as exc:
        pytest.fail(f"The Qwen-Omni GPU test environment is missing a required dependency: {exc}")
    if Qwen3OmniMoeProcessor is None or SamplingParams is None or process_mm_info is None:
        pytest.fail("The installed Qwen-Omni dependency stack is incomplete")


def test_qwen_omni_real_processor_multimodal_and_sampling_contract() -> None:
    """Exercise the external APIs that mocked CPU unit tests cannot validate."""
    _require_real_qwen_stack()
    waveform = _load_short_fixture()
    adapter = QwenOmniASRAdapter(model_id=_MODEL_ID, max_output_tokens=32)
    adapter._processor = Qwen3OmniMoeProcessor.from_pretrained(_MODEL_ID)

    messages = adapter._build_messages(waveform, "English")
    packed = adapter._pack_vllm_inputs(messages)
    sampling_params = SamplingParams(**adapter.sampling_kwargs, max_tokens=adapter.max_output_tokens)

    assert packed["prompt"]
    assert packed["multi_modal_data"]["audio"] is not None
    assert packed["mm_processor_kwargs"] == {"use_audio_in_video": False}
    assert sampling_params.max_tokens == 32


def test_qwen_omni_real_two_gpu_smoke() -> None:
    """Load the real model through Curator and transcribe one bundled sample."""
    _require_real_qwen_stack()
    if torch.cuda.device_count() < 2:
        pytest.fail("Qwen-Omni smoke test requires two visible GPUs")

    adapter = QwenOmniASRAdapter(
        model_id=_MODEL_ID,
        max_output_tokens=64,
        vllm_kwargs={
            "max_model_len": 32768,
            "max_num_seqs": 1,
            "gpu_memory_utilization": 0.95,
            "dtype": "auto",
            "trust_remote_code": True,
            "enable_prefix_caching": True,
            "prefix_caching_hash_algo": "xxhash",
            "limit_mm_per_prompt": {"image": 0, "video": 0, "audio": 1},
            "seed": 1234,
        },
    )
    adapter.load_model(num_gpus=2)
    try:
        results = adapter.transcribe_batch(
            [
                {
                    "waveform": _load_short_fixture(),
                    "sample_rate": _SAMPLE_RATE,
                    "language": "English",
                    "language_code": "en",
                    "task_id": "qwen-omni-gpu-smoke",
                }
            ]
        )
    finally:
        adapter.unload_model()

    assert len(results) == 1
    assert results[0].text.strip()
    assert results[0].skipped is False
