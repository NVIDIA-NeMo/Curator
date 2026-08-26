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

"""Tests for the NeMo implementation of the shared ASR adapter."""

from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from nemo_curator.models.asr import nemo_asr
from nemo_curator.models.asr.base import ASRAdapter
from nemo_curator.models.asr.nemo_asr import NeMoASRAdapter
from nemo_curator.stages.audio.inference.asr.stage import ASRStage
from nemo_curator.stages.audio.inference.batch_policy import BatchPolicy
from nemo_curator.tasks import AudioTask

_MODEL_ID = "nvidia/stt_en_fastconformer_ctc_large"
_SAMPLE_RATE = 16_000
_FIXTURE_PATH = Path(__file__).parents[2] / "fixtures/audio/qwen_omni/audio_1_5s_16khz_mono.wav"


def _item(samples: int = _SAMPLE_RATE, *, sample_rate: int = _SAMPLE_RATE) -> dict[str, object]:
    return {
        "waveform": np.zeros(samples, dtype=np.float32),
        "sample_rate": sample_rate,
        "audio_seconds": float(samples) / float(sample_rate),
    }


def _mock_model(outputs: object) -> MagicMock:
    model = MagicMock()
    model.preprocessor._sample_rate = _SAMPLE_RATE
    model.transcribe.return_value = outputs
    return model


def _load_fixture() -> np.ndarray:
    with wave.open(str(_FIXTURE_PATH), "rb") as wav_file:
        assert wav_file.getframerate() == _SAMPLE_RATE
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        pcm = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")
    return np.ascontiguousarray(pcm.astype(np.float32) / 32768.0)


def test_nemo_adapter_conforms_to_asr_protocol() -> None:
    assert isinstance(NeMoASRAdapter(), ASRAdapter)


def test_download_weights_uses_adapter_model_id() -> None:
    nemo_asr = MagicMock()
    adapter = NeMoASRAdapter(model_id="nvidia/stt_en_fastconformer_ctc_large")
    with patch("nemo_curator.models.asr.nemo_asr._nemo_asr_module", return_value=nemo_asr):
        adapter.download_weights_on_node()

    nemo_asr.models.ASRModel.from_pretrained.assert_called_once_with(
        model_name="nvidia/stt_en_fastconformer_ctc_large",
        return_model_file=True,
    )


def test_load_model_uses_stage_owned_gpu_count_and_is_idempotent() -> None:
    adapter = NeMoASRAdapter()
    model = _mock_model([])

    with patch.object(adapter, "_load_checkpoint", return_value=model) as load:
        adapter.load_model(num_gpus=0)
        adapter.load_model(num_gpus=0)

    assert adapter._model is model
    assert load.call_count == 1
    assert load.call_args.args[0].type == "cpu"


@pytest.mark.parametrize("num_gpus", [-1, 1.5, 2, True])
def test_load_model_rejects_invalid_worker_gpu_counts(num_gpus: object) -> None:
    adapter = NeMoASRAdapter()

    with pytest.raises(ValueError, match="requires num_gpus to be 0 or 1"):
        adapter.load_model(num_gpus=num_gpus)  # type: ignore[arg-type]


def test_load_model_configures_local_attention_when_enabled() -> None:
    adapter = NeMoASRAdapter(enable_local_attention=True, local_attention_context_size=(64, 96))
    model = _mock_model([])

    with patch.object(adapter, "_load_checkpoint", return_value=model):
        adapter.load_model(num_gpus=0)

    model.change_attention_model.assert_called_once_with(
        self_attention_model="rel_pos_local_attn",
        att_context_size=[64, 96],
    )
    model.change_subsampling_conv_chunking_factor.assert_called_once_with(1)


@pytest.mark.parametrize("enabled", [False, True])
def test_load_model_configures_rnnt_cuda_graph_decoder_when_requested(enabled: bool) -> None:
    adapter = NeMoASRAdapter(use_cuda_graph_decoder=enabled)
    model = _mock_model([])
    model.cfg = OmegaConf.create({"decoding": {"strategy": "greedy_batch", "greedy": {}}})

    with patch.object(adapter, "_load_checkpoint", return_value=model):
        adapter.load_model(num_gpus=0)

    decoding_cfg = model.change_decoding_strategy.call_args.kwargs["decoding_cfg"]
    assert decoding_cfg.strategy == "greedy_batch"
    assert decoding_cfg.greedy.use_cuda_graph_decoder is enabled


def test_transcribe_batch_uses_one_exact_nemo_batch() -> None:
    model = _mock_model([SimpleNamespace(text="alpha"), SimpleNamespace(text="beta")])
    adapter = NeMoASRAdapter(num_workers=2)
    adapter._model = model

    results = adapter.transcribe_batch([_item(), _item(samples=2 * _SAMPLE_RATE)])

    assert [result.text for result in results] == ["alpha", "beta"]
    assert all(not result.skipped for result in results)
    kwargs = model.transcribe.call_args.kwargs
    assert kwargs["batch_size"] == 2
    assert kwargs["num_workers"] == 2
    assert len(kwargs["audio"]) == 2


def test_fastconformer_adapter_cost_estimator_prefers_explicit_model_cost() -> None:
    adapter = NeMoASRAdapter()

    assert adapter.estimate_item_cost({"estimated_vram_units": 7, "audio_seconds": 2.0}) == 7.0
    assert adapter.estimate_item_cost({"estimated_encoder_tokens": 5, "audio_seconds": 2.0}) == 5.0
    assert adapter.estimate_item_cost({"audio_seconds": 2.0}) == 2.0
    assert adapter.estimate_item_cost({"audio_seconds": -2.0}) == 0.0
    assert adapter.estimate_item_cost({}) is None


def test_local_policy_outputs_are_exact_fastconformer_model_batches() -> None:
    model = _mock_model([])
    model.transcribe.side_effect = [
        ["long"],
        ["short-a", "short-b"],
    ]
    adapter = NeMoASRAdapter()
    adapter._model = model
    stage = ASRStage(
        adapter_target="nemo_curator.models.asr.nemo_asr.NeMoASRAdapter",
        model_id=_MODEL_ID,
        waveform_key="waveform",
        sample_rate_key="sampling_rate",
        keep_waveform=True,
        batch_policy=BatchPolicy(
            buckets_sec=[0, 2],
            max_items_per_batch_by_bucket=[4, 4],
            max_audio_sec_per_batch=None,
        ),
    )
    stage._adapter = adapter
    tasks = [
        AudioTask(data={"waveform": np.zeros(_SAMPLE_RATE, dtype=np.float32), "sampling_rate": _SAMPLE_RATE}),
        AudioTask(data={"waveform": np.zeros(4 * _SAMPLE_RATE, dtype=np.float32), "sampling_rate": _SAMPLE_RATE}),
        AudioTask(
            data={"waveform": np.zeros(int(1.5 * _SAMPLE_RATE), dtype=np.float32), "sampling_rate": _SAMPLE_RATE}
        ),
    ]

    results = stage.process_batch(tasks)

    assert [call.kwargs["batch_size"] for call in model.transcribe.call_args_list] == [1, 2]
    assert [len(call.kwargs["audio"]) for call in model.transcribe.call_args_list] == [1, 2]
    assert [task.data["pred_text"] for task in results] == ["short-a", "long", "short-b"]


def test_transcribe_batch_preserves_empty_positions() -> None:
    model = _mock_model(["valid"])
    adapter = NeMoASRAdapter()
    adapter._model = model

    results = adapter.transcribe_batch([_item(samples=0), _item()])

    assert [result.text for result in results] == ["", "valid"]
    assert [result.skipped for result in results] == [True, False]
    assert results[0].skip_reason == "empty_audio"


def test_transcribe_batch_requires_upstream_resampling() -> None:
    adapter = NeMoASRAdapter()
    adapter._model = _mock_model([])

    with pytest.raises(ValueError, match="ASRStage must provide 16000 Hz"):
        adapter.transcribe_batch([_item(sample_rate=8_000)])


def test_transcribe_batch_requires_upstream_mono_conversion() -> None:
    adapter = NeMoASRAdapter()
    adapter._model = _mock_model([])
    item = _item()
    item["waveform"] = np.zeros((1, _SAMPLE_RATE), dtype=np.float32)

    with pytest.raises(ValueError, match="mono 1-D waveform"):
        adapter.transcribe_batch([item])


@pytest.mark.parametrize(
    ("outputs", "expected"),
    [
        (([SimpleNamespace(text="tuple")], None), ["tuple"]),
        ([[SimpleNamespace(text="nested")]], ["nested"]),
        (["plain"], ["plain"]),
    ],
)
def test_extract_transcription_texts_matches_nemo_output_shapes(outputs: object, expected: list[str]) -> None:
    assert nemo_asr._extract_nemo_transcription_texts(outputs) == expected


@pytest.mark.gpu
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
