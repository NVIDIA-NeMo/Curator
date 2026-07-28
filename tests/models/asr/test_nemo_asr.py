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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nemo_curator.models.asr.base import ASRAdapter
from nemo_curator.models.asr.nemo_asr import NeMoASRAdapter

_SAMPLE_RATE = 16_000


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


def test_nemo_adapter_conforms_to_asr_protocol() -> None:
    assert isinstance(NeMoASRAdapter(), ASRAdapter)


def test_download_weights_uses_shared_classmethod_contract() -> None:
    nemo_asr = MagicMock()
    with patch("nemo_curator.models.asr.nemo_asr._nemo_asr_module", return_value=nemo_asr):
        NeMoASRAdapter.download_weights_on_node("nvidia/stt_en_fastconformer_ctc_large")

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
def test_normalize_transcriptions_matches_nemo_output_shapes(outputs: object, expected: list[str]) -> None:
    assert NeMoASRAdapter._normalize_transcriptions(outputs) == expected
