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

"""Tests for the NeMo Framework implementation of the shared ASR adapter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nemo_curator.models.asr.base import ASRAdapter
from nemo_curator.models.asr.nemo_asr import NeMoASRAdapter
from nemo_curator.stages.audio.inference.asr import ASRStage
from nemo_curator.tasks import AudioTask

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


def test_nemo_adapter_rejects_unsupported_revision() -> None:
    with pytest.raises(ValueError, match="does not support revision"):
        NeMoASRAdapter(revision="main")


def test_prefetch_downloads_without_constructing_model() -> None:
    nemo_asr = MagicMock()
    with patch("nemo_curator.models.asr.nemo_asr._nemo_asr_module", return_value=nemo_asr):
        NeMoASRAdapter.prefetch_weights("nvidia/stt_en_fastconformer_ctc_large")

    nemo_asr.models.ASRModel.from_pretrained.assert_called_once_with(
        model_name="nvidia/stt_en_fastconformer_ctc_large",
        return_model_file=True,
    )


def test_setup_loads_one_worker_local_model_and_is_idempotent() -> None:
    nemo_asr = MagicMock()
    model = _mock_model([])
    nemo_asr.models.ASRModel.from_pretrained.return_value = model
    adapter = NeMoASRAdapter(device="cpu")

    with patch("nemo_curator.models.asr.nemo_asr._nemo_asr_module", return_value=nemo_asr):
        adapter.setup()
        adapter.setup()

    assert adapter._model is model
    assert nemo_asr.models.ASRModel.from_pretrained.call_count == 1
    kwargs = nemo_asr.models.ASRModel.from_pretrained.call_args.kwargs
    assert kwargs["model_name"] == adapter.model_id
    assert kwargs["map_location"].type == "cpu"


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
    assert "use_lhotse" not in kwargs
    assert len(kwargs["audio"]) == 2
    assert adapter.last_metrics["transcribe_calls"] == 1.0
    assert adapter.last_metrics["requested_batch_size"] == 2.0


def test_transcribe_batch_preserves_skipped_positions() -> None:
    model = _mock_model(["valid"])
    adapter = NeMoASRAdapter()
    adapter._model = model

    results = adapter.transcribe_batch([_item(samples=0), _item(), {"waveform": [1.0], "sample_rate": 0}])

    assert [result.text for result in results] == ["", "valid", ""]
    assert [result.skipped for result in results] == [True, False, True]
    assert model.transcribe.call_args.kwargs["batch_size"] == 1
    assert adapter.last_metrics["utterances_skipped_preprocess"] == 2.0


def test_transcribe_batch_resamples_to_model_rate() -> None:
    model = _mock_model(["resampled"])
    model.preprocessor._sample_rate = 8_000
    adapter = NeMoASRAdapter()
    adapter._model = model
    resampled = np.zeros(8_000, dtype=np.float32)

    with patch("nemo_curator.models.asr.nemo_asr.resample_waveform", return_value=resampled) as resample:
        adapter.transcribe_batch([_item()])

    resample.assert_called_once()
    assert resample.call_args.args[1:] == (_SAMPLE_RATE, 8_000)
    assert model.transcribe.call_args.kwargs["audio"][0] is resampled


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


def test_transcribe_batch_rejects_output_count_mismatch() -> None:
    adapter = NeMoASRAdapter()
    adapter._model = _mock_model(["only one"])

    with pytest.raises(RuntimeError, match="1 transcriptions for 2 valid inputs"):
        adapter.transcribe_batch([_item(), _item()])


def test_asr_stage_drives_nemo_adapter_end_to_end() -> None:
    model = _mock_model(["one", "two"])
    nemo_asr = MagicMock()
    nemo_asr.models.ASRModel.from_pretrained.return_value = model
    stage = ASRStage(
        adapter_target="nemo_curator.models.asr.NeMoASRAdapter",
        model_id="nvidia/stt_en_fastconformer_ctc_large",
        batch_size=2,
        adapter_batch_size=2,
        max_inference_duration_s=60.0,
    )

    with (
        patch("hydra.utils.get_class", return_value=NeMoASRAdapter),
        patch("nemo_curator.models.asr.nemo_asr._nemo_asr_module", return_value=nemo_asr),
    ):
        stage.setup()

    tasks = [
        AudioTask(data={"waveform": np.zeros(_SAMPLE_RATE), "sample_rate": _SAMPLE_RATE}),
        AudioTask(data={"waveform": np.zeros(2 * _SAMPLE_RATE), "sample_rate": _SAMPLE_RATE}),
    ]
    results = stage.process_batch(tasks)

    assert [task.data["pred_text"] for task in results] == ["one", "two"]
    assert model.transcribe.call_args.kwargs["batch_size"] == 2
