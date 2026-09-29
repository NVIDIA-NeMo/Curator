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

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from nemo_curator.models.audio.vad.base import VADSegment
from nemo_curator.models.audio.vad.silero import SileroVADAdapter


def _install_silero_module(monkeypatch: pytest.MonkeyPatch, **members: object) -> None:
    monkeypatch.setitem(sys.modules, "silero_vad", SimpleNamespace(**members))


def test_defaults_preserve_the_existing_silero_stage_contract() -> None:
    adapter = SileroVADAdapter()
    assert adapter.backend == "torch"
    assert adapter.threshold == 0.5
    assert adapter.min_duration_sec == 2.0
    assert adapter.max_duration_sec == 60.0
    assert adapter.min_interval_ms == 500
    assert adapter.speech_pad_ms == 300


@pytest.mark.parametrize("backend", ["openvino", "invalid"])
def test_unknown_runtime_is_rejected(backend: str) -> None:
    with pytest.raises(ValueError, match="Unsupported Silero backend"):
        SileroVADAdapter(backend=backend)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"threshold": 1.1}, "threshold"),
        ({"min_duration_sec": -1}, "min_duration_sec"),
        ({"min_duration_sec": 2, "max_duration_sec": 2}, "max_duration_sec"),
        ({"min_interval_ms": -1}, "min_interval_ms"),
        ({"speech_pad_ms": -1}, "speech_pad_ms"),
    ],
)
def test_invalid_detection_options_are_rejected(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SileroVADAdapter(**kwargs)


def test_prefetch_is_a_noop_for_wheel_bundled_weights() -> None:
    adapter = SileroVADAdapter()
    adapter.download_weights_on_node()
    assert adapter._model is None


def test_torch_cpu_loads_the_official_model_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    model = MagicMock()
    load = MagicMock(return_value=model)
    _install_silero_module(monkeypatch, load_silero_vad=load)
    adapter = SileroVADAdapter()

    adapter.load_model(num_gpus=0)

    load.assert_called_once_with()
    model.to.assert_not_called()
    assert adapter._device == torch.device("cpu")


def test_torch_gpu_moves_the_model_to_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    model = MagicMock()
    model.to.return_value = model
    load = MagicMock(return_value=model)
    _install_silero_module(monkeypatch, load_silero_vad=load)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    adapter = SileroVADAdapter()

    adapter.load_model(num_gpus=1)

    model.to.assert_called_once_with(torch.device("cuda"))
    assert adapter._device == torch.device("cuda")


def test_torch_gpu_rejects_a_worker_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    load = MagicMock()
    _install_silero_module(monkeypatch, load_silero_vad=load)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    adapter = SileroVADAdapter()

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        adapter.load_model(num_gpus=1)
    load.assert_not_called()


def test_onnx_uses_the_official_cpu_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    model = MagicMock()
    load = MagicMock(return_value=model)
    _install_silero_module(monkeypatch, load_silero_vad=load)
    adapter = SileroVADAdapter(backend="onnx")

    adapter.load_model(num_gpus=0)

    load.assert_called_once_with(onnx=True, opset_version=16)
    model.to.assert_not_called()
    assert adapter._device == torch.device("cpu")


def test_onnx_rejects_a_gpu_resource_request(monkeypatch: pytest.MonkeyPatch) -> None:
    load = MagicMock()
    _install_silero_module(monkeypatch, load_silero_vad=load)
    adapter = SileroVADAdapter(backend="onnx")

    with pytest.raises(ValueError, match="must be 0"):
        adapter.load_model(num_gpus=1)
    load.assert_not_called()


def test_more_than_one_gpu_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    load = MagicMock()
    _install_silero_module(monkeypatch, load_silero_vad=load)
    with pytest.raises(ValueError, match="0 or 1"):
        SileroVADAdapter().load_model(num_gpus=2)
    load.assert_not_called()


def test_supported_rates_preserve_official_decimation_behavior_and_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def get_timestamps(waveform: torch.Tensor, model: object, **kwargs: object) -> list[dict[str, int]]:
        calls.append({"waveform": waveform, "model": model, **kwargs})
        rate = int(kwargs["sampling_rate"])
        return [{"start": rate, "end": 2 * rate}]

    _install_silero_module(monkeypatch, get_speech_timestamps=get_timestamps)
    adapter = SileroVADAdapter(
        threshold=0.6,
        min_duration_sec=1.25,
        max_duration_sec=30.0,
        min_interval_ms=250,
        speech_pad_ms=100,
    )
    adapter._model = object()
    adapter._device = torch.device("cpu")

    results = adapter.detect_batch(
        [
            {"waveform": np.zeros(8000, dtype=np.float32), "sample_rate": 8000},
            {"waveform": np.zeros(48000, dtype=np.float32), "sample_rate": 48000},
        ]
    )

    assert [call["sampling_rate"] for call in calls] == [8000, 48000]
    assert [len(call["waveform"]) for call in calls] == [8000, 48000]
    assert results[0].segments == [VADSegment(1.0, 2.0)]
    assert results[1].segments == [VADSegment(1.0, 2.0)]
    assert calls[0]["threshold"] == 0.6
    assert calls[0]["min_speech_duration_ms"] == 1250.0
    assert calls[0]["max_speech_duration_s"] == 30.0
    assert calls[0]["min_silence_duration_ms"] == 250
    assert calls[0]["speech_pad_ms"] == 100


def test_an_unsupported_rate_is_resampled_to_16khz(monkeypatch: pytest.MonkeyPatch) -> None:
    resample_calls: list[tuple[int, int, tuple[int, ...]]] = []

    class _Resample:
        def __init__(self, *, orig_freq: int, new_freq: int) -> None:
            self.orig_freq = orig_freq
            self.new_freq = new_freq

        def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
            resample_calls.append((self.orig_freq, self.new_freq, tuple(waveform.shape)))
            return torch.zeros((1, 160), dtype=torch.float32)

    get_timestamps = MagicMock(return_value=[{"start": 0, "end": 160}])
    _install_silero_module(monkeypatch, get_speech_timestamps=get_timestamps)
    monkeypatch.setitem(
        sys.modules,
        "torchaudio",
        SimpleNamespace(transforms=SimpleNamespace(Resample=_Resample)),
    )
    adapter = SileroVADAdapter()
    adapter._model = object()
    adapter._device = torch.device("cpu")

    result = adapter.detect_batch([{"waveform": np.zeros(441, dtype=np.float32), "sample_rate": 44100}])[0]

    assert resample_calls == [(44100, 16000, (1, 441))]
    assert get_timestamps.call_args.kwargs["sampling_rate"] == 16000
    assert result.segments == [VADSegment(0.0, 0.01)]


def test_postprocessing_failure_is_isolated_to_one_recording(monkeypatch: pytest.MonkeyPatch) -> None:
    def get_timestamps(waveform: torch.Tensor, _model: object, **_kwargs: object) -> list[dict[str, int]]:
        if len(waveform) == 16000:
            message = "bad recording"
            raise RuntimeError(message)
        return [{"start": 0, "end": len(waveform)}]

    _install_silero_module(monkeypatch, get_speech_timestamps=get_timestamps)
    adapter = SileroVADAdapter()
    adapter._model = object()
    adapter._device = torch.device("cpu")

    results = adapter.detect_batch(
        [
            {"waveform": np.zeros(8000, dtype=np.float32), "sample_rate": 8000},
            {"waveform": np.zeros(16000, dtype=np.float32), "sample_rate": 16000},
            {"waveform": np.zeros(32000, dtype=np.float32), "sample_rate": 16000},
        ]
    )

    assert results[0].segments == [VADSegment(0.0, 1.0)]
    assert results[1].segments == []
    assert results[1].error == "RuntimeError: bad recording"
    assert results[2].segments == [VADSegment(0.0, 2.0)]


@pytest.mark.parametrize(
    ("item", "error"),
    [
        ({"sample_rate": 16000}, KeyError),
        ({"waveform": np.zeros(10)}, KeyError),
        ({"waveform": np.zeros((2, 10)), "sample_rate": 16000}, ValueError),
        ({"waveform": np.zeros(10), "sample_rate": 0}, ValueError),
    ],
)
def test_invalid_items_are_rejected(
    item: dict[str, object], error: type[Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_silero_module(monkeypatch, get_speech_timestamps=MagicMock())
    adapter = SileroVADAdapter()
    adapter._model = object()
    adapter._device = torch.device("cpu")
    with pytest.raises(error):
        adapter.detect_batch([item])


def test_empty_batch_does_not_require_a_loaded_model() -> None:
    assert SileroVADAdapter().detect_batch([]) == []


def test_nonempty_batch_requires_a_loaded_model() -> None:
    with pytest.raises(RuntimeError, match="not loaded"):
        SileroVADAdapter().detect_batch([{"waveform": np.zeros(10), "sample_rate": 16000}])


def test_unload_releases_cuda_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    empty_cache = MagicMock()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    adapter = SileroVADAdapter()
    adapter._model = object()
    adapter._device = torch.device("cuda")

    adapter.unload_model()

    assert adapter._model is None
    assert adapter._device is None
    empty_cache.assert_called_once_with()
