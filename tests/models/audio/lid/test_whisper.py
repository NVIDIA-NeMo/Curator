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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.nn import functional

from nemo_curator.models.audio.lid.base import AudioLIDResult
from nemo_curator.models.audio.lid.whisper import WhisperLIDAdapter, WhisperTensorRTEncoder


class _FakeWhisperAudio:
    N_FFT = 400
    HOP_LENGTH = 160

    @staticmethod
    def mel_filters(device: torch.device, n_mels: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(7)
        return torch.rand(n_mels, _FakeWhisperAudio.N_FFT // 2 + 1, generator=generator).to(device)


def _pad_or_trim(audio: torch.Tensor, length: int = 3_200) -> torch.Tensor:
    if audio.shape[-1] > length:
        return audio[..., :length]
    return functional.pad(audio, (0, length - audio.shape[-1]))


class _FakeWhisperModel:
    def __init__(self) -> None:
        self.dims = SimpleNamespace(n_mels=80)
        self.last_input: torch.Tensor | None = None

    def detect_language(self, model_input: torch.Tensor) -> tuple[torch.Tensor, list[dict[str, float]]]:
        self.last_input = model_input
        probabilities = [{"en": 0.9, "fr": 0.1}, {"de": 0.8, "en": 0.2}]
        return torch.tensor([1, 2]), probabilities[: model_input.shape[0]]


class _FakeTensorRTSession:
    def __init__(self, max_batch: int = 2, min_batch: int = 1) -> None:
        self.input_names = ["mel"]
        self.output_names = ["audio_features"]
        self.max_batch = max_batch
        self.min_batch = min_batch
        self.calls: list[tuple[int, ...]] = []
        self.closed = False

    def input_shape_range(self, name: str) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
        assert name == "mel"
        return (self.min_batch, 80, 10), (self.max_batch, 80, 10), (self.max_batch, 80, 10)

    def infer(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mel = inputs["mel"]
        self.calls.append(tuple(mel.shape))
        call_number = len(self.calls)
        return {"audio_features": torch.full((mel.shape[0], 4, 8), float(call_number))}

    def close(self) -> None:
        self.closed = True


def test_tensorrt_encoder_splits_at_profile_batch_and_closes_session() -> None:
    session = _FakeTensorRTSession(max_batch=2)
    encoder = WhisperTensorRTEncoder("unused.plan", session=session)

    features = encoder(torch.zeros(5, 80, 10))

    assert session.calls == [(2, 80, 10), (2, 80, 10), (1, 80, 10)]
    assert features[:, 0, 0].tolist() == [1.0, 1.0, 2.0, 2.0, 3.0]
    encoder.close()
    assert session.closed


def test_tensorrt_encoder_pads_tail_to_profile_minimum_and_trims_output() -> None:
    session = _FakeTensorRTSession(min_batch=2, max_batch=4)
    encoder = WhisperTensorRTEncoder("unused.plan", session=session)

    features = encoder(torch.zeros(5, 80, 10))

    assert session.calls == [(4, 80, 10), (2, 80, 10)]
    assert tuple(features.shape) == (5, 4, 8)


@pytest.mark.parametrize(
    ("input_names", "output_names", "missing"),
    [([], ["audio_features"], "mel"), (["mel"], [], "audio_features")],
)
def test_tensorrt_encoder_validates_tensor_names(
    input_names: list[str], output_names: list[str], missing: str
) -> None:
    session = _FakeTensorRTSession()
    session.input_names = input_names
    session.output_names = output_names

    with pytest.raises(ValueError, match=missing):
        WhisperTensorRTEncoder("unused.plan", session=session)


def test_tensorrt_encoder_closes_owned_session_when_validation_fails() -> None:
    session = _FakeTensorRTSession()
    session.input_names = []

    with (
        patch(
            "nemo_curator.stages.audio.inference.tensorrt_encoder.TensorRTEncoderSession",
            return_value=session,
        ),
        pytest.raises(ValueError, match="mel"),
    ):
        WhisperTensorRTEncoder("invalid.plan")

    assert session.closed


def test_tensorrt_encoder_validates_exact_mel_shape() -> None:
    encoder = WhisperTensorRTEncoder("unused.plan", session=_FakeTensorRTSession())

    with pytest.raises(ValueError, match=r"expects mel \[\*, 80, 10\]"):
        encoder(torch.zeros(2, 80, 11))


def test_batched_log_mel_normalizes_every_sample_independently() -> None:
    whisper = SimpleNamespace(audio=_FakeWhisperAudio)
    adapter = WhisperLIDAdapter()
    adapter._device = torch.device("cpu")
    generator = torch.Generator().manual_seed(11)
    audio = torch.stack(
        [
            torch.randn(3_200, generator=generator) * 0.01,
            torch.randn(3_200, generator=generator),
        ]
    )

    batched = adapter._log_mel_spectrogram(audio, 80, whisper)
    separate = torch.cat([adapter._log_mel_spectrogram(row[None, :], 80, whisper) for row in audio])

    torch.testing.assert_close(batched, separate)


def test_identify_batch_uses_model_dtype_and_preserves_empty_order() -> None:
    whisper = SimpleNamespace(audio=_FakeWhisperAudio, pad_or_trim=_pad_or_trim)
    adapter = WhisperLIDAdapter(model_batch_size=2)
    adapter._device = torch.device("cpu")
    adapter._mel_dtype = torch.float16
    adapter._model = _FakeWhisperModel()
    items = [
        {"waveform": np.ones(1_600, dtype=np.float32)},
        {"waveform": np.array([], dtype=np.float32)},
        {"waveform": np.ones(2_400, dtype=np.float32) * 0.25},
    ]

    with patch("nemo_curator.models.audio.lid.whisper._import_whisper", return_value=whisper):
        results = adapter.identify_batch(items)

    assert [(result.language, result.confidence) for result in results] == [
        ("en", 0.9),
        ("", 0.0),
        ("de", 0.8),
    ]
    assert adapter._model.last_input is not None
    assert adapter._model.last_input.dtype == torch.float16


def test_identify_batch_passes_tensorrt_features_to_decoder() -> None:
    whisper = SimpleNamespace(audio=_FakeWhisperAudio, pad_or_trim=_pad_or_trim)
    adapter = WhisperLIDAdapter(model_batch_size=2)
    adapter._device = torch.device("cpu")
    adapter._model = _FakeWhisperModel()
    adapter._encoder = WhisperTensorRTEncoder("unused.plan", session=_FakeTensorRTSession(max_batch=4))

    with (
        patch("nemo_curator.models.audio.lid.whisper._import_whisper", return_value=whisper),
        patch.object(adapter, "_log_mel_spectrogram", return_value=torch.zeros(1, 80, 10)),
    ):
        adapter.identify_batch([{"waveform": np.ones(800, dtype=np.float32)}])

    assert adapter._model.last_input is not None
    assert tuple(adapter._model.last_input.shape) == (1, 4, 8)


def test_tensorrt_backend_requires_engine_path_and_one_gpu() -> None:
    with pytest.raises(ValueError, match="tensorrt_engine_path"):
        WhisperLIDAdapter(backend="tensorrt")

    adapter = WhisperLIDAdapter(backend="tensorrt", tensorrt_engine_path="encoder.plan")
    with pytest.raises(ValueError, match="exactly 1"):
        adapter.load_model(num_gpus=0)


def test_local_checkpoint_is_validated_without_importing_whisper(tmp_path) -> None:  # noqa: ANN001
    checkpoint = tmp_path / "whisper.pt"
    checkpoint.touch()
    adapter = WhisperLIDAdapter(model_path=str(checkpoint))

    with patch("nemo_curator.models.audio.lid.whisper._import_whisper") as import_whisper:
        adapter.download_weights_on_node()

    import_whisper.assert_not_called()


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"sample_rate": 8_000}, "sample_rate"),
        ({"model_size": ""}, "model_size"),
        ({"model_batch_size": 0}, "model_batch_size"),
        ({"model_batch_size": True}, "model_batch_size"),
        ({"backend": "unknown"}, "backend"),
    ],
)
def test_invalid_model_configuration_fails_before_loading(kwargs: dict[str, object], field: str) -> None:
    with pytest.raises(ValueError, match=field):
        WhisperLIDAdapter(**kwargs)


@pytest.mark.parametrize(("backend", "num_gpus"), [("torch", 0), ("tensorrt", 1)])
def test_model_lifecycle_loads_once_and_releases_owned_resources(backend: str, num_gpus: int, tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    provider = SimpleNamespace(load_model=MagicMock(return_value=model))
    encoder = MagicMock()
    adapter = WhisperLIDAdapter(backend=backend, tensorrt_engine_path="encoder.plan", download_root=str(tmp_path))
    with (
        patch("nemo_curator.models.audio.lid.whisper._import_whisper", return_value=provider),
        patch("nemo_curator.models.audio.lid.whisper.WhisperTensorRTEncoder", return_value=encoder) as encoder_type,
        patch("torch.cuda.is_available", return_value=bool(num_gpus)),
    ):
        adapter.load_model(num_gpus=num_gpus)
        adapter.load_model(num_gpus=num_gpus)

    device = torch.device("cuda" if num_gpus else "cpu")
    provider.load_model.assert_called_once_with("medium", device=device, download_root=str(tmp_path))
    assert not model.training
    assert adapter._model is model
    assert adapter._mel_dtype == (torch.float16 if num_gpus else torch.float32)
    if backend == "tensorrt":
        encoder_type.assert_called_once_with("encoder.plan")
    else:
        encoder_type.assert_not_called()
    adapter._hann_window = torch.ones(400)
    with patch("torch.cuda.empty_cache") as empty_cache:
        adapter.unload_model()

    empty_cache.assert_called_once()
    assert encoder.close.call_count == num_gpus
    assert adapter._model is adapter._encoder is adapter._device is adapter._hann_window is None
    assert adapter._mel_dtype == torch.float32


def test_unavailable_cuda_fails_before_loading_whisper() -> None:
    adapter = WhisperLIDAdapter()
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch.object(adapter, "_load_openai_model") as load_model,
        pytest.raises(RuntimeError, match="CUDA is not available"),
    ):
        adapter.load_model(num_gpus=1)

    load_model.assert_not_called()
    assert adapter._model is None


def test_download_only_constructs_a_temporary_cpu_model() -> None:
    adapter = WhisperLIDAdapter()
    with patch.object(adapter, "_load_openai_model") as load_model:
        adapter.download_weights_on_node()

    load_model.assert_called_once_with("cpu")
    assert adapter._model is None


def test_missing_local_checkpoint_fails_before_provider_load(tmp_path: Path) -> None:
    adapter = WhisperLIDAdapter(model_path=str(tmp_path / "missing.pt"))
    with (
        patch("nemo_curator.models.audio.lid.whisper._import_whisper") as import_whisper,
        pytest.raises(FileNotFoundError, match="Whisper checkpoint not found"),
    ):
        adapter.download_weights_on_node()

    import_whisper.assert_not_called()


def test_nonempty_input_requires_loaded_model_but_empty_batch_does_not() -> None:
    adapter = WhisperLIDAdapter()
    assert adapter.identify_batch([]) == []
    with pytest.raises(RuntimeError, match="call load_model"):
        adapter.identify_batch([{"waveform": np.ones(400, dtype=np.float32)}])


def test_empty_waveforms_do_not_reach_whisper() -> None:
    adapter = WhisperLIDAdapter()
    adapter._device = torch.device("cpu")
    adapter._model = _FakeWhisperModel()
    with patch("nemo_curator.models.audio.lid.whisper._import_whisper"):
        assert adapter.identify_batch([{"waveform": np.array([], dtype=np.float32)}]) == [AudioLIDResult("", 0.0)]

    assert adapter._model.last_input is None


@pytest.mark.parametrize(("probabilities", "message"), [([], "0 predictions"), ([{}], "empty language-probability")])
def test_invalid_model_outputs_fail_instead_of_dropping_rows(probabilities: list, message: str) -> None:
    adapter = WhisperLIDAdapter()
    adapter._device = torch.device("cpu")
    adapter._model = MagicMock()
    adapter._model.dims.n_mels = 80
    adapter._model.detect_language.return_value = None, probabilities
    provider = SimpleNamespace(audio=_FakeWhisperAudio, pad_or_trim=_pad_or_trim)
    with (
        patch("nemo_curator.models.audio.lid.whisper._import_whisper", return_value=provider),
        pytest.raises(RuntimeError, match=message),
    ):
        adapter.identify_batch([{"waveform": np.ones(400, dtype=np.float32)}])
