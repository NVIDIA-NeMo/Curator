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

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.nn import functional

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
