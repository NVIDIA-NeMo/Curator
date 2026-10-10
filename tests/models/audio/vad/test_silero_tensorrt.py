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
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from nemo_curator.models.audio.vad.base import VADSegment
from nemo_curator.models.audio.vad.silero_tensorrt import TensorRTSileroVADAdapter

if TYPE_CHECKING:
    from pathlib import Path


class _FakeSession:
    def __init__(self, max_batch_size: int = 64) -> None:
        self.device = torch.device("cpu")
        self.input_names = ["input", "state"]
        self.output_names = ["output", "stateN"]
        self.batch_sizes: list[int] = []
        self.closed = False
        self.max_batch_size = max_batch_size
        self.profiles = {
            "input": ((1, 576), (min(16, max_batch_size), 576), (max_batch_size, 576)),
            "state": ((1, 2, 128), (min(16, max_batch_size), 2, 128), (max_batch_size, 2, 128)),
        }

    def input_shape_range(self, name: str) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return self.profiles[name]

    def infer(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        state = inputs["state"]
        self.batch_sizes.append(state.shape[0])
        return {
            "output": state[:, 0, :1] + 0.25,
            "stateN": state + 1,
        }

    def close(self) -> None:
        self.closed = True


def _adapter(session: object | None = None, **kwargs: object) -> TensorRTSileroVADAdapter:
    return TensorRTSileroVADAdapter(
        engine_path="/models/silero.plan",
        session=_FakeSession() if session is None else session,
        **kwargs,
    )


def test_engine_path_is_required() -> None:
    with pytest.raises(ValueError, match="engine_path"):
        TensorRTSileroVADAdapter(engine_path="")


def test_prefetch_validates_the_local_engine(tmp_path: Path) -> None:
    engine = tmp_path / "silero.plan"
    adapter = TensorRTSileroVADAdapter(engine_path=str(engine))
    with pytest.raises(FileNotFoundError, match="engine not found"):
        adapter.download_weights_on_node()

    engine.touch()
    adapter.download_weights_on_node()


@pytest.mark.parametrize("num_gpus", [0, 2])
def test_tensorrt_requires_exactly_one_gpu(num_gpus: int) -> None:
    with pytest.raises(ValueError, match=r"exactly one GPU|0 or 1"):
        _adapter().load_model(num_gpus=num_gpus)


@pytest.mark.parametrize(
    ("inputs", "outputs", "message"),
    [
        (["input"], ["output", "stateN"], "inputs must be"),
        (["input", "state", "extra"], ["output", "stateN"], "inputs must be"),
        (["input", "state"], ["output"], "outputs must be"),
        (["input", "state"], ["output", "stateN", "extra"], "outputs must be"),
    ],
)
def test_engine_io_contract_is_exact(inputs: list[str], outputs: list[str], message: str) -> None:
    session = _FakeSession()
    session.input_names = inputs
    session.output_names = outputs
    with pytest.raises(ValueError, match=message):
        _adapter(session).load_model(num_gpus=1)


@pytest.mark.parametrize(
    ("name", "profile", "message"),
    [
        ("input", ((1, 575), (2, 575), (4, 575)), "input profile"),
        ("state", ((1, 2, 127), (2, 2, 127), (4, 2, 127)), "state profile"),
        ("input", ((2, 576), (2, 576), (4, 576)), "identical ordered batch ranges"),
        ("input", ((2, 576), (2, 576), (2, 576)), "batch size 1"),
    ],
)
def test_engine_profile_contract_is_validated(name: str, profile: tuple, message: str) -> None:
    session = _FakeSession(max_batch_size=4)
    session.profiles[name] = profile
    if name == "input" and profile == ((2, 576), (2, 576), (2, 576)):
        session.profiles["state"] = ((2, 2, 128), (2, 2, 128), (2, 2, 128))

    with pytest.raises(ValueError, match=message):
        _adapter(session).load_model(num_gpus=1)


def test_load_model_reuses_the_shared_tensorrt_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = tmp_path / "silero.plan"
    engine.touch()
    session = _FakeSession()
    session_cls = MagicMock(return_value=session)
    from nemo_curator.stages.audio.inference import tensorrt_encoder

    monkeypatch.setattr(tensorrt_encoder, "TensorRTEncoderSession", session_cls)
    adapter = TensorRTSileroVADAdapter(engine_path=str(engine))

    adapter.load_model(num_gpus=1)

    session_cls.assert_called_once_with(str(engine))
    assert adapter._session is session


def test_infer_step_validates_input_and_state_shapes() -> None:
    adapter = _adapter()
    adapter.load_model(num_gpus=1)
    with pytest.raises(ValueError, match=r"\[batch, 576\]"):
        adapter.infer_step(torch.zeros((2, 512)), torch.zeros((2, 2, 128)))
    with pytest.raises(ValueError, match="state must have shape"):
        adapter.infer_step(torch.zeros((2, 576)), torch.zeros((2, 128)))


def test_infer_step_rejects_batch_above_engine_profile() -> None:
    adapter = _adapter(_FakeSession(max_batch_size=2))
    adapter.load_model(num_gpus=1)

    with pytest.raises(ValueError, match="exceeds the loaded engine profile"):
        adapter.infer_step(torch.zeros((3, 576)), torch.zeros((3, 2, 128)))


def test_recurrent_scheduler_compacts_finished_recordings() -> None:
    session = _FakeSession()
    adapter = _adapter(session)
    adapter.load_model(num_gpus=1)

    probabilities = adapter._infer_probabilities([torch.zeros(512), torch.zeros(1024), torch.zeros(1536)])

    assert session.batch_sizes == [3, 2, 1]
    torch.testing.assert_close(probabilities[0], torch.tensor([0.25]))
    torch.testing.assert_close(probabilities[1], torch.tensor([0.25, 1.25]))
    torch.testing.assert_close(probabilities[2], torch.tensor([0.25, 1.25, 2.25]))


def test_recurrent_scheduler_never_exceeds_engine_batch_profile() -> None:
    session = _FakeSession(max_batch_size=2)
    adapter = _adapter(session)
    adapter.load_model(num_gpus=1)

    probabilities = adapter._infer_probabilities(
        [torch.zeros(512), torch.zeros(1024), torch.zeros(1536), torch.zeros(1024), torch.zeros(512)]
    )

    assert session.batch_sizes == [2, 2, 1, 2, 1, 1]
    assert [values.tolist() for values in probabilities] == [
        [0.25],
        [0.25, 1.25],
        [0.25, 1.25, 2.25],
        [0.25, 1.25],
        [0.25],
    ]


def test_ragged_detect_batch_preserves_order_and_uses_official_postprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def get_timestamps(waveform: torch.Tensor, model: object, **kwargs: object) -> list[dict[str, int]]:
        model.reset_states()
        frame_count = (len(waveform) + 511) // 512
        frame_probabilities = [float(model(torch.zeros(512), 16000).item()) for _ in range(frame_count)]
        calls.append({"length": len(waveform), "probabilities": frame_probabilities, **kwargs})
        return [{"start": 0, "end": len(waveform)}]

    monkeypatch.setitem(
        sys.modules,
        "silero_vad",
        SimpleNamespace(get_speech_timestamps=get_timestamps),
    )
    adapter = _adapter(threshold=0.7, min_duration_sec=0.1)
    adapter.load_model(num_gpus=1)

    results = adapter.detect_batch(
        [
            {"waveform": np.zeros(512, dtype=np.float32), "sample_rate": 16000},
            {"waveform": np.zeros(1024, dtype=np.float32), "sample_rate": 16000},
        ]
    )

    assert [call["length"] for call in calls] == [512, 1024]
    assert calls[0]["probabilities"] == [0.25]
    assert calls[1]["probabilities"] == [0.25, 1.25]
    assert calls[0]["sampling_rate"] == 16000
    assert calls[0]["threshold"] == 0.7
    assert results[0].segments == [VADSegment(0.0, 512 / 16000)]
    assert results[1].segments == [VADSegment(0.0, 1024 / 16000)]


def test_postprocessing_failure_is_isolated_after_batch_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    def get_timestamps(waveform: torch.Tensor, _model: object, **_kwargs: object) -> list[dict[str, int]]:
        if len(waveform) == 1024:
            message = "bad recording"
            raise RuntimeError(message)
        return [{"start": 0, "end": len(waveform)}]

    monkeypatch.setitem(
        sys.modules,
        "silero_vad",
        SimpleNamespace(get_speech_timestamps=get_timestamps),
    )
    adapter = _adapter()
    adapter.load_model(num_gpus=1)

    results = adapter.detect_batch(
        [
            {"waveform": np.zeros(512, dtype=np.float32), "sample_rate": 16000},
            {"waveform": np.zeros(1024, dtype=np.float32), "sample_rate": 16000},
            {"waveform": np.zeros(1536, dtype=np.float32), "sample_rate": 16000},
        ]
    )

    assert results[0].segments == [VADSegment(0.0, 512 / 16000)]
    assert results[1].segments == []
    assert results[1].error == "RuntimeError: bad recording"
    assert results[2].segments == [VADSegment(0.0, 1536 / 16000)]


@pytest.mark.parametrize("error", [MemoryError("out of memory"), torch.cuda.OutOfMemoryError("out of memory")])
def test_postprocessing_memory_failures_propagate(error: Exception, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "silero_vad",
        SimpleNamespace(get_speech_timestamps=MagicMock(side_effect=error)),
    )
    adapter = _adapter()
    adapter.load_model(num_gpus=1)

    with pytest.raises(type(error), match="out of memory"):
        adapter.detect_batch([{"waveform": np.zeros(512, dtype=np.float32), "sample_rate": 16000}])


def test_every_non_16khz_input_is_resampled(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int]] = []

    class _Resample:
        def __init__(self, *, orig_freq: int, new_freq: int) -> None:
            calls.append((orig_freq, new_freq))

        def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
            return torch.zeros((1, 512), dtype=torch.float32)

    monkeypatch.setitem(
        sys.modules,
        "torchaudio",
        SimpleNamespace(transforms=SimpleNamespace(Resample=_Resample)),
    )
    monkeypatch.setitem(
        sys.modules,
        "silero_vad",
        SimpleNamespace(get_speech_timestamps=MagicMock(return_value=[])),
    )
    adapter = _adapter()
    adapter.load_model(num_gpus=1)

    adapter.detect_batch([{"waveform": np.zeros(8000), "sample_rate": 8000}])

    assert calls == [(8000, 16000)]


def test_empty_batch_does_not_require_a_loaded_session() -> None:
    assert TensorRTSileroVADAdapter(engine_path="silero.plan").detect_batch([]) == []


def test_nonempty_batch_requires_a_loaded_session() -> None:
    adapter = TensorRTSileroVADAdapter(engine_path="silero.plan")
    with pytest.raises(RuntimeError, match="not loaded"):
        adapter.detect_batch([{"waveform": np.zeros(512), "sample_rate": 16000}])


def test_unload_closes_the_injected_session() -> None:
    session = _FakeSession()
    adapter = _adapter(session)
    adapter.load_model(num_gpus=1)

    adapter.unload_model()

    assert session.closed is True
    assert adapter._session is None
    assert adapter._max_batch_size is None


def test_public_package_resolves_tensorrt_adapter_lazily() -> None:
    from nemo_curator.models.audio import vad

    assert vad.TensorRTSileroVADAdapter is TensorRTSileroVADAdapter


def test_vad_stage_constructs_tensorrt_adapter() -> None:
    from nemo_curator.stages.audio.segmentation.vad_segmentation import VADSegmentationStage

    stage = VADSegmentationStage(
        adapter_target="nemo_curator.models.audio.vad.silero_tensorrt.TensorRTSileroVADAdapter",
        adapter_kwargs={"engine_path": "/models/silero.plan"},
    )

    assert isinstance(stage._create_adapter(), TensorRTSileroVADAdapter)
