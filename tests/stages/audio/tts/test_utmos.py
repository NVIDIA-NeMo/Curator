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

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from nemo_curator.stages.audio.tts.utmos import UTMOSv2AnnotationStage, coerce_prediction_to_float
from nemo_curator.tasks import AudioTask
from tests.stages.audio.tts.test_audio_io import make_tar_segments


class _FakeUtmosModel:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, str]] = []

    def predict(self, data: np.ndarray, sr: int, device: str, num_repetitions: int = 1, verbose: bool = True) -> float:
        self.calls.append((len(data), sr, device))
        self.repetitions = num_repetitions
        assert verbose is False
        if len(data) < 1000:
            msg = "too short"
            raise RuntimeError(msg)
        return 2.5 + len(data) / sr


def _fake_utmosv2_module(model: _FakeUtmosModel, created: list[dict]) -> types.ModuleType:
    module = types.ModuleType("utmosv2")

    def create_model(**kwargs: object) -> _FakeUtmosModel:
        created.append(kwargs)
        return model

    module.create_model = create_model
    return module


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        ({"predicted_mos": 3.25}, 3.25),
        (torch.tensor([[2.0]]), 2.0),
        (np.array([1.5, 9.0]), 1.5),
        (4, 4.0),
    ],
)
def test_coerce_prediction(prediction: object, expected: float) -> None:
    assert coerce_prediction_to_float(prediction) == pytest.approx(expected)


@pytest.mark.parametrize("prediction", [{"mos": 1.0}, np.array([])])
def test_coerce_prediction_rejects(prediction: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        coerce_prediction_to_float(prediction)


def test_scores_segments_at_native_rate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model, created = _FakeUtmosModel(), []
    monkeypatch.setitem(sys.modules, "utmosv2", _fake_utmosv2_module(model, created))
    segments = make_tar_segments(tmp_path, [("a.wav", 1.0, 22050), ("b.wav", 0.01, 16000), ("c.wav", 0.5, 16000)])
    segments[2]["GranaryHifi"]["AudioMetadata"]["tar_member_offset"] = None
    stage = UTMOSv2AnnotationStage(checkpoint_path="/ckpt/fold0.pth", device="cpu")
    stage.setup()
    task = stage.process(AudioTask(data={"orig_sample_id": "v", "segments": segments}))

    out = [s["utmosv2"] for s in task.data["segments"]]
    assert out[0] == {"score": pytest.approx(3.5), "error": None}
    assert out[1] == {"score": None, "error": "utmosv2_failed"}
    assert out[2] == {"score": None, "error": "missing_tar_offsets"}
    assert len(task.data["segments"]) == 3
    assert model.calls[0] == (22050, 22050, "cpu")
    assert created == [{"pretrained": True, "checkpoint_path": "/ckpt/fold0.pth"}]


def test_missing_package_raises_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing score fails the MOS filter condition, so a missing package must not degrade silently."""
    monkeypatch.setitem(sys.modules, "utmosv2", None)  # makes `import utmosv2` raise ImportError
    with pytest.raises(ImportError, match="utmosv2"):
        UTMOSv2AnnotationStage().setup()


def test_missing_package_marks_unavailable_when_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "utmosv2", None)  # makes `import utmosv2` raise ImportError
    stage = UTMOSv2AnnotationStage(allow_missing_package=True)
    stage.setup()
    segment = make_tar_segments(tmp_path, [("a.wav", 0.5, 16000)])[0]
    task = stage.process(AudioTask(data=segment))
    assert task.data["utmosv2"] == {"score": None, "error": "utmosv2_unavailable"}


def test_dotted_output_and_existing_score_preserved(tmp_path: Path) -> None:
    model = _FakeUtmosModel()
    stage = UTMOSv2AnnotationStage(output_key="GranaryHifi.OrigAudioPipeline.UTMOSv2", model=model, device="cpu")
    stage.setup()
    segments = make_tar_segments(tmp_path, [("a.wav", 0.5, 16000), ("b.wav", 0.5, 16000)])
    segments[1]["GranaryHifi"]["OrigAudioPipeline"] = {"UTMOSv2": {"score": 1.0, "error": None}}
    task = stage.process(AudioTask(data={"segments": segments}))
    assert task.data["segments"][0]["GranaryHifi"]["OrigAudioPipeline"]["UTMOSv2"]["score"] == pytest.approx(3.0)
    assert task.data["segments"][1]["GranaryHifi"]["OrigAudioPipeline"]["UTMOSv2"]["score"] == 1.0
    assert len(model.calls) == 1


def test_num_repetitions_is_forwarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model, created = _FakeUtmosModel(), []
    monkeypatch.setitem(sys.modules, "utmosv2", _fake_utmosv2_module(model, created))
    segment = make_tar_segments(tmp_path, [("a.wav", 1.0, 16000)])[0]
    stage = UTMOSv2AnnotationStage(device="cpu", num_repetitions=5)
    stage.setup()
    stage.process(AudioTask(data={"segments": [segment]}))
    assert model.repetitions == 5
