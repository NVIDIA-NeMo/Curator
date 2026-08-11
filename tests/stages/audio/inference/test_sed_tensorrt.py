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

import numpy as np
import pytest
import torch

from nemo_curator.stages.audio.inference.sed import SEDInferenceStage
from nemo_curator.stages.audio.inference.sed_tensorrt import postprocess
from nemo_curator.tasks import AudioTask


def _task(samples: int, task_id: str) -> AudioTask:
    return AudioTask(
        data={
            "waveform": np.zeros(samples, dtype=np.float32),
            "sample_rate": 16000,
            "audio_filepath": f"{task_id}.wav",
        },
        task_id=task_id,
        dataset_name="test",
    )


class _FakeTorchModel:
    def __call__(self, waveforms: torch.Tensor) -> dict[str, torch.Tensor]:
        framewise = torch.zeros((waveforms.shape[0], 33, 527), dtype=torch.float32)
        return {"framewise_output": framewise}


class _FakeTensorRTModel:
    def __init__(self) -> None:
        self.closed = False

    def __call__(self, waveforms: torch.Tensor) -> torch.Tensor:
        return torch.ones((waveforms.shape[0], 33, 527), dtype=torch.float32)

    def close(self) -> None:
        self.closed = True


def test_sed_backend_defaults_are_preserved() -> None:
    stage = SEDInferenceStage()

    assert stage.backend == "torch"
    assert stage.batch_size == 32
    assert stage.tensorrt_engine_path is None


def test_tensorrt_backend_requires_engine_path() -> None:
    with pytest.raises(ValueError, match="tensorrt_engine_path is required"):
        SEDInferenceStage(backend="tensorrt")


def test_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported SED backend"):
        SEDInferenceStage(backend="onnx")  # type: ignore[arg-type]


def test_tensorrt_backend_rejects_unsupported_model() -> None:
    with pytest.raises(ValueError, match="supports only Cnn14_DecisionLevelMax"):
        SEDInferenceStage(
            backend="tensorrt",
            tensorrt_engine_path="cnn14.plan",
            model_type="Cnn14_DecisionLevelAvg",
        )


def test_tensorrt_backend_supports_configurable_batch_size() -> None:
    stage = SEDInferenceStage(
        backend="tensorrt",
        tensorrt_engine_path="cnn14.plan",
        batch_size=16,
    )
    model = _FakeTensorRTModel()
    stage._device = torch.device("cpu")
    stage._model = model

    tasks = stage.process_batch([_task(10240, "a"), _task(10240, "b")])

    assert stage.batch_size == 16
    assert [task.task_id for task in tasks] == ["a", "b"]
    assert all(task.data["_sed_framewise"].shape == (33, 527) for task in tasks)
    assert all(np.all(task.data["_sed_framewise"] == 1.0) for task in tasks)

    stage.teardown()
    assert model.closed


def test_torch_backend_keeps_existing_call_contract() -> None:
    stage = SEDInferenceStage(batch_size=1)
    stage._device = torch.device("cpu")
    stage._model = _FakeTorchModel()

    tasks = stage.process_batch([_task(10240, "a")])

    assert stage.batch_size == 1
    assert tasks[0].data["_sed_framewise"].shape == (33, 527)
    assert np.all(tasks[0].data["_sed_framewise"] == 0.0)


def test_tensorrt_postprocess_matches_panns_geometry() -> None:
    segmentwise = torch.tensor([[[0.1], [0.9]]])

    framewise = postprocess(segmentwise, frames_num=70)

    assert framewise.shape == (1, 70, 1)
    torch.testing.assert_close(framewise[:, :32], torch.full((1, 32, 1), 0.1))
    torch.testing.assert_close(framewise[:, 32:], torch.full((1, 38, 1), 0.9))
