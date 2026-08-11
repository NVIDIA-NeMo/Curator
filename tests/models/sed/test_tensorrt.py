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
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from nemo_curator.models.sed.base import SEDAdapter
from nemo_curator.models.sed.tensorrt import (
    TensorRTPANNsSEDAdapter,
    _trt_dtype_to_torch,
    postprocess,
)
from nemo_curator.stages.audio.inference.sed.stage import SEDInferenceStage

_SR = 16000
_HOP = 320
_CLASSES = 527
_CHECKPOINT = "/weights/Cnn14.pth"
_ENGINE = "/engines/cnn14.plan"


class _FakeTensorRTRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []
        self.closed = False

    def __call__(self, waveforms: torch.Tensor) -> torch.Tensor:
        batch, samples = waveforms.shape
        self.calls.append((batch, samples))
        frames = samples // _HOP
        return torch.ones((batch, frames, _CLASSES), dtype=torch.float32)

    def close(self) -> None:
        self.closed = True


def _adapter(**kwargs: object) -> tuple[TensorRTPANNsSEDAdapter, _FakeTensorRTRuntime]:
    adapter = TensorRTPANNsSEDAdapter(
        checkpoint_path=_CHECKPOINT,
        tensorrt_engine_path=_ENGINE,
        sample_rate=_SR,
        hop_size=_HOP,
        classes_num=_CLASSES,
        **kwargs,
    )
    runtime = _FakeTensorRTRuntime()
    adapter._model = runtime
    adapter._device = torch.device("cuda")
    return adapter, runtime


def _item(seconds: float) -> dict[str, object]:
    return {"waveform": np.zeros(int(seconds * _SR), dtype=np.float32)}


def test_adapter_implements_the_sed_contract() -> None:
    adapter = TensorRTPANNsSEDAdapter(
        checkpoint_path=_CHECKPOINT,
        tensorrt_engine_path=_ENGINE,
    )

    assert isinstance(adapter, SEDAdapter)


def test_adapter_requires_an_engine_path() -> None:
    with pytest.raises(ValueError, match="tensorrt_engine_path is required"):
        TensorRTPANNsSEDAdapter(checkpoint_path=_CHECKPOINT)


def test_adapter_rejects_unsupported_cnn14_variants() -> None:
    with pytest.raises(ValueError, match="supports only Cnn14_DecisionLevelMax"):
        TensorRTPANNsSEDAdapter(
            checkpoint_path=_CHECKPOINT,
            tensorrt_engine_path=_ENGINE,
            model_type="Cnn14_DecisionLevelAvg",
        )


def test_generic_stage_selects_the_tensorrt_adapter() -> None:
    stage = SEDInferenceStage(
        adapter_target="nemo_curator.models.sed.tensorrt.TensorRTPANNsSEDAdapter",
        checkpoint_path=_CHECKPOINT,
        adapter_kwargs={"tensorrt_engine_path": _ENGINE},
    )

    adapter = stage._create_adapter()

    assert isinstance(adapter, TensorRTPANNsSEDAdapter)
    assert adapter.tensorrt_engine_path == _ENGINE


@pytest.mark.parametrize("num_gpus", [0, -1, 1.5, True])
def test_load_model_requires_a_positive_integer_gpu_count(num_gpus: object) -> None:
    adapter = TensorRTPANNsSEDAdapter(
        checkpoint_path=_CHECKPOINT,
        tensorrt_engine_path=_ENGINE,
    )
    with (
        patch("nemo_curator.models.sed.tensorrt.get_model_class") as model_resolver,
        pytest.raises(ValueError, match="requires a positive integer num_gpus"),
    ):
        adapter.load_model(num_gpus=num_gpus)  # type: ignore[arg-type]
    model_resolver.assert_not_called()


def test_load_model_requires_cuda() -> None:
    adapter = TensorRTPANNsSEDAdapter(
        checkpoint_path=_CHECKPOINT,
        tensorrt_engine_path=_ENGINE,
    )
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("nemo_curator.models.sed.tensorrt.get_model_class") as model_resolver,
        pytest.raises(RuntimeError, match="CUDA is not available"),
    ):
        adapter.load_model(num_gpus=1)
    model_resolver.assert_not_called()


def test_load_model_uses_the_checkpoint_frontend_and_tensorrt_runtime(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "cnn14.pth"
    checkpoint_path.touch()
    engine_path = tmp_path / "cnn14.plan"
    adapter = TensorRTPANNsSEDAdapter(
        checkpoint_path=str(checkpoint_path),
        tensorrt_engine_path=str(engine_path),
        sample_rate=22050,
        window_size=2048,
        hop_size=512,
        mel_bins=80,
        fmin=20,
        fmax=10000,
        classes_num=100,
    )
    model = MagicMock()
    model_cls = MagicMock(return_value=model)
    runtime = MagicMock()

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("nemo_curator.models.sed.tensorrt.get_model_class", return_value=model_cls) as resolver,
        patch("torch.load", return_value={"model": {"weight": "value"}}) as torch_load,
        patch("nemo_curator.models.sed.tensorrt.TensorRTSed", return_value=runtime) as runtime_cls,
    ):
        adapter.load_model(num_gpus=1)

    resolver.assert_called_once_with("Cnn14_DecisionLevelMax")
    model_cls.assert_called_once_with(
        sample_rate=22050,
        window_size=2048,
        hop_size=512,
        mel_bins=80,
        fmin=20,
        fmax=10000,
        classes_num=100,
    )
    torch_load.assert_called_once_with(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict.assert_called_once_with({"weight": "value"})
    model.eval.assert_called_once_with()
    runtime_cls.assert_called_once_with(model, str(engine_path))
    assert adapter._model is runtime
    assert adapter._device == torch.device("cuda")


def test_ragged_batch_preserves_the_panns_result_contract() -> None:
    adapter, runtime = _adapter()

    short, long = adapter.infer_batch([_item(1.0), _item(3.0)])

    assert runtime.calls == [(2, 3 * _SR)]
    assert short.framewise_output.shape == long.framewise_output.shape == (3 * _SR // _HOP, _CLASSES)
    assert np.all(short.framewise_output == 1.0)
    assert short.valid_frames == _SR / _HOP
    assert long.valid_frames == 3 * _SR / _HOP
    assert short.original_num_samples == _SR
    assert short.fps == _SR / _HOP


def test_inference_requires_a_loaded_runtime() -> None:
    adapter = TensorRTPANNsSEDAdapter(
        checkpoint_path=_CHECKPOINT,
        tensorrt_engine_path=_ENGINE,
    )

    with pytest.raises(RuntimeError, match=r"load_model\(\) must be called"):
        adapter.infer_batch([_item(1.0)])


def test_unload_closes_the_tensorrt_runtime() -> None:
    adapter, runtime = _adapter()

    adapter.unload_model()

    assert runtime.closed
    assert adapter._model is None
    assert adapter._device is None


def test_tensorrt_postprocess_matches_panns_geometry() -> None:
    segmentwise = torch.tensor([[[0.1], [0.9]]])

    framewise = postprocess(segmentwise, frames_num=70)

    assert framewise.shape == (1, 70, 1)
    torch.testing.assert_close(framewise[:, :32], torch.full((1, 32, 1), 0.1))
    torch.testing.assert_close(framewise[:, 32:], torch.full((1, 38, 1), 0.9))


@pytest.mark.parametrize(
    ("tensorrt_dtype", "torch_dtype"),
    [
        ("DataType.FP16", torch.float16),
        ("DataType.FLOAT", torch.float32),
        ("DataType.INT32", torch.int32),
        ("DataType.BOOL", torch.bool),
    ],
)
def test_tensorrt_dtypes_map_to_torch(tensorrt_dtype: str, torch_dtype: torch.dtype) -> None:
    assert _trt_dtype_to_torch(tensorrt_dtype) == torch_dtype


def test_unknown_tensorrt_dtype_is_rejected() -> None:
    with pytest.raises(TypeError, match="Unsupported TensorRT tensor dtype"):
        _trt_dtype_to_torch("DataType.FP8")
