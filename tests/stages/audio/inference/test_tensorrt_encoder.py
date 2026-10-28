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

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_curator.stages.audio.inference.tensorrt_encoder import (
    TensorRTEncoder,
    TensorRTEncoderSession,
    _trt_dtype_to_torch,
)


class _FakeCudaTensor:
    """Small tensor double covering the CUDA methods used by the TRT session."""

    _next_pointer = 1000

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device_type: str = "cpu",
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = SimpleNamespace(type=device_type)
        self.recorded_streams: list[object] = []
        self.pointer = type(self)._next_pointer
        type(self)._next_pointer += 1

    def to(
        self,
        target: torch.device | torch.dtype,
        *,
        non_blocking: bool = False,
    ) -> "_FakeCudaTensor":
        if isinstance(target, torch.dtype):
            return _FakeCudaTensor(self.shape, dtype=target, device_type=self.device.type)
        return _FakeCudaTensor(self.shape, dtype=self.dtype, device_type=torch.device(target).type)

    def contiguous(self) -> "_FakeCudaTensor":
        return self

    def data_ptr(self) -> int:
        return self.pointer

    def record_stream(self, stream: object) -> None:
        self.recorded_streams.append(stream)


class _FakeCudaStream:
    def __init__(self, cuda_stream: int) -> None:
        self.cuda_stream = cuda_stream
        self.waited_for: list[object] = []

    def wait_stream(self, stream: object) -> None:
        self.waited_for.append(stream)


@pytest.fixture
def trt_session(tmp_path: Path) -> tuple[TensorRTEncoderSession, SimpleNamespace]:
    engine_path = tmp_path / "encoder.plan"
    engine_path.write_bytes(b"serialized-engine")

    context = MagicMock()
    context.set_input_shape.return_value = True
    context.set_tensor_address.return_value = True
    context.get_tensor_shape.side_effect = lambda name: {
        "outputs": (2, 512, 4),
        "encoded_lengths": (2,),
    }[name]
    context.execute_async_v3.return_value = True

    tensor_names = ["audio_signal", "length", "outputs", "encoded_lengths"]
    engine = MagicMock(num_io_tensors=len(tensor_names))
    engine.get_tensor_name.side_effect = tensor_names
    engine.get_tensor_mode.side_effect = lambda name: "input" if name in {"audio_signal", "length"} else "output"
    engine.get_tensor_dtype.side_effect = lambda name: {
        "audio_signal": "DataType.HALF",
        "length": "DataType.INT64",
        "outputs": "DataType.HALF",
        "encoded_lengths": "DataType.INT64",
    }[name]
    engine.get_tensor_profile_shape.return_value = ((1, 80, 8), (8, 80, 800), (16, 80, 4001))
    engine.create_execution_context.return_value = context

    runtime = MagicMock()
    runtime.deserialize_cuda_engine.return_value = engine
    logger_type = MagicMock()
    logger_type.WARNING = "warning"
    trt = SimpleNamespace(
        Logger=logger_type,
        Runtime=MagicMock(return_value=runtime),
        TensorIOMode=SimpleNamespace(INPUT="input"),
    )
    private_stream = _FakeCudaStream(cuda_stream=4242)
    with (
        patch.dict("sys.modules", {"tensorrt": trt}),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.Stream", return_value=private_stream),
    ):
        session = TensorRTEncoderSession(engine_path)

    state = SimpleNamespace(
        context=context,
        engine=engine,
        private_stream=private_stream,
        runtime=runtime,
    )
    return session, state


def _session(
    min_shape: tuple[int, int, int] = (1, 80, 8),
    max_shape: tuple[int, int, int] = (16, 80, 3000),
) -> MagicMock:
    session = MagicMock(
        input_names=["audio_signal", "length"],
        output_names=["outputs", "encoded_lengths"],
    )
    session.input_shape_range.return_value = (min_shape, (8, 80, 800), max_shape)
    return session


@pytest.mark.parametrize(
    ("trt_dtype", "torch_dtype"),
    [
        ("DataType.HALF", torch.float16),
        ("DataType.FLOAT", torch.float32),
        ("DataType.INT64", torch.int64),
        ("DataType.INT32", torch.int32),
        ("DataType.INT8", torch.int8),
        ("DataType.UINT8", torch.uint8),
        ("DataType.BOOL", torch.bool),
    ],
)
def test_tensorrt_dtype_conversion(trt_dtype: object, torch_dtype: torch.dtype) -> None:
    assert _trt_dtype_to_torch(trt_dtype) is torch_dtype


def test_tensorrt_dtype_conversion_rejects_unknown_type() -> None:
    with pytest.raises(TypeError, match="Unsupported TensorRT tensor dtype"):
        _trt_dtype_to_torch("DataType.BF16")


def test_session_requires_cuda(tmp_path: Path) -> None:
    with patch("torch.cuda.is_available", return_value=False), pytest.raises(RuntimeError, match="requires CUDA"):
        TensorRTEncoderSession(tmp_path / "encoder.plan")


def test_session_requires_tensorrt_bindings(tmp_path: Path) -> None:
    engine_path = tmp_path / "encoder.plan"
    engine_path.touch()

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch.dict("sys.modules", {"tensorrt": None}),
        pytest.raises(RuntimeError, match="Python bindings are required"),
    ):
        TensorRTEncoderSession(engine_path)


def test_session_requires_engine_file(tmp_path: Path) -> None:
    fake_trt = SimpleNamespace(Logger=MagicMock(), Runtime=MagicMock())
    fake_trt.Logger.WARNING = "warning"

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch.dict("sys.modules", {"tensorrt": fake_trt}),
        pytest.raises(FileNotFoundError, match="encoder engine not found"),
    ):
        TensorRTEncoderSession(tmp_path / "missing.plan")


def test_session_deserializes_engine_and_discovers_io(
    trt_session: tuple[TensorRTEncoderSession, SimpleNamespace],
) -> None:
    session, state = trt_session

    assert session.input_names == ["audio_signal", "length"]
    assert session.output_names == ["outputs", "encoded_lengths"]
    state.runtime.deserialize_cuda_engine.assert_called_once_with(b"serialized-engine")
    state.engine.create_execution_context.assert_called_once_with()
    assert session.input_shape_range("audio_signal") == ((1, 80, 8), (8, 80, 800), (16, 80, 4001))
    assert session.max_input_shape("audio_signal") == (16, 80, 4001)


def test_session_executes_on_private_stream_and_reuses_output_buffers(
    trt_session: tuple[TensorRTEncoderSession, SimpleNamespace],
) -> None:
    session, state = trt_session
    current_stream = _FakeCudaStream(cuda_stream=111)
    allocations: list[_FakeCudaTensor] = []

    def empty(shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device) -> _FakeCudaTensor:
        output = _FakeCudaTensor(tuple(shape), dtype=dtype, device_type=device.type)
        allocations.append(output)
        return output

    inputs = {
        "audio_signal": _FakeCudaTensor((2, 80, 32), dtype=torch.float32),
        "length": _FakeCudaTensor((2,), dtype=torch.int64),
    }
    with (
        patch("torch.cuda.current_stream", return_value=current_stream),
        patch("torch.cuda.stream", return_value=nullcontext()),
        patch("torch.empty", side_effect=empty),
    ):
        first = session.infer(inputs)
        second = session.infer(inputs)

    assert first == second
    assert len(allocations) == 2
    assert state.context.set_input_shape.call_count == 4
    assert state.context.set_tensor_address.call_count == 8
    assert state.context.execute_async_v3.call_args_list[0].args == (4242,)
    assert state.private_stream.waited_for == [current_stream, current_stream]
    assert current_stream.waited_for == [state.private_stream, state.private_stream]
    assert all(output.recorded_streams == [state.private_stream, state.private_stream] for output in first.values())


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ({"audio_signal": object()}, "Missing TensorRT encoder inputs"),
        ({"audio_signal": object(), "length": object(), "extra": object()}, "Unexpected TensorRT encoder inputs"),
    ],
)
def test_session_rejects_incorrect_input_mapping(
    trt_session: tuple[TensorRTEncoderSession, SimpleNamespace],
    inputs: dict[str, object],
    message: str,
) -> None:
    session, _ = trt_session

    with pytest.raises(KeyError, match=message):
        session.infer(inputs)  # type: ignore[arg-type]


def test_session_rejects_unresolved_output_shape(trt_session: tuple[TensorRTEncoderSession, SimpleNamespace]) -> None:
    session, state = trt_session
    state.context.get_tensor_shape.return_value = (-1, 512, 4)
    state.context.get_tensor_shape.side_effect = None

    with pytest.raises(RuntimeError, match="did not resolve encoder output shape"):
        session._output_buffer("outputs", (-1, 512, 4))


def test_session_rejects_unknown_profile_input(trt_session: tuple[TensorRTEncoderSession, SimpleNamespace]) -> None:
    session, _ = trt_session

    with pytest.raises(KeyError, match="input not found"):
        session.input_shape_range("unknown")


def test_session_close_releases_runtime_objects(trt_session: tuple[TensorRTEncoderSession, SimpleNamespace]) -> None:
    session, _ = trt_session
    session._output_buffers["outputs"] = object()  # type: ignore[assignment]

    session.close()

    assert session._output_buffers == {}
    assert session._context is None
    assert session._engine is None
    assert session._runtime is None


@pytest.mark.parametrize(
    ("input_names", "output_names", "message"),
    [
        (["audio_signal"], ["outputs", "encoded_lengths"], "missing inputs"),
        (["audio_signal", "length"], ["outputs"], "missing outputs"),
    ],
)
def test_encoder_rejects_incomplete_engine_io(
    input_names: list[str],
    output_names: list[str],
    message: str,
) -> None:
    session = MagicMock(input_names=input_names, output_names=output_names)

    with pytest.raises(ValueError, match=message):
        TensorRTEncoder("unused.plan", subsampling_factor=8, session=session)


@pytest.mark.parametrize(
    ("audio_signal", "length"),
    [
        (torch.randn(2, 80), torch.tensor([8, 8])),
        (torch.randn(2, 80, 8), torch.tensor([[8], [8]])),
        (torch.randn(2, 80, 8), torch.tensor([8])),
    ],
)
def test_encoder_rejects_invalid_input_ranks_or_batch_alignment(
    audio_signal: torch.Tensor,
    length: torch.Tensor,
) -> None:
    encoder = TensorRTEncoder("unused.plan", subsampling_factor=8, session=_session())

    with pytest.raises(ValueError, match="expects audio_signal shaped"):
        encoder(audio_signal, length)


def test_encoder_pads_inputs_to_profile_minimum_and_discards_padding_rows() -> None:
    session = _session((4, 80, 16))
    session.infer.return_value = {
        "outputs": torch.randn(4, 512, 4),
        "encoded_lengths": torch.tensor([2, 2, 0, 0]),
    }
    encoder = TensorRTEncoder("unused.plan", subsampling_factor=8, session=session)

    outputs, encoded_lengths = encoder(torch.randn(2, 80, 8), torch.tensor([8, 7]))

    inputs = session.infer.call_args.args[0]
    assert inputs["audio_signal"].shape == (4, 80, 16)
    assert inputs["length"].tolist() == [8, 7, 16, 16]
    assert outputs.shape[0] == 2
    assert encoded_lengths.tolist() == [2, 2]


def test_encoder_rejects_feature_frames_larger_than_profile() -> None:
    encoder = TensorRTEncoder("unused.plan", subsampling_factor=8, session=_session())

    with pytest.raises(ValueError, match="exceeds profile maximum"):
        encoder(torch.randn(2, 80, 3001), torch.full((2,), 3001))


def test_encoder_rejects_wrong_feature_count() -> None:
    encoder = TensorRTEncoder("unused.plan", subsampling_factor=8, session=_session())

    with pytest.raises(ValueError, match="expects 80 input features"):
        encoder(torch.randn(2, 64, 8), torch.full((2,), 8))


def test_encoder_automatically_splits_at_profile_maximum_batch() -> None:
    session = _session(min_shape=(4, 80, 8), max_shape=(8, 80, 3000))

    def infer(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        rows = inputs["audio_signal"].shape[0]
        call_value = float(session.infer.call_count - 1)
        return {
            "outputs": torch.full((rows, 512, 2), call_value),
            "encoded_lengths": torch.full((rows,), 2),
        }

    session.infer.side_effect = infer
    encoder = TensorRTEncoder("unused.plan", subsampling_factor=8, session=session)

    outputs, encoded_lengths = encoder(torch.randn(18, 80, 8), torch.full((18,), 8))

    assert [call.args[0]["audio_signal"].shape for call in session.infer.call_args_list] == [
        (8, 80, 8),
        (8, 80, 8),
        (4, 80, 8),
    ]
    assert outputs[:, 0, 0].tolist() == [0.0] * 8 + [1.0] * 8 + [2.0] * 2
    assert encoded_lengths.tolist() == [2] * 18


def test_encoder_preserves_nemo_freeze_contract() -> None:
    session = _session()
    encoder = TensorRTEncoder("unused.plan", subsampling_factor=8, session=session)

    encoder.train()
    encoder.freeze()
    assert encoder.training is False
    encoder.train()
    encoder.unfreeze(partial=True)
    assert encoder.training is False


def test_encoder_close_releases_session() -> None:
    session = _session()
    encoder = TensorRTEncoder("unused.plan", subsampling_factor=8, session=session)

    encoder.close()

    session.close.assert_called_once_with()
