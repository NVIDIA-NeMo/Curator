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

"""Shared TensorRT bundle validation and execution for exported NeMo encoders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Mapping

_INPUT_NAMES = {"audio_signal", "length"}
_OUTPUT_NAMES = {"outputs", "encoded_lengths"}
_ENCODER_INPUT_RANK = 3
ENGINE_FILENAME = "encoder.plan"
METADATA_FILENAME = "metadata.json"
MODEL_FILENAME = "model.nemo"


def _validate_tensor_names(metadata: dict[str, Any]) -> None:
    for key, expected_names in (("input_names", _INPUT_NAMES), ("output_names", _OUTPUT_NAMES)):
        names = metadata.get(key)
        if (
            not isinstance(names, list)
            or any(not isinstance(name, str) for name in names)
            or set(names) != expected_names
        ):
            msg = f"Unexpected TensorRT encoder {key.replace('_', ' ')}: {names!r}"
            raise ValueError(msg)


def _validate_profile(metadata: dict[str, Any]) -> None:
    profile = metadata.get("profile")
    if not isinstance(profile, dict):
        msg = f"Invalid TensorRT engine profile: {profile!r}"
        raise TypeError(msg)
    points = [profile.get(point) for point in ("min", "opt", "max")]
    if any(not isinstance(point, dict) for point in points):
        msg = f"Invalid TensorRT engine profile points: {points!r}"
        raise TypeError(msg)
    for dimension in ("batch", "feature_frames"):
        values = [point.get(dimension) for point in points]
        if any(type(value) is not int for value in values) or not 1 <= values[0] <= values[1] <= values[2]:
            msg = f"Invalid TensorRT engine profile {dimension}: {values!r}"
            raise ValueError(msg)


def load_engine_metadata(
    engine_dir: str | Path,
    *,
    model_type: str,
    required_positive_ints: tuple[str, ...],
) -> dict[str, Any]:
    """Load and validate a TensorRT encoder bundle manifest."""
    path = Path(engine_dir) / METADATA_FILENAME
    if not path.is_file():
        msg = f"TensorRT engine metadata not found: {path}"
        raise FileNotFoundError(msg)
    try:
        metadata = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        msg = f"Could not read TensorRT engine metadata: {path}"
        raise ValueError(msg) from error
    if not isinstance(metadata, dict):
        msg = f"TensorRT engine metadata must be a JSON object: {path}"
        raise TypeError(msg)

    expected_values = {
        "schema_version": 1,
        "model_type": model_type,
        "precision": "fp16",
        "engine_file": ENGINE_FILENAME,
        "model_file": MODEL_FILENAME,
    }
    for key, expected in expected_values.items():
        if metadata.get(key) != expected:
            display_key = key.replace("_", " ")
            msg = f"Unexpected TensorRT engine metadata {display_key}: {metadata.get(key)!r}; expected {expected!r}"
            raise ValueError(msg)
    _validate_tensor_names(metadata)
    for key in required_positive_ints:
        if type(metadata.get(key)) is not int or metadata[key] < 1:
            msg = f"Invalid TensorRT engine metadata value for {key}: {metadata.get(key)!r}"
            raise ValueError(msg)

    _validate_profile(metadata)
    return metadata


def _trt_dtype_to_torch(dtype: object) -> torch.dtype:
    name = str(dtype).upper()
    mappings = (
        (("FP16", "FLOAT16", ".HALF"), torch.float16),
        (("FP32", "FLOAT32", ".FLOAT"), torch.float32),
        (("INT64",), torch.int64),
        (("INT32",), torch.int32),
        (("UINT8",), torch.uint8),
        (("INT8",), torch.int8),
        (("BOOL",), torch.bool),
    )
    for aliases, torch_dtype in mappings:
        if any(alias in name for alias in aliases):
            return torch_dtype
    msg = f"Unsupported TensorRT tensor dtype: {dtype!r}"
    raise TypeError(msg)


class TensorRTEncoderSession:
    """Persistent TensorRT execution context for an exported NeMo encoder."""

    def __init__(self, engine_path: str | Path) -> None:
        if not torch.cuda.is_available():
            msg = "TensorRT encoder inference requires CUDA"
            raise RuntimeError(msg)
        try:
            import tensorrt as trt
        except ImportError as error:
            msg = "TensorRT Python bindings are required for the TensorRT backend"
            raise RuntimeError(msg) from error

        path = Path(engine_path)
        if not path.is_file():
            msg = f"TensorRT encoder engine not found: {path}"
            raise FileNotFoundError(msg)

        self.device = torch.device("cuda")
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(path.read_bytes())
        if self._engine is None:
            msg = f"Could not deserialize TensorRT encoder engine: {path}"
            raise RuntimeError(msg)
        self._context = self._engine.create_execution_context()
        if self._context is None:
            msg = f"Could not create TensorRT encoder execution context: {path}"
            raise RuntimeError(msg)

        self._stream = torch.cuda.Stream(device=self.device)
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        # Audio lengths produce many dynamic shapes. Keep only one allocation
        # per output so worker GPU memory cannot grow with every observed shape.
        self._output_buffers: dict[str, torch.Tensor] = {}

    def _prepare_input(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type != "cuda":
            tensor = tensor.to(self.device, non_blocking=True)
        expected_dtype = _trt_dtype_to_torch(self._engine.get_tensor_dtype(name))
        if tensor.dtype != expected_dtype:
            tensor = tensor.to(expected_dtype)
        return tensor.contiguous()

    def _output_buffer(self, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        if any(dimension < 0 for dimension in shape):
            msg = f"TensorRT did not resolve encoder output shape for {name!r}: {shape}"
            raise RuntimeError(msg)
        dtype = _trt_dtype_to_torch(self._engine.get_tensor_dtype(name))
        output = self._output_buffers.get(name)
        if output is None or output.shape != shape or output.dtype != dtype:
            output = torch.empty(shape, dtype=dtype, device=self.device)
            self._output_buffers[name] = output
        return output

    def infer(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        missing = set(self.input_names) - set(inputs)
        if missing:
            msg = f"Missing TensorRT encoder inputs: {sorted(missing)}"
            raise KeyError(msg)
        unexpected = set(inputs) - set(self.input_names)
        if unexpected:
            msg = f"Unexpected TensorRT encoder inputs: {sorted(unexpected)}"
            raise KeyError(msg)
        prepared = {name: self._prepare_input(name, inputs[name]) for name in self.input_names}
        current_stream = torch.cuda.current_stream(self.device)
        self._stream.wait_stream(current_stream)

        with torch.cuda.stream(self._stream):
            for name, tensor in prepared.items():
                if self._context.set_input_shape(name, tuple(tensor.shape)) is False:
                    msg = f"Input shape {tuple(tensor.shape)} is outside the TensorRT profile for {name!r}"
                    raise RuntimeError(msg)
                if self._context.set_tensor_address(name, tensor.data_ptr()) is False:
                    msg = f"Failed to bind TensorRT input tensor {name!r}"
                    raise RuntimeError(msg)
                tensor.record_stream(self._stream)

            outputs = {}
            for name in self.output_names:
                output = self._output_buffer(name, tuple(self._context.get_tensor_shape(name)))
                if self._context.set_tensor_address(name, output.data_ptr()) is False:
                    msg = f"Failed to bind TensorRT output tensor {name!r}"
                    raise RuntimeError(msg)
                outputs[name] = output
            if not self._context.execute_async_v3(self._stream.cuda_stream):
                msg = "TensorRT encoder execute_async_v3 failed"
                raise RuntimeError(msg)
            for output in outputs.values():
                output.record_stream(self._stream)

        current_stream.wait_stream(self._stream)
        return outputs

    def close(self) -> None:
        self._output_buffers.clear()
        self._context = None
        self._engine = None
        self._runtime = None

    def input_shape_range(
        self,
        name: str,
        profile_index: int = 0,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Return an input tensor's minimum, optimum, and maximum profile shapes."""
        if name not in self.input_names:
            msg = f"TensorRT encoder input not found: {name!r}"
            raise KeyError(msg)
        profile_shapes = self._engine.get_tensor_profile_shape(name, profile_index)
        return tuple(tuple(int(dimension) for dimension in shape) for shape in profile_shapes)

    def max_input_shape(self, name: str, profile_index: int = 0) -> tuple[int, ...]:
        """Return an input tensor's maximum shape from the serialized engine profile."""
        return self.input_shape_range(name, profile_index)[2]


class TensorRTEncoder(torch.nn.Module):
    """NeMo Conformer encoder adapter backed by a TensorRT session."""

    def __init__(
        self,
        engine_path: str | Path,
        *,
        subsampling_factor: int,
        session: TensorRTEncoderSession | None = None,
    ) -> None:
        super().__init__()
        if session is None:
            session = TensorRTEncoderSession(engine_path)
        self.session = session
        self.subsampling_factor = int(subsampling_factor)

        missing_inputs = _INPUT_NAMES - set(self.session.input_names)
        if missing_inputs:
            msg = f"TensorRT encoder is missing inputs: {sorted(missing_inputs)}"
            raise ValueError(msg)
        missing_outputs = _OUTPUT_NAMES - set(self.session.output_names)
        if missing_outputs:
            msg = f"TensorRT encoder is missing outputs: {sorted(missing_outputs)}"
            raise ValueError(msg)

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if audio_signal.ndim != _ENCODER_INPUT_RANK or length.ndim != 1 or audio_signal.shape[0] != length.shape[0]:
            msg = "TensorRT encoder expects audio_signal shaped [batch, features, frames] and length shaped [batch]"
            raise ValueError(msg)

        batch_size, feature_count, feature_frames = audio_signal.shape
        min_shape, _, max_shape = self.session.input_shape_range("audio_signal")
        if feature_count != min_shape[1]:
            msg = f"TensorRT encoder expects {min_shape[1]} input features, got {feature_count}"
            raise ValueError(msg)
        if feature_frames > max_shape[2]:
            msg = f"TensorRT encoder input shape {tuple(audio_signal.shape)} exceeds profile maximum {max_shape}"
            raise ValueError(msg)

        padded_batch = max(batch_size, min_shape[0])
        padded_frames = max(feature_frames, min_shape[2])
        if padded_batch != batch_size or padded_frames != feature_frames:
            audio_signal = torch.nn.functional.pad(
                audio_signal,
                (0, padded_frames - feature_frames, 0, 0, 0, padded_batch - batch_size),
            )
            length = torch.nn.functional.pad(
                length,
                (0, padded_batch - batch_size),
                value=min_shape[2],
            )

        max_rows = max_shape[0]
        if padded_batch > max_rows:
            outputs = self._infer_split(
                audio_signal,
                length,
                max_rows=max_rows,
                min_shape=min_shape,
            )
        else:
            outputs = self.session.infer({"audio_signal": audio_signal, "length": length})
        if padded_batch == batch_size:
            return outputs["outputs"], outputs["encoded_lengths"]
        return outputs["outputs"][:batch_size], outputs["encoded_lengths"][:batch_size]

    def _infer_split(
        self,
        audio_signal: torch.Tensor,
        length: torch.Tensor,
        *,
        max_rows: int,
        min_shape: tuple[int, ...],
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        batch_size = audio_signal.shape[0]
        for start in range(0, batch_size, max_rows):
            end = min(start + max_rows, batch_size)
            group_signal = audio_signal[start:end]
            group_length = length[start:end]
            group_rows = end - start
            if group_rows < min_shape[0]:
                group_signal = torch.nn.functional.pad(
                    group_signal,
                    (0, 0, 0, 0, 0, min_shape[0] - group_rows),
                )
                group_length = torch.nn.functional.pad(
                    group_length,
                    (0, min_shape[0] - group_rows),
                    value=min_shape[2],
                )
            batch = self.session.infer({"audio_signal": group_signal, "length": group_length})
            for name, output in batch.items():
                output_slice = output[:group_rows]
                if name not in outputs:
                    outputs[name] = torch.empty(
                        (batch_size, *output_slice.shape[1:]),
                        dtype=output_slice.dtype,
                        device=output_slice.device,
                    )
                outputs[name][start:end].copy_(output_slice)
        return outputs

    def freeze(self) -> None:
        self.eval()

    def unfreeze(self, partial: bool = False) -> None:  # noqa: ARG002
        self.eval()

    def close(self) -> None:
        self.session.close()

    def max_input_shape(self, name: str) -> tuple[int, ...]:
        return self.session.max_input_shape(name)
