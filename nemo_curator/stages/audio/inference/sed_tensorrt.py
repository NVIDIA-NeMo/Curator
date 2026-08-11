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

"""TensorRT execution for the CNN14 SED neural core."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional

if TYPE_CHECKING:
    from collections.abc import Mapping


class SedCore(nn.Module):
    """CNN14 neural core; the checkpoint's spectrogram frontend stays in PyTorch."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.bn0 = model.bn0
        self.conv_block1 = model.conv_block1
        self.conv_block2 = model.conv_block2
        self.conv_block3 = model.conv_block3
        self.conv_block4 = model.conv_block4
        self.conv_block5 = model.conv_block5
        self.conv_block6 = model.conv_block6
        self.fc1 = model.fc1
        self.fc_audioset = model.fc_audioset

    def forward(self, logmel: torch.Tensor) -> torch.Tensor:
        value = logmel.transpose(1, 3)
        value = self.bn0(value)
        value = value.transpose(1, 3)
        value = self.conv_block1(value, pool_size=(2, 2), pool_type="avg")
        value = self.conv_block2(value, pool_size=(2, 2), pool_type="avg")
        value = self.conv_block3(value, pool_size=(2, 2), pool_type="avg")
        value = self.conv_block4(value, pool_size=(2, 2), pool_type="avg")
        value = self.conv_block5(value, pool_size=(2, 2), pool_type="avg")
        value = self.conv_block6(value, pool_size=(1, 1), pool_type="avg")
        value = torch.mean(value, dim=3)
        value = functional.max_pool1d(value, kernel_size=3, stride=1, padding=1) + functional.avg_pool1d(
            value, kernel_size=3, stride=1, padding=1
        )
        value = value.transpose(1, 2)
        value = functional.relu(self.fc1(value))
        return torch.sigmoid(self.fc_audioset(value))


def extract_features(model: nn.Module, waveforms: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Run the checkpoint's exact spectrogram and log-mel frontend."""
    spectrogram = model.spectrogram_extractor(waveforms)
    logmel = model.logmel_extractor(spectrogram)
    return logmel, logmel.shape[2]


def postprocess(segmentwise: torch.Tensor, frames_num: int) -> torch.Tensor:
    """Restore PANNs framewise output geometry from CNN14 segment outputs."""
    framewise = segmentwise.repeat_interleave(32, dim=1)
    if framewise.shape[1] < frames_num:
        padding = framewise[:, -1:, :].expand(-1, frames_num - framewise.shape[1], -1)
        framewise = torch.cat((framewise, padding), dim=1)
    return framewise[:, :frames_num, :]


def _trt_dtype_to_torch(dtype: object) -> torch.dtype:
    name = str(dtype).upper()
    mappings = (
        (("FP16", "FLOAT16", ".HALF"), torch.float16),
        (("FP32", "FLOAT32", ".FLOAT"), torch.float32),
        (("INT64",), torch.int64),
        (("INT32",), torch.int32),
        (("INT8",), torch.int8),
        (("BOOL",), torch.bool),
    )
    for aliases, torch_dtype in mappings:
        if any(alias in name for alias in aliases):
            return torch_dtype
    msg = f"Unsupported TensorRT tensor dtype: {dtype!r}"
    raise TypeError(msg)


class TensorRTRunner:
    """Persistent TensorRT runner with shape-specific context memory."""

    def __init__(self, engine_path: str | Path) -> None:
        if not torch.cuda.is_available():
            msg = "TensorRT SED inference requires CUDA"
            raise RuntimeError(msg)

        path = Path(engine_path)
        if not path.is_file():
            msg = f"TensorRT engine not found: {path}"
            raise FileNotFoundError(msg)

        try:
            import tensorrt as trt
        except ImportError as error:
            msg = "TensorRT Python bindings are required for the TensorRT SED backend"
            raise RuntimeError(msg) from error

        self._trt = trt
        self._validate_target(path)
        logger = trt.Logger(trt.Logger.ERROR)
        self._runtime = trt.Runtime(logger)
        self._engine = self._runtime.deserialize_cuda_engine(path.read_bytes())
        if self._engine is None:
            msg = f"Could not deserialize TensorRT engine: {path}"
            raise RuntimeError(msg)
        self._context = self._engine.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
        if self._context is None:
            msg = f"Could not create TensorRT execution context: {path}"
            raise RuntimeError(msg)

        self._device_memory: torch.Tensor | None = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
            else:
                self._output_names.append(name)

    def _validate_target(self, path: Path) -> None:
        metadata_path = path.with_suffix(path.suffix + ".json")
        if not metadata_path.is_file():
            msg = f"Missing engine provenance sidecar: {metadata_path}"
            raise RuntimeError(msg)
        metadata = json.loads(metadata_path.read_text())
        current_cc = list(torch.cuda.get_device_capability())
        if metadata.get("compute_capability") != current_cc:
            msg = (
                "Engine compute capability mismatch: "
                f"built for {metadata.get('compute_capability')}, current GPU is {current_cc}"
            )
            raise RuntimeError(msg)
        if metadata.get("tensorrt_version") != self._trt.__version__:
            msg = (
                "Engine TensorRT version mismatch: "
                f"built with {metadata.get('tensorrt_version')}, runtime is {self._trt.__version__}"
            )
            raise RuntimeError(msg)

    def _bind_inputs(self, inputs: Mapping[str, torch.Tensor]) -> torch.device:
        missing = set(self._input_names) - set(inputs)
        extra = set(inputs) - set(self._input_names)
        if missing or extra:
            msg = f"TensorRT inputs mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
            raise ValueError(msg)

        devices = {inputs[name].device for name in self._input_names}
        if len(devices) != 1:
            msg = f"TensorRT inputs must share one CUDA device, got {devices}"
            raise ValueError(msg)
        device = devices.pop()
        if device.type != "cuda":
            msg = "TensorRT inputs must be CUDA tensors"
            raise ValueError(msg)

        for name in self._input_names:
            tensor = inputs[name]
            expected_dtype = _trt_dtype_to_torch(self._engine.get_tensor_dtype(name))
            if tensor.dtype != expected_dtype:
                msg = f"TensorRT input {name} has dtype {tensor.dtype}; expected {expected_dtype}"
                raise ValueError(msg)
            if not tensor.is_contiguous():
                msg = f"TensorRT input {name} must be contiguous"
                raise ValueError(msg)
            accepted = self._context.set_input_shape(name, tuple(tensor.shape))
            if accepted is False:
                msg = f"Input shape {tuple(tensor.shape)} is outside the TensorRT profile for {name!r}"
                raise ValueError(msg)
            self._context.set_tensor_address(name, tensor.data_ptr())
        return device

    def _prepare_device_memory(self, device: torch.device) -> None:
        required = self._context.update_device_memory_size_for_shapes()
        if required < 0:
            msg = "TensorRT could not determine shape-specific context memory"
            raise RuntimeError(msg)
        if self._device_memory is None or self._device_memory.numel() < required:
            self._device_memory = torch.empty(required, dtype=torch.uint8, device=device)
        accepted = self._context.set_device_memory(self._device_memory.data_ptr(), self._device_memory.numel())
        if accepted is False:
            msg = f"TensorRT rejected {self._device_memory.numel()} bytes of context memory"
            raise RuntimeError(msg)

    def __call__(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        device = self._bind_inputs(inputs)
        self._prepare_device_memory(device)
        outputs: dict[str, torch.Tensor] = {}
        for name in self._output_names:
            shape = tuple(self._context.get_tensor_shape(name))
            if any(dimension < 0 for dimension in shape):
                msg = f"Unresolved TensorRT output shape for {name}: {shape}"
                raise RuntimeError(msg)
            output = torch.empty(
                shape,
                dtype=_trt_dtype_to_torch(self._engine.get_tensor_dtype(name)),
                device=device,
            )
            outputs[name] = output
            self._context.set_tensor_address(name, output.data_ptr())

        stream = torch.cuda.current_stream(device).cuda_stream
        if not self._context.execute_async_v3(stream_handle=stream):
            msg = "TensorRT execute_async_v3 failed"
            raise RuntimeError(msg)
        return outputs

    def close(self) -> None:
        self._device_memory = None
        self._context = None
        self._engine = None
        self._runtime = None


class TensorRTSed:
    """Reusable SED adapter preserving the checkpoint's PyTorch frontend."""

    def __init__(self, model: nn.Module, engine_path: str | Path) -> None:
        self.spectrogram = model.spectrogram_extractor.to("cuda").eval()
        self.logmel = model.logmel_extractor.to("cuda").eval()
        self.runner = TensorRTRunner(engine_path)

    @torch.inference_mode()
    def __call__(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Return framewise probabilities for padded ``[batch, samples]`` input."""
        waveforms = waveforms.to(device="cuda", dtype=torch.float32).contiguous()
        spectrogram = self.spectrogram(waveforms)
        logmel = self.logmel(spectrogram)
        segmentwise = self.runner(logmel=logmel.contiguous())["segmentwise"]
        return postprocess(segmentwise, logmel.shape[2])

    def close(self) -> None:
        self.runner.close()
