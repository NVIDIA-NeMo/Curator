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

import gc
import hashlib
import json
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.nn import functional

from nemo_curator.models.sed import get_model_class
from nemo_curator.models.sed.base import SEDResult
from nemo_curator.models.sed.panns import PANNsSEDAdapter

if TYPE_CHECKING:
    from collections.abc import Mapping


_TENSORRT_MODEL_TYPE = "Cnn14_DecisionLevelMax"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_engine_metadata(
    engine_path: Path,
    expected: Mapping[str, object],
    *,
    compute_capability: list[int],
    tensorrt_version: str,
) -> dict[str, object]:
    """Reject an engine whose immutable build contract differs from this adapter."""
    metadata_path = engine_path.with_suffix(engine_path.suffix + ".json")
    if not metadata_path.is_file():
        msg = f"Missing engine provenance sidecar: {metadata_path}"
        raise RuntimeError(msg)
    metadata = json.loads(metadata_path.read_text())
    if not isinstance(metadata, dict):
        msg = f"Engine provenance sidecar must contain a JSON object: {metadata_path}"
        raise TypeError(msg)

    runtime_expected = {
        "compute_capability": compute_capability,
        "engine_sha256": _sha256(engine_path),
        "tensorrt_version": tensorrt_version,
        **expected,
    }
    mismatches = {
        key: {"engine": metadata.get(key), "runtime": value}
        for key, value in runtime_expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: engine={values['engine']!r}, runtime={values['runtime']!r}" for key, values in mismatches.items()
        )
        msg = f"TensorRT engine contract mismatch ({details})"
        raise RuntimeError(msg)
    return metadata


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

    def __init__(self, engine_path: str | Path, *, expected_metadata: Mapping[str, object]) -> None:
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
        self._metadata = self._validate_target(path, expected_metadata)
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

        if self._input_names != ["logmel"] or self._output_names != ["segmentwise"]:
            msg = (
                "TensorRT engine I/O mismatch; expected input ['logmel'] and output ['segmentwise'], "
                f"got inputs={self._input_names}, outputs={self._output_names}"
            )
            raise RuntimeError(msg)
        self._validate_engine_io()

    def _validate_target(self, path: Path, expected_metadata: Mapping[str, object]) -> dict[str, object]:
        return _validate_engine_metadata(
            path,
            expected_metadata,
            compute_capability=list(torch.cuda.get_device_capability()),
            tensorrt_version=self._trt.__version__,
        )

    def _validate_engine_io(self) -> None:
        expected_shapes = {"logmel": (-1, 1, -1, 64), "segmentwise": (-1, -1, 527)}
        for name, expected_shape in expected_shapes.items():
            actual_shape = tuple(self._engine.get_tensor_shape(name))
            if actual_shape != expected_shape:
                msg = f"TensorRT engine tensor {name!r} has shape {actual_shape}; expected {expected_shape}"
                raise RuntimeError(msg)
            actual_dtype = _trt_dtype_to_torch(self._engine.get_tensor_dtype(name))
            if actual_dtype != torch.float32:
                msg = f"TensorRT engine tensor {name!r} has dtype {actual_dtype}; expected torch.float32"
                raise RuntimeError(msg)

        if self._engine.num_optimization_profiles != 1:
            msg = f"TensorRT SED engine must have exactly one optimization profile, got {self._engine.num_optimization_profiles}"
            raise RuntimeError(msg)
        profile = self._engine.get_tensor_profile_shape("logmel", 0)
        actual_profile = {key: list(shape) for key, shape in zip(("min", "opt", "max"), profile, strict=True)}
        metadata_profiles = self._metadata.get("profiles")
        expected_profile = metadata_profiles.get("logmel") if isinstance(metadata_profiles, dict) else None
        if actual_profile != expected_profile:
            msg = (
                "TensorRT engine profile differs from its provenance sidecar; "
                f"engine={actual_profile}, sidecar={expected_profile!r}"
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
            if not self._context.set_tensor_address(name, tensor.data_ptr()):
                msg = f"TensorRT rejected the device address for input {name!r}"
                raise RuntimeError(msg)
        return device

    def _prepare_device_memory(self, device: torch.device) -> None:
        unresolved = self._context.infer_shapes()
        if unresolved:
            msg = f"TensorRT could not infer shapes for tensors: {sorted(unresolved)}"
            raise RuntimeError(msg)
        required = self._context.update_device_memory_size_for_shapes()
        if required <= 0:
            msg = "TensorRT could not determine shape-specific context memory"
            raise RuntimeError(msg)
        if self._device_memory is None or self._device_memory.numel() < required:
            self._device_memory = torch.empty(required, dtype=torch.uint8, device=device)
        self._context.set_device_memory(self._device_memory.data_ptr(), self._device_memory.numel())

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
            if not self._context.set_tensor_address(name, output.data_ptr()):
                msg = f"TensorRT rejected the device address for output {name!r}"
                raise RuntimeError(msg)

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

    def __init__(
        self,
        model: nn.Module,
        engine_path: str | Path,
        *,
        expected_metadata: Mapping[str, object],
    ) -> None:
        self.spectrogram = model.spectrogram_extractor.to("cuda").eval()
        self.logmel = model.logmel_extractor.to("cuda").eval()
        self.runner = TensorRTRunner(engine_path, expected_metadata=expected_metadata)

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


@dataclass
class TensorRTPANNsSEDAdapter(PANNsSEDAdapter):
    """Run the PANNs CNN14 neural core with a target-specific TensorRT engine.

    Audio preprocessing, checkpoint resolution, batch padding, and the
    canonical ``SEDResult`` contract match ``PANNsSEDAdapter``. The checkpoint's
    spectrogram and log-mel frontend remains in PyTorch; only the CNN14 neural
    core runs in TensorRT. Engines are valid only for the GPU compute capability
    and TensorRT version recorded in their adjacent JSON sidecar.
    """

    tensorrt_engine_path: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.tensorrt_engine_path:
            msg = "tensorrt_engine_path is required for the TensorRT PANNs SED adapter"
            raise ValueError(msg)
        if self.model_type != _TENSORRT_MODEL_TYPE:
            msg = f"The TensorRT PANNs SED adapter supports only {_TENSORRT_MODEL_TYPE}"
            raise ValueError(msg)

    def load_model(self, *, num_gpus: int) -> None:
        """Load the PyTorch frontend and TensorRT runtime on one CUDA device."""
        if isinstance(num_gpus, bool) or not isinstance(num_gpus, Integral) or num_gpus <= 0:
            msg = f"TensorRTPANNsSEDAdapter requires a positive integer num_gpus, got {num_gpus!r}"
            raise ValueError(msg)
        if not torch.cuda.is_available():
            msg = f"TensorRTPANNsSEDAdapter received num_gpus={num_gpus}, but CUDA is not available"
            raise RuntimeError(msg)

        model_cls = get_model_class(self.model_type)
        model = model_cls(
            sample_rate=self.sample_rate,
            window_size=self.window_size,
            hop_size=self.hop_size,
            mel_bins=self.mel_bins,
            fmin=self.fmin,
            fmax=self.fmax,
            classes_num=self.classes_num,
        )
        checkpoint = self._load_checkpoint()
        model.load_state_dict(checkpoint["model"])
        model.eval()

        self._device = torch.device("cuda")
        checkpoint_source = self._resolve_checkpoint_path()
        expected_metadata: dict[str, object] = {
            "schema_version": 1,
            "checkpoint_sha256": _sha256(checkpoint_source),
            "model_type": self.model_type,
            "frontend": {
                "sample_rate": self.sample_rate,
                "window_size": self.window_size,
                "hop_size": self.hop_size,
                "mel_bins": self.mel_bins,
                "fmin": self.fmin,
                "fmax": self.fmax,
                "classes_num": self.classes_num,
            },
        }
        self._model = TensorRTSed(
            model,
            self.tensorrt_engine_path,
            expected_metadata=expected_metadata,
        )
        logger.info(
            "Loaded {} from {} with TensorRT engine {}",
            self.model_type,
            checkpoint_source,
            self.tensorrt_engine_path,
        )

    def unload_model(self) -> None:
        """Release the TensorRT context, PyTorch frontend, and CUDA cache."""
        runtime = self._model
        self._model = None
        self._device = None
        if runtime is not None:
            runtime.close()
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            logger.debug("CUDA cache clear skipped: {}", exc)

    def infer_batch(self, items: list[dict[str, object]]) -> list[SEDResult]:
        """Run one TensorRT call and preserve the PANNs adapter result schema."""
        if self._model is None:
            msg = "TensorRTPANNsSEDAdapter.load_model() must be called before inference"
            raise RuntimeError(msg)

        waveforms = [np.asarray(item["waveform"], dtype=np.float32) for item in items]
        padded = self._pad_to_rectangle(waveforms)
        tensor = torch.from_numpy(padded)
        framewise = self._model(tensor).cpu().numpy()
        fps = float(self.sample_rate) / self.hop_size

        results: list[SEDResult] = []
        for waveform, row in zip(waveforms, framewise):  # noqa: B905 - runtime preserves batch cardinality
            valid_frames = min(int(np.ceil(waveform.size / self.hop_size)), row.shape[0])
            results.append(
                SEDResult(
                    framewise_output=row,
                    fps=fps,
                    valid_frames=valid_frames,
                    original_num_samples=waveform.size,
                )
            )
        logger.info(
            "TensorRT PANNs SED batch: processed {} waveforms (max_samples={}, fps={:.1f})",
            len(waveforms),
            padded.shape[1],
            fps,
        )
        return results
