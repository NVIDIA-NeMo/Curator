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

"""Export Silero's exact 16 kHz inference core and build a TensorRT engine."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import time
from pathlib import Path
from typing import TYPE_CHECKING

from nemo_curator.utils.atomic_io import write_json_atomically

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch


_SAMPLE_RATE = 16000
_WINDOW_SIZE = 512
_CONTEXT_SIZE = 64
_MODEL_INPUT_SIZE = _WINDOW_SIZE + _CONTEXT_SIZE
_STATE_SIZE = 128
_ONNX_OPSET = 16
_INPUT_NAMES = {"input", "state"}
_OUTPUT_NAMES = {"output", "stateN"}
_MODEL_TYPE = "SileroVAD_16k"


def _write_metadata_sidecar(path: Path, metadata: dict[str, object]) -> None:
    """Publish a complete provenance sidecar with one atomic replacement."""
    write_json_atomically(path, metadata, indent=2, sort_keys=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_silero_core():  # noqa: ANN202
    """Rebuild the branch-free 16 kHz core from the official weights."""
    import torch
    from silero_vad import load_silero_vad
    from torch.nn import functional

    class Silero16kInferenceCore(torch.nn.Module):
        """Branch-free 576-sample recurrent inference graph."""

        def __init__(self, source) -> None:  # noqa: ANN001
            super().__init__()
            self.register_buffer("forward_basis", source.stft.forward_basis_buffer.detach().clone())
            self.register_buffer("right_reflect_indices", torch.arange(574, 510, -1, dtype=torch.int64))
            self.encoder = torch.nn.ModuleList()
            for index in range(4):
                source_conv = getattr(source.encoder, str(index)).reparam_conv
                conv = torch.nn.Conv1d(
                    source_conv.in_channels,
                    source_conv.out_channels,
                    source_conv.kernel_size,
                    stride=source_conv.stride,
                    padding=source_conv.padding,
                    dilation=source_conv.dilation,
                )
                conv.load_state_dict(source_conv.state_dict())
                self.encoder.append(conv)

            source_rnn = source.decoder.rnn
            self.rnn_weight_ih = torch.nn.Parameter(source_rnn.weight_ih.detach().clone())
            self.rnn_weight_hh = torch.nn.Parameter(source_rnn.weight_hh.detach().clone())
            self.rnn_bias_ih = torch.nn.Parameter(source_rnn.bias_ih.detach().clone())
            self.rnn_bias_hh = torch.nn.Parameter(source_rnn.bias_hh.detach().clone())
            source_output = getattr(source.decoder.decoder, "2")
            self.output = torch.nn.Conv1d(_STATE_SIZE, 1, 1)
            self.output.load_state_dict(source_output.state_dict())

        def forward(self, audio: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            reflected = torch.index_select(audio, 1, self.right_reflect_indices)
            padded = torch.cat((audio, reflected), dim=1).unsqueeze(1)
            transform = functional.conv1d(padded, self.forward_basis, stride=128)
            real = transform[:, :129, :]
            imaginary = transform[:, 129:, :]
            encoded = torch.sqrt(real.square() + imaginary.square())
            for conv in self.encoder:
                encoded = functional.relu(conv(encoded))

            recurrent_input = encoded.squeeze(-1)
            gates = functional.linear(recurrent_input, self.rnn_weight_ih, self.rnn_bias_ih)
            gates = gates + functional.linear(state[:, 0, :], self.rnn_weight_hh, self.rnn_bias_hh)
            input_gate = torch.sigmoid(gates[:, 0:128])
            forget_gate = torch.sigmoid(gates[:, 128:256])
            candidate = torch.tanh(gates[:, 256:384])
            output_gate = torch.sigmoid(gates[:, 384:512])
            cell = forget_gate * state[:, 1, :] + input_gate * candidate
            hidden = output_gate * torch.tanh(cell)
            probability = torch.sigmoid(self.output(functional.relu(hidden).unsqueeze(-1))).squeeze(-1)
            return probability, torch.stack((hidden, cell), dim=1)

    official = load_silero_vad()._model
    official = official.eval()
    return official, Silero16kInferenceCore(official).eval()


def _validate_pytorch_core(official, core) -> None:  # noqa: ANN001
    """Prove the branch-free graph matches Silero's official 16 kHz core."""
    import torch

    generator = torch.Generator().manual_seed(0)
    for batch_size in (1, 4, 17):
        audio = torch.randn((batch_size, _MODEL_INPUT_SIZE), dtype=torch.float32, generator=generator)
        state = torch.randn((batch_size, 2, _STATE_SIZE), dtype=torch.float32, generator=generator)
        with torch.inference_mode():
            expected_output, expected_state = official(audio, state.transpose(0, 1).contiguous())
            output, next_state = core(audio, state)
        torch.testing.assert_close(output, expected_output, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(
            next_state,
            expected_state.transpose(0, 1).contiguous(),
            rtol=1e-6,
            atol=1e-7,
        )


def _export_silero_onnx(output_path: Path):  # noqa: ANN202
    """Export and validate a dynamic-batch, branch-free ONNX graph."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    official, core = _make_silero_core()
    _validate_pytorch_core(official, core)
    audio = torch.zeros((1, _MODEL_INPUT_SIZE), dtype=torch.float32)
    state = torch.zeros((1, 2, _STATE_SIZE), dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            core,
            (audio, state),
            str(output_path),
            input_names=["input", "state"],
            output_names=["output", "stateN"],
            dynamic_axes={
                "input": {0: "batch"},
                "state": {0: "batch"},
                "output": {0: "batch"},
                "stateN": {0: "batch"},
            },
            opset_version=_ONNX_OPSET,
            do_constant_folding=True,
            dynamo=False,
        )

    exported = onnx.load(output_path)
    onnx.checker.check_model(exported)
    control_flow = [node.name for node in exported.graph.node if node.op_type in {"If", "Loop", "Scan"}]
    if control_flow:
        msg = f"Silero TensorRT graph contains control flow: {control_flow}"
        raise RuntimeError(msg)

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    generator = torch.Generator().manual_seed(1)
    max_abs_error = 0.0
    for batch_size in (1, 8):
        audio = torch.randn((batch_size, _MODEL_INPUT_SIZE), dtype=torch.float32, generator=generator)
        state = torch.randn((batch_size, 2, _STATE_SIZE), dtype=torch.float32, generator=generator)
        with torch.inference_mode():
            expected_output, expected_state = core(audio, state)
        output, next_state = session.run(None, {"input": audio.numpy(), "state": state.numpy()})
        np.testing.assert_allclose(output, expected_output.numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(next_state, expected_state.numpy(), rtol=1e-4, atol=1e-5)
        max_abs_error = max(
            max_abs_error,
            float(np.max(np.abs(output - expected_output.numpy()))),
            float(np.max(np.abs(next_state - expected_state.numpy()))),
        )
    return core, max_abs_error


def _shape_for_batch(network_shape: Sequence[int], batch_size: int) -> tuple[int, ...]:
    """Resolve Silero's dynamic batch dimension for one profile point."""
    return tuple(batch_size if dimension == -1 else dimension for dimension in network_shape)


def _validation_batch_sizes(max_batch_size: int) -> tuple[int, ...]:
    """Exercise batch one plus a representative batch inside the profile."""
    return tuple(dict.fromkeys((1, min(8, max_batch_size))))


def _validate_tensorrt_engine(
    engine_path: Path,
    core: torch.nn.Module,
    *,
    validation_batch_sizes: Sequence[int],
    fp16: bool = False,
) -> float:
    """Compare recurrent TensorRT inference with the exact PyTorch core."""
    import torch

    from nemo_curator.stages.audio.inference.tensorrt_encoder import TensorRTEncoderSession

    rtol, atol = (5e-2, 5e-3) if fp16 else (3e-4, 3e-5)
    generator = torch.Generator().manual_seed(2)
    maximum_error = 0.0
    session = TensorRTEncoderSession(engine_path)
    try:
        if set(session.input_names) != _INPUT_NAMES:
            msg = f"Silero TensorRT engine inputs must be {sorted(_INPUT_NAMES)}, got {sorted(session.input_names)}"
            raise ValueError(msg)
        if set(session.output_names) != _OUTPUT_NAMES:
            msg = f"Silero TensorRT engine outputs must be {sorted(_OUTPUT_NAMES)}, got {sorted(session.output_names)}"
            raise ValueError(msg)
        for batch_size in validation_batch_sizes:
            expected_state = torch.zeros((batch_size, 2, _STATE_SIZE), dtype=torch.float32)
            actual_state = expected_state.to(session.device)
            for _ in range(3):
                audio = torch.randn((batch_size, _MODEL_INPUT_SIZE), dtype=torch.float32, generator=generator)
                with torch.inference_mode():
                    expected_output, expected_state = core(audio, expected_state)
                outputs = session.infer({"input": audio.to(session.device), "state": actual_state})
                actual_output = outputs["output"].cpu()
                actual_state = outputs["stateN"].clone()
                actual_state_cpu = actual_state.cpu()
                torch.testing.assert_close(actual_output, expected_output, rtol=rtol, atol=atol)
                torch.testing.assert_close(actual_state_cpu, expected_state, rtol=rtol, atol=atol)
                maximum_error = max(
                    maximum_error,
                    float((actual_output - expected_output).abs().max()),
                    float((actual_state_cpu - expected_state).abs().max()),
                )
    finally:
        session.close()
    return maximum_error


def _driver_version() -> str | None:
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        value = pynvml.nvmlSystemGetDriverVersion()
        return value.decode() if isinstance(value, bytes) else str(value)
    except pynvml.NVMLError:
        return None


def _build_engine(  # noqa: C901, PLR0915
    args: argparse.Namespace,
    onnx_path: Path,
    core: torch.nn.Module,
) -> tuple[float, str, float]:
    try:
        import tensorrt as trt
    except ImportError as error:
        msg = "TensorRT Python bindings are required to build the Silero engine"
        raise RuntimeError(msg) from error

    logger = trt.Logger(trt.Logger.INFO if args.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        msg = f"Failed to parse {onnx_path}:\n{errors}"
        raise RuntimeError(msg)

    input_names = {network.get_input(index).name for index in range(network.num_inputs)}
    if input_names != _INPUT_NAMES:
        msg = f"Expected Silero ONNX inputs {sorted(_INPUT_NAMES)}, got {sorted(input_names)}"
        raise RuntimeError(msg)
    output_names = {network.get_output(index).name for index in range(network.num_outputs)}
    if output_names != _OUTPUT_NAMES:
        msg = f"Expected Silero ONNX outputs {sorted(_OUTPUT_NAMES)}, got {sorted(output_names)}"
        raise RuntimeError(msg)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gb * (1 << 30))
    # TF32 recurrent error accumulates over long recordings and can shift VAD decisions.
    config.clear_flag(trt.BuilderFlag.TF32)
    if args.fp16:
        if not builder.platform_has_fast_fp16:
            msg = "This GPU does not provide fast FP16 TensorRT kernels"
            raise RuntimeError(msg)
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    profiles = {}
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        network_shape = tuple(tensor.shape)
        minimum = _shape_for_batch(network_shape, args.min_batch)
        optimum = _shape_for_batch(network_shape, args.opt_batch)
        maximum = _shape_for_batch(network_shape, args.max_batch)
        if profile.set_shape(tensor.name, minimum, optimum, maximum) is False:
            msg = f"Could not set TensorRT profile for {tensor.name}: {minimum}/{optimum}/{maximum}"
            raise RuntimeError(msg)
        profiles[tensor.name] = (minimum, optimum, maximum)
    if config.add_optimization_profile(profile) < 0:
        msg = f"TensorRT rejected Silero input profiles: {profiles}"
        raise RuntimeError(msg)

    started = time.monotonic()
    serialized_engine = builder.build_serialized_network(network, config)
    build_seconds = time.monotonic() - started
    if serialized_engine is None:
        msg = "TensorRT failed to build the Silero engine"
        raise RuntimeError(msg)

    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_bytes(serialized_engine)
    try:
        recurrent_max_abs = _validate_tensorrt_engine(
            temporary,
            core,
            validation_batch_sizes=_validation_batch_sizes(args.max_batch),
            fp16=args.fp16,
        )
        temporary.replace(args.output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return build_seconds, trt.__version__, recurrent_max_abs


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Destination .plan/.engine file")
    parser.add_argument(
        "--onnx-output",
        type=Path,
        default=None,
        help="Destination for the generated ONNX graph (default: beside the engine)",
    )
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=16)
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument("--workspace-gb", type=int, default=2)
    parser.add_argument("--fp16", action="store_true", help="Build FP16 instead of the parity-oriented FP32 default")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.min_batch != 1 or not args.min_batch <= args.opt_batch <= args.max_batch:
        parser.error("batch profile must satisfy min-batch = 1 <= opt-batch <= max-batch")
    if args.workspace_gb < 1:
        parser.error("workspace-gb must be at least 1")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    import torch

    args = _parse_args(argv)
    if not torch.cuda.is_available():
        msg = "CUDA is unavailable; build the TensorRT engine on its target GPU"
        raise RuntimeError(msg)
    if args.output.exists() and not args.force:
        msg = f"{args.output} already exists; use --force to rebuild it"
        raise RuntimeError(msg)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx_path = args.onnx_output or args.output.with_suffix(".onnx")
    if onnx_path.resolve() == args.output.resolve():
        msg = "ONNX output and TensorRT engine output must be different files"
        raise ValueError(msg)

    core, onnx_max_abs = _export_silero_onnx(onnx_path)
    build_seconds, tensorrt_version, recurrent_max_abs = _build_engine(args, onnx_path, core)
    metadata = {
        "schema_version": 1,
        "build_seconds": build_seconds,
        "compute_capability": list(torch.cuda.get_device_capability()),
        "context_size": _CONTEXT_SIZE,
        "driver_version": _driver_version(),
        "engine": str(args.output.resolve()),
        "engine_bytes": args.output.stat().st_size,
        "engine_sha256": _sha256(args.output),
        "gpu": torch.cuda.get_device_name(),
        "input_names": ["input", "state"],
        "model_input_size": _MODEL_INPUT_SIZE,
        "model_type": _MODEL_TYPE,
        "onnx": str(onnx_path.resolve()),
        "onnx_bytes": onnx_path.stat().st_size,
        "onnx_max_abs": onnx_max_abs,
        "onnx_opset": _ONNX_OPSET,
        "onnx_sha256": _sha256(onnx_path),
        "output_names": ["output", "stateN"],
        "precision": "fp16" if args.fp16 else "fp32",
        "profiles": {
            "input": {
                "min": [args.min_batch, _MODEL_INPUT_SIZE],
                "opt": [args.opt_batch, _MODEL_INPUT_SIZE],
                "max": [args.max_batch, _MODEL_INPUT_SIZE],
            },
            "state": {
                "min": [args.min_batch, 2, _STATE_SIZE],
                "opt": [args.opt_batch, 2, _STATE_SIZE],
                "max": [args.max_batch, 2, _STATE_SIZE],
            },
        },
        "recurrent_max_abs": recurrent_max_abs,
        "recurrent_validation": {"batches": list(_validation_batch_sizes(args.max_batch)), "steps": 3},
        "sample_rate": _SAMPLE_RATE,
        "silero_vad_version": importlib.metadata.version("silero-vad"),
        "state_size": _STATE_SIZE,
        "tensorrt_version": tensorrt_version,
        "tf32_enabled": False,
        "torch_cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "window_size": _WINDOW_SIZE,
        "workspace_gb": args.workspace_gb,
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    _write_metadata_sidecar(metadata_path, metadata)
    print(
        f"Built {args.output} in {build_seconds:.1f}s; "
        f"ONNX max_abs={onnx_max_abs:.3g}; TensorRT recurrent max_abs={recurrent_max_abs:.3g}"
    )


if __name__ == "__main__":
    main()
