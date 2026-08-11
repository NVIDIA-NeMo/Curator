#!/usr/bin/env python3
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

"""Export CNN14's neural core and build a target-specific TensorRT engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from nemo_curator.stages.audio.inference.sed_tensorrt import SedCore, extract_features, postprocess

_MEL_BINS = 64
_SPLIT_ATOL = 1e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(checkpoint_path: Path):  # noqa: ANN202
    from nemo_curator.stages.audio.inference.sed_models.cnn14 import Cnn14DecisionLevelMax

    model = Cnn14DecisionLevelMax()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    return model.eval().to("cuda")


@torch.inference_mode()
def _export_onnx(checkpoint_path: Path, onnx_path: Path):  # noqa: ANN202
    model = _load_model(checkpoint_path)
    core = SedCore(model).eval()
    generator = torch.Generator(device="cuda").manual_seed(1234)
    waveforms = torch.randn((2, 5 * 16000), generator=generator, device="cuda") * 0.03
    features, frames_num = extract_features(model, waveforms)
    reference = model(waveforms)["framewise_output"]
    split = postprocess(core(features), frames_num)
    split_max_abs = (reference - split).abs().max().item()
    if split_max_abs > _SPLIT_ATOL:
        msg = f"SED core split changed PyTorch output: max_abs={split_max_abs}"
        raise RuntimeError(msg)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        core,
        (features,),
        str(onnx_path),
        input_names=["logmel"],
        output_names=["segmentwise"],
        dynamic_axes={
            "logmel": {0: "batch", 2: "frames"},
            "segmentwise": {0: "batch", 1: "segments"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    return core, features, split_max_abs


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


def _build_engine(args: argparse.Namespace, onnx_path: Path) -> tuple[float, str]:
    try:
        import tensorrt as trt
    except ImportError as error:
        msg = "TensorRT Python bindings are required to build the SED engine"
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
    if input_names != {"logmel"}:
        msg = f"Expected one 'logmel' ONNX input, got {sorted(input_names)}"
        raise RuntimeError(msg)

    config = builder.create_builder_config()
    config.builder_optimization_level = args.optimization_level
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gb * (1 << 30))
    if args.fp16:
        if not builder.platform_has_fast_fp16:
            msg = "This GPU does not provide fast FP16 TensorRT kernels"
            raise RuntimeError(msg)
        config.set_flag(trt.BuilderFlag.FP16)
    if args.no_tf32:
        config.clear_flag(trt.BuilderFlag.TF32)
    else:
        config.set_flag(trt.BuilderFlag.TF32)

    profile = builder.create_optimization_profile()
    minimum = (args.min_batch, 1, args.min_frames, _MEL_BINS)
    optimum = (args.opt_batch, 1, args.opt_frames, _MEL_BINS)
    maximum = (args.max_batch, 1, args.max_frames, _MEL_BINS)
    if profile.set_shape("logmel", minimum, optimum, maximum) is False:
        msg = f"TensorRT rejected logmel profile: {minimum}/{optimum}/{maximum}"
        raise RuntimeError(msg)
    config.add_optimization_profile(profile)

    started = time.monotonic()
    serialized_engine = builder.build_serialized_network(network, config)
    build_seconds = time.monotonic() - started
    if serialized_engine is None:
        msg = "TensorRT failed to build the SED engine"
        raise RuntimeError(msg)

    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_bytes(serialized_engine)
    temporary.replace(args.output)
    return build_seconds, trt.__version__


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Destination .plan/.engine file")
    parser.add_argument(
        "--onnx-output",
        type=Path,
        default=None,
        help="Destination for the generated ONNX graph (default: beside the engine)",
    )
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=16)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--min-frames", type=int, default=33)
    parser.add_argument("--opt-frames", type=int, default=1001)
    parser.add_argument("--max-frames", type=int, default=4001)
    parser.add_argument("--workspace-gb", type=int, default=32)
    parser.add_argument("--optimization-level", type=int, default=5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.min_batch <= args.opt_batch <= args.max_batch:
        parser.error("batch profile must satisfy 1 <= min-batch <= opt-batch <= max-batch")
    if not 1 <= args.min_frames <= args.opt_frames <= args.max_frames:
        parser.error("frame profile must satisfy 1 <= min-frames <= opt-frames <= max-frames")
    return args


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        msg = "CUDA is unavailable; build the TensorRT engine on its target GPU"
        raise RuntimeError(msg)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.output.exists() and not args.force:
        msg = f"{args.output} already exists; use --force to rebuild it"
        raise RuntimeError(msg)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx_path = args.onnx_output or args.output.with_suffix(".onnx")
    _, _, split_max_abs = _export_onnx(args.checkpoint, onnx_path)
    build_seconds, tensorrt_version = _build_engine(args, onnx_path)
    metadata = {
        "build_seconds": build_seconds,
        "builder_optimization_level": args.optimization_level,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "driver_version": _driver_version(),
        "engine": str(args.output.resolve()),
        "engine_bytes": args.output.stat().st_size,
        "engine_sha256": _sha256(args.output),
        "gpu": torch.cuda.get_device_name(),
        "onnx": str(onnx_path.resolve()),
        "onnx_sha256": _sha256(onnx_path),
        "precision": "fp16" if args.fp16 else "fp32",
        "profiles": {
            "logmel": {
                "min": [args.min_batch, 1, args.min_frames, _MEL_BINS],
                "opt": [args.opt_batch, 1, args.opt_frames, _MEL_BINS],
                "max": [args.max_batch, 1, args.max_frames, _MEL_BINS],
            }
        },
        "split_max_abs": split_max_abs,
        "tensorrt_version": tensorrt_version,
        "tf32_enabled": not args.no_tf32,
        "torch_cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "workspace_gb": args.workspace_gb,
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Built {args.output} in {build_seconds:.1f}s; split_max_abs={split_max_abs:.3g}")


if __name__ == "__main__":
    main()
