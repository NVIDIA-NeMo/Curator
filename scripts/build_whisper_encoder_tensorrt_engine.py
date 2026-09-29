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

"""Export Whisper's audio encoder and build a target-specific TensorRT engine.

Language identification runs the encoder over a fixed 30-second window and then a
single decoder step, so the encoder holds nearly all of the compute. Only the batch
dimension is dynamic: ``whisper.pad_or_trim`` fixes the Mel length at ``N_FRAMES``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import time
from pathlib import Path

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _import_whisper():  # noqa: ANN202
    try:
        import whisper
    except ImportError as error:
        msg = "OpenAI Whisper is required to export the encoder. Install: pip install openai-whisper"
        raise RuntimeError(msg) from error
    return whisper


def _load_model(checkpoint: str):  # noqa: ANN202
    whisper = _import_whisper()
    return whisper.load_model(checkpoint, device="cuda").eval()


@torch.inference_mode()
def _export_onnx(model: object, onnx_path: Path, n_mels: int, n_frames: int) -> None:
    # The encoder is exported as-is. Its only shape assertion compares the post-convolution
    # length against the positional embedding, which holds for every batch at fixed n_frames.
    encoder = model.encoder.eval()  # type: ignore[attr-defined]
    generator = torch.Generator(device="cuda").manual_seed(1234)
    mel = torch.randn((2, n_mels, n_frames), generator=generator, device="cuda", dtype=torch.float32)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        # Whisper prefers fused scaled_dot_product_attention, which exports poorly.
        # Its own context manager falls back to the explicit attention math.
        disable_sdpa = getattr(_import_whisper().model, "disable_sdpa", None)
        if disable_sdpa is not None:
            stack.enter_context(disable_sdpa())
        torch.onnx.export(
            encoder,
            (mel,),
            str(onnx_path),
            input_names=["mel"],
            output_names=["audio_features"],
            dynamic_axes={"mel": {0: "batch"}, "audio_features": {0: "batch"}},
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )


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


def _build_engine(args: argparse.Namespace, onnx_path: Path, n_mels: int, n_frames: int) -> tuple[float, str]:
    try:
        import tensorrt as trt
    except ImportError as error:
        msg = "TensorRT Python bindings are required to build the Whisper encoder engine"
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
    if input_names != {"mel"}:
        msg = f"Expected one 'mel' ONNX input, got {sorted(input_names)}"
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
    minimum = (args.min_batch, n_mels, n_frames)
    optimum = (args.opt_batch, n_mels, n_frames)
    maximum = (args.max_batch, n_mels, n_frames)
    if profile.set_shape("mel", minimum, optimum, maximum) is False:
        msg = f"TensorRT rejected mel profile: {minimum}/{optimum}/{maximum}"
        raise RuntimeError(msg)
    config.add_optimization_profile(profile)

    started = time.monotonic()
    serialized_engine = builder.build_serialized_network(network, config)
    build_seconds = time.monotonic() - started
    if serialized_engine is None:
        msg = "TensorRT failed to build the Whisper encoder engine"
        raise RuntimeError(msg)

    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_bytes(serialized_engine)
    temporary.replace(args.output)
    return build_seconds, trt.__version__


@torch.inference_mode()
def _verify(  # noqa: PLR0913
    model: object,
    engine_path: Path,
    n_mels: int,
    n_frames: int,
    batch: int,
    *,
    fp16: bool,
) -> dict[str, float]:
    """Compare TensorRT encoder features and detected languages against PyTorch.

    Random Mels are not speech, so the languages themselves are meaningless; what the
    check establishes is that both encoders drive the decoder to the same token.
    """
    from nemo_curator.models.audio.lid.whisper import WhisperTensorRTEncoder

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(4321)
    mel = torch.randn((batch, n_mels, n_frames), generator=generator, device=device, dtype=torch.float32)

    torch_mel = mel.to(torch.float16) if fp16 else mel
    torch_features = model.encoder(torch_mel)  # type: ignore[attr-defined]
    encoder = WhisperTensorRTEncoder(engine_path)
    trt_features = encoder(mel)

    difference = (torch_features.float() - trt_features.float()).abs()
    scale = torch_features.float().abs().amax().clamp(min=1e-6)

    torch_tokens, _ = model.detect_language(torch_features)  # type: ignore[attr-defined]
    trt_tokens, _ = model.detect_language(trt_features)  # type: ignore[attr-defined]
    matches = (torch_tokens == trt_tokens).sum().item()
    agreement = matches / int(torch_tokens.numel())

    encoder.close()
    return {
        "feature_max_abs_diff": float(difference.amax()),
        "feature_mean_abs_diff": float(difference.mean()),
        "feature_relative_max_diff": float(difference.amax() / scale),
        "random_mel_language_agreement": agreement,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Whisper model name or local .pt path passed to whisper.load_model",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination .plan file")
    parser.add_argument(
        "--onnx-output",
        type=Path,
        default=None,
        help="Destination for the generated ONNX graph (default: beside the engine)",
    )
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=16)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--workspace-gb", type=int, default=32)
    parser.add_argument("--optimization-level", type=int, default=5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.min_batch <= args.opt_batch <= args.max_batch:
        parser.error("batch profile must satisfy 1 <= min-batch <= opt-batch <= max-batch")
    return args


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        msg = "CUDA is unavailable; build the TensorRT engine on its target GPU"
        raise RuntimeError(msg)
    if args.output.exists() and not args.force:
        msg = f"{args.output} already exists; use --force to rebuild it"
        raise RuntimeError(msg)

    whisper = _import_whisper()
    n_frames = int(whisper.audio.N_FRAMES)
    model = _load_model(args.checkpoint)
    n_mels = int(model.dims.n_mels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx_path = args.onnx_output or args.output.with_suffix(".onnx")
    _export_onnx(model, onnx_path, n_mels, n_frames)
    build_seconds, tensorrt_version = _build_engine(args, onnx_path, n_mels, n_frames)

    verification: dict[str, float] = {}
    if not args.skip_verify:
        verification = _verify(model, args.output, n_mels, n_frames, args.opt_batch, fp16=args.fp16)

    checkpoint_path = Path(args.checkpoint)
    metadata = {
        "build_seconds": build_seconds,
        "builder_optimization_level": args.optimization_level,
        "checkpoint": str(checkpoint_path.resolve()) if checkpoint_path.is_file() else args.checkpoint,
        "checkpoint_sha256": _sha256(checkpoint_path) if checkpoint_path.is_file() else None,
        "compute_capability": list(torch.cuda.get_device_capability()),
        "driver_version": _driver_version(),
        "engine": str(args.output.resolve()),
        "engine_bytes": args.output.stat().st_size,
        "engine_sha256": _sha256(args.output),
        "gpu": torch.cuda.get_device_name(),
        "model_type": "whisper_encoder",
        "n_audio_state": int(model.dims.n_audio_state),
        "n_frames": n_frames,
        "n_mels": n_mels,
        "onnx": str(onnx_path.resolve()),
        "onnx_sha256": _sha256(onnx_path),
        "precision": "fp16" if args.fp16 else "fp32",
        "profiles": {
            "mel": {
                "min": [args.min_batch, n_mels, n_frames],
                "opt": [args.opt_batch, n_mels, n_frames],
                "max": [args.max_batch, n_mels, n_frames],
            }
        },
        "tensorrt_version": tensorrt_version,
        "tf32_enabled": not args.no_tf32,
        "torch_cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "verification": verification,
        "workspace_gb": args.workspace_gb,
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    summary = f"Built {args.output} in {build_seconds:.1f}s"
    if verification:
        summary += (
            f"; feature_max_abs_diff={verification['feature_max_abs_diff']:.3g}"
            f", language_agreement={verification['random_mel_language_agreement']:.3f}"
        )
    print(summary)


if __name__ == "__main__":
    main()
