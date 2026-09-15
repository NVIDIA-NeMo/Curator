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

"""Shared export, build, and validation helpers for NeMo TensorRT encoders."""

# ruff: noqa: INP001

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

    import torch

logger = logging.getLogger(__name__)


def export_encoder(
    model: torch.nn.Module,
    onnx_path: Path,
    *,
    feature_count: int,
    example_frames: int,
) -> None:
    import torch

    audio_signal = torch.randn(
        (1, feature_count, example_frames),
        device="cuda",
        dtype=torch.float16,
    )
    length = torch.full((1,), example_frames, device="cuda", dtype=torch.int64)
    model.encoder.export(
        str(onnx_path),
        input_example=(audio_signal, length),
        do_constant_folding=False,
        onnx_opset_version=17,
        check_trace=False,
        dynamic_axes={
            "audio_signal": {0: "batch", 2: "feature_frames"},
            "length": {0: "batch"},
            "outputs": {0: "batch", 2: "encoded_frames"},
            "encoded_lengths": {0: "batch"},
        },
        use_dynamo=False,
    )


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    *,
    feature_count: int,
    args: argparse.Namespace,
) -> str:
    try:
        import tensorrt as trt
    except ImportError as error:
        msg = "TensorRT Python bindings are required to build the encoder engine"
        raise RuntimeError(msg) from error

    logger = trt.Logger(trt.Logger.INFO if args.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        msg = f"Failed to parse {onnx_path}:\n{errors}"
        raise RuntimeError(msg)

    input_names = {network.get_input(index).name for index in range(network.num_inputs)}
    if input_names != {"audio_signal", "length"}:
        msg = f"Unexpected exported encoder inputs: {sorted(input_names)}"
        raise RuntimeError(msg)
    output_names = {network.get_output(index).name for index in range(network.num_outputs)}
    if output_names != {"outputs", "encoded_lengths"}:
        msg = f"Unexpected exported encoder outputs: {sorted(output_names)}"
        raise RuntimeError(msg)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gb * (1 << 30))
    if not builder.platform_has_fast_fp16:
        msg = "This GPU does not provide fast FP16 TensorRT kernels"
        raise RuntimeError(msg)
    config.set_flag(trt.BuilderFlag.FP16)
    config.builder_optimization_level = 5

    profile = builder.create_optimization_profile()
    audio_profile_status = profile.set_shape(
        "audio_signal",
        (args.min_batch, feature_count, args.min_frames),
        (args.opt_batch, feature_count, args.opt_frames),
        (args.max_batch, feature_count, args.max_frames),
    )
    length_profile_status = profile.set_shape(
        "length",
        (args.min_batch,),
        (args.opt_batch,),
        (args.max_batch,),
    )
    if audio_profile_status is False or length_profile_status is False:
        msg = "Could not set the TensorRT optimization profile"
        raise RuntimeError(msg)
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        msg = "TensorRT failed to build the encoder engine"
        raise RuntimeError(msg)
    engine_path.write_bytes(serialized_engine)
    return trt.__version__


def validate_engine(  # noqa: PLR0913
    model: torch.nn.Module,
    engine_path: Path,
    *,
    feature_count: int,
    min_batch: int,
    min_frames: int,
    opt_frames: int,
    tolerances: tuple[float, float] = (5e-2, 5e-2),
) -> None:
    import torch

    from nemo_curator.stages.audio.inference.tensorrt_encoder import TensorRTEncoderSession

    validation_frames = max(min_frames, min(opt_frames, 256))
    generator = torch.Generator(device="cuda").manual_seed(0)
    audio_signal = torch.randn(
        (min_batch, feature_count, validation_frames),
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    length = torch.full((min_batch,), validation_frames, device="cuda", dtype=torch.int64)
    with torch.inference_mode():
        expected_outputs, expected_lengths = model.encoder(audio_signal=audio_signal, length=length)

    session = TensorRTEncoderSession(engine_path)
    try:
        actual = session.infer({"audio_signal": audio_signal, "length": length})
        rtol, atol = tolerances
        torch.testing.assert_close(actual["outputs"], expected_outputs, rtol=rtol, atol=atol)
        torch.testing.assert_close(actual["encoded_lengths"], expected_lengths)
    finally:
        session.close()


def build_encoder_bundle(  # noqa: PLR0913
    model: torch.nn.Module,
    model_path: Path,
    output_dir: Path,
    *,
    args: argparse.Namespace,
    metadata: dict[str, object],
    temporary_prefix: str,
    parity_message: str,
    tolerances: tuple[float, float] = (5e-2, 5e-2),
) -> None:
    """Build and validate a bundle before publishing any final artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_names = ("encoder.plan", "model.nemo", "metadata.json")
    destinations = [output_dir / name for name in artifact_names]
    existing = [str(path) for path in destinations if path.exists()]
    if existing:
        msg = f"Refusing to overwrite existing bundle artifacts: {existing}"
        raise FileExistsError(msg)

    feature_count = int(metadata["feature_count"])
    with tempfile.TemporaryDirectory(prefix=temporary_prefix, dir=output_dir) as temporary_dir:
        staging_dir = Path(temporary_dir)
        onnx_path = staging_dir / "encoder.onnx"
        engine_path = staging_dir / "encoder.plan"
        export_encoder(
            model,
            onnx_path,
            feature_count=feature_count,
            example_frames=args.min_frames,
        )
        tensorrt_version = build_engine(
            onnx_path,
            engine_path,
            feature_count=feature_count,
            args=args,
        )
        validate_engine(
            model,
            engine_path,
            feature_count=feature_count,
            min_batch=args.min_batch,
            min_frames=args.min_frames,
            opt_frames=args.opt_frames,
            tolerances=tolerances,
        )

        shutil.copy2(model_path, staging_dir / "model.nemo")
        metadata.update(
            {
                "schema_version": 1,
                "precision": "fp16",
                "engine_file": "encoder.plan",
                "model_file": "model.nemo",
                "source_model": model_path.name,
                "input_names": ["audio_signal", "length"],
                "output_names": ["outputs", "encoded_lengths"],
                "profile": {
                    "min": {"batch": args.min_batch, "feature_frames": args.min_frames},
                    "opt": {"batch": args.opt_batch, "feature_frames": args.opt_frames},
                    "max": {"batch": args.max_batch, "feature_frames": args.max_frames},
                },
                "onnx_opset": 17,
                "tensorrt_version": tensorrt_version,
            }
        )
        (staging_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

        for name in artifact_names:
            (staging_dir / name).replace(output_dir / name)

    logger.info(parity_message)
    logger.info("Wrote TensorRT encoder bundle to %s", output_dir)
