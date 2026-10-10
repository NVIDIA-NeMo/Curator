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

"""Export a streaming Sortformer checkpoint and build its TensorRT bundle."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import tempfile
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nemo_curator.models.audio.speaker_diarization.export_sortformer_onnx import (
    INPUT_NAMES,
    OUTPUT_NAMES,
    export_checkpoint,
)
from nemo_curator.models.audio.speaker_diarization.sortformer_tensorrt import _validate_streaming_geometry
from nemo_curator.utils.atomic_io import write_json_atomically

_MODEL_TYPE = "streaming_sortformer"
_DEFAULT_RUNTIME_MODULE = Path("/opt/riva/backends/sortformer_modules.py")
_FEATURE_DIM = 128
_SUPPORTED_NORMALIZATION_MODES = frozenset({"none", "per_feature", "all_features"})


@dataclass(frozen=True)
class _BundlePaths:
    engine: Path
    config: Path
    mel_basis: Path
    runtime_module: Path
    learned_silence: Path
    onnx: Path


def _bundle_paths(engine: Path, onnx: Path) -> _BundlePaths:
    prefix = engine.stem
    return _BundlePaths(
        engine=engine,
        config=engine.with_suffix(".json"),
        mel_basis=engine.with_name(f"{prefix}.mel_basis.npy"),
        runtime_module=engine.with_name(f"{prefix}.sortformer_modules.py"),
        learned_silence=engine.with_name(f"{prefix}.learnable_sil_emb.npy"),
        onnx=onnx,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_config_bytes(nemo_path: Path) -> bytes:
    """Read model_config.yaml without extracting untrusted archive paths."""
    with tarfile.open(nemo_path, "r:*") as archive:
        for member in archive:
            if member.name.lstrip("./") != "model_config.yaml":
                continue
            extracted = archive.extractfile(member)
            if extracted is not None:
                return extracted.read()
    msg = f"model_config.yaml was not found in {nemo_path}"
    raise FileNotFoundError(msg)


def _nested(config: dict[str, Any], *keys: str, default: object = None) -> object:
    value: object = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _checkpoint_normalization(value: object) -> str:
    """Canonicalize NeMo's frontend normalization without guessing at unsupported forms."""
    if value is None or value is False:
        return "none"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "na":
            normalized = "none"
        if normalized in _SUPPORTED_NORMALIZATION_MODES:
            return normalized
    msg = (
        "Unsupported Sortformer preprocessor normalization "
        f"{value!r}; supported modes are {sorted(_SUPPORTED_NORMALIZATION_MODES)}"
    )
    raise ValueError(msg)


def _model_parameters(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_speakers = _nested(config, "sortformer_modules", "num_spks")
    if checkpoint_speakers is None:
        checkpoint_speakers = config.get("max_num_of_spks", 4)
    if args.num_speakers is not None and int(args.num_speakers) != int(checkpoint_speakers):
        msg = f"num-speakers={args.num_speakers} does not match the checkpoint output width {checkpoint_speakers}"
        raise ValueError(msg)
    speakers = checkpoint_speakers if args.num_speakers is None else args.num_speakers
    spkcache_len = args.spkcache_len
    if spkcache_len is None:
        spkcache_len = _nested(config, "sortformer_modules", "spkcache_len", default=160)
    preprocessor = config.get("preprocessor", {})
    if not isinstance(preprocessor, dict):
        msg = f"Invalid Sortformer preprocessor config: {preprocessor!r}"
        raise TypeError(msg)
    sample_rate = int(preprocessor.get("sample_rate", config.get("sample_rate", 16_000)))
    window_size = float(preprocessor.get("window_size", 0.025))
    window_stride = float(preprocessor.get("window_stride", 0.01))
    parameters = {
        "num_speakers": int(speakers),
        "spkcache_len": int(spkcache_len),
        "fifo_len": int(args.fifo_len),
        "chunk_len": int(args.chunk_len),
        "emb_dim": int(_nested(config, "model_defaults", "fc_d_model", default=512)),
        "subsampling_factor": int(_nested(config, "encoder", "subsampling_factor", default=8)),
        "spkcache_refresh_rate": int(args.spkcache_refresh_rate),
        "max_batch_size": int(args.max_batch_size),
        "sample_rate": sample_rate,
        "n_fft": int(preprocessor.get("n_fft", 512)),
        "win_length": round(sample_rate * window_size),
        "hop_length": round(sample_rate * window_stride),
        "preemphasis": float(preprocessor.get("preemph", 0.97)),
        "log_guard": float(2**-24),
        "normalization": _checkpoint_normalization(preprocessor.get("normalize", "per_feature")),
        "center_chunk_frames": int(args.center_chunk_frames),
        "left_context_frames": int(args.left_context_frames),
        "right_context_frames": int(args.right_context_frames),
        "output_step_ms": int(args.output_step_ms),
    }
    _validate_streaming_geometry(parameters)
    return parameters


def _force_sensitive_layers_to_fp32(network: Any, trt: Any) -> None:  # noqa: ANN401
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        layer_name = (layer.name or "").lower()
        if layer.type in (trt.LayerType.SHAPE, trt.LayerType.CONSTANT):
            continue
        if layer.type == trt.LayerType.SOFTMAX or any(token in layer_name for token in ("layernorm", "norm", "ln")):
            layer.precision = trt.float32
            for output_index in range(layer.num_outputs):
                layer.set_output_type(output_index, trt.float32)


def _configure_precision(network: Any, builder: Any, build_config: Any, args: argparse.Namespace, trt: Any) -> None:  # noqa: ANN401
    if args.precision == "bf16":
        build_config.set_flag(trt.BuilderFlag.BF16)
        build_config.set_flag(trt.BuilderFlag.FP16)
    elif args.precision == "fp16":
        if not builder.platform_has_fast_fp16:
            msg = "This GPU does not provide fast FP16 TensorRT kernels"
            raise RuntimeError(msg)
        build_config.set_flag(trt.BuilderFlag.FP16)
    else:
        return
    build_config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    _force_sensitive_layers_to_fp32(network, trt)


def _profile_shapes(
    parameters: dict[str, Any],
    *,
    opt_batch_size: int,
) -> dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
    maximum_batch = int(parameters["max_batch_size"])
    chunk_len = int(parameters["chunk_len"])
    cache_len = int(parameters["spkcache_len"])
    fifo_len = int(parameters["fifo_len"])
    embedding_dim = int(parameters["emb_dim"])
    batches = (1, opt_batch_size, maximum_batch)
    return {
        "chunk": tuple((batch, chunk_len, _FEATURE_DIM) for batch in batches),
        "chunk_lengths": tuple((batch,) for batch in batches),
        "spkcache": (
            (1, 1, embedding_dim),
            (opt_batch_size, cache_len, embedding_dim),
            (maximum_batch, cache_len, embedding_dim),
        ),
        "spkcache_lengths": tuple((batch,) for batch in batches),
        "fifo": (
            (1, 1, embedding_dim),
            (opt_batch_size, max(1, fifo_len // 2), embedding_dim),
            (maximum_batch, max(1, fifo_len), embedding_dim),
        ),
        "fifo_lengths": tuple((batch,) for batch in batches),
    }


def _build_engine(
    onnx_path: Path,
    engine_path: Path,
    parameters: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[float, str]:
    try:
        import tensorrt as trt
    except ImportError as exc:
        msg = "TensorRT Python bindings are required to build the Sortformer engine"
        raise RuntimeError(msg) from exc

    logger = trt.Logger(trt.Logger.INFO if args.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        msg = f"Failed to parse {onnx_path}:\n{errors}"
        raise RuntimeError(msg)
    actual_inputs = {network.get_input(index).name for index in range(network.num_inputs)}
    actual_outputs = {network.get_output(index).name for index in range(network.num_outputs)}
    if actual_inputs != set(INPUT_NAMES) or actual_outputs != set(OUTPUT_NAMES):
        msg = f"Unexpected Sortformer ONNX contract: inputs={sorted(actual_inputs)}, outputs={sorted(actual_outputs)}"
        raise RuntimeError(msg)

    build_config = builder.create_builder_config()
    build_config.builder_optimization_level = args.optimization_level
    build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gib * (1 << 30))
    _configure_precision(network, builder, build_config, args, trt)

    profile = builder.create_optimization_profile()
    for name, shapes in _profile_shapes(parameters, opt_batch_size=args.opt_batch_size).items():
        profile.set_shape(name, *shapes)
    if build_config.add_optimization_profile(profile) < 0:
        msg = "TensorRT rejected the Sortformer optimization profile"
        raise RuntimeError(msg)

    started = time.monotonic()
    serialized = builder.build_serialized_network(network, build_config)
    build_seconds = time.monotonic() - started
    if serialized is None:
        msg = "TensorRT failed to build the Sortformer engine"
        raise RuntimeError(msg)
    temporary = engine_path.with_suffix(engine_path.suffix + ".part")
    temporary.write_bytes(serialized)
    temporary.replace(engine_path)
    return build_seconds, trt.__version__


def _mel_basis(parameters: dict[str, Any]) -> np.ndarray:
    try:
        from librosa.filters import mel as librosa_mel
    except ImportError as exc:
        msg = "librosa is required to generate the NeMo Sortformer mel basis"
        raise RuntimeError(msg) from exc
    return librosa_mel(
        sr=int(parameters["sample_rate"]),
        n_fft=int(parameters["n_fft"]),
        n_mels=_FEATURE_DIM,
        fmin=0,
        fmax=None,
        norm="slaney",
    ).astype(np.float32)


def _copy_runtime_module(source: Path | None, destination: Path) -> Path:
    source = source or _DEFAULT_RUNTIME_MODULE
    if not source.is_file():
        msg = (
            "Riva's matching sortformer_modules.py was not found; provide "
            f"--runtime-module or build inside its Riva NIM image: {source}"
        )
        raise FileNotFoundError(msg)
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)
    return destination


def _uses_learned_silence(config: dict[str, Any]) -> bool:
    return bool(_nested(config, "sortformer_modules", "use_learnable_sil_emb", default=False))


def _validate_learned_silence(path: Path, parameters: dict[str, Any], *, required: bool) -> bool:
    if not path.is_file():
        if required:
            msg = "Checkpoint enables use_learnable_sil_emb but its parameter was not exported"
            raise RuntimeError(msg)
        return False
    value = np.load(path, allow_pickle=False)
    expected_shape = (int(parameters["emb_dim"]),)
    if value.dtype != np.float32 or value.shape != expected_shape:
        msg = (
            f"Invalid learned silence artifact {path}: dtype={value.dtype}, shape={value.shape}; "
            f"expected float32/{expected_shape}"
        )
        raise ValueError(msg)
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Destination .plan/.engine file")
    parser.add_argument("--runtime-module", type=Path)
    parser.add_argument("--onnx-output", type=Path)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max-batch-size", type=int, default=512)
    parser.add_argument("--opt-batch-size", type=int, default=128)
    parser.add_argument("--chunk-len", type=int, default=128)
    parser.add_argument("--spkcache-len", type=int)
    parser.add_argument("--fifo-len", type=int, default=80)
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--spkcache-refresh-rate", type=int, default=0)
    parser.add_argument("--center-chunk-frames", type=int, default=112)
    parser.add_argument("--left-context-frames", type=int, default=16)
    parser.add_argument("--right-context-frames", type=int, default=0)
    parser.add_argument("--output-step-ms", type=int, default=80)
    parser.add_argument("--workspace-gib", type=int, default=16)
    parser.add_argument("--optimization-level", type=int, default=5)
    parser.add_argument("--export-device", default="cpu")
    parser.add_argument("--no-bf16-roundtrip", action="store_true")
    parser.add_argument("--skip-native-validation", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.opt_batch_size <= args.max_batch_size:
        parser.error("batch profile must satisfy 1 <= opt-batch-size <= max-batch-size")
    positive_names = (
        "chunk_len",
        "center_chunk_frames",
        "output_step_ms",
        "workspace_gib",
    )
    if any(getattr(args, name) < 1 for name in positive_names):
        parser.error("chunk, FIFO, center, output-step, and workspace values must be positive")
    if (
        args.fifo_len < 0
        or args.left_context_frames < 0
        or args.right_context_frames < 0
        or args.spkcache_refresh_rate < 0
    ):
        parser.error("FIFO, context frames, and cache refresh rate must be non-negative")
    if args.spkcache_len is not None and args.spkcache_len < 1:
        parser.error("spkcache-len must be positive when provided")
    if args.num_speakers is not None and args.num_speakers < 1:
        parser.error("num-speakers must be positive when provided")
    if args.left_context_frames + args.center_chunk_frames + args.right_context_frames > args.chunk_len:
        parser.error("left + center + right context frames must not exceed chunk-len")
    return args


def _load_yaml_config(nemo_model: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        msg = "PyYAML is required to read the Sortformer checkpoint config"
        raise RuntimeError(msg) from exc
    config = yaml.safe_load(_model_config_bytes(nemo_model))
    if not isinstance(config, dict):
        msg = f"Sortformer model_config.yaml must contain an object: {nemo_model}"
        raise TypeError(msg)
    return config


def _runtime_config(  # noqa: PLR0913 - provenance inputs are intentionally explicit
    args: argparse.Namespace,
    paths: _BundlePaths,
    parameters: dict[str, Any],
    *,
    build_seconds: float,
    tensorrt_version: str,
    lowering_error: float | None,
    has_learned_silence: bool,
) -> dict[str, object]:
    profile_shapes = {
        name: {point: list(shape) for point, shape in zip(("min", "opt", "max"), shapes, strict=True)}
        for name, shapes in _profile_shapes(parameters, opt_batch_size=args.opt_batch_size).items()
    }
    config: dict[str, object] = {
        "schema_version": 2,
        "model_type": _MODEL_TYPE,
        **parameters,
        "precision": args.precision,
        "mel_basis": paths.mel_basis.name,
        "mel_basis_sha256": _sha256(paths.mel_basis),
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "engine_file": paths.engine.name,
        "runtime_module": paths.runtime_module.name,
        "runtime_module_sha256": _sha256(paths.runtime_module),
        "source_nemo": str(args.nemo_model.resolve()),
        "source_nemo_sha256": _sha256(args.nemo_model),
        "engine_sha256": _sha256(paths.engine),
        "onnx_sha256": _sha256(paths.onnx),
        "build_seconds": build_seconds,
        "builder_optimization_level": args.optimization_level,
        "workspace_gib": args.workspace_gib,
        "profiles": profile_shapes,
        "tensorrt_version": tensorrt_version,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
    }
    if lowering_error is not None:
        config["native_export_max_abs"] = lowering_error
    if has_learned_silence:
        config["learnable_sil_emb"] = paths.learned_silence.name
        config["learnable_sil_emb_sha256"] = _sha256(paths.learned_silence)
    return config


def _validate_staged_bundle(paths: _BundlePaths, parameters: dict[str, Any]) -> None:
    from nemo_curator.models.audio.speaker_diarization.sortformer_tensorrt import (
        TensorRTSortformerAdapter,
    )

    adapter = TensorRTSortformerAdapter(
        model_id=str(paths.engine),
        sample_rate=int(parameters["sample_rate"]),
        engine_path=str(paths.engine),
        config_path=str(paths.config),
        runtime_module_path=str(paths.runtime_module),
    )
    # Loading is the publication gate: it verifies bundle checksums, actually
    # deserializes the staged plan, validates every binding/profile, and loads
    # the matching streaming-state assets.  Metadata-only validation can let a
    # corrupt or incompatible engine become the final bundle commit marker.
    try:
        adapter.load_model(num_gpus=1)
    finally:
        adapter.unload_model()


def _build_bundle(args: argparse.Namespace, paths: _BundlePaths) -> bool:
    """Build and validate every artifact inside one unpublished directory."""
    config = _load_yaml_config(args.nemo_model)
    parameters = _model_parameters(config, args)
    lowering_error = export_checkpoint(
        args.nemo_model,
        paths.onnx,
        device=args.export_device,
        bf16_roundtrip=args.precision == "bf16" and not args.no_bf16_roundtrip,
        validate_native=not args.skip_native_validation,
        learnable_silence_output=paths.learned_silence,
    )
    has_learned_silence = _validate_learned_silence(
        paths.learned_silence,
        parameters,
        required=_uses_learned_silence(config),
    )
    build_seconds, tensorrt_version = _build_engine(paths.onnx, paths.engine, parameters, args)
    np.save(paths.mel_basis, _mel_basis(parameters), allow_pickle=False)
    _copy_runtime_module(args.runtime_module, paths.runtime_module)
    runtime_config = _runtime_config(
        args,
        paths,
        parameters,
        build_seconds=build_seconds,
        tensorrt_version=tensorrt_version,
        lowering_error=lowering_error,
        has_learned_silence=has_learned_silence,
    )
    write_json_atomically(paths.config, runtime_config, indent=2, sort_keys=True)
    _validate_staged_bundle(paths, parameters)
    return has_learned_silence


def _publish_bundle(
    staged: _BundlePaths,
    final: _BundlePaths,
    *,
    has_learned_silence: bool,
    publish_onnx: bool,
) -> None:
    """Publish the validated config last so it is the bundle commit marker."""
    artifacts = []
    if publish_onnx:
        artifacts.append((staged.onnx, final.onnx))
    artifacts.extend(
        [
            (staged.engine, final.engine),
            (staged.mel_basis, final.mel_basis),
            (staged.runtime_module, final.runtime_module),
        ]
    )
    if has_learned_silence:
        artifacts.append((staged.learned_silence, final.learned_silence))
    artifacts.append((staged.config, final.config))
    for source, destination in artifacts:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)


def main() -> None:
    args = _parse_args()
    if not args.nemo_model.is_file():
        raise FileNotFoundError(args.nemo_model)
    if not torch.cuda.is_available():
        msg = "CUDA is unavailable; build the TensorRT engine on its target GPU"
        raise RuntimeError(msg)
    onnx_output = args.onnx_output or args.output.with_suffix(".onnx")
    if onnx_output.resolve() == args.output.resolve():
        msg = f"Sortformer bundle output paths must be distinct: {[args.output, onnx_output]}"
        raise ValueError(msg)
    final_paths = _bundle_paths(args.output, onnx_output)
    final_candidates = [
        final_paths.engine,
        final_paths.config,
        final_paths.mel_basis,
        final_paths.runtime_module,
        final_paths.learned_silence,
    ]
    if args.onnx_output is not None:
        final_candidates.append(final_paths.onnx)
    if len({path.resolve() for path in final_candidates}) != len(final_candidates):
        msg = f"Sortformer bundle output paths must be distinct: {final_candidates}"
        raise ValueError(msg)
    existing = [path for path in final_candidates if path.exists()]
    if existing and not args.force:
        msg = f"Sortformer bundle artifacts already exist; use --force to rebuild: {existing}"
        raise RuntimeError(msg)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        staging_dir = Path(
            stack.enter_context(tempfile.TemporaryDirectory(prefix=".sortformer-bundle-", dir=args.output.parent))
        )
        staged_onnx = staging_dir / final_paths.onnx.name
        if args.onnx_output is not None:
            final_paths.onnx.parent.mkdir(parents=True, exist_ok=True)
            onnx_staging_dir = Path(
                stack.enter_context(
                    tempfile.TemporaryDirectory(prefix=".sortformer-onnx-", dir=final_paths.onnx.parent)
                )
            )
            staged_onnx = onnx_staging_dir / final_paths.onnx.name
        staged_paths = _bundle_paths(
            staging_dir / final_paths.engine.name,
            staged_onnx,
        )
        has_learned_silence = _build_bundle(args, staged_paths)
        _publish_bundle(
            staged_paths,
            final_paths,
            has_learned_silence=has_learned_silence,
            publish_onnx=args.onnx_output is not None,
        )


if __name__ == "__main__":
    main()
