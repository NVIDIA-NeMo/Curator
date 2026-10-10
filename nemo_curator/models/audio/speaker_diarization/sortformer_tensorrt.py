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

"""Streaming Sortformer adapter backed by a shared TensorRT session."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import inspect
import json
import math
import sys
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from loguru import logger

from nemo_curator.models.audio.speaker_diarization.base import (
    DiarizationResult,
    DiarizationSegment,
)
from nemo_curator.models.audio.speaker_diarization.sortformer import (
    _DEFAULT_MODEL_ID,
    _DEFAULT_SAMPLE_RATE,
    _torch_module,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from types import ModuleType

    import torch


_FEATURE_DIM = 128
_DEFAULT_LONG_AUDIO_SECONDS = 60 * 60
_DEFAULT_STFT_BLOCK_SECONDS = 60 * 60
_NORMALIZATION_EPSILON = 1.0e-5
_SUPPORTED_NORMALIZATION_MODES = frozenset({"none", "per_feature", "all_features"})
_FEATURE_INPUT_RANK = 2
_SEQUENCE_INPUT_RANK = 3
_REQUIRED_INPUTS = {
    "chunk",
    "chunk_lengths",
    "spkcache",
    "spkcache_lengths",
    "fifo",
    "fifo_lengths",
}
_REQUIRED_OUTPUTS = {
    "predictions",
    "pred_lengths",
    "chunk_embs",
    "chunk_emb_lengths",
}
_POSITIVE_CONFIG_INTS = (
    "sample_rate",
    "n_fft",
    "win_length",
    "hop_length",
    "chunk_len",
    "emb_dim",
    "num_speakers",
    "subsampling_factor",
    "spkcache_len",
    "max_batch_size",
    "center_chunk_frames",
    "output_step_ms",
)
_NONNEGATIVE_CONFIG_INTS = (
    "fifo_len",
    "spkcache_refresh_rate",
    "left_context_frames",
    "right_context_frames",
)


class _TensorRTSession(Protocol):
    device: torch.device
    input_names: list[str]
    output_names: list[str]

    def infer(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...

    def input_shape_range(
        self,
        name: str,
        profile_index: int = 0,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]: ...

    def close(self) -> None: ...


def _soundfile_module() -> Any:  # noqa: ANN401
    try:
        import soundfile
    except ImportError as exc:
        msg = "Sortformer TensorRT file input requires soundfile from the audio extra"
        raise ImportError(msg) from exc
    return soundfile


def _binarize(sequence: torch.Tensor, frame_step: float, threshold: float) -> torch.Tensor:
    """Convert one speaker's frame probabilities to reference-compatible spans."""
    torch = _torch_module()
    if sequence.numel() == 0:
        return torch.empty((0, 2), device=sequence.device)
    active = sequence > threshold
    padded = torch.nn.functional.pad(active.float(), (1, 1))
    transitions = padded[1:] - padded[:-1]
    starts = torch.where(transitions > threshold)[0]
    ends = torch.where(transitions < -threshold)[0]
    start_times = starts.float() * frame_step
    end_times = ends.float() * frame_step
    valid = end_times > start_times
    return torch.stack((start_times[valid], end_times[valid]), dim=1)


def _load_runtime_module(path: Path) -> ModuleType:
    """Load the matching Riva state-management module from an explicit path."""
    if not path.is_file():
        msg = f"Sortformer TensorRT runtime module not found: {path}"
        raise FileNotFoundError(msg)
    module_name = f"_nemo_curator_sortformer_runtime_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        msg = f"Could not load Sortformer TensorRT runtime module: {path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    if not callable(getattr(module, "SortformerModules", None)):
        msg = f"Sortformer TensorRT runtime module does not define SortformerModules: {path}"
        raise TypeError(msg)
    return module


def _create_state_modules(
    state_module: ModuleType,
    config: dict[str, Any],
    learned_silence: torch.Tensor | None,
) -> object:
    """Construct current or legacy Riva streaming-state helpers."""
    torch = _torch_module()
    module_args = {
        "spkcache_refresh_rate": int(config["spkcache_refresh_rate"]),
        "spkcache_len": int(config["spkcache_len"]),
        "fifo_len": int(config["fifo_len"]),
        "fc_d_model": int(config["emb_dim"]),
        "num_spks": int(config["num_speakers"]),
        "dtype": torch.float32,
    }
    parameters = inspect.signature(state_module.SortformerModules).parameters
    supports_learned_silence = "learnable_sil_emb" in parameters
    if supports_learned_silence:
        module_args["learnable_sil_emb"] = learned_silence
    modules = state_module.SortformerModules(**module_args)
    silence_frames = int(getattr(modules, "spkcache_sil_frames_per_spk", 0))
    minimum_spkcache_len = int(config["num_speakers"]) * silence_frames
    if int(config["spkcache_len"]) < minimum_spkcache_len:
        msg = (
            "Sortformer TensorRT spkcache_len must reserve at least "
            f"{silence_frames} silence frames for each speaker; expected at least {minimum_spkcache_len}"
        )
        raise ValueError(msg)
    if learned_silence is not None and not supports_learned_silence:

        def learned_silence_profile(embeddings: torch.Tensor, _predictions: torch.Tensor) -> torch.Tensor:
            return learned_silence.unsqueeze(0).expand(embeddings.shape[0], -1)

        modules._get_silence_profile = learned_silence_profile
    return modules


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        msg = f"Sortformer TensorRT config not found: {path}"
        raise FileNotFoundError(msg)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Could not read Sortformer TensorRT config: {path}"
        raise ValueError(msg) from exc
    if not isinstance(config, dict):
        msg = f"Sortformer TensorRT config must be a JSON object: {path}"
        raise TypeError(msg)
    return config


def _validate_config_numbers(config: dict[str, Any]) -> None:
    for key in _POSITIVE_CONFIG_INTS:
        value = config.get(key)
        if type(value) is not int or value < 1:
            msg = f"Invalid Sortformer TensorRT config value for {key}: {value!r}"
            raise ValueError(msg)
    for key in _NONNEGATIVE_CONFIG_INTS:
        value = config.get(key)
        if type(value) is not int or value < 0:
            msg = f"Invalid Sortformer TensorRT config value for {key}: {value!r}"
            raise ValueError(msg)
    for key in ("preemphasis", "log_guard"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            msg = f"Invalid Sortformer TensorRT config value for {key}: {value!r}"
            raise ValueError(msg)
    if float(config["log_guard"]) <= 0:
        msg = f"Invalid Sortformer TensorRT config value for log_guard: {config['log_guard']!r}"
        raise ValueError(msg)


def _validate_streaming_geometry(config: dict[str, Any]) -> None:
    subsampling = int(config["subsampling_factor"])
    center_frames = int(config["center_chunk_frames"])
    unaligned = [name for name in ("center_chunk_frames", "left_context_frames") if int(config[name]) % subsampling]
    if unaligned:
        msg = (
            "Sortformer TensorRT center and left-context frames must be divisible by "
            f"subsampling_factor={subsampling}: {unaligned}"
        )
        raise ValueError(msg)
    emitted_frames = center_frames // subsampling
    fifo_len = int(config["fifo_len"])
    if 0 < fifo_len < emitted_frames:
        msg = f"Sortformer TensorRT fifo_len must be 0 or at least {emitted_frames}, got {fifo_len}"
        raise ValueError(msg)


def _validate_config_geometry(config: dict[str, Any], *, sample_rate: int, config_path: Path) -> None:
    if config["sample_rate"] != sample_rate:
        msg = (
            f"Sortformer TensorRT config sample_rate={config['sample_rate']} does not match "
            f"adapter sample_rate={sample_rate}: {config_path}"
        )
        raise ValueError(msg)


def _validate_config_paths(config: dict[str, Any]) -> None:
    if config["win_length"] > config["n_fft"]:
        msg = "Sortformer TensorRT win_length must not exceed n_fft"
        raise ValueError(msg)
    window_frames = config["left_context_frames"] + config["center_chunk_frames"] + config["right_context_frames"]
    if window_frames > config["chunk_len"]:
        msg = f"Sortformer TensorRT context window exceeds chunk_len: {window_frames} > {config['chunk_len']}"
        raise ValueError(msg)
    for key in ("mel_basis",):
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            msg = f"Invalid Sortformer TensorRT config path for {key}: {value!r}"
            raise ValueError(msg)
    learned_silence = config.get("learnable_sil_emb")
    if learned_silence is not None and (not isinstance(learned_silence, str) or not learned_silence.strip()):
        msg = f"Invalid Sortformer TensorRT config path for learnable_sil_emb: {learned_silence!r}"
        raise ValueError(msg)


def _validate_config_normalization(config: dict[str, Any]) -> None:
    value = config.get("normalization")
    if type(value) is not str or value not in _SUPPORTED_NORMALIZATION_MODES:
        msg = (
            "Sortformer TensorRT config must declare a supported normalization contract; "
            f"got {value!r}, expected one of {sorted(_SUPPORTED_NORMALIZATION_MODES)}"
        )
        raise ValueError(msg)


def _validate_config(config: dict[str, Any], *, sample_rate: int, config_path: Path) -> None:
    _validate_config_numbers(config)
    _validate_streaming_geometry(config)
    _validate_config_geometry(config, sample_rate=sample_rate, config_path=config_path)
    _validate_config_paths(config)
    _validate_config_normalization(config)


def _config_artifact(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else config_path.parent / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_optional_checksum(config: dict[str, Any], key: str, path: Path) -> None:
    expected = config.get(key)
    if expected is not None and (not isinstance(expected, str) or _sha256(path) != expected):
        msg = f"Sortformer TensorRT artifact checksum {key} does not match: {path}"
        raise ValueError(msg)


def _validate_bundle_metadata(
    config: dict[str, Any],
    *,
    config_path: Path,
    engine_path: Path,
    runtime_module_path: Path,
) -> None:
    expected_lists = {"input_names": _REQUIRED_INPUTS, "output_names": _REQUIRED_OUTPUTS}
    for key, expected in expected_lists.items():
        value = config.get(key)
        if value is not None and (
            not isinstance(value, list) or any(not isinstance(item, str) for item in value) or set(value) != expected
        ):
            msg = f"Unexpected Sortformer TensorRT config {key}: {value!r}"
            raise ValueError(msg)
    precision = config.get("precision")
    if precision is not None and precision not in {"bf16", "fp16", "fp32"}:
        msg = f"Unsupported Sortformer TensorRT config precision: {precision!r}"
        raise ValueError(msg)
    linked_paths = {
        "engine_file": engine_path,
        "runtime_module": runtime_module_path,
    }
    for key, actual_path in linked_paths.items():
        configured = config.get(key)
        if configured is not None and (
            not isinstance(configured, str)
            or _config_artifact(config_path, configured).resolve() != actual_path.resolve()
        ):
            msg = f"Sortformer TensorRT config {key} does not match {actual_path}: {configured!r}"
            raise ValueError(msg)
    _validate_optional_checksum(config, "engine_sha256", engine_path)
    _validate_optional_checksum(config, "runtime_module_sha256", runtime_module_path)


def _validate_mel_basis(path: Path, config: dict[str, Any]) -> None:
    if not path.is_file():
        msg = f"Sortformer mel basis not found: {path}"
        raise FileNotFoundError(msg)
    mel_basis = np.load(path, allow_pickle=False)
    expected_shape = (_FEATURE_DIM, int(config["n_fft"]) // 2 + 1)
    if mel_basis.shape != expected_shape or mel_basis.dtype != np.float32:
        msg = (
            f"Invalid Sortformer mel basis dtype/shape: {mel_basis.dtype}/{mel_basis.shape}; "
            f"expected float32/{expected_shape}"
        )
        raise ValueError(msg)
    if not np.isfinite(mel_basis).all():
        msg = f"Sortformer mel basis contains non-finite values: {path}"
        raise ValueError(msg)


def _validate_learned_silence(path: Path, config: dict[str, Any]) -> None:
    if not path.is_file():
        msg = f"Sortformer learned silence embedding not found: {path}"
        raise FileNotFoundError(msg)
    learned_silence = np.load(path, allow_pickle=False)
    expected_shape = (int(config["emb_dim"]),)
    if learned_silence.shape != expected_shape or learned_silence.dtype != np.float32:
        msg = (
            "Invalid Sortformer learned silence embedding: "
            f"dtype={learned_silence.dtype}, shape={learned_silence.shape}; "
            f"expected dtype=float32, shape={expected_shape}"
        )
        raise ValueError(msg)
    if not np.isfinite(learned_silence).all():
        msg = f"Sortformer learned silence embedding contains non-finite values: {path}"
        raise ValueError(msg)


def _validated_duration(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        msg = f"TensorRTSortformerAdapter.{name} must be finite, got {value!r}"
        raise ValueError(msg)
    result = float(value)
    if result <= 0:
        msg = f"TensorRTSortformerAdapter.{name} must be positive, got {value!r}"
        raise ValueError(msg)
    return result


def _validate_session_names(session: _TensorRTSession) -> None:
    inputs = set(getattr(session, "input_names", ()))
    outputs = set(getattr(session, "output_names", ()))
    if inputs != _REQUIRED_INPUTS:
        msg = f"Sortformer TensorRT engine inputs must be {sorted(_REQUIRED_INPUTS)}, got {sorted(inputs)}"
        raise ValueError(msg)
    if outputs != _REQUIRED_OUTPUTS:
        msg = f"Sortformer TensorRT engine outputs must be {sorted(_REQUIRED_OUTPUTS)}, got {sorted(outputs)}"
        raise ValueError(msg)
    device = getattr(session, "device", None)
    if getattr(device, "type", None) != "cuda":
        msg = f"Sortformer TensorRT session must use CUDA, got {device!r}"
        raise ValueError(msg)


def _validate_sequence_profile(
    session: _TensorRTSession,
    name: str,
    *,
    requested_batch_size: int,
    required_frames: int,
    required_features: int,
) -> None:
    minimum, _optimum, maximum = session.input_shape_range(name)
    if len(minimum) != _SEQUENCE_INPUT_RANK or len(maximum) != _SEQUENCE_INPUT_RANK:
        msg = f"Sortformer TensorRT input {name!r} must have rank {_SEQUENCE_INPUT_RANK}"
        raise ValueError(msg)
    if minimum[0] > 1 or maximum[0] < requested_batch_size:
        msg = (
            f"Sortformer TensorRT input {name!r} batch profile {minimum[0]}..{maximum[0]} "
            f"does not support 1..{requested_batch_size}"
        )
        raise ValueError(msg)
    required_minimum_frames = 1 if name in {"spkcache", "fifo"} else required_frames
    if minimum[1] > required_minimum_frames or maximum[1] < required_frames or minimum[2] != required_features:
        msg = (
            f"Sortformer TensorRT input {name!r} profile {minimum}..{maximum} "
            f"does not support (*, {required_minimum_frames}..{required_frames}, {required_features})"
        )
        raise ValueError(msg)
    if maximum[2] != required_features:
        msg = f"Sortformer TensorRT input {name!r} has inconsistent feature dimension"
        raise ValueError(msg)


def _validate_length_profile(session: _TensorRTSession, name: str, *, requested_batch_size: int) -> None:
    minimum, _optimum, maximum = session.input_shape_range(name)
    if len(minimum) != 1 or len(maximum) != 1 or minimum[0] > 1 or maximum[0] < requested_batch_size:
        msg = f"Sortformer TensorRT input {name!r} has incompatible batch profile"
        raise ValueError(msg)


@dataclass
class TensorRTSortformerAdapter:
    """Run the full streaming Sortformer graph through TensorRT.

    The engine contains the acoustic encoder and speaker prediction graph. The
    JSON file supplies frontend and streaming geometry, while the matching Riva
    ``sortformer_modules.py`` manages speaker-cache state. All three files are
    target-specific inputs produced by ``build_sortformer_tensorrt_engine``.
    """

    model_id: str = _DEFAULT_MODEL_ID
    sample_rate: int = _DEFAULT_SAMPLE_RATE
    engine_path: str | None = None
    config_path: str | None = None
    runtime_module_path: str | None = None
    inference_batch_size: int | None = None
    speech_threshold: float = 0.5
    long_audio_seconds: float = float(_DEFAULT_LONG_AUDIO_SECONDS)
    stft_block_seconds: float = float(_DEFAULT_STFT_BLOCK_SECONDS)

    _session: _TensorRTSession | None = field(default=None, init=False, repr=False)
    _config: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _modules: Any = field(default=None, init=False, repr=False)
    _mel_basis: Any = field(default=None, init=False, repr=False)
    _window: Any = field(default=None, init=False, repr=False)
    _stft_block_samples: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            msg = "TensorRTSortformerAdapter.model_id must be non-empty"
            raise ValueError(msg)
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, Integral) or self.sample_rate <= 0:
            msg = f"TensorRTSortformerAdapter.sample_rate must be a positive integer, got {self.sample_rate!r}"
            raise ValueError(msg)
        for name in ("engine_path", "config_path", "runtime_module_path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                msg = f"TensorRTSortformerAdapter.{name} is required"
                raise ValueError(msg)
        if self.inference_batch_size is not None and (
            isinstance(self.inference_batch_size, bool)
            or not isinstance(self.inference_batch_size, Integral)
            or self.inference_batch_size <= 0
        ):
            msg = (
                "TensorRTSortformerAdapter.inference_batch_size must be a positive integer or None, "
                f"got {self.inference_batch_size!r}"
            )
            raise ValueError(msg)
        if (
            isinstance(self.speech_threshold, bool)
            or not isinstance(self.speech_threshold, Real)
            or not math.isfinite(float(self.speech_threshold))
            or not 0 < float(self.speech_threshold) < 1
        ):
            msg = "TensorRTSortformerAdapter.speech_threshold must be between 0 and 1"
            raise ValueError(msg)
        self.sample_rate = int(self.sample_rate)
        if self.inference_batch_size is not None:
            self.inference_batch_size = int(self.inference_batch_size)
        self.speech_threshold = float(self.speech_threshold)
        self.long_audio_seconds = _validated_duration("long_audio_seconds", self.long_audio_seconds)
        self.stft_block_seconds = _validated_duration("stft_block_seconds", self.stft_block_seconds)

    @property
    def _engine_file(self) -> Path:
        return Path(self.engine_path or "").expanduser()

    @property
    def _config_file(self) -> Path:
        return Path(self.config_path or "").expanduser()

    @property
    def _runtime_module_file(self) -> Path:
        return Path(self.runtime_module_path or "").expanduser()

    def _validated_bundle(self) -> tuple[dict[str, Any], Path, Path | None]:
        engine_path = self._engine_file
        if not engine_path.is_file():
            msg = f"Sortformer TensorRT engine not found: {engine_path}"
            raise FileNotFoundError(msg)
        runtime_module_path = self._runtime_module_file
        if not runtime_module_path.is_file():
            msg = f"Sortformer TensorRT runtime module not found: {runtime_module_path}"
            raise FileNotFoundError(msg)

        config_path = self._config_file
        config = _load_config(config_path)
        _validate_config(config, sample_rate=self.sample_rate, config_path=config_path)
        _validate_bundle_metadata(
            config,
            config_path=config_path,
            engine_path=engine_path,
            runtime_module_path=runtime_module_path,
        )
        max_batch_size = int(config["max_batch_size"])
        requested_batch_size = self.inference_batch_size or max_batch_size
        if requested_batch_size > max_batch_size:
            msg = (
                f"Sortformer TensorRT inference batch size must be between 1 and {max_batch_size}, "
                f"got {requested_batch_size}"
            )
            raise ValueError(msg)

        mel_path = _config_artifact(config_path, config["mel_basis"])
        _validate_mel_basis(mel_path, config)
        _validate_optional_checksum(config, "mel_basis_sha256", mel_path)

        learned_silence_path = None
        if config.get("learnable_sil_emb"):
            learned_silence_path = _config_artifact(config_path, config["learnable_sil_emb"])
            _validate_learned_silence(learned_silence_path, config)
            _validate_optional_checksum(config, "learnable_sil_emb_sha256", learned_silence_path)
        return config, mel_path, learned_silence_path

    def download_weights_on_node(self) -> None:
        """Validate the complete target-specific runtime bundle."""
        self._validated_bundle()

    def _validate_session(self, session: _TensorRTSession, config: dict[str, Any]) -> None:
        _validate_session_names(session)
        requested_batch_size = self.inference_batch_size or int(config["max_batch_size"])
        expected_shapes = {
            "chunk": (int(config["chunk_len"]), _FEATURE_DIM),
            "spkcache": (int(config["spkcache_len"]), int(config["emb_dim"])),
            "fifo": (max(1, int(config["fifo_len"])), int(config["emb_dim"])),
        }
        for name, (required_frames, required_features) in expected_shapes.items():
            _validate_sequence_profile(
                session,
                name,
                requested_batch_size=requested_batch_size,
                required_frames=required_frames,
                required_features=required_features,
            )
        for name in ("chunk_lengths", "spkcache_lengths", "fifo_lengths"):
            _validate_length_profile(session, name, requested_batch_size=requested_batch_size)

    def load_model(self, *, num_gpus: int) -> None:
        """Load one shared TensorRT session and its streaming state helpers."""
        if isinstance(num_gpus, bool) or not isinstance(num_gpus, Integral) or num_gpus != 1:
            msg = f"TensorRTSortformerAdapter.load_model requires exactly one GPU, got {num_gpus!r}"
            raise ValueError(msg)
        if self._session is not None:
            return
        torch = _torch_module()
        if not torch.cuda.is_available():
            msg = "TensorRTSortformerAdapter received num_gpus=1, but CUDA is not available"
            raise RuntimeError(msg)

        config, mel_path, learned_silence_path = self._validated_bundle()
        block_samples = int(self.sample_rate * self.stft_block_seconds)
        block_samples -= block_samples % int(config["hop_length"])
        if block_samples <= 0:
            msg = "Sortformer TensorRT STFT block is shorter than one feature hop"
            raise ValueError(msg)
        from nemo_curator.stages.audio.inference.tensorrt_encoder import TensorRTEncoderSession

        session = TensorRTEncoderSession(self._engine_file)
        try:
            self._validate_session(session, config)
            state_module = _load_runtime_module(self._runtime_module_file)
            mel_basis = torch.from_numpy(np.load(mel_path, allow_pickle=False)).to(
                device=session.device,
                dtype=torch.float32,
            )
            learned_silence = None
            if learned_silence_path is not None:
                learned_silence = torch.from_numpy(np.load(learned_silence_path, allow_pickle=False)).to(
                    device=session.device,
                    dtype=torch.float32,
                )
            modules = _create_state_modules(state_module, config, learned_silence)
            window = torch.hann_window(
                int(config["win_length"]),
                periodic=False,
                dtype=torch.float32,
                device=session.device,
            )
        except Exception:
            session.close()
            raise

        self._session = session
        self._config = config
        self._modules = modules
        self._mel_basis = mel_basis
        self._window = window
        self._stft_block_samples = block_samples
        self.inference_batch_size = self.inference_batch_size or int(config["max_batch_size"])
        logger.info("Loaded Sortformer TensorRT engine {}", self._engine_file)

    def unload_model(self) -> None:
        """Close TensorRT and release frontend and streaming-state tensors."""
        session = self._session
        self._session = None
        self._config = None
        self._modules = None
        self._mel_basis = None
        self._window = None
        self._stft_block_samples = None
        if session is not None:
            session.close()
        gc.collect()
        try:
            torch = _torch_module()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _extract_features(
        self,
        waveform: torch.Tensor,
        logical_length: int,
        frame_offset: int = 0,
    ) -> torch.Tensor:
        torch = _torch_module()
        if self._config is None or self._window is None or self._mel_basis is None:
            msg = "TensorRTSortformerAdapter is not initialized"
            raise RuntimeError(msg)
        preemphasis = float(self._config["preemphasis"])
        with torch.inference_mode():
            signal = waveform.reshape(1, -1).to(device=self._window.device, non_blocking=True)
            signal = torch.cat((signal[:, :1], signal[:, 1:] - preemphasis * signal[:, :-1]), dim=1)
            spectrum = torch.stft(
                signal,
                n_fft=int(self._config["n_fft"]),
                hop_length=int(self._config["hop_length"]),
                win_length=int(self._config["win_length"]),
                window=self._window,
                center=True,
                pad_mode="constant",
                return_complex=True,
            )
            mel = torch.log(
                torch.matmul(self._mel_basis.unsqueeze(0), spectrum.abs().square()) + float(self._config["log_guard"])
            )
            end = frame_offset + logical_length
            return mel[0, :, frame_offset:end].transpose(0, 1).to(device="cpu").contiguous()

    def _features(self, waveforms: list[torch.Tensor]) -> list[torch.Tensor]:
        if self._config is None:
            msg = "TensorRTSortformerAdapter is not initialized"
            raise RuntimeError(msg)
        hop_length = int(self._config["hop_length"])
        return [
            self._normalize_features(
                self._extract_features(waveform, max(1, waveform.numel() // hop_length)),
            )
            for waveform in waveforms
        ]

    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the checkpoint's full-recording NeMo normalization contract."""
        torch = _torch_module()
        if self._config is None:
            msg = "TensorRTSortformerAdapter is not initialized"
            raise RuntimeError(msg)
        mode = self._config.get("normalization")
        if type(mode) is not str or mode not in _SUPPORTED_NORMALIZATION_MODES:
            msg = f"Unsupported Sortformer TensorRT normalization contract: {mode!r}"
            raise ValueError(msg)
        if features.ndim != _FEATURE_INPUT_RANK or features.shape[1] != _FEATURE_DIM:
            msg = f"Sortformer features must have shape [frames, {_FEATURE_DIM}], got {tuple(features.shape)}"
            raise ValueError(msg)
        if mode == "none" or features.shape[0] == 0:
            return features

        if mode == "per_feature":
            mean = features.mean(dim=0, keepdim=True)
            centered = features - mean
            if features.shape[0] == 1:
                standard_deviation = torch.zeros_like(mean)
            else:
                standard_deviation = torch.sqrt(centered.square().sum(dim=0, keepdim=True) / (features.shape[0] - 1))
            standard_deviation = standard_deviation.masked_fill(standard_deviation.isnan(), 0.0)
            return centered / (standard_deviation + _NORMALIZATION_EPSILON)

        mean = features.mean()
        standard_deviation = features.std() + _NORMALIZATION_EPSILON
        return (features - mean) / standard_deviation

    def _normalize_feature_blocks(self, feature_blocks: Iterator[torch.Tensor]) -> torch.Tensor:
        """Concatenate bounded raw blocks and normalize over one logical recording."""
        torch = _torch_module()
        blocks = list(feature_blocks)
        if not blocks:
            return torch.empty((0, _FEATURE_DIM), dtype=torch.float32)
        return self._normalize_features(torch.cat(blocks, dim=0))

    def _load_file_waveform(self, path: str) -> torch.Tensor:
        """Load one regular-sized file, downmixing and resampling when needed."""
        torch = _torch_module()
        soundfile = _soundfile_module()
        with soundfile.SoundFile(path) as audio_file:
            source_rate = int(audio_file.samplerate)
            data = audio_file.read(dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data).mean(dim=1)
        if source_rate != self.sample_rate:
            try:
                import torchaudio
            except ImportError as exc:
                msg = "Resampling regular Sortformer file input requires torchaudio"
                raise ImportError(msg) from exc
            waveform = torchaudio.functional.resample(waveform, source_rate, self.sample_rate)
        return waveform.contiguous()

    def _source_duration_and_empty(self, source: torch.Tensor | str) -> tuple[float, bool]:
        if isinstance(source, str):
            info = _soundfile_module().info(source)
            if info.samplerate <= 0:
                msg = f"Invalid sample rate {info.samplerate} for Sortformer input {source}"
                raise ValueError(msg)
            return info.frames / info.samplerate, info.frames == 0
        return source.numel() / self.sample_rate, source.numel() == 0

    def _regular_waveforms(self, sources: list[torch.Tensor | str]) -> list[torch.Tensor]:
        return [self._load_file_waveform(source) if isinstance(source, str) else source for source in sources]

    def _waveform_feature_blocks(self, waveform: torch.Tensor) -> Iterator[torch.Tensor]:
        torch = _torch_module()
        if self._config is None or self._stft_block_samples is None:
            msg = "TensorRTSortformerAdapter is not initialized"
            raise RuntimeError(msg)
        hop_length = int(self._config["hop_length"])
        context_hops = math.ceil((int(self._config["n_fft"]) / 2) / hop_length)
        context_samples = context_hops * hop_length
        total_samples = waveform.numel()
        for start in range(0, total_samples, self._stft_block_samples):
            end = min(start + self._stft_block_samples, total_samples)
            logical_length = (end - start) // hop_length
            if logical_length == 0:
                continue
            read_start = max(0, start - context_samples)
            read_end = min(total_samples, end + context_samples)
            signal = waveform[read_start:read_end]
            if start < context_samples:
                signal = torch.nn.functional.pad(signal, (context_samples - start, 0))
            yield self._extract_features(signal, logical_length, context_hops)

    def _file_feature_blocks(self, path: str) -> Iterator[torch.Tensor]:
        """Read only bounded, context-overlapped windows from one long file."""
        torch = _torch_module()
        if self._config is None or self._stft_block_samples is None:
            msg = "TensorRTSortformerAdapter is not initialized"
            raise RuntimeError(msg)
        soundfile = _soundfile_module()
        hop_length = int(self._config["hop_length"])
        context_hops = math.ceil((int(self._config["n_fft"]) / 2) / hop_length)
        context_samples = context_hops * hop_length
        with soundfile.SoundFile(path) as audio_file:
            if audio_file.samplerate != self.sample_rate:
                msg = (
                    f"Long Sortformer inputs must use the engine sample rate "
                    f"({self.sample_rate} Hz), got {audio_file.samplerate} Hz for {path}"
                )
                raise ValueError(msg)
            total_samples = len(audio_file)
            for start in range(0, total_samples, self._stft_block_samples):
                end = min(start + self._stft_block_samples, total_samples)
                logical_length = (end - start) // hop_length
                if logical_length == 0:
                    continue
                read_start = max(0, start - context_samples)
                read_end = min(total_samples, end + context_samples)
                audio_file.seek(read_start)
                data = audio_file.read(read_end - read_start, dtype="float32", always_2d=True)
                signal = torch.from_numpy(data).mean(dim=1)
                if start < context_samples:
                    signal = torch.nn.functional.pad(signal, (context_samples - start, 0))
                yield self._extract_features(signal, logical_length, context_hops)

    def _infer_batch(
        self,
        batch_states: list[Any],
        feature_windows: list[torch.Tensor],
        left_embeddings: list[int],
        right_embeddings: list[int],
        end_flags: list[int],
    ) -> tuple[list[Any], list[torch.Tensor]]:
        torch = _torch_module()
        if self._config is None or self._session is None or self._modules is None:
            msg = "TensorRTSortformerAdapter is not initialized"
            raise RuntimeError(msg)
        chunk_len = int(self._config["chunk_len"])
        self._modules.sync_pending_compression_batched(batch_states)

        chunks = torch.zeros(
            (len(feature_windows), chunk_len, _FEATURE_DIM),
            dtype=torch.float32,
            device=self._session.device,
        )
        chunk_lengths = [min(window.shape[0], chunk_len) for window in feature_windows]
        for index, (window, length) in enumerate(zip(feature_windows, chunk_lengths, strict=True)):
            chunks[index, :length] = window[:length].to(device=self._session.device, non_blocking=True)
        speaker_lengths = [state.spkcache_len_cached for state in batch_states]
        fifo_lengths = [state.fifo_len_cached for state in batch_states]
        max_speaker_length = max(1, *speaker_lengths)
        max_fifo_length = max(1, *fifo_lengths)
        fifo = []
        for state in batch_states:
            item = state.fifo[0, :max_fifo_length]
            if item.shape[0] < max_fifo_length:
                item = torch.nn.functional.pad(item, (0, 0, 0, max_fifo_length - item.shape[0]))
            fifo.append(item)
        outputs = self._session.infer(
            {
                "chunk": chunks,
                "chunk_lengths": torch.tensor(chunk_lengths, dtype=torch.int64, device=self._session.device),
                "spkcache": torch.stack(
                    [state.spkcache[0, :max_speaker_length] for state in batch_states],
                ),
                "spkcache_lengths": torch.tensor(
                    speaker_lengths,
                    dtype=torch.int64,
                    device=self._session.device,
                ),
                "fifo": torch.stack(fifo),
                "fifo_lengths": torch.tensor(fifo_lengths, dtype=torch.int64, device=self._session.device),
            }
        )
        chunk_embedding_lengths = outputs["chunk_emb_lengths"]
        if chunk_embedding_lengths.numel() != len(batch_states):
            msg = (
                "Sortformer TensorRT chunk_emb_lengths must contain one value per batch row, "
                f"got shape {tuple(chunk_embedding_lengths.shape)} for batch {len(batch_states)}"
            )
            raise RuntimeError(msg)
        embedding_lengths = [int(value) for value in chunk_embedding_lengths.detach().cpu().reshape(-1).tolist()]
        maximum_embedding_length = outputs["chunk_embs"].shape[1]
        if any(length < 1 or length > maximum_embedding_length for length in embedding_lengths):
            msg = (
                "Sortformer TensorRT chunk_emb_lengths contains a value outside the chunk_embs sequence, "
                f"got {embedding_lengths} for length {maximum_embedding_length}"
            )
            raise RuntimeError(msg)
        predictions = self._modules.apply_mask_to_preds(outputs["predictions"], outputs["pred_lengths"])
        updated_states, chunk_predictions, _ = self._modules.streaming_update_batched(
            batch_states=batch_states,
            chunk_embs=outputs["chunk_embs"],
            chunk_emb_lengths=embedding_lengths,
            preds=predictions,
            lc_list=left_embeddings,
            rc_list=right_embeddings,
            end_flags=end_flags,
        )
        probability_chunks = []
        for index, embedding_length in enumerate(embedding_lengths):
            output_length = max(0, embedding_length - left_embeddings[index] - right_embeddings[index])
            probability_chunks.append(chunk_predictions[index, :output_length].float().cpu())
        return updated_states, probability_chunks

    def _infer_probabilities(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        torch = _torch_module()
        if self._config is None or self._session is None or self._modules is None:
            msg = "TensorRTSortformerAdapter is not initialized"
            raise RuntimeError(msg)
        center_frames = int(self._config["center_chunk_frames"])
        left_frames = int(self._config["left_context_frames"])
        right_frames = int(self._config["right_context_frames"])
        subsampling = int(self._config["subsampling_factor"])
        batch_size = int(self.inference_batch_size or 1)

        states = [self._modules.init_streaming_state(self._session.device) for _ in features]
        positions = [0] * len(features)
        probabilities: list[list[torch.Tensor]] = [[] for _ in features]
        while True:
            active = [index for index in range(len(features)) if positions[index] < features[index].shape[0]]
            if not active:
                break
            for batch_start in range(0, len(active), batch_size):
                batch_active = active[batch_start : batch_start + batch_size]
                batch_states = [states[index] for index in batch_active]
                feature_windows = []
                left_embeddings = []
                right_embeddings = []
                end_flags = []
                for item_index in batch_active:
                    feature = features[item_index]
                    center_start = positions[item_index]
                    center_end = min(center_start + center_frames, feature.shape[0])
                    window_start = max(0, center_start - left_frames)
                    window_end = min(feature.shape[0], center_end + right_frames)
                    feature_windows.append(feature[window_start:window_end])
                    left_embeddings.append((center_start - window_start + subsampling - 1) // subsampling)
                    right_embeddings.append((window_end - center_end + subsampling - 1) // subsampling)
                    end_flags.append(int(center_end == feature.shape[0]))
                    positions[item_index] = center_end
                updated_states, probability_chunks = self._infer_batch(
                    batch_states,
                    feature_windows,
                    left_embeddings,
                    right_embeddings,
                    end_flags,
                )
                for batch_index, item_index in enumerate(batch_active):
                    states[item_index] = updated_states[batch_index]
                    probabilities[item_index].append(probability_chunks[batch_index])

        num_speakers = int(self._config["num_speakers"])
        return [
            torch.cat(parts) if parts else torch.empty((0, num_speakers), dtype=torch.float32)
            for parts in probabilities
        ]

    def _segments(self, probabilities: torch.Tensor) -> list[DiarizationSegment]:
        if self._config is None:
            msg = "TensorRTSortformerAdapter is not initialized"
            raise RuntimeError(msg)
        frame_step = float(self._config["output_step_ms"]) / 1000
        segments: list[DiarizationSegment] = []
        active_speakers = (probabilities > self.speech_threshold).any(dim=0).nonzero().flatten().tolist()
        for speaker in active_speakers:
            for start, end in _binarize(
                probabilities[:, speaker],
                frame_step,
                self.speech_threshold,
            ).tolist():
                segments.append(
                    DiarizationSegment(
                        start=round(float(start), 2),
                        end=round(float(end), 2),
                        speaker=f"speaker_{speaker}",
                    )
                )
        return sorted(segments, key=lambda segment: (segment["start"], segment["speaker"]))

    def _prepare_inputs(
        self,
        items: list[dict[str, Any]],
    ) -> tuple[list[int], list[torch.Tensor | str], list[DiarizationResult]]:
        torch = _torch_module()
        results = [DiarizationResult(segments=[]) for _ in items]
        valid_indices = []
        sources: list[torch.Tensor | str] = []
        for index, item in enumerate(items):
            has_waveform = item.get("waveform") is not None
            has_filepath = item.get("audio_filepath") is not None
            if has_waveform == has_filepath:
                msg = f"Diarization stage item {index} must provide exactly one of waveform or audio_filepath"
                raise ValueError(msg)
            if has_filepath:
                filepath = item["audio_filepath"]
                if not isinstance(filepath, (str, Path)) or not str(filepath).strip():
                    msg = f"Diarization stage item {index} has an invalid audio_filepath: {filepath!r}"
                    raise ValueError(msg)
                valid_indices.append(index)
                sources.append(str(filepath))
                continue

            item_sample_rate = item.get("sample_rate")
            if (
                isinstance(item_sample_rate, bool)
                or not isinstance(item_sample_rate, Integral)
                or int(item_sample_rate) != self.sample_rate
            ):
                msg = (
                    f"Diarization stage must provide {self.sample_rate} Hz audio for {self.model_id!r}; "
                    f"received {item_sample_rate!r}"
                )
                raise ValueError(msg)
            waveform = np.asarray(item["waveform"], dtype=np.float32)
            if waveform.ndim != 1:
                msg = f"Diarization stage must provide a mono 1-D waveform, got shape {waveform.shape}"
                raise ValueError(msg)
            if not np.isfinite(waveform).all():
                msg = f"Diarization stage item {index} contains non-finite audio samples"
                raise ValueError(msg)
            if waveform.size == 0:
                continue
            valid_indices.append(index)
            sources.append(torch.from_numpy(np.ascontiguousarray(waveform)))
        return valid_indices, sources, results

    def diarize_batch(self, items: list[dict[str, Any]]) -> list[DiarizationResult]:
        """Diarize stage-prepared waveforms, streaming only long outliers."""
        if not items:
            return []
        if self._session is None:
            msg = "TensorRTSortformerAdapter is not initialized; call load_model() first"
            raise RuntimeError(msg)
        valid_indices, sources, results = self._prepare_inputs(items)
        if not sources:
            return results

        regular_positions = []
        long_positions = []
        for position, source in enumerate(sources):
            duration, empty = self._source_duration_and_empty(source)
            if empty:
                continue
            (long_positions if duration > self.long_audio_seconds else regular_positions).append(position)
        if regular_positions:
            regular_sources = [sources[position] for position in regular_positions]
            regular_waveforms = self._regular_waveforms(regular_sources)
            probabilities = self._infer_probabilities(self._features(regular_waveforms))
            for position, item_probabilities in zip(regular_positions, probabilities, strict=True):
                results[valid_indices[position]] = DiarizationResult(segments=self._segments(item_probabilities))

        for position in long_positions:
            source = sources[position]
            feature_blocks = (
                self._file_feature_blocks(source) if isinstance(source, str) else self._waveform_feature_blocks(source)
            )
            normalized_features = self._normalize_feature_blocks(feature_blocks)
            probabilities = self._infer_probabilities([normalized_features])[0]
            results[valid_indices[position]] = DiarizationResult(segments=self._segments(probabilities))
        return results
