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

"""NeMo Streaming Sortformer behind the diarization adapter contract."""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import numpy as np
from loguru import logger

from nemo_curator.models.audio.speaker_diarization.base import (
    DiarizationResult,
    DiarizationSegment,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel
    from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor


_DEFAULT_MODEL_ID = "nvidia/diar_streaming_sortformer_4spk-v2.1"
_DEFAULT_SAMPLE_RATE = 16_000
_DEFAULT_STFT_BLOCK_SECONDS = 60 * 60
_SUPPORTED_PRECISIONS = {"fp32", "fp16", "bf16"}


def _torch_module() -> Any:  # noqa: ANN401
    """Import torch only when an adapter operation needs it."""
    try:
        import torch
    except ImportError as exc:
        msg = "NeMoSortformerAdapter requires the audio_common extra: uv sync --extra audio_common"
        raise ImportError(msg) from exc
    return torch


def _sortformer_model_class() -> type:
    """Resolve NeMo's model class without importing NeMo on package import."""
    try:
        from nemo.collections.asr.models import SortformerEncLabelModel
    except ImportError as exc:
        msg = "NeMoSortformerAdapter requires the audio_common extra: uv sync --extra audio_common"
        raise ImportError(msg) from exc
    return SortformerEncLabelModel


def _snapshot_download(repo_id: str, cache_dir: str | None) -> str:
    """Download one Hugging Face repository without importing Hub eagerly."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        msg = "NeMoSortformerAdapter requires huggingface-hub from the audio_common extra"
        raise ImportError(msg) from exc
    return snapshot_download(repo_id=repo_id, cache_dir=cache_dir)


def _segment_values(segment: object) -> tuple[object, object, object] | None:
    """Extract provider-independent values from one supported segment shape."""
    if isinstance(segment, str):
        parts = segment.strip().split()
        if len(parts) < 2:  # noqa: PLR2004
            return None
        return parts[0], parts[1], parts[2] if len(parts) > 2 else "unknown"  # noqa: PLR2004
    if isinstance(segment, dict) and {"start", "end", "speaker"} <= segment.keys():
        return segment["start"], segment["end"], segment["speaker"]
    if hasattr(segment, "start") and hasattr(segment, "end"):
        return (
            segment.start,
            segment.end,
            getattr(segment, "speaker", getattr(segment, "label", "unknown")),
        )
    if isinstance(segment, (tuple, list)) and len(segment) >= 3:  # noqa: PLR2004
        return segment[0], segment[1], segment[2]
    return None


def parse_sortformer_segments(raw_segments: Iterable[object]) -> list[DiarizationSegment]:
    """Convert NeMo segment variants to finite, manifest-ready dictionaries."""
    segments: list[DiarizationSegment] = []
    for raw_segment in raw_segments:
        values = _segment_values(raw_segment)
        if values is None:
            logger.warning("Unrecognised Sortformer segment format: {!r}", raw_segment)
            continue
        try:
            start = float(values[0])
            end = float(values[1])
        except (TypeError, ValueError):
            logger.warning("Invalid Sortformer segment timestamps: {!r}", raw_segment)
            continue
        if not math.isfinite(start) or not math.isfinite(end):
            logger.warning("Non-finite Sortformer segment timestamps: {!r}", raw_segment)
            continue
        segments.append(
            DiarizationSegment(
                start=start,
                end=end,
                speaker=str(values[2]),
            )
        )
    return segments


def _extract_nemo_features_in_blocks(
    preprocessor: AudioToMelSpectrogramPreprocessor,
    waveform: torch.Tensor,
    sample_count: int,
    *,
    block_seconds: float = _DEFAULT_STFT_BLOCK_SECONDS,
) -> torch.Tensor:
    """Run the streaming Sortformer preprocessor in bounded STFT blocks."""
    torch = _torch_module()
    featurizer = preprocessor.featurizer
    hop_length = int(featurizer.hop_length)
    context_hops = math.ceil((int(featurizer.n_fft) / 2) / hop_length)
    block_samples = int(block_seconds * int(preprocessor._sample_rate))
    block_samples -= block_samples % hop_length
    if block_samples <= 0:
        msg = f"Sortformer STFT block duration {block_seconds!r} is shorter than one feature hop"
        raise ValueError(msg)

    sample_count = min(int(sample_count), int(waveform.numel()))
    expected_frames = sample_count // hop_length
    if expected_frames == 0:
        return torch.empty((int(featurizer.nfilt), 0), dtype=torch.float32)

    context_samples = context_hops * hop_length
    first_buffer = next(iter(preprocessor.buffers()), None)
    device = first_buffer.device if first_buffer is not None else waveform.device
    blocks: list[Any] = []

    # NeMo's default ``per_feature`` normalization is defined over the whole
    # logical recording. Running the complete preprocessor independently for
    # each bounded STFT block would therefore change the model inputs. Extract
    # raw log-mel blocks first, then normalize the concatenated logical frames
    # exactly once below. Always restore the shared worker-local preprocessor,
    # including when feature extraction raises.
    normalize_type = featurizer.normalize
    try:
        featurizer.normalize = None
        for start in range(0, sample_count, block_samples):
            end = min(start + block_samples, sample_count)
            logical_frames = end // hop_length - start // hop_length
            if logical_frames == 0:
                continue

            read_start = max(0, start - context_samples)
            read_end = min(sample_count, end + context_samples)
            signal = waveform.reshape(-1)[read_start:read_end]
            if start < context_samples:
                signal = torch.nn.functional.pad(signal, (context_samples - start, 0))

            signal = signal.reshape(1, -1).to(device=device, dtype=torch.float32, non_blocking=True)
            signal_length = torch.tensor([signal.shape[1]], dtype=torch.long, device=device)
            processed, _ = preprocessor(input_signal=signal, length=signal_length)
            block = processed[0, :, context_hops : context_hops + logical_frames]
            blocks.append(block.to(device="cpu"))
    finally:
        featurizer.normalize = normalize_type

    features = torch.cat(blocks, dim=1)
    if features.shape[1] != expected_frames:
        msg = f"Bounded NeMo STFT produced {features.shape[1]} frames; expected {expected_frames}"
        raise RuntimeError(msg)

    if normalize_type:
        from nemo.collections.asr.parts.preprocessing.features import normalize_batch

        features_on_device = features.unsqueeze(0).to(device=device, non_blocking=True)
        feature_lengths = torch.tensor([expected_frames], dtype=torch.long, device=device)
        features_on_device, _, _ = normalize_batch(
            features_on_device,
            feature_lengths,
            normalize_type=normalize_type,
        )
        features = features_on_device[0].to(device="cpu")
    return features


def _bounded_nemo_process_signal(
    model: SortformerEncLabelModel,
    audio_signal: torch.Tensor,
    audio_signal_length: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replacement for NeMo ``process_signal`` that bounds each STFT call."""
    torch = _torch_module()
    source = audio_signal.detach().to(device="cpu")
    lengths = audio_signal_length.detach().to(device="cpu", dtype=torch.long)
    block_seconds = float(getattr(model, "_curator_stft_block_seconds", _DEFAULT_STFT_BLOCK_SECONDS))
    feature_list = [
        _extract_nemo_features_in_blocks(
            model.preprocessor,
            source[index],
            int(lengths[index].item()),
            block_seconds=block_seconds,
        )
        for index in range(source.shape[0])
    ]
    feature_lengths = torch.tensor([item.shape[1] for item in feature_list], dtype=torch.long)
    max_feature_length = int(feature_lengths.max().item())
    feature_dim = feature_list[0].shape[0]
    pad_value = float(getattr(model.preprocessor.featurizer, "pad_value", 0.0))
    processed = torch.full(
        (len(feature_list), feature_dim, max_feature_length),
        pad_value,
        dtype=feature_list[0].dtype,
    )
    for index, item in enumerate(feature_list):
        processed[index, :, : item.shape[1]] = item
    return processed.to(model.device), feature_lengths.to(model.device)


@dataclass
class NeMoSortformerAdapter:
    """Run NeMo Streaming Sortformer on stage-prepared mono waveforms.

    ``model_path`` takes precedence over ``model_id``. If neither denotes a
    local ``.nemo`` file, ``model_id`` is resolved as a Hugging Face repository
    and its first lexically sorted ``.nemo`` file is used.

    Streaming fields default to Curator's maintained v2.1 preset. Set an
    individual field to ``None`` to retain that checkpoint value.
    """

    DEFAULT_MODEL_ID: ClassVar[str] = _DEFAULT_MODEL_ID
    DEFAULT_SAMPLE_RATE: ClassVar[int] = _DEFAULT_SAMPLE_RATE

    model_id: str = _DEFAULT_MODEL_ID
    model_path: str | None = None
    cache_dir: str | None = None
    sample_rate: int = _DEFAULT_SAMPLE_RATE
    preloaded_model: Any = field(default=None, repr=False)

    chunk_len: int | None = 340
    chunk_left_context: int | None = 1
    chunk_right_context: int | None = 40
    fifo_len: int | None = 40
    spkcache_update_period: int | None = 300
    spkcache_len: int | None = 188

    inference_batch_size: int = 1
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    compile_encoder: bool = False
    bounded_stft: bool = True
    stft_block_seconds: float = float(_DEFAULT_STFT_BLOCK_SECONDS)
    max_positional_encoding_length: int | None = 30_000
    strict: bool = False

    _model: Any = field(default=None, init=False, repr=False)
    _device: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or (not self.model_id.strip() and self.model_path is None):
            msg = "NeMoSortformerAdapter.model_id must be non-empty when model_path is not set"
            raise ValueError(msg)
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, Integral) or self.sample_rate <= 0:
            msg = f"NeMoSortformerAdapter.sample_rate must be a positive integer, got {self.sample_rate!r}"
            raise ValueError(msg)
        if self.precision not in _SUPPORTED_PRECISIONS:
            msg = f"Unsupported Sortformer precision: {self.precision!r}"
            raise ValueError(msg)
        if (
            isinstance(self.inference_batch_size, bool)
            or not isinstance(self.inference_batch_size, Integral)
            or self.inference_batch_size <= 0
        ):
            msg = (
                "NeMoSortformerAdapter.inference_batch_size must be a positive integer, "
                f"got {self.inference_batch_size!r}"
            )
            raise ValueError(msg)
        self._validate_streaming_values()
        if isinstance(self.max_positional_encoding_length, bool) or (
            self.max_positional_encoding_length is not None
            and (
                not isinstance(self.max_positional_encoding_length, Integral)
                or self.max_positional_encoding_length <= 0
            )
        ):
            msg = (
                "NeMoSortformerAdapter.max_positional_encoding_length must be a positive integer or None, "
                f"got {self.max_positional_encoding_length!r}"
            )
            raise ValueError(msg)
        if (
            isinstance(self.stft_block_seconds, bool)
            or not isinstance(self.stft_block_seconds, Real)
            or not math.isfinite(float(self.stft_block_seconds))
            or self.stft_block_seconds <= 0
        ):
            msg = (
                "NeMoSortformerAdapter.stft_block_seconds must be finite and positive, "
                f"got {self.stft_block_seconds!r}"
            )
            raise ValueError(msg)
        self.sample_rate = int(self.sample_rate)
        self.inference_batch_size = int(self.inference_batch_size)
        self.stft_block_seconds = float(self.stft_block_seconds)
        if self.max_positional_encoding_length is not None:
            self.max_positional_encoding_length = int(self.max_positional_encoding_length)

    def _validate_streaming_values(self) -> None:
        minimums = {
            "chunk_len": 1,
            "chunk_left_context": 0,
            "chunk_right_context": 0,
            "fifo_len": 0,
            "spkcache_update_period": 0,
            "spkcache_len": 1,
        }
        for name, minimum in minimums.items():
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                msg = f"NeMoSortformerAdapter.{name} must be an integer >= {minimum} or None, got {value!r}"
                raise ValueError(msg)
            setattr(self, name, int(value))

    @staticmethod
    def _find_nemo_file(directory: Path, *, source: str) -> Path:
        nemo_files = sorted(directory.rglob("*.nemo"))
        if not nemo_files:
            msg = f"No .nemo file found in {directory} for {source}"
            raise FileNotFoundError(msg)
        return nemo_files[0]

    def _local_model_path(self) -> Path | None:
        if self.model_path is not None:
            return Path(self.model_path).expanduser()
        candidate = Path(self.model_id).expanduser()
        if candidate.is_file() or candidate.suffix == ".nemo":
            return candidate
        if candidate.is_dir():
            return self._find_nemo_file(candidate, source=f"model_id={self.model_id!r}")
        return None

    def _resolve_model_path(self) -> Path:
        local_path = self._local_model_path()
        if local_path is not None:
            if not local_path.is_file():
                msg = f"Sortformer .nemo checkpoint not found: {local_path}"
                raise FileNotFoundError(msg)
            return local_path.resolve()

        repo_dir = Path(_snapshot_download(self.model_id, self.cache_dir))
        return self._find_nemo_file(repo_dir, source=f"model_id={self.model_id!r}")

    def download_weights_on_node(self) -> None:
        """Resolve the local checkpoint without constructing a NeMo model."""
        if self.preloaded_model is not None:
            return
        self._resolve_model_path()

    def _restore_model(self, model_path: Path, device: Any) -> Any:  # noqa: ANN401
        model_cls = _sortformer_model_class()
        return model_cls.restore_from(
            restore_path=str(model_path),
            map_location=device,
            strict=self.strict,
        )

    @staticmethod
    def _preloaded_model_device(model: Any, torch: Any, *, fallback: Any) -> Any:  # noqa: ANN401
        """Infer an injected model's placement without moving caller-owned state."""
        for accessor_name in ("parameters", "buffers"):
            accessor = getattr(model, accessor_name, None)
            if not callable(accessor):
                continue
            try:
                tensor = next(iter(accessor()))
            except StopIteration:
                continue
            device = getattr(tensor, "device", None)
            if device is not None:
                return torch.device(device)

        model_device = getattr(model, "device", None)
        if isinstance(model_device, (str, torch.device)):
            return torch.device(model_device)
        return torch.device(fallback)

    def load_model(self, *, num_gpus: int) -> None:
        """Load and configure one worker-local Sortformer checkpoint."""
        if self._model is not None:
            return
        if isinstance(num_gpus, bool) or not isinstance(num_gpus, Integral) or num_gpus not in {0, 1}:
            msg = f"NeMoSortformerAdapter requires num_gpus to be 0 or 1, got {num_gpus!r}"
            raise ValueError(msg)

        torch = _torch_module()
        model = self.preloaded_model
        if model is None:
            if num_gpus and not torch.cuda.is_available():
                msg = "NeMoSortformerAdapter received num_gpus=1, but CUDA is not available"
                raise RuntimeError(msg)
            if not num_gpus and self.precision != "fp32":
                msg = f"NeMoSortformerAdapter precision={self.precision!r} requires one GPU"
                raise ValueError(msg)
            self._device = torch.device("cuda" if num_gpus else "cpu")
            model = self._restore_model(self._resolve_model_path(), self._device)
            move_model = True
        else:
            self._device = self._preloaded_model_device(
                model,
                torch,
                fallback="cuda" if num_gpus else "cpu",
            )
            if self.precision != "fp32" and self._device.type != "cuda":
                msg = (
                    f"NeMoSortformerAdapter precision={self.precision!r} requires the preloaded model "
                    f"to be on CUDA; found {self._device}"
                )
                raise ValueError(msg)
            move_model = False
        self._model = model
        try:
            if move_model:
                move = getattr(model, "to", None)
                if callable(move):
                    move(self._device)
            model.eval()
            self._configure_streaming(model)
            self._extend_positional_encoding(model)
            self._enable_bounded_stft(model)
            self._compile_encoder(model)
        except Exception:
            self.unload_model()
            raise

    def _configure_streaming(self, model: Any) -> None:  # noqa: ANN401
        modules = getattr(model, "sortformer_modules", None)
        if modules is None:
            msg = f"NeMo checkpoint {self.model_id!r} does not expose sortformer_modules"
            raise TypeError(msg)
        overrides = {
            "chunk_len": self.chunk_len,
            "chunk_left_context": self.chunk_left_context,
            "chunk_right_context": self.chunk_right_context,
            "fifo_len": self.fifo_len,
            "spkcache_update_period": self.spkcache_update_period,
            "spkcache_len": self.spkcache_len,
        }
        applied = False
        for name, value in overrides.items():
            if value is None:
                continue
            if name == "spkcache_update_period" and not hasattr(modules, name):
                continue
            setattr(modules, name, value)
            applied = True
        validate = getattr(modules, "_check_streaming_parameters", None)
        if applied and callable(validate):
            validate()

    def _extend_positional_encoding(self, model: Any) -> None:  # noqa: ANN401
        if self.max_positional_encoding_length is None:
            return
        pos_enc = getattr(getattr(model, "encoder", None), "pos_enc", None)
        if pos_enc is None or not callable(getattr(pos_enc, "extend_pe", None)):
            logger.warning("Sortformer encoder has no extendable positional encoding; skipping extension")
            return
        try:
            parameter = next(model.parameters())
            pos_enc.extend_pe(
                self.max_positional_encoding_length,
                parameter.device,
                parameter.dtype,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not extend Sortformer positional encoding: {}", exc)

    def _enable_bounded_stft(self, model: Any) -> None:  # noqa: ANN401
        if (
            not self.bounded_stft
            or not bool(getattr(model, "streaming_mode", False))
            or "_curator_original_process_signal" in vars(model)
        ):
            return
        process_signal = getattr(model, "process_signal", None)
        if not callable(process_signal) or getattr(model, "preprocessor", None) is None:
            logger.warning("Sortformer does not expose streaming signal preprocessing; bounded STFT disabled")
            return
        model._curator_original_process_signal = process_signal
        model._curator_stft_block_seconds = self.stft_block_seconds
        model.process_signal = MethodType(_bounded_nemo_process_signal, model)

    def _compile_encoder(self, model: Any) -> None:  # noqa: ANN401
        if not self.compile_encoder:
            return
        encoder = getattr(model, "encoder", None)
        if encoder is None:
            msg = f"NeMo checkpoint {self.model_id!r} does not expose an encoder to compile"
            raise TypeError(msg)
        model.encoder = _torch_module().compile(encoder, dynamic=False)

    @staticmethod
    def _restore_bounded_stft(model: Any) -> None:  # noqa: ANN401
        if "_curator_original_process_signal" in vars(model):
            model.process_signal = model._curator_original_process_signal
            del model._curator_original_process_signal
        if "_curator_stft_block_seconds" in vars(model):
            del model._curator_stft_block_seconds

    def unload_model(self) -> None:
        """Restore patched methods and release model and CUDA cache state."""
        model = self._model
        if model is not None:
            self._restore_bounded_stft(model)
        self._model = None
        self._device = None
        gc.collect()
        try:
            torch = _torch_module()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _run_diarization(
        self,
        audio: list[np.ndarray] | list[str],
        *,
        sample_rate: int | None,
    ) -> object:
        kwargs = {
            "audio": audio,
            "batch_size": min(self.inference_batch_size, len(audio)),
        }
        if sample_rate is not None:
            kwargs["sample_rate"] = sample_rate
        if self.precision == "fp32":
            return self._model.diarize(**kwargs)

        torch = _torch_module()
        dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        device_type = getattr(self._device, "type", str(self._device).split(":", maxsplit=1)[0])
        with torch.autocast(device_type=device_type, dtype=dtype):
            return self._model.diarize(**kwargs)

    def _prepare_inputs(
        self,
        items: list[dict[str, Any]],
    ) -> tuple[list[int], list[np.ndarray], list[int], list[str], list[DiarizationResult]]:
        """Separate waveform and direct-path inputs while retaining positions."""
        results = [DiarizationResult(segments=[]) for _ in items]
        waveform_indices: list[int] = []
        waveforms: list[np.ndarray] = []
        filepath_indices: list[int] = []
        filepaths: list[str] = []
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
                filepath_indices.append(index)
                filepaths.append(str(filepath))
                continue

            waveform_value = item["waveform"]
            waveform = np.asarray(waveform_value, dtype=np.float32)
            if waveform.ndim != 1:
                msg = f"Diarization stage must provide a mono 1-D waveform, got shape {waveform.shape}"
                raise ValueError(msg)
            if not np.isfinite(waveform).all():
                msg = f"Diarization stage item {index} contains non-finite audio samples"
                raise ValueError(msg)
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
            if waveform.size == 0:
                continue
            waveform_indices.append(index)
            waveforms.append(np.ascontiguousarray(waveform))
        return waveform_indices, waveforms, filepath_indices, filepaths, results

    @staticmethod
    def _scatter_outputs(
        indices: list[int],
        inputs: list[np.ndarray] | list[str],
        raw_outputs: object,
        results: list[DiarizationResult],
    ) -> None:
        if not isinstance(raw_outputs, (list, tuple)):
            msg = f"NeMo Sortformer returned unsupported output type {type(raw_outputs).__name__}"
            raise TypeError(msg)
        if len(raw_outputs) != len(inputs):
            msg = f"NeMo Sortformer returned {len(raw_outputs)} results for {len(inputs)} valid inputs"
            raise RuntimeError(msg)
        for index, raw_segments in zip(indices, raw_outputs, strict=True):
            results[index] = DiarizationResult(
                segments=parse_sortformer_segments(cast("Iterable[object]", raw_segments))
            )

    def diarize_batch(self, items: list[dict[str, Any]]) -> list[DiarizationResult]:
        """Diarize one stage-prepared batch while preserving every position."""
        if not items:
            return []
        if self._model is None:
            msg = "NeMoSortformerAdapter is not initialized; call load_model() first"
            raise RuntimeError(msg)

        waveform_indices, waveforms, filepath_indices, filepaths, results = self._prepare_inputs(items)
        if waveforms:
            self._scatter_outputs(
                waveform_indices,
                waveforms,
                self._run_diarization(waveforms, sample_rate=self.sample_rate),
                results,
            )
        if filepaths:
            self._scatter_outputs(
                filepath_indices,
                filepaths,
                self._run_diarization(filepaths, sample_rate=None),
                results,
            )
        return results


_parse_sortformer_segments = parse_sortformer_segments
