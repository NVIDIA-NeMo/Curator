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

"""OpenAI Whisper implementation of the audio-LID adapter."""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger
from torch.nn.utils.rnn import pad_sequence

from .base import (
    AudioLIDResult,
    _validate_sample_rate,
    _validate_single_device_gpu_count,
    _waveform_from_item,
)

if TYPE_CHECKING:
    from types import ModuleType

_WHISPER_SAMPLE_RATE = 16_000
_MEL_INPUT_RANK = 3


def _import_whisper() -> ModuleType:
    """Import OpenAI Whisper only when its adapter is selected."""
    try:
        import whisper
    except ImportError as exc:
        msg = "WhisperLIDAdapter requires an audio extra: uv sync --extra audio_cpu or --extra audio_cuda12"
        raise ImportError(msg) from exc
    return whisper


class WhisperTensorRTEncoder:
    """Adapt the shared TensorRT session to Whisper's encoder contract."""

    _INPUT = "mel"
    _OUTPUT = "audio_features"

    def __init__(self, engine_path: str | Path, session: Any = None) -> None:  # noqa: ANN401
        if session is None:
            from nemo_curator.stages.audio.inference.tensorrt_encoder import TensorRTEncoderSession

            session = TensorRTEncoderSession(engine_path)
        self.session = session
        for name, names in ((self._INPUT, session.input_names), (self._OUTPUT, session.output_names)):
            if name not in names:
                msg = f"Whisper encoder engine is missing tensor {name!r}; found {sorted(names)}"
                raise ValueError(msg)

        min_shape, _, max_shape = session.input_shape_range(self._INPUT)
        min_shape = tuple(min_shape)
        max_shape = tuple(max_shape)
        if (
            len(min_shape) != _MEL_INPUT_RANK
            or len(max_shape) != _MEL_INPUT_RANK
            or any(
                isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
                for dimension in (*min_shape, *max_shape)
            )
        ):
            msg = f"Whisper encoder has invalid mel profile: min={min_shape!r}, max={max_shape!r}"
            raise ValueError(msg)
        if min_shape[1:] != max_shape[1:]:
            msg = f"Whisper encoder requires fixed mel dimensions, got min={min_shape!r}, max={max_shape!r}"
            raise ValueError(msg)
        self.min_batch = min_shape[0]
        self.max_batch, self.n_mels, self.n_frames = max_shape

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        return self.forward(mel)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Encode one Mel batch, splitting rows at the engine profile limit."""
        if mel.ndim != _MEL_INPUT_RANK:
            msg = f"Whisper encoder expects mel shaped [batch, n_mels, n_frames], got {tuple(mel.shape)}"
            raise ValueError(msg)
        batch, n_mels, n_frames = mel.shape
        if n_mels != self.n_mels or n_frames != self.n_frames:
            msg = f"Whisper encoder engine expects mel [*, {self.n_mels}, {self.n_frames}], got {tuple(mel.shape)}"
            raise ValueError(msg)
        if batch <= self.max_batch:
            return self._infer_group(mel)

        # TensorRTEncoderSession intentionally reuses output buffers. Clone each
        # group before the next infer call overwrites it.
        groups: list[torch.Tensor] = []
        for start in range(0, batch, self.max_batch):
            encoded = self._infer_group(mel[start : start + self.max_batch])
            groups.append(encoded.clone())
        return torch.cat(groups, dim=0)

    def _infer_group(self, mel: torch.Tensor) -> torch.Tensor:
        rows = mel.shape[0]
        if rows < self.min_batch:
            mel = torch.nn.functional.pad(mel, (0, 0, 0, 0, 0, self.min_batch - rows))
        encoded = self.session.infer({self._INPUT: mel})[self._OUTPUT]
        return encoded[:rows]

    def close(self) -> None:
        """Release the shared session's persistent TensorRT buffers."""
        self.session.close()


@dataclass
class WhisperLIDAdapter:
    """Identify speech with OpenAI Whisper's language-token decoder step."""

    model_size: str = "medium"
    sample_rate: int = _WHISPER_SAMPLE_RATE
    model_path: str | None = None
    download_root: str | None = None
    fp16: bool = True
    backend: str = "torch"
    tensorrt_engine_path: str | None = None
    model_batch_size: int = 8
    _model: Any = field(default=None, init=False, repr=False)
    _device: torch.device | None = field(default=None, init=False, repr=False)
    _mel_dtype: torch.dtype = field(default=torch.float32, init=False, repr=False)
    _hann_window: torch.Tensor | None = field(default=None, init=False, repr=False)
    _encoder: WhisperTensorRTEncoder | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.sample_rate = _validate_sample_rate(self.sample_rate, owner=type(self).__name__)
        if self.sample_rate != _WHISPER_SAMPLE_RATE:
            msg = f"WhisperLIDAdapter requires sample_rate={_WHISPER_SAMPLE_RATE}, got {self.sample_rate}"
            raise ValueError(msg)
        if not isinstance(self.model_size, str) or not self.model_size.strip():
            msg = "WhisperLIDAdapter.model_size must be a non-empty string"
            raise ValueError(msg)
        if (
            isinstance(self.model_batch_size, bool)
            or not isinstance(self.model_batch_size, int)
            or self.model_batch_size <= 0
        ):
            msg = f"WhisperLIDAdapter.model_batch_size must be a positive integer, got {self.model_batch_size!r}"
            raise ValueError(msg)
        if self.backend not in {"torch", "tensorrt"}:
            msg = f"WhisperLIDAdapter.backend must be 'torch' or 'tensorrt', got {self.backend!r}"
            raise ValueError(msg)
        if self.backend == "tensorrt" and not self.tensorrt_engine_path:
            msg = "WhisperLIDAdapter.tensorrt_engine_path is required when backend='tensorrt'"
            raise ValueError(msg)

    def _checkpoint(self) -> str:
        if self.model_path is None:
            return self.model_size
        path = Path(self.model_path).expanduser()
        if not path.is_file():
            msg = f"Whisper checkpoint not found: {path}"
            raise FileNotFoundError(msg)
        return str(path)

    def _load_openai_model(self, device: str | torch.device) -> object:
        kwargs: dict[str, Any] = {"device": device}
        if self.download_root is not None:
            kwargs["download_root"] = self.download_root
        return _import_whisper().load_model(self._checkpoint(), **kwargs)

    def download_weights_on_node(self) -> None:
        """Populate Whisper's checkpoint cache without retaining a model."""
        if self.model_path is not None:
            self._checkpoint()
            return
        model = self._load_openai_model("cpu")
        del model
        gc.collect()

    def load_model(self, *, num_gpus: int) -> None:
        """Load Whisper and, when selected, its TensorRT audio encoder."""
        if self._model is not None:
            return
        gpu_count = _validate_single_device_gpu_count(
            num_gpus,
            owner=type(self).__name__,
            gpu_required=self.backend == "tensorrt",
        )
        if gpu_count and not torch.cuda.is_available():
            msg = "WhisperLIDAdapter received num_gpus=1, but CUDA is not available"
            raise RuntimeError(msg)

        self._device = torch.device("cuda" if gpu_count else "cpu")
        model = self._load_openai_model(self._device)
        model.eval()
        self._model = model
        self._mel_dtype = torch.float16 if self.fp16 and self._device.type == "cuda" else torch.float32
        if self.backend == "tensorrt":
            self._encoder = WhisperTensorRTEncoder(str(self.tensorrt_engine_path))
        logger.info(
            "Loaded Whisper LID model {} with {} encoder on {}", self._checkpoint(), self.backend, self._device
        )

    def unload_model(self) -> None:
        """Release Whisper, the optional TensorRT session, and cached windows."""
        if self._encoder is not None:
            self._encoder.close()
        self._encoder = None
        self._model = None
        self._device = None
        self._mel_dtype = torch.float32
        self._hann_window = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            logger.debug("CUDA cache clear skipped: {}", exc)

    def _encode(self, mel_batch: torch.Tensor) -> torch.Tensor:
        if self._encoder is None:
            return mel_batch
        return self._encoder(mel_batch).to(mel_batch.dtype)

    def _log_mel_spectrogram(
        self,
        audio_batch: torch.Tensor,
        n_mels: int,
        whisper: ModuleType,
    ) -> torch.Tensor:
        """Compute vectorized log-Mels with independent normalization per row."""
        if self._device is None:
            msg = "WhisperLIDAdapter device is not initialized"
            raise RuntimeError(msg)

        n_fft = int(whisper.audio.N_FFT)
        hop_length = int(whisper.audio.HOP_LENGTH)
        if (
            self._hann_window is None
            or self._hann_window.device != self._device
            or self._hann_window.dtype != audio_batch.dtype
        ):
            self._hann_window = torch.hann_window(n_fft, device=self._device, dtype=audio_batch.dtype)

        stft = torch.stft(
            audio_batch,
            n_fft,
            hop_length,
            window=self._hann_window,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs().square()
        filters = whisper.audio.mel_filters(self._device, n_mels).to(dtype=magnitudes.dtype)
        mel_spec = filters @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        per_sample_max = log_spec.amax(dim=(-2, -1), keepdim=True)
        log_spec = torch.maximum(log_spec, per_sample_max - 8.0)
        return ((log_spec + 4.0) / 4.0).to(dtype=self._mel_dtype)

    def identify_batch(self, items: list[dict[str, Any]]) -> list[AudioLIDResult]:  # noqa: C901
        """Run Whisper language detection in bounded model batches."""
        if not items:
            return []
        if self._model is None or self._device is None:
            msg = "WhisperLIDAdapter is not initialized; call load_model() first"
            raise RuntimeError(msg)

        whisper = _import_whisper()
        results = [AudioLIDResult("", 0.0) for _ in items]
        valid_indices: list[int] = []
        waveforms: list[torch.Tensor] = []
        for index, item in enumerate(items):
            waveform = _waveform_from_item(item, owner=type(self).__name__)
            if waveform.size:
                valid_indices.append(index)
                waveforms.append(torch.from_numpy(waveform))
        if not waveforms:
            return results

        n_mels = int(self._model.dims.n_mels)
        for start in range(0, len(waveforms), self.model_batch_size):
            chunk = waveforms[start : start + self.model_batch_size]
            chunk_indices = valid_indices[start : start + self.model_batch_size]
            audio_batch = pad_sequence(chunk, batch_first=True, padding_value=0.0)
            audio_batch = audio_batch.to(self._device, non_blocking=self._device.type == "cuda")
            audio_batch = whisper.pad_or_trim(audio_batch)
            mel_batch = self._log_mel_spectrogram(audio_batch, n_mels, whisper)
            with torch.inference_mode():
                _tokens, probabilities = self._model.detect_language(self._encode(mel_batch))

            if isinstance(probabilities, dict):
                probabilities = [probabilities]
            if len(probabilities) != len(chunk_indices):
                msg = f"Whisper returned {len(probabilities)} predictions for {len(chunk_indices)} inputs"
                raise RuntimeError(msg)
            for item_index, language_probabilities in zip(chunk_indices, probabilities, strict=True):
                if not language_probabilities:
                    msg = "Whisper returned an empty language-probability mapping"
                    raise RuntimeError(msg)
                language = max(language_probabilities, key=language_probabilities.get)
                results[item_index] = AudioLIDResult(language, float(language_probabilities[language]))
        return results
