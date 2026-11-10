# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert Sound Expert posteriors into intervals that can be removed from VAD speech."""

from __future__ import annotations

import contextlib
import os
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

Interval = tuple[float, float]
DEFAULT_PEE_MODEL = "nvidia/ParallelExpertEncoder"
DEFAULT_SOUND_EVENT_TOKENS = (
    "<ev:breath>",
    "<ev:crying>",
    "<ev:effort_strain>",
    "<ev:gesture_noise>",
    "<ev:laughter>",
    "<ev:mouth_noise>",
    "<ev:music_vocal>",
    "<ev:pause_timing>",
    "<ev:shout_exclaim>",
    "<ev:throat_cough>",
)
SUPPORTED_BACKENDS = {"auto", "ctc", "pee"}
SUPPORTED_DTYPES = {"float32", "float16", "bfloat16"}
SAMPLE_RATE = 16000


@dataclass(frozen=True)
class SoundExpertConfig:
    """Configuration for deriving approximate spans from Sound Expert event peaks."""

    model: str | None = None
    backend: str = "auto"
    device: str = "auto"
    compute_dtype: str = "bfloat16"
    threshold: float = 0.5
    frame_pad_sec: float = 0.2
    merge_gap_sec: float = 0.1
    min_event_duration_sec: float = 0.0
    min_residual_speech_sec: float = 0.05
    chunk_duration_sec: float = 15.0
    chunk_overlap_sec: float = 1.0
    event_tokens: tuple[str, ...] = DEFAULT_SOUND_EVENT_TOKENS

    @property
    def enabled(self) -> bool:
        return bool(self.model)


@dataclass(frozen=True)
class SoundEventPrediction:
    """Union and per-token event intervals in recording-relative seconds."""

    intervals: tuple[Interval, ...] = ()
    by_event: dict[str, tuple[Interval, ...]] | None = None


def merge_intervals(intervals: Sequence[Interval], *, maximum_gap: float = 0.0) -> list[Interval]:
    """Merge overlapping intervals and intervals separated by at most ``maximum_gap``."""
    merged: list[Interval] = []
    for start, end in sorted((float(start), float(end)) for start, end in intervals):
        if end <= start:
            continue
        if merged and start - merged[-1][1] <= maximum_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def subtract_intervals(
    intervals: Sequence[Interval],
    exclusions: Sequence[Interval],
    *,
    minimum_duration: float = 0.0,
) -> list[Interval]:
    """Subtract exclusion spans from speech spans without changing the time base."""
    cleaned_exclusions = merge_intervals(exclusions)
    output: list[Interval] = []
    for interval_start, interval_end in merge_intervals(intervals):
        cursor = interval_start
        for exclusion_start, exclusion_end in cleaned_exclusions:
            if exclusion_end <= cursor:
                continue
            if exclusion_start >= interval_end:
                break
            if exclusion_start > cursor:
                piece_end = min(interval_end, exclusion_start)
                if piece_end - cursor >= minimum_duration:
                    output.append((cursor, piece_end))
            cursor = max(cursor, exclusion_end)
            if cursor >= interval_end:
                break
        if interval_end - cursor >= minimum_duration:
            output.append((cursor, interval_end))
    return output


def _scores_to_intervals(
    scores: Sequence[float],
    *,
    frame_duration: float,
    audio_duration: float,
    config: SoundExpertConfig,
) -> list[Interval]:
    active_start: float | None = None
    intervals: list[Interval] = []
    for frame_index, score in enumerate(scores):
        start = min(audio_duration, frame_index * frame_duration)
        end = min(audio_duration, (frame_index + 1) * frame_duration)
        if float(score) >= config.threshold:
            if active_start is None:
                active_start = start
        elif active_start is not None:
            if start - active_start >= config.min_event_duration_sec:
                intervals.append(
                    (
                        max(0.0, active_start - config.frame_pad_sec),
                        min(audio_duration, start + config.frame_pad_sec),
                    )
                )
            active_start = None
        if end >= audio_duration:
            break
    if active_start is not None and audio_duration - active_start >= config.min_event_duration_sec:
        intervals.append((max(0.0, active_start - config.frame_pad_sec), audio_duration))
    return intervals


def event_probabilities_to_prediction(
    probabilities: Sequence[Sequence[float]],
    event_tokens: Sequence[str],
    *,
    frame_duration: float,
    audio_duration: float,
    config: SoundExpertConfig,
) -> SoundEventPrediction:
    """Turn per-frame event probabilities into padded approximate event intervals."""
    if frame_duration <= 0 or audio_duration < 0:
        msg = "frame duration must be positive and audio duration non-negative"
        raise ValueError(msg)
    if probabilities and any(len(frame) != len(event_tokens) for frame in probabilities):
        msg = "every probability frame must have one value per event token"
        raise ValueError(msg)

    by_event: dict[str, tuple[Interval, ...]] = {}
    combined: list[Interval] = []
    for event_index, token in enumerate(event_tokens):
        event_intervals = _scores_to_intervals(
            [float(frame[event_index]) for frame in probabilities],
            frame_duration=frame_duration,
            audio_duration=audio_duration,
            config=config,
        )
        merged = tuple(merge_intervals(event_intervals, maximum_gap=config.merge_gap_sec))
        by_event[token] = merged
        combined.extend(merged)
    return SoundEventPrediction(
        intervals=tuple(merge_intervals(combined, maximum_gap=config.merge_gap_sec)),
        by_event=by_event,
    )


def _canonical_event_token(value: str) -> str:
    token = value.strip()
    if token.startswith("<ev:") and token.endswith(">"):
        return token
    return f"<ev:{token.removeprefix('ev:').strip('<>')}>"


def _parallel_expert_target(path: Path) -> bool:
    if not path.is_file() or path.suffix != ".nemo":
        return False
    try:
        with tarfile.open(path, mode="r") as archive:
            member = next(
                (item for item in archive.getmembers() if os.path.basename(item.name) == "model_config.yaml"),
                None,
            )
            if member is None:
                return False
            stream = archive.extractfile(member)
            return stream is not None and "ParallelExpertEncoderPT" in stream.read().decode("utf-8")
    except (OSError, tarfile.TarError, UnicodeDecodeError):
        return False


class SoundExpertDetector:
    """Run a standalone Sound Expert CTC model or an embedded PEE Sound Expert."""

    def __init__(self, config: SoundExpertConfig) -> None:
        self.config = config
        self._validate_config()
        try:
            import torch
        except ImportError as exc:
            msg = "Sound Expert inference requires torch"
            raise RuntimeError(msg) from exc

        self._torch = torch
        device_name = ("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else config.device
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            msg = f"Sound Expert requested {device_name}, but CUDA is unavailable"
            raise RuntimeError(msg)
        self._device = torch.device(device_name)
        self._dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.compute_dtype]
        self._backend = self._resolve_backend()
        self._model: Any = None
        self._pee: Any = None
        self._preprocessor: Any = None
        self._all_event_tokens: tuple[str, ...] = ()
        self._selected_event_indexes: tuple[int, ...] = ()
        self._ctc_event_ids: tuple[int, ...] = ()
        if self._backend == "pee":
            self._load_pee()
        else:
            self._load_ctc()
        self._configure_event_selection()

    @property
    def metadata(self) -> dict[str, object]:
        return {
            **asdict(self.config),
            "backend_resolved": self._backend,
            "selected_event_tokens": list(self.event_tokens),
            "note": "Point tags are expanded by frame_pad_sec; they are approximate spans.",
        }

    @property
    def event_tokens(self) -> tuple[str, ...]:
        return tuple(self._all_event_tokens[index] for index in self._selected_event_indexes)

    def _validate_config(self) -> None:
        if not self.config.enabled:
            msg = "Sound Expert model is not configured"
            raise ValueError(msg)
        if self.config.backend not in SUPPORTED_BACKENDS:
            msg = f"unsupported Sound Expert backend: {self.config.backend}"
            raise ValueError(msg)
        if self.config.compute_dtype not in SUPPORTED_DTYPES:
            msg = f"unsupported Sound Expert compute dtype: {self.config.compute_dtype}"
            raise ValueError(msg)
        if not 0.0 <= self.config.threshold <= 1.0:
            msg = "Sound Expert threshold must be in [0, 1]"
            raise ValueError(msg)
        durations = (
            self.config.frame_pad_sec,
            self.config.merge_gap_sec,
            self.config.min_event_duration_sec,
            self.config.min_residual_speech_sec,
            self.config.chunk_overlap_sec,
        )
        if min(durations) < 0 or self.config.chunk_duration_sec <= 0:
            msg = "Sound Expert durations must be non-negative and chunk duration positive"
            raise ValueError(msg)
        if self.config.chunk_overlap_sec >= self.config.chunk_duration_sec:
            msg = "Sound Expert chunk overlap must be shorter than chunk duration"
            raise ValueError(msg)
        if not self.config.event_tokens:
            msg = "at least one Sound Expert event token must be selected"
            raise ValueError(msg)

    def _resolve_backend(self) -> str:
        if self.config.backend != "auto":
            return self.config.backend
        model = str(self.config.model)
        if model == DEFAULT_PEE_MODEL or "ParallelExpertEncoder" in model:
            return "pee"
        if _parallel_expert_target(Path(model).expanduser()):
            return "pee"
        return "ctc"

    def _load_ctc(self) -> None:
        try:
            from nemo.collections.asr.models import ASRModel, EncDecCTCModel, EncDecCTCModelBPE
        except ImportError as exc:
            msg = "standalone Sound Expert inference requires nemo_toolkit[asr]"
            raise RuntimeError(msg) from exc

        model_ref = str(self.config.model)
        model_path = Path(model_ref).expanduser()
        if model_path.is_file():
            cfg = ASRModel.restore_from(restore_path=str(model_path), return_config=True)
            cfg = cfg.get("model", cfg)
            model_class = EncDecCTCModelBPE if cfg.get("tokenizer") else EncDecCTCModel
            model = model_class.restore_from(restore_path=str(model_path), map_location="cpu")
        else:
            model = ASRModel.from_pretrained(model_name=model_ref, map_location="cpu")
        if not all(hasattr(model, name) for name in ("preprocessor", "encoder", "decoder")):
            msg = f"{model_ref} is not a compatible CTC Sound Expert model"
            raise TypeError(msg)
        self._model = model.to(self._device)
        self._model.eval()
        self._model.freeze()
        vocabulary = list(self._model.cfg.decoder.get("vocabulary") or ())
        self._all_event_tokens = tuple(str(token) for token in vocabulary if str(token).startswith("<ev:"))
        self._ctc_event_ids = tuple(index for index, token in enumerate(vocabulary) if str(token).startswith("<ev:"))

    def _load_pee(self) -> None:
        try:
            from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
            from nemo.collections.asr.modules.parallel_expert_encoder import ParallelExpertEncoderPT
        except ImportError as exc:
            msg = "PEE Sound Expert inference requires a NeMo release that provides ParallelExpertEncoderPT"
            raise RuntimeError(msg) from exc

        model_ref = str(self.config.model)
        model_path = Path(model_ref).expanduser()
        if model_path.is_file():
            shell = ParallelExpertEncoderPT.restore_from(
                restore_path=str(model_path),
                map_location="cpu",
                strict=True,
            )
        else:
            shell = ParallelExpertEncoderPT.from_pretrained(
                model_name=model_ref,
                map_location="cpu",
                strict=True,
            )
        self._pee = shell.encoder.to(self._device)
        self._pee.eval()
        self._pee.freeze()
        feature_count = int(shell.cfg.sound_expert_cfg.get("feat_in", 128))
        self._preprocessor = AudioToMelSpectrogramPreprocessor(
            sample_rate=SAMPLE_RATE,
            normalize=None,
            window_size=0.025,
            window_stride=0.01,
            window="hann",
            features=feature_count,
            n_fft=512,
            log=True,
            frame_splicing=1,
            dither=0.0,
            pad_to=0,
            pad_value=0.0,
        ).to(self._device)
        self._preprocessor.eval()
        self._all_event_tokens = tuple(str(token) for token in self._pee.sound_event_tokens)

    def _configure_event_selection(self) -> None:
        requested = tuple(_canonical_event_token(token) for token in self.config.event_tokens)
        missing = sorted(set(requested) - set(self._all_event_tokens))
        if missing:
            msg = (
                f"Sound Expert model does not provide requested event tokens {missing}; "
                f"available={list(self._all_event_tokens)}"
            )
            raise ValueError(msg)
        self._selected_event_indexes = tuple(self._all_event_tokens.index(token) for token in requested)

    def _autocast(self) -> contextlib.AbstractContextManager:
        enabled = self._device.type == "cuda" and self._dtype != self._torch.float32
        if not enabled:
            return contextlib.nullcontext()
        return self._torch.autocast(device_type="cuda", dtype=self._dtype)

    def _predict_chunk_ctc(self, chunk: torch.Tensor) -> list[list[float]]:
        signal = chunk.unsqueeze(0).to(self._device)
        length = self._torch.tensor([chunk.numel()], dtype=self._torch.long, device=self._device)
        with self._torch.inference_mode(), self._autocast():
            result = self._model(input_signal=signal, input_signal_length=length)
            logits = result[0]
            encoded_length = int(result[1][0].item())
            probabilities = self._torch.softmax(logits[0, :encoded_length].float(), dim=-1)
            event_probabilities = probabilities[:, list(self._ctc_event_ids)]
            event_probabilities = event_probabilities[:, list(self._selected_event_indexes)]
        return event_probabilities.cpu().tolist()

    def _predict_chunk_pee(self, chunk: torch.Tensor) -> list[list[float]]:
        signal = chunk.unsqueeze(0).to(self._device)
        length = self._torch.tensor([chunk.numel()], dtype=self._torch.long, device=self._device)
        with self._torch.inference_mode():
            mel, mel_length = self._preprocessor(input_signal=signal, length=length)
        with self._torch.inference_mode(), self._autocast():
            result = self._pee(mel.to(self._device), mel_length.to(self._device), return_experts=True)
            experts = result[-1]
            sound_states, sound_length = experts["sound"]
            event_probabilities, _ = self._pee._sound_tag_posteriors(sound_states)
            valid_length = min(int(sound_length[0].item()), int(event_probabilities.shape[1]))
            event_probabilities = event_probabilities[0, :valid_length, list(self._selected_event_indexes)]
        return event_probabilities.float().cpu().tolist()

    def predict_waveform(self, waveform: torch.Tensor) -> SoundEventPrediction:
        """Predict approximate event spans for a mono 16-kHz waveform."""
        waveform = waveform.detach().float().cpu().reshape(-1)
        total_samples = int(waveform.numel())
        if total_samples == 0:
            return SoundEventPrediction(by_event=dict.fromkeys(self.event_tokens, ()))
        chunk_samples = max(1, round(self.config.chunk_duration_sec * SAMPLE_RATE))
        overlap_samples = max(0, round(self.config.chunk_overlap_sec * SAMPLE_RATE))
        step_samples = chunk_samples - overlap_samples
        by_event: dict[str, list[Interval]] = {token: [] for token in self.event_tokens}

        for chunk_start in range(0, total_samples, step_samples):
            chunk_end = min(total_samples, chunk_start + chunk_samples)
            chunk = waveform[chunk_start:chunk_end]
            probabilities = (
                self._predict_chunk_pee(chunk) if self._backend == "pee" else self._predict_chunk_ctc(chunk)
            )
            chunk_duration = chunk.numel() / SAMPLE_RATE
            frame_duration = chunk_duration / len(probabilities) if probabilities else chunk_duration
            prediction = event_probabilities_to_prediction(
                probabilities,
                self.event_tokens,
                frame_duration=frame_duration,
                audio_duration=chunk_duration,
                config=self.config,
            )
            offset = chunk_start / SAMPLE_RATE
            for token, intervals in (prediction.by_event or {}).items():
                by_event[token].extend((offset + start, offset + end) for start, end in intervals)
            if chunk_end == total_samples:
                break

        merged_by_event = {
            token: tuple(merge_intervals(intervals, maximum_gap=self.config.merge_gap_sec))
            for token, intervals in by_event.items()
        }
        combined = [interval for intervals in merged_by_event.values() for interval in intervals]
        return SoundEventPrediction(
            intervals=tuple(merge_intervals(combined, maximum_gap=self.config.merge_gap_sec)),
            by_event=merged_by_event,
        )
