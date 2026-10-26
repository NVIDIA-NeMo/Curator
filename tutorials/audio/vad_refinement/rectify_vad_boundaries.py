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

"""Refine MFA words with constrained Silero/Sortformer atomic voting."""

from __future__ import annotations

import argparse
import json
import os
from bisect import bisect_right
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from textgrid import IntervalTier, TextGrid

from tutorials.audio.vad_refinement.sound_expert_intervals import (
    DEFAULT_SOUND_EVENT_TOKENS,
    SoundExpertConfig,
    SoundExpertDetector,
    subtract_intervals,
)
from tutorials.audio.vad_refinement.textgrid_rttm import (
    SILENCE_MARKS,
    speech_intervals_from_rows,
    write_refined_textgrids,
)

if TYPE_CHECKING:
    import torch

Interval = tuple[float, float]
Word = list[float | str]
SAMPLE_RATE = 16000
RTTM_MIN_FIELDS = 8
TIME_EPSILON = 1e-9
REQUIRED_MANIFEST_FIELDS = {
    "session_id",
    "speaker_id",
    "audio_filepath",
    "offset",
    "duration",
    "mfa_textgrid",
}


@dataclass(frozen=True)
class VADConfig:
    threshold: float = 0.5
    min_speech_duration_ms: int = 100
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 0


@dataclass(frozen=True)
class SortformerVADConfig:
    model: str | None = None
    device: str = "auto"
    onset: float = 0.5
    offset: float = 0.5
    min_speech_duration_ms: int = 100
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 0

    @property
    def enabled(self) -> bool:
        return bool(self.model)


@dataclass(frozen=True)
class VADPredictions:
    silero: tuple[Interval, ...]
    secondary: tuple[Interval, ...]
    secondary_name: str = "nemo"
    sound_events: tuple[Interval, ...] = ()
    sound_events_by_type: dict[str, tuple[Interval, ...]] = field(default_factory=dict)
    silero_raw: tuple[Interval, ...] = ()
    secondary_raw: tuple[Interval, ...] = ()


class VADDetector(Protocol):
    metadata: dict[str, object]

    def __call__(self, audio_path: Path) -> VADPredictions: ...


def merge_intervals(intervals: list[Interval], *, maximum_gap: float = 0.0) -> list[Interval]:
    merged: list[Interval] = []
    for start, end in sorted((float(start), float(end)) for start, end in intervals):
        if end <= start:
            continue
        if merged and start - merged[-1][1] <= maximum_gap + 1e-9:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def frame_probabilities_to_intervals(
    probabilities: list[float],
    frame_duration: float,
    audio_duration: float,
    config: SortformerVADConfig,
) -> list[Interval]:
    """Apply hysteresis, duration filtering, and padding to frame probabilities."""
    active_start: float | None = None
    raw_intervals: list[Interval] = []
    for frame_index, probability in enumerate(probabilities):
        frame_time = min(audio_duration, frame_index * frame_duration)
        if active_start is None and probability >= config.onset:
            active_start = frame_time
        elif active_start is not None and probability < config.offset:
            if frame_time > active_start:
                raw_intervals.append((active_start, frame_time))
            active_start = None
    if active_start is not None and audio_duration > active_start:
        raw_intervals.append((active_start, audio_duration))

    joined = merge_intervals(
        raw_intervals,
        maximum_gap=config.min_silence_duration_ms / 1000,
    )
    minimum_speech = config.min_speech_duration_ms / 1000
    padding = config.speech_pad_ms / 1000
    padded = [
        (max(0.0, start - padding), min(audio_duration, end + padding))
        for start, end in joined
        if end - start >= minimum_speech
    ]
    return merge_intervals(padded)


def _load_audio_16k(audio_path: Path) -> torch.Tensor:
    try:
        import soundfile
        import torch
        import torchaudio
    except ImportError as exc:
        msg = "VAD inference requires soundfile, torch, and torchaudio"
        raise RuntimeError(msg) from exc

    samples, sample_rate = soundfile.read(
        str(audio_path),
        dtype="float32",
        always_2d=True,
    )
    waveform = torch.from_numpy(samples.mean(axis=1))
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
    return waveform.contiguous()


class SileroVADDetector:
    def __init__(self, config: VADConfig) -> None:
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad
        except ImportError as exc:
            msg = "Silero prediction requires silero-vad"
            raise RuntimeError(msg) from exc
        self.config = config
        self._get_speech_timestamps = get_speech_timestamps
        self._model = load_silero_vad()

    def predict_waveform(self, waveform: torch.Tensor) -> list[Interval]:
        timestamps = self._get_speech_timestamps(
            waveform,
            self._model,
            sampling_rate=SAMPLE_RATE,
            threshold=self.config.threshold,
            min_speech_duration_ms=self.config.min_speech_duration_ms,
            min_silence_duration_ms=self.config.min_silence_duration_ms,
            speech_pad_ms=self.config.speech_pad_ms,
        )
        return [
            (
                float(timestamp["start"]) / SAMPLE_RATE,
                float(timestamp["end"]) / SAMPLE_RATE,
            )
            for timestamp in timestamps
            if timestamp["end"] > timestamp["start"]
        ]


class SortformerVADDetector:
    def __init__(self, config: SortformerVADConfig) -> None:
        if not config.enabled:
            msg = "Sortformer VAD requires a model path or model name"
            raise ValueError(msg)
        try:
            import torch
            from nemo.collections.asr.models import SortformerEncLabelModel
        except ImportError as exc:
            msg = "Sortformer prediction requires a NeMo release with Sortformer support"
            raise RuntimeError(msg) from exc
        self.config = config
        self._torch = torch
        device_name = ("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else config.device
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            msg = f"Sortformer requested {device_name}, but CUDA is unavailable"
            raise RuntimeError(msg)
        self._device = torch.device(device_name)
        model_path = Path(str(config.model)).expanduser()
        if model_path.is_file():
            model = SortformerEncLabelModel.restore_from(
                restore_path=str(model_path),
                map_location="cpu",
            )
        else:
            model = SortformerEncLabelModel.from_pretrained(
                model_name=str(config.model),
                map_location="cpu",
            )
        self._model = model.to(self._device)
        self._model.eval()
        self._model.freeze()
        self._frame_duration = float(self._model.cfg.preprocessor.window_stride) * int(
            self._model.output_subsampling_factor
        )

    @property
    def metadata(self) -> dict[str, object]:
        return {
            **asdict(self.config),
            "frame_duration_sec": self._frame_duration,
            "activity_reduction": "max_across_speakers",
        }

    def predict_waveform(self, waveform: torch.Tensor) -> list[Interval]:
        waveform = waveform.detach().float().reshape(-1)
        input_signal = waveform.unsqueeze(0).to(self._device)
        input_length = self._torch.tensor(
            [waveform.numel()],
            dtype=self._torch.long,
            device=self._device,
        )
        with self._torch.inference_mode():
            predictions = self._model(
                audio_signal=input_signal,
                audio_signal_length=input_length,
            )
            probabilities = predictions[0].amax(dim=-1).float().cpu().tolist()
        return frame_probabilities_to_intervals(
            probabilities,
            self._frame_duration,
            waveform.numel() / SAMPLE_RATE,
            self.config,
        )


class SileroSortformerVADDetector:
    """Run Silero and Sortformer, then optionally remove sound events."""

    def __init__(
        self,
        silero_config: VADConfig,
        sortformer_config: SortformerVADConfig,
        sound_config: SoundExpertConfig,
    ) -> None:
        self.silero_config = silero_config
        self.sortformer_config = sortformer_config
        self.sound_config = sound_config
        self._silero = SileroVADDetector(silero_config)
        self._sortformer = SortformerVADDetector(sortformer_config)
        self._sound = SoundExpertDetector(sound_config) if sound_config.enabled else None

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "silero": asdict(self.silero_config),
            "secondary_vad": "sortformer",
            "sortformer": self._sortformer.metadata,
            "sound_expert": self._sound.metadata if self._sound is not None else None,
        }

    def __call__(self, audio_path: Path) -> VADPredictions:
        waveform = _load_audio_16k(audio_path)
        silero_raw = tuple(self._silero.predict_waveform(waveform))
        secondary_raw = tuple(self._sortformer.predict_waveform(waveform))
        if self._sound is None:
            return VADPredictions(
                silero=silero_raw,
                secondary=secondary_raw,
                secondary_name="sortformer",
            )
        sound = self._sound.predict_waveform(waveform)
        return VADPredictions(
            silero=tuple(
                subtract_intervals(
                    silero_raw,
                    sound.intervals,
                    minimum_duration=self.sound_config.min_residual_speech_sec,
                )
            ),
            secondary=tuple(
                subtract_intervals(
                    secondary_raw,
                    sound.intervals,
                    minimum_duration=self.sound_config.min_residual_speech_sec,
                )
            ),
            secondary_name="sortformer",
            sound_events=sound.intervals,
            sound_events_by_type=sound.by_event or {},
            silero_raw=silero_raw,
            secondary_raw=secondary_raw,
        )


class _IntervalIndex:
    def __init__(self, intervals: list[Interval]) -> None:
        self.intervals = merge_intervals(intervals)
        self.starts = [start for start, _ in self.intervals]

    def contains(self, timestamp: float) -> bool:
        index = bisect_right(self.starts, timestamp) - 1
        return index >= 0 and timestamp < self.intervals[index][1]

    def containing(self, timestamp: float) -> tuple[int, Interval] | None:
        index = bisect_right(self.starts, timestamp) - 1
        if index < 0 or timestamp >= self.intervals[index][1]:
            return None
        return index, self.intervals[index]


def _local_intervals(
    intervals: tuple[Interval, ...],
    offset: float,
    duration: float,
) -> list[Interval]:
    segment_end = offset + duration
    return merge_intervals(
        [
            (
                max(offset, float(start)) - offset,
                min(segment_end, float(end)) - offset,
            )
            for start, end in intervals
            if min(segment_end, float(end)) > max(offset, float(start))
        ]
    )


def atomic_vote_consensus(
    words: list[Word],
    silero_intervals: list[Interval],
    sortformer_intervals: list[Interval],
    duration: float,
) -> tuple[list[Interval], int]:
    """Vote on maximal atomic regions with MFA words as hard speech anchors."""
    mfa_intervals = merge_intervals([(float(start), float(end)) for start, end, _ in words])
    silero_intervals = merge_intervals(silero_intervals)
    sortformer_intervals = merge_intervals(sortformer_intervals)
    boundaries = {0.0, duration}
    for intervals in (mfa_intervals, silero_intervals, sortformer_intervals):
        for start, end in intervals:
            boundaries.add(max(0.0, min(duration, start)))
            boundaries.add(max(0.0, min(duration, end)))
    mfa = _IntervalIndex(mfa_intervals)
    silero = _IntervalIndex(silero_intervals)
    sortformer = _IntervalIndex(sortformer_intervals)
    consensus: list[Interval] = []
    atomic_regions = 0
    for start, end in pairwise(sorted(boundaries)):
        if end <= start:
            continue
        atomic_regions += 1
        midpoint = (start + end) / 2
        output_speech = mfa.contains(midpoint) or (silero.contains(midpoint) and sortformer.contains(midpoint))
        if output_speech:
            consensus.append((start, end))
    return merge_intervals(consensus), atomic_regions


def _validate_projection(original: list[Word], projected: list[Word]) -> None:
    if len(original) != len(projected):
        msg = f"word count changed: {len(original)} -> {len(projected)}"
        raise RuntimeError(msg)
    for source, destination in zip(original, projected, strict=True):
        if str(source[2]) != str(destination[2]):
            msg = f"word label changed: {source!r} -> {destination!r}"
            raise RuntimeError(msg)
        if float(destination[0]) > float(source[0]) + TIME_EPSILON:
            msg = f"word start contracted: {source!r} -> {destination!r}"
            raise RuntimeError(msg)
        if float(destination[1]) < float(source[1]) - TIME_EPSILON:
            msg = f"word end contracted: {source!r} -> {destination!r}"
            raise RuntimeError(msg)
    for left, right in pairwise(projected):
        if float(left[1]) > float(right[0]) + TIME_EPSILON:
            msg = f"projected words overlap: {left!r}, {right!r}"
            raise RuntimeError(msg)


def project_consensus_to_words(
    words: list[Word],
    consensus: list[Interval],
    offset: float,
) -> tuple[list[Word], list[dict[str, object]], set[int]]:
    """Widen words to anchored consensus components without contraction."""
    index = _IntervalIndex(consensus)
    projected = [word.copy() for word in words]
    changes: list[dict[str, object]] = []
    anchored_components: set[int] = set()
    previous_end = 0.0
    for word_index, word in enumerate(words):
        original_start = float(word[0])
        original_end = float(word[1])
        match = index.containing((original_start + original_end) / 2)
        if match is None:
            msg = f"MFA word is outside protected consensus: {word!r}"
            raise RuntimeError(msg)
        component_index, (component_start, component_end) = match
        anchored_components.add(component_index)
        next_start = float(words[word_index + 1][0]) if word_index + 1 < len(words) else component_end
        refined_start = max(previous_end, min(original_start, component_start))
        refined_end = min(next_start, max(original_end, component_end))
        projected[word_index][0] = refined_start
        projected[word_index][1] = refined_end
        previous_end = refined_end
        if abs(refined_start - original_start) > TIME_EPSILON or abs(refined_end - original_end) > TIME_EPSILON:
            changes.append(
                {
                    "word_index": word_index,
                    "word": str(word[2]),
                    "original_start": offset + original_start,
                    "original_end": offset + original_end,
                    "refined_start": offset + refined_start,
                    "refined_end": offset + refined_end,
                    "consensus_start": offset + component_start,
                    "consensus_end": offset + component_end,
                }
            )
    _validate_projection(words, projected)
    return projected, changes, anchored_components


def _marked_words(path: Path) -> list[list[float | str]]:
    textgrid = TextGrid.fromFile(str(path))
    return [
        [float(interval.minTime), float(interval.maxTime), interval.mark]
        for interval in textgrid.getFirst("words").intervals
        if interval.mark.strip().casefold() not in SILENCE_MARKS
    ]


def _textgrid_source(row: dict[str, Any]) -> tuple[Path, list[list[float | str]]]:
    primary = Path(str(row["mfa_textgrid"]))
    words = _marked_words(primary)
    if words:
        return primary, words
    fallback_value = row.get("fallback_textgrid")
    if fallback_value:
        fallback = Path(str(fallback_value))
        words = _marked_words(fallback)
        if words:
            return fallback, words
    return primary, []


def _safe_segment_name(row: dict[str, Any], source_path: Path) -> str:
    value = str(row.get("segment_id") or source_path.stem)
    safe = "".join(character if character.isalnum() or character in {"-", "_", "."} else "_" for character in value)
    return safe or "segment"


def refine_textgrid_boundaries(
    row: dict[str, Any],
    predictions: VADPredictions,
    *,
    output_path: Path | None = None,
) -> tuple[Path, list[dict[str, object]], dict[str, object]]:
    """Apply one constrained equal-weight atomic vote to TextGrid words."""
    source_path, words = _textgrid_source(row)
    output_path = output_path or source_path.with_name(f"{source_path.stem}_vad_refined.TextGrid")
    duration = float(row["duration"])
    offset = float(row["offset"])
    silero = _local_intervals(predictions.silero, offset, duration)
    sortformer = _local_intervals(predictions.secondary, offset, duration)
    consensus, atomic_regions = atomic_vote_consensus(
        words,
        silero,
        sortformer,
        duration,
    )
    projected, word_changes, anchored_components = project_consensus_to_words(
        words,
        consensus,
        offset,
    )

    textgrid = TextGrid(minTime=0.0, maxTime=duration)
    tier = IntervalTier(name="words", minTime=0.0, maxTime=duration)
    for start, end, mark in projected:
        if float(end) > float(start):
            tier.add(float(start), float(end), str(mark))
    textgrid.append(tier)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    textgrid.write(str(output_path))
    vote_audit = {
        "algorithm": "constrained_equal_weight_atomic_vote",
        "weights": {"mfa": 1.0, "silero": 1.0, "sortformer": 1.0},
        "hard_constraint": "preserve_mfa_words",
        "atomic_regions": atomic_regions,
        "consensus_components": len(consensus),
        "anchored_components": len(anchored_components),
        "discarded_unanchored_components": len(consensus) - len(anchored_components),
    }
    return output_path, word_changes, vote_audit


def _rttm_line(recording_id: str, speaker_id: str, start: float, end: float) -> str:
    return f"SPEAKER {recording_id} 1 {start:.6f} {end - start:.6f} <NA> <NA> {speaker_id} <NA> <NA>"


def _write_rttm(path: Path, recording_id: str, intervals: dict[str, tuple[Interval, ...]]) -> None:
    lines = sorted(
        (start, _rttm_line(recording_id, speaker_id, start, end))
        for speaker_id, speaker_intervals in intervals.items()
        for start, end in speaker_intervals
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        "\n".join(line for _, line in lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_rttm(path: Path) -> tuple[str, dict[str, list[Interval]]]:
    recording_id = ""
    by_speaker: dict[str, list[Interval]] = defaultdict(list)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < RTTM_MIN_FIELDS or fields[0] != "SPEAKER":
            msg = f"{path}:{line_number}: malformed RTTM line"
            raise ValueError(msg)
        start = float(fields[3])
        duration = float(fields[4])
        if duration <= 0:
            continue
        recording_id = recording_id or fields[1]
        by_speaker[fields[7]].append((start, start + duration))
    if not recording_id:
        msg = f"RTTM contains no speech intervals: {path}"
        raise ValueError(msg)
    return recording_id, dict(by_speaker)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _validate_rttm_token(value: object, field_name: str, path: Path, line_number: int) -> str:
    token = str(value)
    if not token or any(character.isspace() for character in token):
        msg = f"{path}:{line_number}: {field_name} must be a non-empty RTTM token without whitespace"
        raise ValueError(msg)
    return token


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        msg = f"manifest is empty: {path}"
        raise ValueError(msg)
    session_ids = {str(row.get("session_id", "")) for row in rows}
    if len(session_ids) != 1 or not next(iter(session_ids)):
        msg = f"manifest must contain exactly one non-empty session_id: {path}"
        raise ValueError(msg)
    session_id = next(iter(session_ids))
    if session_id in {".", ".."} or "/" in session_id or "\\" in session_id:
        msg = f"session_id cannot contain path separators: {session_id!r}"
        raise ValueError(msg)
    for line_number, row in enumerate(rows, start=1):
        missing = sorted(REQUIRED_MANIFEST_FIELDS - row.keys())
        if missing:
            msg = f"{path}:{line_number}: missing required fields: {missing}"
            raise ValueError(msg)
        _validate_rttm_token(row["session_id"], "session_id", path, line_number)
        speaker_id = _validate_rttm_token(row["speaker_id"], "speaker_id", path, line_number)
        if "/" in speaker_id or "\\" in speaker_id:
            msg = f"{path}:{line_number}: speaker_id cannot contain path separators"
            raise ValueError(msg)
        if float(row["duration"]) <= 0 or float(row["offset"]) < 0:
            msg = f"{path}:{line_number}: duration must be positive and offset non-negative"
            raise ValueError(msg)
    return rows


def _speaker_audio_paths(rows: list[dict[str, Any]]) -> dict[str, Path]:
    source_paths: dict[str, Path] = {}
    for row in rows:
        speaker_id = str(row["speaker_id"])
        path = Path(str(row["audio_filepath"])).expanduser()
        if speaker_id in source_paths and source_paths[speaker_id] != path:
            msg = f"speaker {speaker_id!r} maps to multiple audio files"
            raise ValueError(msg)
        source_paths[speaker_id] = path
    missing = {speaker_id: path for speaker_id, path in source_paths.items() if not path.is_file()}
    if missing:
        msg = f"missing audio files: {missing}"
        raise FileNotFoundError(msg)
    return source_paths


def _write_debug_rttms(
    output_dir: Path,
    session_id: str,
    predictions: dict[str, VADPredictions],
) -> dict[str, str]:
    labels: dict[str, dict[str, tuple[Interval, ...]]] = {
        "silero": {speaker: value.silero for speaker, value in predictions.items()},
        next(iter(predictions.values())).secondary_name: {
            speaker: value.secondary for speaker, value in predictions.items()
        },
    }
    if any(value.sound_events for value in predictions.values()):
        labels["sound_events"] = {speaker: value.sound_events for speaker, value in predictions.items()}
        labels["silero_raw"] = {speaker: value.silero_raw for speaker, value in predictions.items()}
        secondary_name = next(iter(predictions.values())).secondary_name
        labels[f"{secondary_name}_raw"] = {speaker: value.secondary_raw for speaker, value in predictions.items()}
    outputs: dict[str, str] = {}
    for label, intervals in labels.items():
        path = output_dir / f"{session_id}_{label}.rttm"
        _write_rttm(path, session_id, intervals)
        outputs[label] = str(path.resolve())
    return outputs


def process_manifest(
    manifest_path: Path,
    output_root: Path,
    detector: VADDetector,
) -> dict[str, object]:
    """Refine one session manifest and write non-destructive outputs."""
    rows = load_manifest(manifest_path)
    session_id = str(rows[0]["session_id"])
    session_dir = output_root / session_id
    segment_grid_dir = session_dir / "textgrids" / "segments"
    source_paths = _speaker_audio_paths(rows)
    predictions = {speaker_id: detector(audio_path) for speaker_id, audio_path in sorted(source_paths.items())}

    speaker_audit: dict[str, dict[str, object]] = {
        speaker_id: {
            "audio_filepath": str(source_paths[speaker_id].resolve()),
            "silero_intervals": value.silero,
            "silero_raw_intervals": value.silero_raw or value.silero,
            f"{value.secondary_name}_intervals": value.secondary,
            f"{value.secondary_name}_raw_intervals": value.secondary_raw or value.secondary,
            "sound_event_intervals": value.sound_events,
            "sound_events_by_type": value.sound_events_by_type,
            "segments": [],
        }
        for speaker_id, value in predictions.items()
    }
    output_rows: list[dict[str, Any]] = []
    for index, source_row in enumerate(rows):
        row = dict(source_row)
        speaker_id = str(row["speaker_id"])
        source_grid = Path(str(row["mfa_textgrid"]))
        segment_name = _safe_segment_name(row, source_grid)
        output_path = segment_grid_dir / f"{index:06d}_{segment_name}_vad_refined.TextGrid"
        refined_path, word_changes, vote_audit = refine_textgrid_boundaries(
            row,
            predictions[speaker_id],
            output_path=output_path,
        )
        row["vad_refined_textgrid"] = str(refined_path.resolve())
        row["rttm_textgrid"] = str(refined_path.resolve())
        row["speech_intervals_source"] = "constrained_equal_weight_atomic_vote"
        output_rows.append(row)
        speaker_audit[speaker_id]["segments"].append(
            {
                "segment_id": str(row.get("segment_id") or index),
                "source_textgrid": str(source_grid.resolve()),
                "refined_textgrid": str(refined_path.resolve()),
                "word_changes": word_changes,
                "atomic_vote": vote_audit,
            }
        )

    aligned = speech_intervals_from_rows(
        output_rows,
        maximum_pause=0.0,
    )
    for speaker_id in source_paths:
        aligned.setdefault(speaker_id, ())
    final_intervals = aligned

    rttm_dir = session_dir / "rttm"
    session_rttm = rttm_dir / f"{session_id}_vad_refined.rttm"
    _write_rttm(session_rttm, session_id, final_intervals)
    speaker_rttms: dict[str, str] = {}
    for speaker_id, intervals in sorted(final_intervals.items()):
        recording_id = "".join(
            character if character.isalnum() or character in {"-", "_", "."} else "_"
            for character in source_paths[speaker_id].stem
        ).strip("._")
        recording_id = recording_id or "recording"
        speaker_path = rttm_dir / f"{recording_id}_vad_refined.rttm"
        _write_rttm(speaker_path, recording_id, {speaker_id: intervals})
        speaker_rttms[speaker_id] = str(speaker_path.resolve())
        speaker_audit[speaker_id]["aligned_intervals"] = aligned[speaker_id]
        speaker_audit[speaker_id]["final_intervals"] = intervals

    write_refined_textgrids(output_rows, session_dir / "textgrids", session_id)
    output_manifest = session_dir / "manifest_vad_refined.jsonl"
    _write_jsonl(output_manifest, output_rows)
    debug_rttms = _write_debug_rttms(session_dir / "vad_debug", session_id, predictions)
    summary: dict[str, object] = {
        "session_id": session_id,
        "source_manifest": str(manifest_path.resolve()),
        "output_manifest": str(output_manifest.resolve()),
        "session_rttm": str(session_rttm.resolve()),
        "speaker_rttms": speaker_rttms,
        "debug_rttms": debug_rttms,
        "detectors": detector.metadata,
        "refinement_algorithm": {
            "name": "constrained_equal_weight_atomic_vote",
            "weights": {"mfa": 1.0, "silero": 1.0, "sortformer": 1.0},
            "hard_constraint": "preserve_mfa_words",
            "postprocessing": "none",
        },
        "speakers": speaker_audit,
    }
    _write_json(session_dir / "audit.json", summary)
    return summary


def discover_manifests(root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in root.glob(pattern) if path.is_file())


def _add_detector_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--silero-threshold", type=float, default=0.5)
    parser.add_argument("--silero-min-speech-duration-ms", type=int, default=100)
    parser.add_argument("--silero-min-silence-duration-ms", type=int, default=100)
    parser.add_argument("--silero-speech-pad-ms", type=int, default=0)
    parser.add_argument("--sortformer-model", required=True)
    parser.add_argument("--sortformer-device", default="auto")
    parser.add_argument("--sortformer-onset", type=float, default=0.5)
    parser.add_argument("--sortformer-offset", type=float, default=0.5)
    parser.add_argument("--sortformer-min-speech-duration-ms", type=int, default=100)
    parser.add_argument("--sortformer-min-silence-duration-ms", type=int, default=100)
    parser.add_argument("--sortformer-speech-pad-ms", type=int, default=0)


def _add_sound_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sound-expert-model", default=None)
    parser.add_argument("--sound-expert-backend", choices=("auto", "ctc", "pee"), default="auto")
    parser.add_argument("--sound-expert-device", default="auto")
    parser.add_argument(
        "--sound-expert-compute-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--sound-expert-threshold", type=float, default=0.5)
    parser.add_argument("--sound-expert-frame-pad-sec", type=float, default=0.2)
    parser.add_argument("--sound-expert-merge-gap-sec", type=float, default=0.1)
    parser.add_argument("--sound-expert-min-event-duration-sec", type=float, default=0.0)
    parser.add_argument("--sound-expert-min-residual-speech-sec", type=float, default=0.05)
    parser.add_argument("--sound-expert-chunk-duration-sec", type=float, default=15.0)
    parser.add_argument("--sound-expert-chunk-overlap-sec", type=float, default=1.0)
    parser.add_argument(
        "--sound-expert-events",
        nargs="+",
        default=list(DEFAULT_SOUND_EVENT_TOKENS),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, action="append")
    source.add_argument("--manifest-root", type=Path)
    parser.add_argument("--manifest-pattern", default="**/*.jsonl")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    _add_detector_arguments(parser)
    _add_sound_arguments(parser)
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, shard-count)")
    thresholds = (
        args.silero_threshold,
        args.sortformer_onset,
        args.sortformer_offset,
        args.sound_expert_threshold,
    )
    if not all(0 <= value <= 1 for value in thresholds):
        parser.error("thresholds must be in [0, 1]")
    durations = (
        args.silero_min_speech_duration_ms,
        args.silero_min_silence_duration_ms,
        args.silero_speech_pad_ms,
        args.sortformer_min_speech_duration_ms,
        args.sortformer_min_silence_duration_ms,
        args.sortformer_speech_pad_ms,
        args.sound_expert_frame_pad_sec,
        args.sound_expert_merge_gap_sec,
        args.sound_expert_min_event_duration_sec,
        args.sound_expert_min_residual_speech_sec,
        args.sound_expert_chunk_overlap_sec,
    )
    if min(durations) < 0:
        parser.error("duration parameters must be non-negative")
    if not 0 <= args.sound_expert_chunk_overlap_sec < args.sound_expert_chunk_duration_sec:
        parser.error("Sound Expert chunk overlap must be shorter than chunk duration")
    return args


def main() -> int:
    args = parse_args()
    manifests = (
        args.manifest if args.manifest is not None else discover_manifests(args.manifest_root, args.manifest_pattern)
    )
    manifests = [path for index, path in enumerate(sorted(manifests)) if index % args.shard_count == args.shard_index]
    if not manifests:
        print("No manifests found")
        return 0

    detector = SileroSortformerVADDetector(
        VADConfig(
            threshold=args.silero_threshold,
            min_speech_duration_ms=args.silero_min_speech_duration_ms,
            min_silence_duration_ms=args.silero_min_silence_duration_ms,
            speech_pad_ms=args.silero_speech_pad_ms,
        ),
        SortformerVADConfig(
            model=args.sortformer_model,
            device=args.sortformer_device,
            onset=args.sortformer_onset,
            offset=args.sortformer_offset,
            min_speech_duration_ms=args.sortformer_min_speech_duration_ms,
            min_silence_duration_ms=args.sortformer_min_silence_duration_ms,
            speech_pad_ms=args.sortformer_speech_pad_ms,
        ),
        SoundExpertConfig(
            model=args.sound_expert_model,
            backend=args.sound_expert_backend,
            device=args.sound_expert_device,
            compute_dtype=args.sound_expert_compute_dtype,
            threshold=args.sound_expert_threshold,
            frame_pad_sec=args.sound_expert_frame_pad_sec,
            merge_gap_sec=args.sound_expert_merge_gap_sec,
            min_event_duration_sec=args.sound_expert_min_event_duration_sec,
            min_residual_speech_sec=args.sound_expert_min_residual_speech_sec,
            chunk_duration_sec=args.sound_expert_chunk_duration_sec,
            chunk_overlap_sec=args.sound_expert_chunk_overlap_sec,
            event_tokens=tuple(args.sound_expert_events),
        ),
    )
    failed = 0
    for completed, manifest_path in enumerate(manifests, start=1):
        try:
            rows = load_manifest(manifest_path)
            session_id = str(rows[0]["session_id"])
            audit_path = args.output_root / session_id / "audit.json"
            if audit_path.is_file() and not args.overwrite:
                print(f"[{completed}/{len(manifests)}] {session_id}: already complete")
                continue
            summary = process_manifest(manifest_path, args.output_root, detector)
            print(f"[{completed}/{len(manifests)}] {session_id}: {summary['session_rttm']}")
        except Exception as exc:  # noqa: BLE001
            print(f"[{completed}/{len(manifests)}] {manifest_path}: ERROR: {exc}")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
