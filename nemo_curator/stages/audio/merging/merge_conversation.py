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

"""Merge complete conversations with silence stripping and manifest overlaps."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask
from nemo_curator.utils.operation_utils import make_temporary_dir

_MIN_FIELDS = 5
_PEAK_LIMIT = 0.99
_MIN_RANDOMIZE_PAUSE = 0.5


@dataclass
class _SampleMapping:
    source_start: int
    source_end: int
    output_start: int


@dataclass
class _PreparedTurn:
    entry: dict[str, Any]
    audio: np.ndarray
    sample_rate: int
    mappings: list[_SampleMapping]
    speech: list[tuple[int, int]]
    fallback: bool
    start: int = 0


class MergeConversationSDPStage(ProcessingStage[AudioTask, AudioTask]):
    """Merge nested conversation turns into audio and matching annotations.

    Each input ``AudioTask`` must contain ``conversation_id`` and a ``turns``
    list of dictionaries. Each turn requires ``audio_filepath``, ``speaker``
    and a unique integer ``turn_index``. Optional fields are ``rttm_filepath``,
    ``ctm_filepath``, ``overlap``, ``text`` (or ``utterance``),
    ``reference_audio``, ``reference_voice`` and ``mfa_skipped``. A turn's
    optional ``conversation_id`` must match its parent. Produce this shape
    before scheduling, for example with one complete conversation per JSONL
    row read by ``ManifestReader``. Independent turn tasks are rejected;
    scheduler batches never define a conversation boundary.

    RTTM speech is clipped to the source waveform and overlapping intervals
    are merged. Short intra-turn pauses are retained; longer pauses are cut.
    Missing or unusable RTTM retains the full audio with an identity mapping
    and sets ``mfa_fallback``. CTM words are clipped to retained source audio;
    words entirely in removed audio are omitted. All timestamps follow the
    actual sample placement. Positive ``overlap`` overlaps the preceding
    conversation end (capped by the new turn length and half the elapsed
    conversation); negative values insert a capped pause. Same-speaker
    overlaps add their samples. A common gain limits speaker channels to
    0.99 before PCM encoding; the mono mix is limited separately.

    Outputs under ``output_conversations_dir/conversation_id`` are
    ``mixed.wav``, ``multichannel.wav``, ``all.rttm``, optional ``all.ctm``
    and ``segments.seglst.json``. ``speaker_artifacts`` maps original labels
    to channel indices and WAV/RTTM/CTM paths under ``speakers/`` with generated
    filenames. ``speaker_references`` preserves reference metadata by label.
    ``duration`` is the written sample count divided by sample rate;
    ``offset`` is zero. A missing CTM is reported as an empty path.

    Unsafe IDs and malformed turn structures raise ``ValueError`` before
    writes. Unreadable/empty audio, inconsistent sample rates, existing output
    destinations and write failures filter the whole conversation (``[]``),
    leaving the input task unchanged. Artifacts are built in a temporary
    directory and published together. Use a fresh output root for reruns.

    Args:
        output_conversations_dir: Root for complete conversation directories.
        max_pause_duration: Maximum inserted inter-turn silence, in seconds.
        max_intra_turn_pause: Maximum source pause retained between speech segments.
        randomize_pauses: Randomize capped inter-turn pauses longer than 0.5 seconds.
        seglst_offset: Nonnegative padding around turn bounds in the segment list.
    """

    name = "MergeConversationSDPStage"

    def __init__(
        self,
        output_conversations_dir: str,
        max_pause_duration: float = 2.0,
        max_intra_turn_pause: float = 1.0,
        randomize_pauses: bool = False,
        seglst_offset: float = 0.1,
    ):
        super().__init__()
        for key, value in (
            ("max_pause_duration", max_pause_duration),
            ("max_intra_turn_pause", max_intra_turn_pause),
            ("seglst_offset", seglst_offset),
        ):
            if not math.isfinite(value) or value < 0:
                msg = f"{key} must be finite and nonnegative"
                raise ValueError(msg)
        self.output_conversations_dir = Path(output_conversations_dir).expanduser().resolve()
        self.max_pause_duration = max_pause_duration
        self.max_intra_turn_pause = max_intra_turn_pause
        self.randomize_pauses = randomize_pauses
        self.seglst_offset = seglst_offset
        self._rng = random.Random()  # noqa: S311

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], ["conversation_id", "turns"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [
            "conversation_id",
            "audio_filepath",
            "rttm_filepath",
            "ctm_filepath",
            "seglst_filepath",
            "duration",
            "num_speakers",
            "offset",
            "mfa_fallback",
            "speaker_references",
            "speaker_artifacts",
        ]

    def process(self, task: AudioTask) -> AudioTask | list[AudioTask]:
        conversation_id, entries = self._validate_conversation(task.data)
        if not entries:
            return []
        destination = self.output_conversations_dir / conversation_id
        try:
            self._require_fresh_destination(destination)
            turns = [self._prepare_turn(entry) for entry in entries]
            total_samples = self._place_turns(turns)
            speakers = list(dict.fromkeys(turn.entry["speaker"] for turn in turns))
            # Original labels never become path components, including reserved aggregate names.
            stems = {speaker: f"speaker_{i:04d}" for i, speaker in enumerate(speakers)}
            self.output_conversations_dir.mkdir(parents=True, exist_ok=True)
            with make_temporary_dir(target_dir=self.output_conversations_dir, prefix=".merge-") as temporary:
                staging = temporary / "conversation"
                staging.mkdir()
                (staging / "speakers").mkdir()
                self._write_audio(staging, turns, speakers, stems, total_samples)
                with_ctm = self._write_annotations(staging, conversation_id, turns, stems)
                self._generate_seglst(staging / "segments.seglst.json", conversation_id, turns, total_samples)
                result = self._result(
                    destination, conversation_id, turns, stems, with_ctm, total_samples=total_samples
                )
                self._require_fresh_destination(destination)
                staging.rename(destination)
        except (OSError, RuntimeError, ValueError):
            logger.exception(f"Failed to merge conversation {conversation_id}")
            return []
        task.data.clear()
        task.data.update(result)
        return task

    @staticmethod
    def _require_fresh_destination(destination: Path) -> None:
        if destination.exists() or destination.is_symlink():
            msg = f"Conversation destination already exists: {destination}"
            raise FileExistsError(msg)

    @staticmethod
    def _validate_conversation(data: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        entries = data.get("turns")
        if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
            msg = "Expected a turns list containing all turns of one conversation"
            raise ValueError(msg)
        conversation_id = data.get("conversation_id")
        if (
            not isinstance(conversation_id, str)
            or not conversation_id
            or conversation_id in {".", ".."}
            or any(char in conversation_id for char in "/\\\x00")
            or any(char.isspace() for char in conversation_id)
        ):
            msg = "conversation_id must be a nonempty safe path component without whitespace"
            raise ValueError(msg)
        indices = set()
        for entry in entries:
            if entry.get("conversation_id", conversation_id) != conversation_id:
                msg = "Turn conversation_id does not match its parent"
                raise ValueError(msg)
            speaker = entry.get("speaker")
            if (
                not isinstance(speaker, str)
                or not speaker
                or any(char.isspace() or char == "\x00" for char in speaker)
            ):
                msg = "Each turn requires a nonempty speaker label without whitespace"
                raise ValueError(msg)
            index = entry.get("turn_index")
            if isinstance(index, bool) or not isinstance(index, int) or index in indices:
                msg = "Each turn requires a unique integer turn_index"
                raise ValueError(msg)
            indices.add(index)
            MergeConversationSDPStage._validate_paths(entry)
            overlap = entry.get("overlap", 0)
            if not isinstance(overlap, (int, float)) or not math.isfinite(overlap):
                msg = "Turn overlap must be finite"
                raise ValueError(msg)
        return conversation_id, sorted(entries, key=lambda entry: entry["turn_index"])

    @staticmethod
    def _validate_paths(entry: dict[str, Any]) -> None:
        if not isinstance(entry.get("audio_filepath"), str) or not entry["audio_filepath"]:
            msg = "Each turn requires a nonempty audio_filepath string"
            raise ValueError(msg)
        for key in ("rttm_filepath", "ctm_filepath"):
            if key in entry and not isinstance(entry[key], str):
                msg = f"{key} must be a string"
                raise ValueError(msg)

    @staticmethod
    def _read_timestamps(path: str, *, ctm: bool = False) -> list[tuple[float, float, str]]:
        rows = []
        if not path:
            return rows
        try:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    fields = line.split()
                    if len(fields) < _MIN_FIELDS or (not ctm and fields[0] != "SPEAKER"):
                        continue
                    try:
                        start = float(fields[2] if ctm else fields[3])
                        duration = float(fields[3] if ctm else fields[4])
                    except ValueError:
                        continue
                    if (
                        math.isfinite(start)
                        and math.isfinite(duration)
                        and duration > 0
                        and math.isfinite(start + duration)
                    ):
                        rows.append((start, duration, fields[4] if ctm else ""))
        except (OSError, UnicodeError):
            logger.warning(f"Could not read annotation file: {path}")
        return sorted(rows, key=lambda row: row[0])

    def _prepare_turn(self, entry: dict[str, Any]) -> _PreparedTurn:
        audio, sr = sf.read(entry.get("audio_filepath", ""), dtype="float64", always_2d=True)
        if not len(audio) or not np.isfinite(audio).all():
            msg = "Turn audio must contain finite samples"
            raise ValueError(msg)
        audio = audio.mean(axis=1)
        speech: list[tuple[int, int]] = []
        for start, duration, _ in self._read_timestamps(entry.get("rttm_filepath", "")):
            left = round(max(0, min(len(audio) / sr, start)) * sr)
            right = round(max(0, min(len(audio) / sr, start + duration)) * sr)
            if left >= right:
                continue
            if speech and left <= speech[-1][1]:
                speech[-1] = (speech[-1][0], max(right, speech[-1][1]))
            else:
                speech.append((left, right))
        fallback = not speech
        if fallback:
            speech = [(0, len(audio))]
        retained: list[tuple[int, int]] = []
        for left, right in speech:
            if retained and left - retained[-1][1] <= self.max_intra_turn_pause * sr:
                retained[-1] = (retained[-1][0], right)
            else:
                retained.append((left, right))
        mappings = []
        position = 0
        for left, right in retained:
            mappings.append(_SampleMapping(left, right, position))
            position += right - left
        adjusted_speech = [
            (mapping.output_start + left - mapping.source_start, mapping.output_start + right - mapping.source_start)
            for left, right in speech
            for mapping in mappings
            if mapping.source_start <= left and right <= mapping.source_end
        ]
        return _PreparedTurn(
            entry,
            np.concatenate([audio[left:right] for left, right in retained]),
            sr,
            mappings,
            adjusted_speech,
            fallback,
        )

    def _place_turns(self, turns: list[_PreparedTurn]) -> int:
        if len({turn.sample_rate for turn in turns}) != 1:
            msg = "Conversation sample rates differ; resample turns before merging"
            raise ValueError(msg)
        position = 0
        sr = turns[0].sample_rate
        for i, turn in enumerate(turns):
            overlap = turn.entry.get("overlap", 0) if i else 0
            if overlap < 0:
                pause = min(-overlap, self.max_pause_duration)
                if self.randomize_pauses and pause > _MIN_RANDOMIZE_PAUSE:
                    pause = self._rng.uniform(0.3, pause)
                turn.start = position + round(pause * sr)
            else:
                backup = min(round(min(overlap, len(turn.audio) / sr) * sr), position // 2)
                turn.start = position - backup
            position = max(position, turn.start + len(turn.audio))
        return position

    @staticmethod
    def _write_audio(
        directory: Path,
        turns: list[_PreparedTurn],
        speakers: list[str],
        stems: dict[str, str],
        total_samples: int,
    ) -> None:
        channels = np.zeros((total_samples, len(speakers)), dtype=np.float64)
        indices = {speaker: i for i, speaker in enumerate(speakers)}
        for turn in turns:
            channels[turn.start : turn.start + len(turn.audio), indices[turn.entry["speaker"]]] += turn.audio
        peak = np.abs(channels).max()
        if peak > _PEAK_LIMIT:
            channels *= _PEAK_LIMIT / peak
        sr = turns[0].sample_rate
        for speaker, index in indices.items():
            sf.write(directory / "speakers" / f"{stems[speaker]}.wav", channels[:, index], sr)
        sf.write(directory / "multichannel.wav", channels, sr)
        mixed = channels.sum(axis=1)
        peak = np.abs(mixed).max()
        if peak > _PEAK_LIMIT:
            mixed *= _PEAK_LIMIT / peak
        sf.write(directory / "mixed.wav", mixed, sr)

    def _mapped_words(self, turn: _PreparedTurn) -> list[tuple[float, float, str]]:
        words = []
        sr = turn.sample_rate
        source_duration = turn.mappings[-1].source_end / sr
        for start, duration, word in self._read_timestamps(turn.entry.get("ctm_filepath", ""), ctm=True):
            left = round(max(0, min(source_duration, start)) * sr)
            right = round(max(0, min(source_duration, start + duration)) * sr)
            intersections = [
                (
                    max(left, mapping.source_start) - mapping.source_start + mapping.output_start,
                    min(right, mapping.source_end) - mapping.source_start + mapping.output_start,
                )
                for mapping in turn.mappings
                if max(left, mapping.source_start) < min(right, mapping.source_end)
            ]
            if intersections:
                output_start, output_end = intersections[0][0], intersections[-1][1]
                words.append(((turn.start + output_start) / sr, (output_end - output_start) / sr, word))
        return words

    def _write_annotations(
        self,
        directory: Path,
        conversation_id: str,
        turns: list[_PreparedTurn],
        stems: dict[str, str],
    ) -> set[str]:
        rttm: dict[str, list[tuple[float, str]]] = {speaker: [] for speaker in stems}
        ctm: dict[str, list[tuple[float, str]]] = {speaker: [] for speaker in stems}
        for turn in turns:
            speaker = turn.entry["speaker"]
            sr = turn.sample_rate
            for left, right in turn.speech:
                start, duration = (turn.start + left) / sr, (right - left) / sr
                rttm[speaker].append(
                    (start, f"SPEAKER {conversation_id} 1 {start:.6f} {duration:.6f} <NA> <NA> {speaker} <NA> <NA>")
                )
            for start, duration, word in self._mapped_words(turn):
                ctm[speaker].append((start, f"{conversation_id} 1 {start:.6f} {duration:.6f} {word}"))
        with_ctm = set()
        for extension, speaker_rows in (("rttm", rttm), ("ctm", ctm)):
            combined = []
            for speaker, rows in speaker_rows.items():
                if rows:
                    self._write_lines(directory / "speakers" / f"{stems[speaker]}.{extension}", rows)
                    combined.extend(rows)
                    if extension == "ctm":
                        with_ctm.add(speaker)
            if combined:
                self._write_lines(directory / f"all.{extension}", combined)
        return with_ctm

    @staticmethod
    def _write_lines(path: Path, rows: list[tuple[float, str]]) -> None:
        path.write_text("".join(line + "\n" for _, line in sorted(rows, key=lambda row: row[0])), encoding="utf-8")

    def _generate_seglst(
        self,
        path: Path,
        conversation_id: str,
        turns: list[_PreparedTurn],
        total_samples: int,
    ) -> None:
        duration = total_samples / turns[0].sample_rate
        segments = []
        for turn in turns:
            segments.append(
                {
                    "session_id": conversation_id,
                    "speaker": turn.entry["speaker"],
                    "start_time": max(0, (turn.start + turn.speech[0][0]) / turn.sample_rate - self.seglst_offset),
                    "end_time": min(
                        duration, (turn.start + turn.speech[-1][1]) / turn.sample_rate + self.seglst_offset
                    ),
                    "words": turn.entry.get("text", turn.entry.get("utterance", "")),
                }
            )
        path.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _result(  # noqa: PLR0913
        directory: Path,
        conversation_id: str,
        turns: list[_PreparedTurn],
        stems: dict[str, str],
        with_ctm: set[str],
        *,
        total_samples: int,
    ) -> dict[str, Any]:
        references = {}
        for turn in turns:
            references.setdefault(
                turn.entry["speaker"],
                {
                    "reference_audio": turn.entry.get("reference_audio", ""),
                    "reference_voice": turn.entry.get("reference_voice", ""),
                },
            )
        artifacts = {
            speaker: {
                "channel": i,
                "audio_filepath": str(directory / "speakers" / f"{stem}.wav"),
                "rttm_filepath": str(directory / "speakers" / f"{stem}.rttm"),
                "ctm_filepath": str(directory / "speakers" / f"{stem}.ctm") if speaker in with_ctm else "",
            }
            for i, (speaker, stem) in enumerate(stems.items())
        }
        return {
            "conversation_id": conversation_id,
            "audio_filepath": str(directory / "mixed.wav"),
            "rttm_filepath": str(directory / "all.rttm"),
            "ctm_filepath": str(directory / "all.ctm") if with_ctm else "",
            "seglst_filepath": str(directory / "segments.seglst.json"),
            "duration": total_samples / turns[0].sample_rate,
            "num_speakers": len(stems),
            "offset": 0,
            "mfa_fallback": any(turn.fallback or turn.entry.get("mfa_skipped", False) for turn in turns),
            "speaker_references": references,
            "speaker_artifacts": artifacts,
        }
