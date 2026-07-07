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

"""
Merge Conversation Stage -- SDP-style (silence stripping + manifest overlaps).

Produces per-conversation output directories containing:
  - Per-speaker single-channel WAVs (full conversation duration)
  - Multi-channel WAV (channel N = speaker N)
  - Mixed mono WAV (sum of all channels)
  - Per-speaker RTTM and combined RTTM
  - Per-speaker word-level CTM and combined CTM
  - Segment-level seglst JSON derived from RTTM timestamps
"""

from __future__ import annotations

import contextlib
import json
import os
import random as _random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


@dataclass
class _TurnTimeline:
    """Pre-computed timeline for one turn, shared by RTTM and CTM merging."""

    time_offset: float
    rttm_segments: list[tuple[float, float]] = field(default_factory=list)
    adjusted_segments: list[tuple[float, float]] = field(default_factory=list)
    local_offset: float = 0.0


_MIN_RTTM_FIELDS = 5
_MIN_CTM_FIELDS = 5
_MIN_RANDOMIZE_PAUSE = 0.5
_PEAK_NORMALIZE_THRESHOLD = 0.99


class MergeConversationSDPStage(ProcessingStage[AudioTask, AudioTask]):
    """Merge conversation turns using SDP-style silence-stripping and manifest overlaps.

    Each batch of ``AudioTask`` objects is expected to contain all turns of
    **one** conversation (conversation-batched mode). The stage produces a
    per-conversation folder under ``output_conversations_dir`` with
    multi-channel audio, per-speaker RTTMs/CTMs, a segment list, and a
    mixed mono WAV.

    This stage only supports :meth:`process_batch`; calling :meth:`process`
    raises ``NotImplementedError``.

    Args:
        output_conversations_dir: Root directory for per-conversation output folders.
        max_pause_duration: Maximum pause duration (seconds) between turns.
        max_intra_turn_pause: Maximum intra-turn pause to preserve when
            stripping silence from RTTM segments.
        randomize_pauses: Whether to randomize inter-turn pauses.
        seglst_offset: Padding (seconds) around segment boundaries in the
            generated segment list JSON.
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
        self.output_conversations_dir = Path(output_conversations_dir)
        self.max_pause_duration = max_pause_duration
        self.max_intra_turn_pause = max_intra_turn_pause
        self.randomize_pauses = randomize_pauses
        self.seglst_offset = seglst_offset
        self._rng = _random.Random()  # noqa: S311

    def setup(self, worker_metadata: object = None) -> None:  # noqa: ARG002
        self.output_conversations_dir.mkdir(parents=True, exist_ok=True)

    def teardown(self) -> None:
        pass

    # ------------------------------------------------------------------
    # ProcessingStage interface
    # ------------------------------------------------------------------

    def process(self, task: AudioTask) -> AudioTask:
        msg = "MergeConversationSDPStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Merge all turns of one conversation into per-speaker outputs.

        Args:
            tasks: List of AudioTask objects, each representing one
                conversation turn. All tasks must share the same
                ``conversation_id``.

        Returns:
            A single-element list containing one AudioTask with the merged
            conversation metadata, or an empty list on failure.
        """
        if not tasks:
            return []

        entries = [task.data for task in tasks]
        conversation_id = entries[0].get("conversation_id", "unknown")

        for entry in entries:
            if entry.get("conversation_id") != conversation_id:
                logger.warning(
                    f"Mixed conversation IDs in batch! Expected {conversation_id}, "
                    f"got {entry.get('conversation_id')}"
                )

        sorted_turns = sorted(entries, key=lambda x: x.get("turn_index", 0))

        merged_entry = self._merge_conversation(conversation_id, sorted_turns)

        if merged_entry is not None:
            return [
                AudioTask(
                    data=merged_entry,
                    task_id=tasks[0].task_id,
                    dataset_name=tasks[0].dataset_name,
                )
            ]

        logger.error(f"Failed to merge conversation {conversation_id}")
        return []

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_rttm_timestamps(rttm_path: str) -> list[tuple[float, float]]:
        """Return sorted ``(start, duration)`` tuples from an RTTM file."""
        timestamps: list[tuple[float, float]] = []
        if not rttm_path or not os.path.exists(rttm_path):
            return timestamps
        try:
            with open(rttm_path, encoding="utf-8") as f:
                for raw_line in f:
                    stripped = raw_line.strip()
                    if not stripped or not stripped.startswith("SPEAKER"):
                        continue
                    parts = stripped.split()
                    if len(parts) >= _MIN_RTTM_FIELDS:
                        try:
                            timestamps.append((float(parts[3]), float(parts[4])))
                        except (ValueError, IndexError):
                            continue
        except OSError as exc:
            logger.error(f"Error reading RTTM {rttm_path}: {exc}")
        timestamps.sort(key=lambda x: x[0])
        return timestamps

    @staticmethod
    def _parse_ctm_words(ctm_path: str) -> list[tuple[float, float, str]]:
        """Return sorted ``(start, duration, word)`` tuples from a CTM file."""
        words: list[tuple[float, float, str]] = []
        if not ctm_path or not os.path.exists(ctm_path):
            return words
        try:
            with open(ctm_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= _MIN_CTM_FIELDS:
                        try:
                            words.append((float(parts[2]), float(parts[3]), parts[4]))
                        except (ValueError, IndexError):
                            continue
        except OSError as exc:
            logger.error(f"Error reading CTM {ctm_path}: {exc}")
        words.sort(key=lambda x: x[0])
        return words

    @staticmethod
    def _find_nearest_segment(
        w_start: float,
        segments: list[tuple[float, float]],
    ) -> int:
        """Return index of the RTTM segment nearest to *w_start*."""
        best_idx, best_dist = 0, float("inf")
        for idx, (seg_start, seg_dur) in enumerate(segments):
            seg_end = seg_start + seg_dur
            if seg_start <= w_start <= seg_end:
                return idx
            dist = min(abs(w_start - seg_start), abs(w_start - seg_end))
            if dist < best_dist:
                best_idx, best_dist = idx, dist
        return best_idx

    # ------------------------------------------------------------------
    # Speaking-segment extraction (SDP logic)
    # ------------------------------------------------------------------

    def extract_speaking_segments(
        self,
        audio_filepath: str,
        rttm_filepath: str,
    ) -> tuple[np.ndarray | None, int | None]:
        """Extract speaking segments from *audio_filepath* guided by RTTM.

        Pauses between RTTM segments that are <= ``self.max_intra_turn_pause``
        are preserved; longer pauses are removed.

        Returns ``(concatenated_audio, sample_rate)`` or ``(None, None)`` on
        failure.
        """
        timestamps = self._parse_rttm_timestamps(rttm_filepath)
        if not timestamps:
            logger.warning(f"No speaking segments in RTTM: {rttm_filepath}")
            return None, None

        try:
            audio_data, sr = sf.read(audio_filepath)
        except (OSError, RuntimeError) as exc:
            logger.error(f"Error reading audio {audio_filepath}: {exc}")
            return None, None

        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        segments: list[np.ndarray] = []
        max_pause = self.max_intra_turn_pause

        for i, (start_time, duration) in enumerate(timestamps):
            start_idx = max(0, int(start_time * sr))
            end_idx = min(len(audio_data), int((start_time + duration) * sr))

            if start_idx < end_idx:
                segments.append(audio_data[start_idx:end_idx])

                if i < len(timestamps) - 1:
                    next_start = timestamps[i + 1][0]
                    current_end = start_time + duration
                    pause_dur = next_start - current_end

                    if 0 < pause_dur <= max_pause:
                        pause_start = end_idx
                        pause_end = min(len(audio_data), int(next_start * sr))
                        if pause_start < pause_end:
                            segments.append(audio_data[pause_start:pause_end])

        if not segments:
            logger.warning(f"No valid speaking segments extracted from {audio_filepath}")
            return None, None

        return np.concatenate(segments), sr

    # ------------------------------------------------------------------
    # Unified timeline computation (shared by RTTM + CTM merging)
    # ------------------------------------------------------------------

    def _compute_timeline(
        self,
        turns: list[dict],
        actual_overlaps: list[float],
    ) -> list[_TurnTimeline]:
        """Compute per-turn adjusted RTTM segment positions.

        This is the single source of truth for timestamp offsets. Both
        ``_merge_rttm_files`` and ``_merge_ctm_files`` consume the result
        instead of independently recomputing offsets.
        """
        timeline: list[_TurnTimeline] = []
        current_time_offset = 0.0

        for i, entry in enumerate(turns):
            rttm_file = entry.get("rttm_filepath", "")
            overlap = (
                actual_overlaps[i]
                if i < len(actual_overlaps)
                else entry.get("overlap", 0)
            )

            if i > 0:
                if overlap < 0:
                    current_time_offset += abs(overlap)
                elif overlap > 0:
                    max_backup = current_time_offset * 0.5
                    actual_backup = min(overlap, max_backup)
                    current_time_offset = max(0, current_time_offset - actual_backup)

            segments = self._parse_rttm_timestamps(rttm_file) if rttm_file else []

            local_offset = 0.0
            max_pause = self.max_intra_turn_pause
            adjusted_segs: list[tuple[float, float]] = []

            for seg_idx, (seg_start, seg_dur) in enumerate(segments):
                adjusted_start = current_time_offset + local_offset
                adjusted_end = adjusted_start + seg_dur
                adjusted_segs.append((adjusted_start, adjusted_end))

                local_offset += seg_dur

                if seg_idx < len(segments) - 1:
                    next_start = segments[seg_idx + 1][0]
                    cur_end = seg_start + seg_dur
                    pause_dur = next_start - cur_end
                    if 0 < pause_dur <= max_pause:
                        local_offset += pause_dur

            timeline.append(_TurnTimeline(
                time_offset=current_time_offset,
                rttm_segments=segments,
                adjusted_segments=adjusted_segs,
                local_offset=local_offset,
            ))

            if segments:
                offset_before = current_time_offset
                current_time_offset += local_offset
                if current_time_offset <= offset_before and local_offset > 0:
                    current_time_offset = offset_before + local_offset

        return timeline

    # ------------------------------------------------------------------
    # Top-level merge orchestrator
    # ------------------------------------------------------------------

    def _merge_conversation(  # noqa: C901, PLR0912, PLR0915
        self,
        conversation_id: str,
        turns: list[dict],
    ) -> dict | None:
        """Merge all turns into per-speaker outputs under a conversation folder."""

        valid_turns: list[dict] = []
        for entry in turns:
            audio_file = entry.get("audio_filepath", "")
            if audio_file and os.path.isfile(audio_file):
                valid_turns.append(entry)
            else:
                logger.warning(
                    f"Skipping turn {entry.get('turn_index', '?')} "
                    f"(speaker={entry.get('speaker', '?')}): "
                    f"missing audio_filepath"
                )

        if not valid_turns:
            logger.error(f"No valid turns for conversation {conversation_id}")
            return None

        turns = valid_turns

        try:
            conv_dir = self.output_conversations_dir / conversation_id
            conv_dir.mkdir(parents=True, exist_ok=True)

            speakers: set[str] = set()
            for entry in turns:
                speaker = entry.get("speaker", "")
                if speaker:
                    speakers.add(speaker)

            actual_overlaps = self._merge_audio_files(conv_dir, turns)
            logger.info(f"Merged audio saved to {conv_dir}")

            timeline = self._compute_timeline(turns, actual_overlaps)

            per_turn_merged_segments: list[list[tuple[float, float]]] = []
            try:
                per_turn_merged_segments = self._merge_rttm_files(
                    conv_dir, conversation_id, turns, timeline
                )
                logger.info(f"Merged RTTMs saved to {conv_dir}")
            except (OSError, RuntimeError):
                logger.exception("Error merging RTTMs")

            try:
                self._merge_ctm_files(
                    conv_dir, conversation_id, turns, timeline
                )
                logger.info(f"Merged CTMs saved to {conv_dir}")
            except (OSError, RuntimeError):
                logger.exception("Error merging CTMs")

            mixed_path = conv_dir / "mixed.wav"
            merged_duration = 0.0
            with contextlib.suppress(OSError):
                info = sf.info(str(mixed_path))
                merged_duration = info.duration

            seglst_path = conv_dir / "segments.seglst.json"
            try:
                self._generate_seglst(
                    seglst_path, conversation_id, turns,
                    per_turn_merged_segments, max_duration=merged_duration,
                )
                logger.info(f"Seglst saved: {seglst_path}")
            except (OSError, RuntimeError):
                logger.exception("Error generating seglst")

            mfa_fallback = any(entry.get("mfa_skipped", False) for entry in turns)

            speaker_references: dict[str, dict[str, str]] = {}
            for entry in turns:
                spk = entry.get("speaker", "")
                if spk and spk not in speaker_references:
                    speaker_references[spk] = {
                        "reference_audio": entry.get("reference_audio", ""),
                        "reference_voice": entry.get("reference_voice", ""),
                    }

            merged_entry: dict[str, Any] = {
                "conversation_id": conversation_id,
                "audio_filepath": str(mixed_path),
                "rttm_filepath": (
                    str(conv_dir / "all.rttm")
                    if (conv_dir / "all.rttm").exists()
                    else ""
                ),
                "ctm_filepath": (
                    str(conv_dir / "all.ctm")
                    if (conv_dir / "all.ctm").exists()
                    else ""
                ),
                "seglst_filepath": str(seglst_path),
                "duration": merged_duration,
                "num_speakers": len(speakers),
                "offset": 0,
                "mfa_fallback": mfa_fallback,
                "speaker_references": speaker_references,
            }

        except (OSError, RuntimeError):
            logger.exception(f"Error merging conversation {conversation_id}")
            raise
        else:
            return merged_entry

    # ------------------------------------------------------------------
    # Audio merging -- per-speaker buffers
    # ------------------------------------------------------------------

    def _merge_audio_files(  # noqa: C901, PLR0912, PLR0915
        self,
        conv_dir: Path,
        conversation_entries: list[dict],
    ) -> list[float]:
        """Merge per-turn audio into per-speaker channels.

        Produces:
          - ``{conv_dir}/speaker_X.wav`` -- single-channel, full duration
          - ``{conv_dir}/multichannel.wav`` -- N-channel WAV
          - ``{conv_dir}/mixed.wav`` -- mono sum of all channels

        Returns a list of *actual* overlap/pause values used (one per turn;
        first entry is always 0.0).
        """
        if not conversation_entries:
            return []

        speaker_buffers: dict[str, np.ndarray] = {}
        current_position = 0
        total_length = 0
        sample_rate: int | None = None
        actual_overlaps: list[float] = []
        sorted_speakers: list[str] = []

        for i, entry in enumerate(conversation_entries):
            audio_file = entry.get("audio_filepath", "")

            try:
                rttm_file = entry.get("rttm_filepath", "")
                if rttm_file:
                    audio_data, sr = self.extract_speaking_segments(audio_file, rttm_file)
                    if audio_data is None:
                        logger.warning(
                            f"Failed to extract speaking segments from {audio_file}, "
                            "falling back to full audio"
                        )
                        audio_data, sr = sf.read(audio_file)
                        if len(audio_data.shape) > 1:
                            audio_data = audio_data.mean(axis=1)
                else:
                    audio_data, sr = sf.read(audio_file)
                    if len(audio_data.shape) > 1:
                        audio_data = audio_data.mean(axis=1)

                if sample_rate is None:
                    sample_rate = sr
                elif sr != sample_rate:
                    logger.warning(f"Sample rate mismatch: {sr} vs {sample_rate}")

                speaker = entry.get("speaker", f"speaker_{i}")
                if speaker not in sorted_speakers:
                    sorted_speakers.append(speaker)

                overlap = entry.get("overlap", 0)

                if i == 0:
                    start_position = 0
                    actual_overlaps.append(0.0)
                elif overlap < 0:
                    pause_duration = min(abs(overlap), self.max_pause_duration)
                    if self.randomize_pauses and pause_duration > _MIN_RANDOMIZE_PAUSE:
                        pause_duration = self._rng.uniform(
                            0.3, min(pause_duration, self.max_pause_duration)
                        )
                    pause_samples = int(pause_duration * sample_rate)
                    actual_overlaps.append(-pause_duration)
                    start_position = current_position + pause_samples
                elif overlap > 0:
                    audio_duration_sec = len(audio_data) / sample_rate
                    max_overlap_sec = min(
                        audio_duration_sec,
                        current_position / sample_rate * 0.5,
                    )
                    actual_overlap = min(overlap, max_overlap_sec)
                    overlap_samples = int(actual_overlap * sample_rate)
                    actual_overlaps.append(actual_overlap)
                    start_position = max(0, current_position - overlap_samples)
                else:
                    actual_overlaps.append(0.0)
                    start_position = current_position

                end_position = start_position + len(audio_data)

                if speaker not in speaker_buffers:
                    speaker_buffers[speaker] = np.zeros(end_position, dtype=np.float64)
                buf = speaker_buffers[speaker]
                if end_position > len(buf):
                    extended = np.zeros(end_position, dtype=np.float64)
                    extended[: len(buf)] = buf
                    speaker_buffers[speaker] = extended
                    buf = speaker_buffers[speaker]

                buf[start_position:end_position] = audio_data
                current_position = max(current_position, end_position)
                total_length = max(total_length, end_position)

            except (OSError, RuntimeError) as exc:
                logger.error(f"Error processing audio file {audio_file}: {exc}")
                actual_overlaps.append(entry.get("overlap", 0))
                continue

        if not speaker_buffers or sample_rate is None:
            logger.error("No valid audio data to save")
            return actual_overlaps

        for spk, buf in speaker_buffers.items():
            if len(buf) < total_length:
                extended = np.zeros(total_length, dtype=np.float64)
                extended[: len(buf)] = buf
                speaker_buffers[spk] = extended
            elif len(buf) > total_length:
                speaker_buffers[spk] = buf[:total_length]

        sorted_speakers = [s for s in sorted_speakers if s in speaker_buffers]
        for spk in sorted_speakers:
            sf.write(str(conv_dir / f"{spk}.wav"), speaker_buffers[spk], sample_rate)

        multichannel = np.column_stack(
            [speaker_buffers[spk] for spk in sorted_speakers]
        )
        sf.write(str(conv_dir / "multichannel.wav"), multichannel, sample_rate)

        mixed = np.sum(
            [speaker_buffers[spk] for spk in sorted_speakers], axis=0
        )
        peak = np.abs(mixed).max()
        if peak > _PEAK_NORMALIZE_THRESHOLD:
            mixed = mixed * (_PEAK_NORMALIZE_THRESHOLD / peak)
        sf.write(str(conv_dir / "mixed.wav"), mixed, sample_rate)

        logger.debug(
            f"Saved audio: {len(sorted_speakers)} speakers, "
            f"{total_length / sample_rate:.2f}s, {len(actual_overlaps)} turns"
        )
        return actual_overlaps

    # ------------------------------------------------------------------
    # RTTM merging -- per-speaker + combined
    # ------------------------------------------------------------------

    def _merge_rttm_files(
        self,
        conv_dir: Path,
        conversation_id: str,
        conversation_entries: list[dict],
        timeline: list[_TurnTimeline],
    ) -> list[list[tuple[float, float]]]:
        """Merge per-turn RTTMs into per-speaker and combined RTTM files.

        Uses the pre-computed *timeline* (from ``_compute_timeline``) so that
        offsets are computed exactly once and shared with CTM merging.

        Returns a list (one per turn) of ``[(adjusted_start, adjusted_end), ...]``
        tuples representing the repositioned RTTM segments for seglst generation.
        """
        speaker_rttm_lines: dict[str, list[str]] = {}
        per_turn_merged_segments: list[list[tuple[float, float]]] = []

        for i, entry in enumerate(conversation_entries):
            speaker = entry.get("speaker", f"speaker_{i}")
            tl = timeline[i]

            if not tl.rttm_segments:
                per_turn_merged_segments.append([])
                continue

            if speaker not in speaker_rttm_lines:
                speaker_rttm_lines[speaker] = []

            turn_segments: list[tuple[float, float]] = []

            for seg_idx, (_seg_start, seg_dur) in enumerate(tl.rttm_segments):
                adj_start, adj_end = tl.adjusted_segments[seg_idx]

                speaker_rttm_lines[speaker].append(
                    f"SPEAKER {conversation_id} 1 "
                    f"{adj_start:.3f} {seg_dur:.3f} "
                    f"<NA> <NA> {speaker} <NA> <NA>"
                )
                turn_segments.append((adj_start, adj_end))

            per_turn_merged_segments.append(turn_segments)

            logger.debug(
                f"Turn {i} ({speaker}): {len(tl.rttm_segments)} segs, "
                f"local_dur={tl.local_offset:.3f}s, "
                f"offset={tl.time_offset:.3f}s"
            )

        if speaker_rttm_lines:
            for spk, lines in speaker_rttm_lines.items():
                with open(conv_dir / f"{spk}.rttm", "w", encoding="utf-8") as f:
                    f.writelines(rttm_line + "\n" for rttm_line in lines)

            all_lines: list[str] = []
            for lines in speaker_rttm_lines.values():
                all_lines.extend(lines)
            all_lines.sort(key=lambda rttm_line: float(rttm_line.split()[3]))
            with open(conv_dir / "all.rttm", "w", encoding="utf-8") as f:
                f.writelines(rttm_line + "\n" for rttm_line in all_lines)

        return per_turn_merged_segments

    # ------------------------------------------------------------------
    # CTM merging -- per-speaker + combined
    # ------------------------------------------------------------------

    def _merge_ctm_files(
        self,
        conv_dir: Path,
        conversation_id: str,
        conversation_entries: list[dict],
        timeline: list[_TurnTimeline],
    ) -> None:
        """Merge per-turn CTMs using the shared *timeline*.

        Each CTM word is assigned to the nearest RTTM segment rather than
        requiring strict containment, so no words are silently dropped.
        """
        speaker_ctm_lines: dict[str, list[tuple[float, str]]] = {}

        for i, entry in enumerate(conversation_entries):
            ctm_file = entry.get("ctm_filepath", "")
            speaker = entry.get("speaker", f"speaker_{i}")
            tl = timeline[i]

            ctm_words = self._parse_ctm_words(ctm_file) if ctm_file else []

            if not tl.rttm_segments or not ctm_words:
                continue

            if speaker not in speaker_ctm_lines:
                speaker_ctm_lines[speaker] = []

            for w_start, w_dur, word in ctm_words:
                seg_idx = self._find_nearest_segment(w_start, tl.rttm_segments)
                seg_start, _seg_dur = tl.rttm_segments[seg_idx]
                adj_seg_start = tl.adjusted_segments[seg_idx][0]

                word_offset_in_seg = max(0.0, w_start - seg_start)
                adjusted_word_start = adj_seg_start + word_offset_in_seg

                speaker_ctm_lines[speaker].append((
                    adjusted_word_start,
                    f"{conversation_id} 1 {adjusted_word_start:.3f} {w_dur:.3f} {word}",
                ))

        if speaker_ctm_lines:
            for spk, ctm_entries in speaker_ctm_lines.items():
                ctm_entries.sort(key=lambda x: x[0])
                with open(conv_dir / f"{spk}.ctm", "w", encoding="utf-8") as f:
                    f.writelines(ctm_line + "\n" for _, ctm_line in ctm_entries)

            all_entries: list[tuple[float, str]] = []
            for ctm_entries in speaker_ctm_lines.values():
                all_entries.extend(ctm_entries)
            all_entries.sort(key=lambda x: x[0])
            with open(conv_dir / "all.ctm", "w", encoding="utf-8") as f:
                f.writelines(ctm_line + "\n" for _, ctm_line in all_entries)

    # ------------------------------------------------------------------
    # Segment list generation from RTTM
    # ------------------------------------------------------------------

    def _generate_seglst(
        self,
        output_path: Path,
        conversation_id: str,
        turns: list[dict],
        per_turn_merged_segments: list[list[tuple[float, float]]],
        max_duration: float = 0.0,
    ) -> None:
        """Generate a segment list JSON from RTTM-derived timestamps.

        Each segment corresponds to one turn. Boundaries are derived from the
        first and last RTTM segment of that turn, padded by ``seglst_offset``
        and clamped to *max_duration* (the WAV file length).
        """
        seglst: list[dict[str, Any]] = []
        offset = self.seglst_offset

        for i, entry in enumerate(turns):
            speaker = entry.get("speaker", f"speaker_{i}")
            utterance = entry.get("text", entry.get("utterance", ""))
            turn_segs = (
                per_turn_merged_segments[i]
                if i < len(per_turn_merged_segments)
                else []
            )

            if turn_segs:
                seg_start = max(0.0, turn_segs[0][0] - offset)
                seg_end = turn_segs[-1][1] + offset
                if max_duration > 0:
                    seg_end = min(seg_end, max_duration)
            else:
                seg_start = 0.0
                seg_end = 0.0

            seglst.append({
                "session_id": conversation_id,
                "speaker": speaker,
                "start_time": round(seg_start, 3),
                "end_time": round(seg_end, 3),
                "words": utterance,
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(seglst, f, indent=2, ensure_ascii=False)
