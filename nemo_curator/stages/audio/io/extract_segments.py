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
Audio segment extraction stage.

Extracts audio segments from original source files based on manifest
entries produced by NeMo Curator audio pipelines.  Auto-detects the
pipeline combo from the manifest schema and applies the appropriate
extraction strategy:

  Combo 2 (no VAD / VAD only):
    Extracts each segment by ``original_start_ms`` / ``original_end_ms``.
    Output: ``{original_filename}_segment_{NNN}.{format}``

  Combo 3 (speaker diarization):
    Extracts each speaking interval from ``diar_segments`` per speaker.
    Output: ``{original_filename}_speaker_{X}_segment_{NNN}.{format}``

  Combo 4 (VAD + speaker):
    Extracts each speaker-segment by timestamps.
    Output: ``{original_filename}_speaker_{X}_segment_{NNN}.{format}``

Example:
    from nemo_curator.stages.audio.io.extract_segments import SegmentExtractionStage

    stage = SegmentExtractionStage(
        output_dir="/data/extracted",
        output_format="flac",
    )

    # Standalone usage (post-pipeline):
    stage.extract_from_manifest("manifest.jsonl")

    # Or as a pipeline stage:
    pipeline.add_stage(stage)
"""

from __future__ import annotations

import csv
import fcntl
import glob
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import soundfile as sf
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import numpy as np

from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, IOSpec, StageContract
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

DEFAULT_OUTPUT_FORMAT = "wav"
_LOCK_FILENAME = ".segment_extraction.lock"
_STATE_FILENAME = ".segment_extraction_state.json"

SOUNDFILE_FORMATS = {
    "wav": "PCM_16",
    "flac": "PCM_16",
    "ogg": "VORBIS",
}

_CSV_STRUCTURAL_KEYS = frozenset(
    {
        "filename",
        "original_file",
        "original_start_ms",
        "original_end_ms",
        "duration_ms",
        "start_sec",
        "end_sec",
        "duration",
        "segment_index",
        "speaker_id",
        "num_speakers",
        "speaking_duration",
        "diar_segments",
    }
)

Interval = tuple[int, int, Any]  # (start_ms, end_ms, duration_sec)


# ------------------------------------------------------------------
# Pure helper functions
# ------------------------------------------------------------------


def _extract_scores(entry: dict, exclude: frozenset[str] = frozenset()) -> dict:
    """Extract quality/filter score fields from a manifest entry.

    Returns all keys that are not structural CSV columns (timestamps,
    duration, speaker info), with float values rounded for readability.
    Since TimestampMapper already whitelist-filters the manifest output,
    anything remaining is a quality score or user-defined field.
    ``exclude`` drops additional keys (the stage's configurable
    ``output_key`` bookkeeping must never leak into the CSV schema).
    """
    return {
        k: round(v, 4) if isinstance(v, float) else v
        for k, v in entry.items()
        if k not in _CSV_STRUCTURAL_KEYS and k not in exclude
    }


def _get_speaker_label(entry: dict) -> tuple[str, str]:
    """Return (speaker_id, speaker_num) from a manifest entry."""
    speaker_id = entry.get("speaker_id", "unknown")
    speaker_num = speaker_id.replace("speaker_", "") if "speaker_" in speaker_id else speaker_id
    return speaker_id, speaker_num


def _read_segment(filepath: str, start_ms: int, end_ms: int, sample_rate: int) -> np.ndarray:
    """Read a slice of audio from a file."""
    start_sample = int(start_ms * sample_rate / 1000)
    end_sample = int(end_ms * sample_rate / 1000)
    audio, _ = sf.read(filepath, start=start_sample, stop=end_sample, dtype="float32")
    return audio


def _intervals_from_timestamps(entry: dict) -> list[Interval]:
    start_ms = entry.get("original_start_ms", 0)
    end_ms = entry.get("original_end_ms", 0)
    dur = entry.get("duration", (end_ms - start_ms) / 1000)
    return [(start_ms, end_ms, dur)]


def _intervals_from_diar_segments(entry: dict) -> list[Interval]:
    diar_segments = entry.get("diar_segments", [])
    if not diar_segments:
        speaker_id = entry.get("speaker_id", "unknown")
        logger.warning(f"  {speaker_id}: no diar_segments, skipping")
        return []
    valid_segments = []
    for segment in diar_segments:
        if isinstance(segment, Mapping):
            start, end = segment.get("start"), segment.get("end")
        else:
            try:
                start, end = segment[0], segment[1]
            except (TypeError, IndexError, KeyError):
                continue
        try:
            start_value, end_value = float(start), float(end)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start_value) or not math.isfinite(end_value) or end_value <= start_value:
            continue
        try:
            if end - start <= 0:
                continue
        except TypeError:
            start, end = start_value, end_value
        valid_segments.append((start, end))
    valid_segments.sort(key=lambda bounds: bounds[0])
    return [(int(start * 1000), int(end * 1000), end - start) for start, end in valid_segments]


def _expand_nested_speaker_entries(entries: list[dict]) -> tuple[list[dict], list[tuple[dict, list[dict]]]]:
    """Normalize dictionary diarization segments into canonical per-speaker rows."""
    expanded = []
    expanded_groups = []
    for entry in entries:
        segments = entry.get("diar_segments") or []
        nested_speakers = {
            str(segment["speaker"])
            for segment in segments
            if isinstance(segment, Mapping) and segment.get("speaker") is not None
        }
        if not nested_speakers:
            expanded.append(entry)
            continue
        speaker_entries = []
        for speaker_id in sorted(nested_speakers):
            speaker_entry = dict(entry)
            speaker_entry["speaker_id"] = speaker_id
            speaker_entry["diar_segments"] = [
                segment
                for segment in segments
                if isinstance(segment, Mapping) and str(segment.get("speaker")) == speaker_id
            ]
            expanded.append(speaker_entry)
            speaker_entries.append(speaker_entry)
        expanded_groups.append((entry, speaker_entries))
    return expanded, expanded_groups


def _base_metadata(  # noqa: PLR0913
    filename: str,
    original_file: str,
    entry: dict,
    seg_idx: int,
    start_ms: int,
    end_ms: int,
    dur: float,
    exclude: frozenset[str] = frozenset(),
) -> dict:
    row: dict = {
        "filename": filename,
        "original_file": original_file,
        "segment_index": seg_idx,
        "start_sec": round(start_ms / 1000, 3),
        "end_sec": round(end_ms / 1000, 3),
        "duration": round(dur, 3),
    }
    speaker_id = entry.get("speaker_id")
    if speaker_id is not None:
        row["speaker_id"] = speaker_id
    num_speakers = entry.get("num_speakers")
    if num_speakers is not None:
        row["num_speakers"] = num_speakers
    row.update(_extract_scores(entry, exclude))
    return row


def detect_combo(entries: list) -> int:
    """Detect which pipeline combo produced the manifest.

    Returns 2, 3, or 4.  Since TimestampMapper always emits
    ``original_start_ms``/``original_end_ms``, combos 1 and 2 are
    indistinguishable and both use timestamp-based extraction.

    Returns:
        2: segments by timestamps (combos 1 and 2)
        3: speaker diarization segments
        4: speaker-segments by timestamps
    """
    if not entries:
        return 2

    first = entries[0]
    has_nested_speaker = any(
        isinstance(segment, Mapping) and segment.get("speaker") is not None
        for segment in (first.get("diar_segments") or [])
    )
    has_speaker = "speaker_id" in first or has_nested_speaker
    has_diar = "diar_segments" in first

    if has_speaker and has_diar:
        return 3
    if has_speaker:
        return 4
    return 2


def load_manifest(manifest_path: str) -> list:
    """Load a single manifest.jsonl file and return list of entries."""
    entries = []
    with open(manifest_path) as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num} in {manifest_path}: {e}")
    return entries


def load_manifests(input_path: str, output_dir: str) -> list:
    """Load entries from a single jsonl file or a directory of jsonl files."""
    if os.path.isfile(input_path):
        return load_manifest(input_path)

    if not os.path.isdir(input_path):
        logger.error(f"Input path not found: {input_path}")
        return []

    jsonl_files = sorted(glob.glob(os.path.join(input_path, "*.jsonl")))
    if not jsonl_files:
        logger.error(f"No .jsonl files found in {input_path}")
        return []

    logger.info(f"Found {len(jsonl_files)} jsonl files in {input_path}")

    all_entries = []
    for jf in jsonl_files:
        all_entries.extend(load_manifest(jf))

    logger.info(f"Combined {len(all_entries)} entries from {len(jsonl_files)} file(s)")

    if all_entries:
        os.makedirs(output_dir, exist_ok=True)
        combined_path = os.path.join(output_dir, "manifest.jsonl")
        with open(combined_path, "w") as f:
            f.writelines(json.dumps(e) + "\n" for e in all_entries)
        logger.info(f"Saved combined manifest to {combined_path}")

    return all_entries


def _write_metadata_csv(output_dir: str, metadata_rows: list[dict]) -> str:
    """Write metadata.csv from collected metadata rows."""
    if not metadata_rows:
        return ""

    all_keys: list[str] = []
    seen: set[str] = set()
    for row in metadata_rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    csv_path = os.path.join(output_dir, "metadata.csv")
    fd, temp_path = tempfile.mkstemp(prefix=".metadata.", suffix=".csv.tmp", dir=output_dir)
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(metadata_rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, csv_path)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise

    return csv_path


@contextmanager
def _output_lock(output_dir: str) -> Iterator[None]:
    lock_path = os.path.join(output_dir, _LOCK_FILENAME)
    with open(lock_path, "a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


# ------------------------------------------------------------------
# Stage
# ------------------------------------------------------------------


@dataclass
class SegmentExtractionStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """Extract audio segments from original files based on manifest entries.

    Receives ``AudioTask`` objects whose ``data`` dicts are manifest
    entries (produced by ``TimestampMapperStage``).  For each entry the
    stage reads the audio slice from the original file and writes it as
    a standalone segment file.

    Each entry's pipeline combo is detected independently. Entries are
    grouped by combo and ``original_file`` so heterogeneous batches are
    handled without making behavior depend on row order.

    This is an IO stage: ``process()`` raises ``NotImplementedError``
    and all work is done in ``process_batch()``, following the same
    pattern as ``AudioToDocumentStage`` and ``ALMManifestWriterStage``.

    Args:
        output_dir: Directory where extracted segment files are written.
        output_format: Audio format — ``wav``, ``flac``, or ``ogg``.
    """

    name: str = "SegmentExtraction"
    BATCH_ONLY = True  # process() raises; only process_batch is implemented (agent-discovery hint)
    output_dir: str = ""
    output_format: str = DEFAULT_OUTPUT_FORMAT
    batch_size: int = 64
    is_resumable = True
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    output_key: str = field(default="extracted_path", kw_only=True)

    def __post_init__(self) -> None:
        super().__init__()
        if not self.output_dir:
            msg = "output_dir is required for SegmentExtractionStage"
            raise ValueError(msg)
        if self.output_format not in SOUNDFILE_FORMATS:
            msg = f"output_format must be one of {list(SOUNDFILE_FORMATS)}, got {self.output_format!r}"
            raise ValueError(msg)
        if not isinstance(self.output_key, str) or not self.output_key:
            msg = "output_key must be a non-empty string"
            raise ValueError(msg)
        self._metadata_by_filename: dict[str, dict] = {}
        self._segment_reservations: dict[str, dict[str, Any]] = {}
        self._task_ids_by_entry: dict[int, str] = {}
        self._segment_counter: dict[str, int] = defaultdict(int)
        self._speaker_segment_counter: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], ["original_file"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_key]

    def describe(self) -> StageContract:
        return StageContract(
            reads_one_of=[
                IOSpec(data_keys=["original_file", "original_start_ms", "original_end_ms"], accepts=["file"]),
                IOSpec(data_keys=["original_file", "diar_segments", "speaker_id"], accepts=["file"]),
            ],
            writes=IOSpec(data_keys=[self.output_key], produces=["disk"]),
            gates=Gates(
                writes_to_disk=True,
                output_path_params=["output_dir"],
                requires_stable_task_id=True,
                # Stable reservations make checkpoint retries idempotent, but the
                # legacy per-source counters still make filenames depend on sibling rows.
                per_row_independent=False,
            ),
        )

    def num_workers(self) -> int | None:
        return 1

    def process(self, task: AudioTask) -> AudioTask:
        msg = "SegmentExtractionStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []

        os.makedirs(self.output_dir, exist_ok=True)

        entries = [task.data for task in tasks]
        with _output_lock(self.output_dir):
            self._load_output_state()
            self._task_ids_by_entry = {id(task.data): task.task_id for task in tasks}
            try:
                extracted, total_dur, speaker_counts, metadata_rows = self._extract_entries(
                    entries,
                    record_output_paths=True,
                )
            finally:
                self._task_ids_by_entry = {}

            for row in metadata_rows:
                self._metadata_by_filename[row["filename"]] = row
            _write_metadata_csv(self.output_dir, list(self._metadata_by_filename.values()))

        logger.info(f"[{self.name}] Extracted {extracted} segments ({total_dur:.1f}s) from {len(tasks)} entries")
        if speaker_counts:
            for speaker, count in sorted(speaker_counts.items()):
                logger.debug(f"  {speaker}: {count} segments")

        return tasks

    def _extract_entries(
        self,
        entries: list[dict],
        *,
        record_output_paths: bool = False,
    ) -> tuple[int, float, dict[str, int], list[dict]]:
        extractors = {
            2: self._extract_by_timestamps,
            3: self._extract_speaker_diar,
            4: self._extract_speaker_timestamps,
        }
        entries_by_combo: dict[int, list[dict]] = defaultdict(list)
        for entry in entries:
            entries_by_combo[detect_combo([entry])].append(entry)

        total_extracted = 0
        total_duration = 0.0
        total_speaker_counts: dict[str, int] = defaultdict(int)
        all_metadata_rows: list[dict] = []
        for combo in (2, 3, 4):
            if not entries_by_combo[combo]:
                continue
            extracted, duration, speaker_counts, metadata_rows = extractors[combo](
                entries_by_combo[combo],
                record_output_paths=record_output_paths,
            )
            total_extracted += extracted
            total_duration += duration
            all_metadata_rows.extend(metadata_rows)
            for speaker, count in speaker_counts.items():
                total_speaker_counts[speaker] += count
        return total_extracted, total_duration, total_speaker_counts, all_metadata_rows

    @property
    def _state_path(self) -> str:
        return os.path.join(self.output_dir, _STATE_FILENAME)

    def _load_output_state(self) -> None:
        self._metadata_by_filename = {}
        self._segment_reservations = {}
        self._segment_counter = defaultdict(int)
        self._speaker_segment_counter = defaultdict(lambda: defaultdict(int))

        metadata_path = os.path.join(self.output_dir, "metadata.csv")
        if os.path.exists(metadata_path):
            with open(metadata_path, newline="") as metadata_file:
                for row in csv.DictReader(metadata_file):
                    filename = row.get("filename")
                    if filename:
                        self._metadata_by_filename[filename] = row
                        self._advance_counter(row)

        if not os.path.exists(self._state_path):
            return
        try:
            with open(self._state_path, encoding="utf-8") as state_file:
                state = json.load(state_file)
        except (OSError, json.JSONDecodeError, TypeError) as error:
            message = f"[{self.name}] Cannot safely resume from unreadable extraction state: {error}"
            raise RuntimeError(message) from error
        segments = state.get("segments", {}) if isinstance(state, dict) else {}
        if not isinstance(segments, dict):
            message = f"[{self.name}] Cannot safely resume from malformed extraction state"
            raise TypeError(message)
        for resume_key, record in segments.items():
            if not isinstance(resume_key, str) or not isinstance(record, dict) or not isinstance(record.get("filename"), str):
                message = f"[{self.name}] Cannot safely resume from malformed extraction reservation"
                raise TypeError(message)
            self._segment_reservations[resume_key] = record
            self._advance_counter(record)

    def _advance_counter(self, record: dict[str, Any]) -> None:
        filename = record.get("filename")
        original_file = record.get("original_file")
        if not isinstance(filename, str) or not isinstance(original_file, str):
            return
        try:
            _, index_suffix = filename.rsplit("_segment_", 1)
        except ValueError:
            return
        try:
            index = int(index_suffix.split(".", 1)[0]) + 1
        except ValueError:
            return
        name = Path(original_file).stem
        speaker_id = record.get("speaker_id")
        if speaker_id is not None:
            speaker_key = str(speaker_id)
            self._speaker_segment_counter[name][speaker_key] = max(
                self._speaker_segment_counter[name][speaker_key], index
            )
        else:
            self._segment_counter[name] = max(self._segment_counter[name], index)

    def _persist_output_state(self) -> None:
        fd, temp_path = tempfile.mkstemp(prefix=".segment_extraction_state.", suffix=".json.tmp", dir=self.output_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as state_file:
                json.dump({"version": 1, "segments": self._segment_reservations}, state_file, sort_keys=True)
                state_file.flush()
                os.fsync(state_file.fileno())
            os.replace(temp_path, self._state_path)
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def _segment_resume_key(
        self,
        entry: dict[str, Any],
        original_file: str,
        start_ms: int,
        end_ms: int,
        segment_index: int,
    ) -> str | None:
        task_id = self._task_ids_by_entry.get(id(entry))
        if not task_id:
            return None
        identity = {
            "task_id": task_id,
            "original_file": os.path.abspath(original_file),
            "speaker_id": entry.get("speaker_id"),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "segment_index": segment_index,
            "output_format": self.output_format,
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _reserved_filename(
        self,
        resume_key: str,
        original_file: str,
        entry: dict[str, Any],
        make_filename: Callable[[], str],
    ) -> tuple[str, bool]:
        existing = self._segment_reservations.get(resume_key)
        if existing is not None:
            return existing["filename"], True
        filename = make_filename()
        record: dict[str, Any] = {"filename": filename, "original_file": original_file}
        if entry.get("speaker_id") is not None:
            record["speaker_id"] = entry["speaker_id"]
        self._segment_reservations[resume_key] = record
        self._persist_output_state()
        return filename, False

    def _write_audio_atomically(
        self, output_path: str, audio: np.ndarray, sample_rate: int, *, reuse_existing: bool
    ) -> None:
        if reuse_existing:
            try:
                existing = sf.info(output_path)
            except (OSError, RuntimeError, sf.LibsndfileError):
                existing = None
            if existing is not None and existing.samplerate == sample_rate and existing.frames == len(audio):
                return

        fd, temp_path = tempfile.mkstemp(prefix=f".{Path(output_path).name}.", suffix=".tmp", dir=self.output_dir)
        os.close(fd)
        try:
            sf.write(
                temp_path,
                audio,
                sample_rate,
                format=self.output_format.upper(),
                subtype=SOUNDFILE_FORMATS[self.output_format],
            )
            os.replace(temp_path, output_path)
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    # ------------------------------------------------------------------
    # Combo extractors (instance methods using self.output_dir/format)
    # ------------------------------------------------------------------

    def _extract_by_timestamps(
        self,
        entries: list[dict],
        *,
        record_output_paths: bool = False,
    ) -> tuple[int, float, dict[str, int], list[dict]]:
        """Combo 2: extract by original_start_ms / original_end_ms."""

        def _make_filename(name: str, _entry: dict, _seg_idx: int) -> str:
            idx = self._segment_counter[name]
            self._segment_counter[name] += 1
            return f"{name}_segment_{idx:03d}.{self.output_format}"

        return self._extract_file_segments(
            entries,
            sort_key=lambda x: x.get("original_start_ms", 0),
            get_intervals=_intervals_from_timestamps,
            make_filename=_make_filename,
            record_output_paths=record_output_paths,
        )

    def _extract_speaker_diar(  # noqa: C901
        self,
        entries: list[dict],
        *,
        record_output_paths: bool = False,
    ) -> tuple[int, float, dict[str, int], list[dict]]:
        """Combo 3: extract each diar_segment per speaker."""

        def _make_filename(name: str, entry: dict, _seg_idx: int) -> str:
            speaker_id, speaker_num = _get_speaker_label(entry)
            idx = self._speaker_segment_counter[name][speaker_id]
            self._speaker_segment_counter[name][speaker_id] += 1
            return f"{name}_speaker_{speaker_num}_segment_{idx:03d}.{self.output_format}"

        expanded_entries, expanded_groups = _expand_nested_speaker_entries(entries)
        inherited_paths: dict[int, list[Any]] = {}
        for parent, speaker_entries in expanded_groups:
            parent_task_id = self._task_ids_by_entry.get(id(parent))
            if parent_task_id is not None:
                for speaker_entry in speaker_entries:
                    self._task_ids_by_entry[id(speaker_entry)] = parent_task_id
            if record_output_paths:
                existing = parent.get(self.output_key)
                inherited_paths[id(parent)] = (
                    list(existing) if isinstance(existing, list) else ([] if existing is None else [existing])
                )
                for speaker_entry in speaker_entries:
                    speaker_entry.pop(self.output_key, None)
        result = self._extract_file_segments(
            expanded_entries,
            sort_key=lambda x: x.get("speaker_id", ""),
            get_intervals=_intervals_from_diar_segments,
            make_filename=_make_filename,
            record_output_paths=record_output_paths,
        )
        if record_output_paths:
            for parent, speaker_entries in expanded_groups:
                output_paths = inherited_paths[id(parent)]
                for speaker_entry in speaker_entries:
                    for path in speaker_entry.get(self.output_key, []):
                        if path not in output_paths:
                            output_paths.append(path)
                parent[self.output_key] = output_paths
        return result

    def _extract_speaker_timestamps(
        self,
        entries: list[dict],
        *,
        record_output_paths: bool = False,
    ) -> tuple[int, float, dict[str, int], list[dict]]:
        """Combo 4: extract speaker-segments by timestamps."""

        def _make_filename(name: str, entry: dict, _seg_idx: int) -> str:
            speaker_id, speaker_num = _get_speaker_label(entry)
            idx = self._speaker_segment_counter[name][speaker_id]
            self._speaker_segment_counter[name][speaker_id] += 1
            return f"{name}_speaker_{speaker_num}_segment_{idx:03d}.{self.output_format}"

        return self._extract_file_segments(
            entries,
            sort_key=lambda x: (x.get("speaker_id", ""), x.get("original_start_ms", 0)),
            get_intervals=_intervals_from_timestamps,
            make_filename=_make_filename,
            record_output_paths=record_output_paths,
        )

    # ------------------------------------------------------------------
    # Shared extraction engine
    # ------------------------------------------------------------------

    def _extract_file_segments(  # noqa: C901, PLR0912
        self,
        entries: list[dict],
        *,
        sort_key: Callable[[dict], Any],
        get_intervals: Callable[[dict], list[Interval]],
        make_filename: Callable[[str, dict, int], str],
        record_output_paths: bool = False,
    ) -> tuple[int, float, dict[str, int], list[dict]]:
        """Group-by-file -> read -> write -> metadata loop."""
        by_file: dict[str, list] = defaultdict(list)
        for entry in entries:
            by_file[entry.get("original_file", "")].append(entry)

        extracted = 0
        total_dur = 0.0
        speaker_counts: dict[str, int] = defaultdict(int)
        metadata_rows: list[dict] = []
        written_paths: dict[int, list[Any]] = {}
        if record_output_paths:
            for entry in entries:
                existing = entry.get(self.output_key)
                written_paths[id(entry)] = (
                    list(existing) if isinstance(existing, list) else ([] if existing is None else [existing])
                )

        for original_file, file_entries in by_file.items():
            if not os.path.exists(original_file):
                logger.error(f"Original file not found: {original_file}")
                continue

            info = sf.info(original_file)
            original_name = Path(original_file).stem
            file_entries.sort(key=sort_key)
            logger.info(f"\nProcessing: {original_name} ({len(file_entries)} entries)")

            for entry in file_entries:
                intervals = get_intervals(entry)
                for seg_idx, (start_ms, end_ms, dur) in enumerate(intervals):
                    if record_output_paths:
                        resume_key = self._segment_resume_key(entry, original_file, start_ms, end_ms, seg_idx)
                        if resume_key is None:
                            out_filename = make_filename(original_name, entry, seg_idx)
                            is_retry = False
                        else:
                            out_filename, is_retry = self._reserved_filename(
                                resume_key,
                                original_file,
                                entry,
                                lambda name=original_name, item=entry, index=seg_idx: make_filename(
                                    name, item, index
                                ),
                            )
                    else:
                        out_filename = make_filename(original_name, entry, seg_idx)
                        is_retry = False
                    output_path = os.path.join(self.output_dir, out_filename)

                    try:
                        audio = _read_segment(original_file, start_ms, end_ms, info.samplerate)
                        self._write_audio_atomically(
                            output_path,
                            audio,
                            info.samplerate,
                            reuse_existing=is_retry,
                        )
                        if record_output_paths and output_path not in written_paths[id(entry)]:
                            written_paths[id(entry)].append(output_path)
                        extracted += 1
                        total_dur += float(dur)

                        speaker_id = entry.get("speaker_id")
                        if speaker_id:
                            speaker_counts[speaker_id] += 1

                        metadata_rows.append(
                            _base_metadata(
                                out_filename,
                                original_file,
                                entry,
                                seg_idx,
                                start_ms,
                                end_ms,
                                dur,
                                exclude=frozenset({self.output_key}) if record_output_paths else frozenset(),
                            )
                        )
                        logger.debug(f"  {out_filename} ({start_ms}-{end_ms}ms, {dur:.2f}s)")
                    except Exception as e:
                        logger.error(f"  Failed to extract {out_filename}: {e}")
                        if record_output_paths:
                            raise

        if record_output_paths:
            for entry in entries:
                entry[self.output_key] = written_paths[id(entry)]

        return extracted, total_dur, speaker_counts, metadata_rows

    # ------------------------------------------------------------------
    # Standalone convenience methods (post-pipeline usage)
    # ------------------------------------------------------------------

    def extract_from_manifest(self, input_path: str) -> None:
        """Load a manifest file (or directory of JSONL files) and extract all segments.

        This is a convenience method for standalone usage outside
        of a pipeline.  It handles manifest loading, combo detection,
        CSV metadata, and summary JSON — equivalent to the old
        ``extract_segments()`` function.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"Loading manifest: {input_path}")
        entries = load_manifests(input_path, self.output_dir)
        logger.info(f"Found {len(entries)} entries total")

        if not entries:
            logger.error("No entries found in manifest")
            return

        combo_names = {
            2: "Segments by timestamps",
            3: "Speaker diarization segments",
            4: "Speaker-segments by timestamps",
        }
        detected_combos = sorted({detect_combo([entry]) for entry in entries})
        logger.info(f"Detected: {', '.join(combo_names[combo] for combo in detected_combos)}")

        total_extracted, total_dur, speaker_counts, metadata_rows = self._extract_entries(entries)

        csv_path = _write_metadata_csv(self.output_dir, metadata_rows)

        summary = {
            "manifest_path": input_path,
            "output_dir": self.output_dir,
            "total_segments": total_extracted,
            "total_duration_sec": round(total_dur, 2),
            "output_format": self.output_format,
        }
        if speaker_counts:
            summary["segments_by_speaker"] = dict(speaker_counts)

        summary_path = os.path.join(self.output_dir, "extraction_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n{'=' * 60}")
        logger.info("EXTRACTION COMPLETE")
        logger.info(f"{'=' * 60}")
        logger.info(f"  Combo: {', '.join(combo_names[combo] for combo in detected_combos)}")
        logger.info(f"  Total segments: {total_extracted}")
        logger.info(f"  Total duration: {total_dur:.2f}s ({total_dur / 60:.1f} min)")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Format: {self.output_format}")
        if speaker_counts:
            logger.info("  Segments by speaker:")
            for speaker, count in sorted(speaker_counts.items()):
                logger.info(f"    {speaker}: {count} segments")
        if csv_path:
            logger.info(f"  Metadata CSV: {csv_path}")
        logger.info(f"  Summary: {summary_path}")


# ------------------------------------------------------------------
# Backward-compatible free functions (delegate to stage)
# ------------------------------------------------------------------


def extract_segments_by_timestamps(
    entries: list,
    output_dir: str,
    output_format: str,
) -> tuple[int, float, dict[str, int], list[dict]]:
    """Extract segments by original_start_ms / original_end_ms, sorted by start time."""
    stage = SegmentExtractionStage(output_dir=output_dir, output_format=output_format)
    return stage._extract_by_timestamps(entries)


def extract_speaker_diar_segments(
    entries: list,
    output_dir: str,
    output_format: str,
) -> tuple[int, float, dict[str, int], list[dict]]:
    """Extract individual speaking intervals from diar_segments per speaker."""
    stage = SegmentExtractionStage(output_dir=output_dir, output_format=output_format)
    return stage._extract_speaker_diar(entries)


def extract_speaker_segments_by_timestamps(
    entries: list,
    output_dir: str,
    output_format: str,
) -> tuple[int, float, dict[str, int], list[dict]]:
    """Extract speaker-segments using original_start_ms / original_end_ms."""
    stage = SegmentExtractionStage(output_dir=output_dir, output_format=output_format)
    return stage._extract_speaker_timestamps(entries)


def extract_segments(input_path: str, output_dir: str, output_format: str = DEFAULT_OUTPUT_FORMAT) -> None:
    """Extract segments from original audio files based on manifest."""
    stage = SegmentExtractionStage(output_dir=output_dir, output_format=output_format)
    stage.extract_from_manifest(input_path)
