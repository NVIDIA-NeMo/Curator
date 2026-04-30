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

"""Stages for cutting long-form diarized audio into snippets for ALM pretraining.

The pipeline these stages compose into reads a JSONL manifest whose rows
each describe one long-form audio file plus a diarized + transcribed
``segments`` list, drops overlapping segments, packs the survivors into
bounded-duration snippets that never split a segment, slices the source
audio into mono resampled snippet files, and emits a per-snippet JSONL
manifest plus a metrics summary JSON.

The snippet manifest is intended as the foundation for audio LLM
pretraining data: each row's ``segments`` (with snippet-relative
timestamps) can be used to construct interleaved audio/text continuation
data, ASR training data, TTS training data, and speaker-diarization
training data without re-cutting the source audio.
"""

from __future__ import annotations

import copy
import glob
import json
import math
import os
import shutil
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask, _EmptyTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata


# ----------------------------------------------------------------------
# Constants & format mapping
# ----------------------------------------------------------------------


_SOUNDFILE_SUBTYPES = {
    "wav": "PCM_16",
    "flac": "PCM_16",
    "ogg": "VORBIS",
}

_PRETRAIN_META_KEY = "pretrain_long_form"
_PLAN_DATA_KEY = "_snippet_plan"
_HISTOGRAM_BIN_WIDTH_SEC = 30.0
_MANIFEST_SHARD_EXT = "jsonl"
# Metrics shards are JSONL (one record per task processed by a replica). The
# format avoids relying on `teardown()` -- the Xenna executor never calls it
# (actors are killed with `ray.kill()`), so an in-memory-only aggregator that
# flushed in teardown would always produce an empty summary. See
# `PretrainMetricsAggregatorStage.process` for the per-task record schema.
_METRICS_SHARD_EXT = "jsonl"


def _make_shard_path(output_path: str, ext: str) -> str:
    """Per-worker unique shard path next to ``output_path``.

    Each writer/aggregator worker computes one of these in ``setup()``;
    the merger glob-matches the same pattern after pipeline completion.
    """
    return f"{output_path}.shard-{os.getpid()}-{uuid.uuid4().hex[:8]}.{ext}"


def _glob_shards(output_path: str, ext: str) -> list[str]:
    return sorted(glob.glob(f"{output_path}.shard-*.{ext}"))


def _delete_shards(output_path: str, ext: str) -> int:
    n = 0
    for s in _glob_shards(output_path, ext):
        try:
            os.remove(s)
            n += 1
        except OSError as e:
            logger.warning(f"failed to remove shard {s}: {e}")
    return n


# ----------------------------------------------------------------------
# Pure helpers (unit-testable without Ray / soundfile)
# ----------------------------------------------------------------------


def _segment_text(seg: dict) -> str:
    """Return the best non-empty text representation of a segment."""
    return (seg.get("text_ITN") or seg.get("text") or "").strip()


def filter_empty_segments(segments: list[dict]) -> tuple[list[dict], int]:
    """Drop segments with no text and no words.

    Returns ``(kept, dropped_count)``.  Order is preserved.
    """
    kept: list[dict] = []
    dropped = 0
    for seg in segments:
        if _segment_text(seg) or seg.get("words"):
            kept.append(seg)
        else:
            dropped += 1
    return kept, dropped


def find_overlapping_indices(segments: list[dict], min_overlap_sec: float) -> set[int]:
    """Indices of segments that overlap any other segment.

    Two segments are considered overlapping (and both indices are
    returned) iff they share at least ``min_overlap_sec`` seconds of
    intersection OR one fully contains the other.  Brief touch-ups
    smaller than ``min_overlap_sec`` where neither covers the other are
    not flagged.
    """
    n = len(segments)
    bad: set[int] = set()
    for i in range(n):
        si, ei = segments[i]["start"], segments[i]["end"]
        for j in range(i + 1, n):
            sj, ej = segments[j]["start"], segments[j]["end"]
            if ej <= si or sj >= ei:
                continue
            overlap = min(ei, ej) - max(si, sj)
            i_contains_j = si <= sj and ei >= ej
            j_contains_i = sj <= si and ej >= ei
            if overlap >= min_overlap_sec or i_contains_j or j_contains_i:
                bad.add(i)
                bad.add(j)
    return bad


def plan_snippets(
    segments: list[dict],
    max_duration_sec: float,
    min_duration_sec: float,
    max_segment_gap_in_snippet: float,
) -> tuple[list[dict], dict[str, int]]:
    """Greedy contiguous packing of segments into snippets.

    Walks ``segments`` (assumed sorted by ``start``) and grows a current
    snippet while:

    1. its span ``[first.start, last.end]`` stays within
       ``max_duration_sec``, AND
    2. the gap from the last accepted segment's ``end`` to the next
       segment's ``start`` is at most ``max_segment_gap_in_snippet``.

    Either constraint failing closes the current snippet and opens a new
    one with the current segment.  Single segments longer than
    ``max_duration_sec`` are emitted as a one-segment candidate and then
    dropped under ``too_long``.

    The gap constraint matters for ALM pretraining: two segments
    separated by a long silence often belong to semantically distinct
    conversations (e.g. a topic change, an ad break, two takes recorded
    back to back), and a snippet that bridges them would teach the model
    to associate unrelated content.  Closing the snippet at long gaps
    keeps each training example semantically coherent.

    Returns ``(snippets, drop_counts)`` where each snippet is a dict with
    keys ``start``, ``end``, ``segments`` (the actual segment dicts) and
    drop counts keys are ``too_long``, ``too_short``, ``no_text``.
    """
    drop_counts = {"too_long": 0, "too_short": 0, "no_text": 0}
    if not segments:
        return [], drop_counts

    candidates: list[dict] = []
    cur: dict | None = None
    for seg in segments:
        if cur is None:
            cur = {"start": seg["start"], "end": seg["end"], "segments": [seg]}
            continue
        gap = seg["start"] - cur["end"]
        within_duration = seg["end"] - cur["start"] <= max_duration_sec
        within_gap = gap <= max_segment_gap_in_snippet
        if within_duration and within_gap:
            cur["end"] = seg["end"]
            cur["segments"].append(seg)
        else:
            candidates.append(cur)
            cur = {"start": seg["start"], "end": seg["end"], "segments": [seg]}
    if cur is not None:
        candidates.append(cur)

    snippets: list[dict] = []
    for cand in candidates:
        duration = cand["end"] - cand["start"]
        if duration > max_duration_sec:
            drop_counts["too_long"] += 1
            continue
        if duration < min_duration_sec:
            drop_counts["too_short"] += 1
            continue
        text = " ".join(_segment_text(s) for s in cand["segments"]).strip()
        if not text:
            drop_counts["no_text"] += 1
            continue
        snippets.append(cand)
    return snippets, drop_counts


def relativize_segments(
    segments: list[dict], snippet_start: float, snippet_end: float
) -> list[dict]:
    """Return shallow-copied segments with timestamps shifted to snippet-relative.

    Each segment-level and word-level ``start``/``end`` is shifted by
    ``-snippet_start`` and clamped to ``[0, snippet_end - snippet_start]``.
    Real diarization data has small (~10 ms) jitter where words are
    annotated as starting fractionally before their parent segment or
    ending fractionally after, so unclamped values can slip outside
    ``[0, duration]`` even though the snippet boundaries themselves
    align with segment boundaries; clamping keeps downstream consumers
    from having to handle that.

    Other fields are reused by reference -- treat the returned segments
    as read-only.
    """
    duration = max(0.0, snippet_end - snippet_start)

    def _shift_clamp(t: float) -> float:
        return min(duration, max(0.0, t - snippet_start))

    out: list[dict] = []
    for seg in segments:
        new_seg = dict(seg)
        new_seg["start"] = _shift_clamp(seg["start"])
        new_seg["end"] = _shift_clamp(seg["end"])
        words = seg.get("words")
        if words:
            new_words = []
            for w in words:
                new_w = dict(w)
                if "start" in w:
                    new_w["start"] = _shift_clamp(w["start"])
                if "end" in w:
                    new_w["end"] = _shift_clamp(w["end"])
                new_words.append(new_w)
            new_seg["words"] = new_words
        out.append(new_seg)
    return out


def make_snippet_id(original_id: str, start_sec: float, end_sec: float) -> str:
    """Snippet id format: ``<id>_{st:.3f}_{en:.3f}``.

    Millisecond precision avoids id collisions between adjacent short
    snippets that would round to the same value at 2-decimal precision.
    """
    return f"{original_id}_{start_sec:.3f}_{end_sec:.3f}"


def histogram_30s(durations: list[float]) -> dict[str, int]:
    """Bucket snippet durations into fixed-width 30s bins.

    Returns an ordered mapping ``{"0-30": n, "30-60": n, ...}`` covering
    every bin from 0 up to and including the bin containing the longest
    duration.  Empty input returns an empty dict.
    """
    if not durations:
        return {}
    max_idx = max(int(d // _HISTOGRAM_BIN_WIDTH_SEC) for d in durations)
    counts: list[int] = [0] * (max_idx + 1)
    for d in durations:
        idx = int(d // _HISTOGRAM_BIN_WIDTH_SEC)
        counts[idx] += 1
    bin_w = int(_HISTOGRAM_BIN_WIDTH_SEC)
    return {f"{i * bin_w}-{(i + 1) * bin_w}": counts[i] for i in range(max_idx + 1)}


def _resolve_audio_path(audio_dir: str, value: str) -> str:
    """Resolve a manifest's audio path against ``audio_dir`` by basename.

    The pipeline accepts a directory of audio files plus a JSONL whose
    ``audio_filepath`` may be relative (``./foo.m4a``) or absolute; we
    always re-anchor to ``audio_dir`` using the basename so manifests
    stay portable across hosts.
    """
    return os.path.join(audio_dir, os.path.basename(value))


def _is_origin_stub(task: AudioTask) -> bool:
    """A stub task from the extractor that carries per-original metrics for an
    input that produced zero snippets.  Has no snippet_id."""
    return task.data.get("snippet_id") is None


# ----------------------------------------------------------------------
# Stage 1: read JSONL manifest, fan out into AudioTasks
# ----------------------------------------------------------------------


@dataclass
class ReadLongFormManifestStage(ProcessingStage[_EmptyTask, AudioTask]):
    """Read a JSONL manifest of long-form audios; emit one AudioTask per row.

    Each line in ``input_manifest`` is parsed as JSON and re-emitted as
    an ``AudioTask`` whose ``data`` is the parsed dict with its audio
    path re-anchored to ``audio_dir``.

    This is the entry-point ``_EmptyTask -> list[AudioTask]`` fan-out
    stage following the same pattern as
    ``CreateInitialManifestReadSpeechStage``.

    Args:
        input_manifest: Path to the JSONL file.
        audio_dir: Directory containing the source audio files; the row's
            ``audio_filepath`` value is replaced with
            ``audio_dir / basename(audio_filepath)``.
        audio_filepath_key: JSONL field that holds the path to the audio
            file (default ``"audio_filepath"``).
        dataset_name: Optional dataset tag stamped on emitted tasks.
    """

    input_manifest: str = ""
    audio_dir: str = ""
    audio_filepath_key: str = "audio_filepath"
    dataset_name: str = "long_form_audio"

    name: str = "ReadLongFormManifest"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self) -> None:
        if not self.input_manifest:
            msg = "input_manifest is required for ReadLongFormManifestStage"
            raise ValueError(msg)
        if not self.audio_dir:
            msg = "audio_dir is required for ReadLongFormManifestStage"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key, "id", "segments"]

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"max_workers_per_node": 1, "num_workers": 1}

    def process(self, _: _EmptyTask) -> list[AudioTask]:
        t0 = time.perf_counter()
        if not os.path.isfile(self.input_manifest):
            msg = f"Manifest not found: {self.input_manifest}"
            raise FileNotFoundError(msg)

        tasks: list[AudioTask] = []
        with open(self.input_manifest, encoding="utf-8") as f:
            for lineno, raw in enumerate(f, 1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.error(f"[{self.name}] line {lineno}: invalid JSON ({e}); skipping")
                    continue

                original_path = entry.get(self.audio_filepath_key)
                if not original_path:
                    logger.warning(f"[{self.name}] line {lineno}: missing {self.audio_filepath_key!r}; skipping")
                    continue
                entry[self.audio_filepath_key] = _resolve_audio_path(self.audio_dir, original_path)

                tasks.append(
                    AudioTask(
                        task_id=f"{entry.get('id', f'line_{lineno}')}",
                        dataset_name=self.dataset_name,
                        data=entry,
                        filepath_key=self.audio_filepath_key,
                    )
                )

        self._log_metrics(
            {
                "manifest_load_time": time.perf_counter() - t0,
                "manifest_rows": float(len(tasks)),
            }
        )
        logger.info(f"[{self.name}] loaded {len(tasks)} rows from {self.input_manifest}")
        return tasks


# ----------------------------------------------------------------------
# Stage 2: drop empty + overlapping segments
# ----------------------------------------------------------------------


@dataclass
class OverlapFilterStage(ProcessingStage[AudioTask, AudioTask]):
    """Drop empty segments and overlapping segment pairs.

    First filters segments that have neither text nor words.  Then drops
    every segment that overlaps any other surviving segment, where
    "overlap" means intersection ≥ ``min_overlap_sec`` OR one fully
    contains the other.  Both members of an overlapping pair are
    discarded -- this version keeps no overlap-resolution heuristic.

    Per-original counters are stamped onto ``task._metadata`` under the
    ``pretrain_long_form`` key so the final aggregator can build a
    per-original metrics breakdown.
    """

    min_overlap_sec: float = 0.5

    name: str = "OverlapFilter"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], ["segments"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], ["segments"]

    def process(self, task: AudioTask) -> AudioTask:
        t0 = time.perf_counter()
        segments = list(task.data.get("segments") or [])
        original_count = len(segments)
        # Wall-clock span of the source recording: last segment's end minus
        # first segment's start. Comparable to `out_duration_sec` (which is
        # also a span, including inter-segment silences) so the input/output
        # totals can be diffed meaningfully. min/max instead of [-1].end /
        # [0].start because the input JSONL is not guaranteed to be sorted.
        original_duration = (
            max(s["end"] for s in segments) - min(s["start"] for s in segments) if segments else 0.0
        )

        kept_after_empty, dropped_empty = filter_empty_segments(segments)
        kept_after_empty.sort(key=lambda s: (s["start"], s["end"]))
        bad = find_overlapping_indices(kept_after_empty, self.min_overlap_sec)
        kept = [s for i, s in enumerate(kept_after_empty) if i not in bad]
        dropped_overlap = len(bad)

        task.data["segments"] = kept

        meta = task._metadata.setdefault(_PRETRAIN_META_KEY, {})
        meta["original_seg_count"] = original_count
        meta["original_seg_duration"] = float(original_duration)
        meta["dropped_empty"] = dropped_empty
        meta["dropped_overlap"] = dropped_overlap
        meta["kept_after_filter_count"] = len(kept)

        self._log_metrics(
            {
                "overlap_filter_time": time.perf_counter() - t0,
                "input_segments": float(original_count),
                "dropped_empty": float(dropped_empty),
                "dropped_overlap": float(dropped_overlap),
                "output_segments": float(len(kept)),
            }
        )
        return task


# ----------------------------------------------------------------------
# Stage 3: plan snippet boundaries (no I/O)
# ----------------------------------------------------------------------


@dataclass
class SnippetCutPlannerStage(ProcessingStage[AudioTask, AudioTask]):
    """Compute snippet cut boundaries for one input audio.

    Pure planning -- no audio I/O.  Produces a list of snippet specs
    each holding ``start``, ``end`` (absolute seconds in the source
    audio) and the contained ``segments``.  The plan is stored under
    ``task.data["_snippet_plan"]`` for the downstream extractor to act
    on.  Drop counts (``too_long``, ``too_short``, ``no_text``) are
    written to ``task._metadata['pretrain_long_form']``.
    """

    max_duration_sec: float = 30.0
    min_duration_sec: float = 0.5
    # Two segments separated by more than this many seconds of silence are
    # assumed to belong to semantically distinct conversations and are
    # never grouped into the same snippet (see plan_snippets docstring).
    max_segment_gap_in_snippet: float = 30.0

    name: str = "SnippetCutPlanner"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self) -> None:
        if self.max_duration_sec <= 0:
            msg = "max_duration_sec must be > 0"
            raise ValueError(msg)
        if self.min_duration_sec < 0:
            msg = "min_duration_sec must be >= 0"
            raise ValueError(msg)
        if self.min_duration_sec > self.max_duration_sec:
            msg = "min_duration_sec must be <= max_duration_sec"
            raise ValueError(msg)
        if self.max_segment_gap_in_snippet < 0:
            msg = "max_segment_gap_in_snippet must be >= 0"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], ["segments"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [_PLAN_DATA_KEY]

    def process(self, task: AudioTask) -> AudioTask:
        t0 = time.perf_counter()
        segments = list(task.data.get("segments") or [])
        snippets, drop_counts = plan_snippets(
            segments,
            self.max_duration_sec,
            self.min_duration_sec,
            self.max_segment_gap_in_snippet,
        )
        task.data[_PLAN_DATA_KEY] = snippets

        meta = task._metadata.setdefault(_PRETRAIN_META_KEY, {})
        meta["dropped_too_long"] = drop_counts["too_long"]
        meta["dropped_too_short"] = drop_counts["too_short"]
        meta["dropped_no_text"] = drop_counts["no_text"]
        meta["planned_snippets"] = len(snippets)

        self._log_metrics(
            {
                "plan_time": time.perf_counter() - t0,
                "planned_snippets": float(len(snippets)),
                "dropped_too_long": float(drop_counts["too_long"]),
                "dropped_too_short": float(drop_counts["too_short"]),
                "dropped_no_text": float(drop_counts["no_text"]),
            }
        )
        if not snippets:
            logger.warning(f"[{self.name}] {task.task_id}: planner produced 0 snippets (drop counts={drop_counts})")
        return task


# ----------------------------------------------------------------------
# Stage 4: read source audio, extract snippets, write files
# ----------------------------------------------------------------------


@dataclass
class SnippetExtractionStage(ProcessingStage[AudioTask, AudioTask]):
    """Slice the source audio per snippet plan, mono-resample, and write.

    For each planned snippet:

    1. Read just the slice ``[start, end]`` from the source file.
    2. Channel-average to mono if the source has > 1 channel.
    3. Resample to ``target_sample_rate`` using torchaudio if the source
       rate differs.
    4. Write to ``output_dir/<snippet_id>.<output_format>``.
    5. Emit one ``AudioTask`` per snippet with the source row's metadata
       carried over (minus ``alignment``), the new ``snippet_id``,
       updated ``audio_filepath`` / ``duration``, and segments
       relativized to the snippet start.

    If the input produced zero snippets, a single "stub" ``AudioTask``
    is emitted (``snippet_id=None``, no audio file written) so that
    per-original metrics can still flow to the aggregator.

    Dry-run mode (``dry_run=True``): skips steps 1-4 entirely (no
    ``soundfile`` reads, no resampling, no file writes), and step 5 uses
    the planned ``end - start`` as the snippet ``duration`` instead of
    the post-resample frame count.  The emitted ``audio_filepath`` still
    points at where the file *would* have been written, but the file
    does not exist on disk.  Useful for quickly previewing the manifest
    and metrics on real data before committing to a full run.
    """

    output_dir: str = ""
    target_sample_rate: int = 16000
    output_format: str = "flac"
    audio_filepath_key: str = "audio_filepath"
    dry_run: bool = False

    name: str = "SnippetExtraction"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self) -> None:
        if not self.output_dir:
            msg = "output_dir is required for SnippetExtractionStage"
            raise ValueError(msg)
        if self.output_format not in _SOUNDFILE_SUBTYPES:
            msg = f"output_format must be one of {sorted(_SOUNDFILE_SUBTYPES)}, got {self.output_format!r}"
            raise ValueError(msg)
        if self.target_sample_rate <= 0:
            msg = "target_sample_rate must be > 0"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key, _PLAN_DATA_KEY]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key, "snippet_id", "duration", "segments"]

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    def process(self, task: AudioTask) -> list[AudioTask]:
        t0 = time.perf_counter()
        # Read the plan without mutating the input task -- Xenna may preempt
        # and replay the same task through this stage; popping would leave the
        # retried task without a plan and fail validate_input on retry.
        plan: list[dict] = list(task.data.get(_PLAN_DATA_KEY) or [])
        if not plan:
            return [self._make_stub_task(task)]

        original_id = str(task.data.get("id") or task.task_id)

        if self.dry_run:
            outputs = self._dry_run_emit(task, plan, original_id)
        else:
            outputs = self._extract_emit(task, plan, original_id)

        total_dur = 0.0 if _is_origin_stub(outputs[0]) else sum(t.data["duration"] for t in outputs)
        self._log_metrics(
            {
                "extract_time": time.perf_counter() - t0,
                "snippets_written": float(len(outputs) if not _is_origin_stub(outputs[0]) else 0),
                "snippets_total_duration": float(total_dur),
            }
        )
        return outputs

    def _extract_emit(
        self, task: AudioTask, plan: list[dict], original_id: str
    ) -> list[AudioTask]:
        import soundfile as sf

        source_path = task.data.get(self.audio_filepath_key)
        if not source_path or not os.path.exists(source_path):
            logger.error(
                f"[{self.name}] {task.task_id}: source audio missing at {source_path!r}; "
                f"emitting stub for {len(plan)} planned snippets"
            )
            return [self._make_stub_task(task)]
        try:
            info = sf.info(source_path)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{self.name}] {task.task_id}: cannot read header of {source_path}: {e}")
            return [self._make_stub_task(task)]

        outputs: list[AudioTask] = []
        for snippet in plan:
            emitted = self._extract_one_snippet(task, snippet, source_path, info, original_id)
            if emitted is not None:
                outputs.append(emitted)
        if not outputs:
            outputs.append(self._make_stub_task(task))
        return outputs

    def _dry_run_emit(
        self, task: AudioTask, plan: list[dict], original_id: str
    ) -> list[AudioTask]:
        """Emit snippet metadata only, without reading or writing audio.

        The ``audio_filepath`` of each emitted task points at where the
        snippet *would* have been written; that file does not exist on
        disk in dry-run mode.  Snippet ``duration`` is the planned
        ``end - start`` (vs. the resampled-frame-count duration the real
        path would compute -- the difference is at most one frame at
        ``target_sample_rate``).
        """
        outputs: list[AudioTask] = []
        for snippet in plan:
            start_sec = float(snippet["start"])
            end_sec = float(snippet["end"])
            snippet_id = make_snippet_id(original_id, start_sec, end_sec)
            out_path = os.path.join(self.output_dir, f"{snippet_id}.{self.output_format}")
            outputs.append(
                self._make_snippet_task(
                    task=task,
                    snippet=snippet,
                    snippet_id=snippet_id,
                    out_path=out_path,
                    duration=end_sec - start_sec,
                )
            )
        if not outputs:
            outputs.append(self._make_stub_task(task))
        return outputs

    def _extract_one_snippet(
        self,
        task: AudioTask,
        snippet: dict,
        source_path: str,
        info: Any,  # noqa: ANN401  (soundfile._SoundFileInfo)
        original_id: str,
    ) -> AudioTask | None:
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio.functional as taf

        source_sr = info.samplerate
        start_sec = float(snippet["start"])
        end_sec = float(snippet["end"])
        start_frame = max(0, math.floor(start_sec * source_sr))
        end_frame = min(info.frames, math.ceil(end_sec * source_sr))
        if end_frame <= start_frame:
            logger.warning(
                f"[{self.name}] {task.task_id}: empty frame range [{start_frame}, {end_frame}); skipping snippet"
            )
            return None

        try:
            audio, _ = sf.read(source_path, start=start_frame, stop=end_frame, dtype="float32", always_2d=True)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{self.name}] {task.task_id}: failed to read slice [{start_sec:.2f}, {end_sec:.2f}]: {e}")
            return None

        wave = torch.from_numpy(np.ascontiguousarray(audio.T))
        if wave.shape[0] > 1:
            wave = wave.mean(dim=0, keepdim=True)
        if source_sr != self.target_sample_rate:
            wave = taf.resample(wave, source_sr, self.target_sample_rate)
        mono = wave.squeeze(0).contiguous().numpy()
        actual_duration = mono.shape[0] / float(self.target_sample_rate)

        snippet_id = make_snippet_id(original_id, start_sec, end_sec)
        out_path = os.path.join(self.output_dir, f"{snippet_id}.{self.output_format}")
        try:
            sf.write(
                out_path,
                mono,
                self.target_sample_rate,
                subtype=_SOUNDFILE_SUBTYPES[self.output_format],
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{self.name}] failed to write {out_path}: {e}")
            return None

        return self._make_snippet_task(
            task=task,
            snippet=snippet,
            snippet_id=snippet_id,
            out_path=out_path,
            duration=actual_duration,
        )

    def _make_snippet_task(
        self,
        task: AudioTask,
        snippet: dict,
        snippet_id: str,
        out_path: str,
        duration: float,
    ) -> AudioTask:
        new_data = dict(task.data)
        new_data.pop("alignment", None)
        new_data.pop(_PLAN_DATA_KEY, None)
        # Drop source-file-specific fields that don't apply to the snippet.
        new_data.pop("audio_size", None)
        new_data.pop("resampled_audio_filepath", None)
        new_data["snippet_id"] = snippet_id
        new_data[self.audio_filepath_key] = out_path
        new_data["duration"] = duration
        # Update audio-property fields only if the source row had them.
        if "actual_duration" in new_data:
            new_data["actual_duration"] = duration
        if "proposed_duration" in new_data:
            new_data["proposed_duration"] = duration
        if "audio_sample_rate" in new_data:
            new_data["audio_sample_rate"] = self.target_sample_rate
        if "audio_num_channels" in new_data:
            new_data["audio_num_channels"] = 1
        # Reset to "" — a downstream pipeline is expected to set this correctly.
        if "swift_audio_filepath" in new_data:
            new_data["swift_audio_filepath"] = ""
        new_data["segments"] = relativize_segments(
            snippet["segments"], snippet["start"], snippet["end"]
        )
        if "text" in new_data:
            new_data["text"] = " ".join(_segment_text(s) for s in snippet["segments"]).strip()
        return AudioTask(
            task_id=f"{task.task_id}::{snippet_id}",
            dataset_name=task.dataset_name,
            data=new_data,
            filepath_key=self.audio_filepath_key,
            _metadata=copy.deepcopy(task._metadata),
            _stage_perf=list(task._stage_perf),
        )

    def _make_stub_task(self, task: AudioTask) -> AudioTask:
        original_id = task.data.get("id")
        stub_data: dict = {
            "id": original_id,
            "snippet_id": None,
            self.audio_filepath_key: None,
            "duration": 0.0,
            "segments": [],
        }
        return AudioTask(
            task_id=f"{task.task_id}::stub",
            dataset_name=task.dataset_name,
            data=stub_data,
            _metadata=copy.deepcopy(task._metadata),
            _stage_perf=list(task._stage_perf),
        )


# ----------------------------------------------------------------------
# Stage 5: append snippet records to a JSONL manifest
# ----------------------------------------------------------------------


@dataclass
class SnippetManifestWriterStage(ProcessingStage[AudioTask, AudioTask]):
    """Append each (non-stub) snippet's ``data`` as a JSONL line.

    Single-replica writer; the file is truncated once on driver setup
    so reruns produce a clean output.  Origin-stub tasks (no
    ``snippet_id``) are passed through unchanged so the metrics
    aggregator can still see them.
    """

    output_path: str = ""

    name: str = "SnippetManifestWriter"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self) -> None:
        if not self.output_path:
            msg = "output_path is required for SnippetManifestWriterStage"
            raise ValueError(msg)
        self._shard_path: str | None = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        parent = os.path.dirname(self.output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        parent = os.path.dirname(self.output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # Each replica writes its own shard; finalize_audio_pretrain_outputs
        # merges them after pipeline.run().
        self._shard_path = _make_shard_path(self.output_path, _MANIFEST_SHARD_EXT)
        logger.info(f"[{self.name}] writing manifest shard to {self._shard_path}")

    def process(self, task: AudioTask) -> AudioTask:
        if not _is_origin_stub(task) and self._shard_path is not None:
            with open(self._shard_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(task.data, ensure_ascii=False) + "\n")
        return task


# ----------------------------------------------------------------------
# Stage 6: aggregate metrics across all snippets/originals
# ----------------------------------------------------------------------


@dataclass
class PretrainMetricsAggregatorStage(ProcessingStage[AudioTask, AudioTask]):
    """Per-replica metrics aggregator.

    Each ``process()`` call appends one JSONL record to a per-replica
    shard.  ``finalize_audio_pretrain_outputs`` reads every shard after
    ``pipeline.run()`` returns and aggregates the records into the final
    summary JSON.

    The per-task append shape (vs. accumulating in memory and flushing in
    ``teardown()``) is required for correctness under Xenna: Xenna kills
    stage actors with ``ray.kill()`` and never invokes any teardown hook,
    so an in-memory-only aggregator silently produces an empty summary.

    Record schema (one line per task seen):

    * ``id`` -- original audio id
    * ``in_segments``, ``in_duration_sec``, ``dropped`` -- per-original
      input-side counters; written on every record (identical across
      records for the same original); the merger keeps the first.
    * ``is_stub`` -- True iff this is the extractor's zero-snippet stub.
    * ``out_segments``, ``out_duration_sec`` -- this snippet's
      contribution; zero for stubs.

    The merger sums ``out_*`` across non-stub records per id and counts
    them as ``out_snippets``.
    """

    output_path: str = ""

    name: str = "PretrainMetricsAggregator"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self) -> None:
        if not self.output_path:
            msg = "output_path is required for PretrainMetricsAggregatorStage"
            raise ValueError(msg)
        self._shard_path: str | None = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        parent = os.path.dirname(self.output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        parent = os.path.dirname(self.output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._shard_path = _make_shard_path(self.output_path, _METRICS_SHARD_EXT)
        logger.info(f"[{self.name}] writing metrics shard to {self._shard_path}")

    def process(self, task: AudioTask) -> AudioTask:
        if self._shard_path is None:
            return task
        original_id = str(task.data.get("id") or "")
        if not original_id:
            return task
        meta = task._metadata.get(_PRETRAIN_META_KEY, {})
        is_stub = _is_origin_stub(task)
        record = {
            "id": original_id,
            "in_segments": int(meta.get("original_seg_count", 0)),
            "in_duration_sec": float(meta.get("original_seg_duration", 0.0)),
            "dropped": {
                "empty": int(meta.get("dropped_empty", 0)),
                "overlap": int(meta.get("dropped_overlap", 0)),
                "too_long": int(meta.get("dropped_too_long", 0)),
                "too_short": int(meta.get("dropped_too_short", 0)),
                "no_text": int(meta.get("dropped_no_text", 0)),
            },
            "is_stub": is_stub,
            "out_segments": 0 if is_stub else len(task.data.get("segments") or []),
            "out_duration_sec": 0.0 if is_stub else float(task.data.get("duration", 0.0)),
        }
        with open(self._shard_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return task


# ----------------------------------------------------------------------
# Post-pipeline shard merging
# ----------------------------------------------------------------------


def prepare_audio_pretrain_outputs(output_manifest_path: str, metrics_path: str) -> None:
    """Delete any pre-existing shards from prior runs.

    Call this once on the driver, BEFORE ``pipeline.run()``.  Multi-worker
    backends would race on cleanup if we did it inside a stage's
    ``setup()``, so we keep cleanup driver-only.
    """
    n_man = _delete_shards(output_manifest_path, _MANIFEST_SHARD_EXT)
    n_met = _delete_shards(metrics_path, _METRICS_SHARD_EXT)
    if n_man or n_met:
        logger.info(
            f"prepare_audio_pretrain_outputs: removed {n_man} stale manifest "
            f"shard(s) and {n_met} stale metrics shard(s) from prior runs"
        )


def finalize_audio_pretrain_outputs(output_manifest_path: str, metrics_path: str) -> None:
    """Merge per-worker shards into the final manifest and metrics JSON.

    Call once on the driver, AFTER ``pipeline.run()`` returns
    successfully.  Reads all manifest + metrics shards written by the
    writer / aggregator stages, concatenates / combines them, writes
    the final user-facing files at the user-provided paths, and removes
    the shards.
    """
    _merge_manifest_shards(output_manifest_path)
    _merge_metrics_shards(metrics_path)


def _merge_manifest_shards(output_path: str) -> None:
    shards = _glob_shards(output_path, _MANIFEST_SHARD_EXT)
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for s in shards:
            with open(s, encoding="utf-8") as f:
                shutil.copyfileobj(f, out)
    for s in shards:
        try:
            os.remove(s)
        except OSError as e:
            logger.warning(f"failed to remove manifest shard {s}: {e}")
    logger.info(f"merged {len(shards)} manifest shard(s) into {output_path}")


def _merge_metrics_shards(metrics_path: str) -> None:
    shards = _glob_shards(metrics_path, _METRICS_SHARD_EXT)
    per_original: dict[str, dict[str, Any]] = {}
    durations: list[float] = []
    for s in shards:
        with open(s, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                r = json.loads(line)
                pid = r["id"]
                entry = per_original.get(pid)
                if entry is None:
                    # First record wins for input-side fields. They're
                    # identical across every record for a given original
                    # (they come from `_metadata`, copied through fan-out).
                    entry = {
                        "id": pid,
                        "in_segments": int(r.get("in_segments", 0)),
                        "in_duration_sec": float(r.get("in_duration_sec", 0.0)),
                        "dropped": dict(r.get("dropped") or {}),
                        "out_snippets": 0,
                        "out_segments": 0,
                        "out_duration_sec": 0.0,
                    }
                    per_original[pid] = entry
                if not r.get("is_stub", False):
                    entry["out_snippets"] += 1
                    entry["out_segments"] += int(r.get("out_segments", 0))
                    entry["out_duration_sec"] += float(r.get("out_duration_sec", 0.0))
                    durations.append(float(r.get("out_duration_sec", 0.0)))

    summary = _build_final_summary(per_original, durations)
    parent = os.path.dirname(metrics_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    for s in shards:
        try:
            os.remove(s)
        except OSError as e:
            logger.warning(f"failed to remove metrics shard {s}: {e}")
    logger.info(f"merged {len(shards)} metrics shard(s) into {metrics_path}")


def _build_final_summary(
    per_original: dict[str, dict[str, Any]], durations: list[float]
) -> dict[str, Any]:
    totals_dropped: dict[str, int] = defaultdict(int)
    in_segments = 0
    in_duration = 0.0
    out_snippets = 0
    out_segments = 0
    out_duration = 0.0
    for entry in per_original.values():
        in_segments += int(entry.get("in_segments", 0))
        in_duration += float(entry.get("in_duration_sec", 0.0))
        out_snippets += int(entry.get("out_snippets", 0))
        out_segments += int(entry.get("out_segments", 0))
        out_duration += float(entry.get("out_duration_sec", 0.0))
        for k, v in (entry.get("dropped") or {}).items():
            totals_dropped[k] += int(v)

    return {
        "num_input_audios": len(per_original),
        "num_output_snippets": out_snippets,
        "input_total_segments": in_segments,
        "input_total_duration_sec": round(in_duration, 3),
        "output_total_segments": out_segments,
        "output_total_duration_sec": round(out_duration, 3),
        "dropped": dict(totals_dropped),
        "snippet_duration_histogram_30s": histogram_30s(durations),
        "per_original": list(per_original.values()),
    }
