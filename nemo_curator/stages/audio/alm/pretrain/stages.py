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
import io
import json
import math
import os
import tarfile
import time
import uuid
from collections import Counter, defaultdict
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
# Per-replica audio tar shards; merged into the user-facing tar in
# `finalize_audio_pretrain_outputs`. The extractor holds an open
# `TarFile` on the instance for the worker's lifetime (re-opening in
# append mode for every snippet would force `tarfile` to re-scan the
# archive to find the end-of-archive marker, making writes O(n^2)). A
# worker killed by `ray.kill()` mid-process won't write the trailing
# zero-blocks, but `tarfile.open(..., "r")` reads such truncated
# archives by walking valid headers until it hits EOF, so the merger
# tolerates them.
_TAR_SHARD_EXT = "tar"
# Cap on how many filtered snippet texts are retained for the metrics summary.
# Bounds per-source metadata, shard size, and the final summary list size --
# the same constant is applied per-source in the filter stage and globally in
# the shard merger. Large enough to be diagnostic but small enough that the
# metrics JSON stays human-readable on pathological inputs.
_MAX_FILTERED_TEXT_EXAMPLES = 1000


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
    """Return the segment's ``text`` field, stripped, or an empty string.

    The pipeline used to also consult ``text_ITN`` as a higher-priority
    source, but ``text_ITN`` is unreliable in real upstream data (often
    empty or stale even when ``text`` is populated), so the helper now
    reads ``text`` exclusively.  ``text_ITN`` is still carried through
    on output segments unchanged via the shallow copy in
    ``relativize_segments``; it just no longer drives any decision.
    """
    return (seg.get("text") or "").strip()


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

    Precondition: ``segments`` must be non-overlapping (sorted by ``start``
    with each ``end <= next.start``).  ``OverlapFilterStage`` guarantees
    this upstream in the pipeline.  If overlapping segments are passed in,
    ``gap`` becomes negative and the gap constraint is silently bypassed,
    grouping content that should belong to separate snippets.
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
    """Snippet id format: ``<id>-<st_int>_<st_ms>-<en_int>_<en_ms>``.

    Millisecond precision avoids id collisions between adjacent short
    snippets that would round to the same value at 2-decimal precision.

    The id is intentionally free of any ``.`` character so that the
    resulting filename ``<snippet_id>.<ext>`` (e.g. ``.flac``) survives
    WebDataset-style grouping, which uses the first ``.`` after the
    sample basename as the boundary between sample key and extensions.
    A snippet id like ``X_11.708_13.970`` would otherwise be parsed by
    WebDataset as a multi-piece compound key with extensions ``708``,
    ``970``, ``flac``. Using ``-`` as the field separator and ``_`` as
    the decimal mark keeps both human readability and tar-friendliness.
    """
    start_str = f"{start_sec:.3f}".replace(".", "_")
    end_str = f"{end_sec:.3f}".replace(".", "_")
    return f"{original_id}-{start_str}-{end_str}"


def histogram_30s(durations: list[float]) -> dict[str, int]:
    """Bucket snippet durations into fixed-width 30s bins.

    Returns an ordered mapping ``{"0-30": n, "30-60": n, ...}`` covering
    every bin from 0 up to and including the bin containing the longest
    duration.  Empty input returns an empty dict.

    Bins are kept contiguous from 0 by design: a leading bin may have a
    count of 0 (e.g. a single 30.0s snippet lands in ``30-60`` and yields
    ``{"0-30": 0, "30-60": 1}``).  Downstream consumers should treat the
    output as a dense histogram, not a sparse one.
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


# ----------------------------------------------------------------------
# Repetition filter helpers (pure, no HF / no Ray)
# ----------------------------------------------------------------------


def _count_ngrams(token_ids: list[int], n: int) -> Counter[tuple[int, ...]]:
    """Count contiguous n-gram frequencies in a token id sequence."""
    if n <= 0 or len(token_ids) < n:
        return Counter()
    return Counter(tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1))


def _find_offending_ngrams(
    counts: Counter[tuple[int, ...]], max_count: int
) -> set[tuple[int, ...]]:
    """Return n-grams whose frequency strictly exceeds ``max_count``."""
    return {ng for ng, c in counts.items() if c > max_count}


def _locate_ngram_char_ranges(
    token_ids: list[int],
    offsets: list[tuple[int, int]],
    offending: set[tuple[int, ...]],
    n: int,
) -> list[tuple[int, int]]:
    """Char-range spans for every position where an offending n-gram starts."""
    if not offending or len(token_ids) < n:
        return []
    ranges: list[tuple[int, int]] = []
    for i in range(len(token_ids) - n + 1):
        ng = tuple(token_ids[i : i + n])
        if ng in offending:
            start = offsets[i][0]
            end = offsets[i + n - 1][1]
            if end > start:
                ranges.append((start, end))
    return ranges


def _merge_char_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or touching char ranges; input may be unsorted."""
    if not ranges:
        return []
    sorted_ranges = sorted(ranges)
    merged: list[tuple[int, int]] = [sorted_ranges[0]]
    for start, end in sorted_ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _format_red(text: str, ranges: list[tuple[int, int]]) -> str:
    """Wrap each char range in loguru ``<red>...</red>`` markup.

    Literal ``<`` in the surrounding text is escaped to ``\\<`` so
    loguru's tag parser leaves it alone.  ``ranges`` must be merged and
    sorted (use :func:`_merge_char_ranges`).
    """
    if not ranges:
        return text.replace("<", r"\<")
    pieces: list[str] = []
    cursor = 0
    for start, end in ranges:
        if start > cursor:
            pieces.append(text[cursor:start].replace("<", r"\<"))
        pieces.append("<red>")
        pieces.append(text[start:end].replace("<", r"\<"))
        pieces.append("</red>")
        cursor = end
    if cursor < len(text):
        pieces.append(text[cursor:].replace("<", r"\<"))
    return "".join(pieces)


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

    max_duration_sec: float = 600.0
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
# Stage 4: filter snippets whose joined text shows n-gram repetition
# ----------------------------------------------------------------------


@dataclass
class SnippetRepetitionFilterStage(ProcessingStage[AudioTask, AudioTask]):
    """Drop planned snippets whose text shows suspicious n-gram repetition.

    Whisper-style ASR sometimes degenerates into repeating the same
    short phrase for many seconds; the resulting transcript looks fine
    locally but contains the same n-gram of token ids dozens of times.
    Such snippets are unsuitable for pretraining.

    For every planned snippet (read from ``task.data["_snippet_plan"]``)
    we join the segment ``text`` fields with the same formula the
    extractor uses, tokenize with the configured HuggingFace fast
    tokenizer, count n-gram frequencies over the resulting token-id
    sequence, and drop the snippet if any n-gram appears strictly more
    than ``ngram_max_count`` times.  Filtered snippets are logged with
    the offending occurrences highlighted in red (loguru color tags).

    Snippets whose tokenized text has fewer than ``ngram_n`` tokens are
    kept unchanged (no n-grams to evaluate; the planner already enforces
    a minimum-duration threshold).

    Sits between :class:`SnippetCutPlannerStage` and
    :class:`SnippetExtractionStage` so filtered snippets never incur
    audio decode / resample / file-write cost.
    """

    tokenizer_path: str = ""
    ngram_n: int = 10
    ngram_max_count: int = 3

    name: str = "SnippetRepetitionFilter"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self) -> None:
        if not self.tokenizer_path:
            msg = "tokenizer_path is required for SnippetRepetitionFilterStage"
            raise ValueError(msg)
        if self.ngram_n < 1:
            msg = "ngram_n must be >= 1"
            raise ValueError(msg)
        if self.ngram_max_count < 1:
            msg = "ngram_max_count must be >= 1"
            raise ValueError(msg)
        self._tokenizer: Any = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [_PLAN_DATA_KEY]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [_PLAN_DATA_KEY]

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=True)
        if not getattr(self._tokenizer, "is_fast", False):
            msg = (
                f"SnippetRepetitionFilterStage requires a fast tokenizer with offset mapping; "
                f"loaded tokenizer at {self.tokenizer_path!r} is not fast"
            )
            raise RuntimeError(msg)
        logger.info(
            f"[{self.name}] loaded tokenizer from {self.tokenizer_path} "
            f"(n={self.ngram_n}, max_count={self.ngram_max_count})"
        )

    def process(self, task: AudioTask) -> AudioTask:
        t0 = time.perf_counter()
        plan: list[dict] = list(task.data.get(_PLAN_DATA_KEY) or [])
        kept: list[dict] = []
        dropped_texts: list[str] = []
        for snippet in plan:
            text = " ".join(_segment_text(s) for s in snippet["segments"]).strip()
            if self._snippet_is_repetitive(text, snippet, task.task_id):
                dropped_texts.append(text)
            else:
                kept.append(snippet)

        task.data[_PLAN_DATA_KEY] = kept

        meta = task._metadata.setdefault(_PRETRAIN_META_KEY, {})
        meta["dropped_repetition"] = len(dropped_texts)
        meta["kept_after_repetition_filter"] = len(kept)
        # Override the planner's count so downstream consumers (and
        # logging) see the post-filter snippet count.
        meta["planned_snippets"] = len(kept)
        # Retain up to N example texts per source for the metrics summary;
        # the shard merger applies a second global cap of the same size.
        # Assigned (not appended) so re-execution under Ray Data fan-out is
        # idempotent -- the same source's plan can flow through this stage
        # more than once without accumulating duplicate texts.
        meta["filtered_repetition_texts"] = dropped_texts[:_MAX_FILTERED_TEXT_EXAMPLES]

        self._log_metrics(
            {
                "repetition_filter_time": time.perf_counter() - t0,
                "snippets_scanned": float(len(plan)),
                "snippets_filtered_repetition": float(len(dropped_texts)),
            }
        )
        return task

    def _snippet_is_repetitive(self, text: str, snippet: dict, task_id: str) -> bool:
        """Tokenize ``text`` and decide whether to drop the snippet.

        On drop, emit a colorized warning showing the offending n-gram
        occurrences highlighted in red.
        """
        if not text:
            return False
        encoding = self._tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        token_ids: list[int] = list(encoding["input_ids"])
        offsets: list[tuple[int, int]] = [tuple(o) for o in encoding["offset_mapping"]]
        if len(token_ids) < self.ngram_n:
            return False
        counts = _count_ngrams(token_ids, self.ngram_n)
        offending = _find_offending_ngrams(counts, self.ngram_max_count)
        if not offending:
            return False
        ranges = _merge_char_ranges(
            _locate_ngram_char_ranges(token_ids, offsets, offending, self.ngram_n)
        )
        colorized = _format_red(text, ranges)
        worst_count = max(counts[ng] for ng in offending)
        logger.opt(colors=True).warning(
            f"[{self.name}] {task_id}: dropping snippet "
            f"[{snippet.get('start', 0):.2f}, {snippet.get('end', 0):.2f}] "
            f"(n={self.ngram_n}, offending_ngrams={len(offending)}, max_count={worst_count}): "
            f"{colorized}"
        )
        return True


# ----------------------------------------------------------------------
# Stage 5: read source audio, extract snippets, write files
# ----------------------------------------------------------------------


@dataclass
class SnippetExtractionStage(ProcessingStage[AudioTask, AudioTask]):
    """Slice the source audio per snippet plan, mono-resample, and write into a tar.

    For each planned snippet:

    1. Read just the slice ``[start, end]`` from the source file.
    2. Channel-average to mono if the source has > 1 channel.
    3. Resample to ``target_sample_rate`` using torchaudio if the source
       rate differs.
    4. Encode the mono waveform in-memory (via ``soundfile`` to a
       ``BytesIO``) and append it as ``<snippet_id>.<output_format>`` to
       this replica's tar shard (``output_audio_tar_path.shard-...``);
       all replicas' shards are merged into ``output_audio_tar_path`` by
       :func:`finalize_audio_pretrain_outputs`.
    5. Emit one ``AudioTask`` per snippet with the source row's metadata
       carried over (minus ``alignment``), the new ``snippet_id``,
       ``audio_filepath`` set to the **tar-internal basename**
       (``<snippet_id>.<output_format>``), updated ``duration``, and
       segments relativized to the snippet start.

    The tar-internal basename matches webdataset / Energon convention:
    sample key is ``<snippet_id>`` (everything before the first ``.``),
    extension is ``<output_format>``.  ``make_snippet_id`` already
    avoids ``.`` characters so the snippet id never spuriously splits.

    If the input produced zero snippets, a single "stub" ``AudioTask``
    is emitted (``snippet_id=None``, no audio written) so that
    per-original metrics can still flow to the aggregator.

    Dry-run mode (``dry_run=True``): skips steps 1-4 entirely (no
    ``soundfile`` reads, no resampling, no tar writes -- not even a tar
    shard is opened), and step 5 uses the planned ``end - start`` as the
    snippet ``duration`` instead of the post-resample frame count.  The
    emitted ``audio_filepath`` still uses the basename form for parity
    with real runs.  Useful for previewing the manifest and metrics on
    real data before committing to a full run.
    """

    output_dir: str = ""
    output_audio_tar_path: str = ""
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
        if not self.output_audio_tar_path:
            msg = "output_audio_tar_path is required for SnippetExtractionStage"
            raise ValueError(msg)
        if self.output_format not in _SOUNDFILE_SUBTYPES:
            msg = f"output_format must be one of {sorted(_SOUNDFILE_SUBTYPES)}, got {self.output_format!r}"
            raise ValueError(msg)
        if self.target_sample_rate <= 0:
            msg = "target_sample_rate must be > 0"
            raise ValueError(msg)
        self._tar_shard_path: str | None = None
        self._tar: Any = None  # tarfile.TarFile, opened lazily in setup()

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
        parent = os.path.dirname(self.output_audio_tar_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        parent = os.path.dirname(self.output_audio_tar_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if self.dry_run:
            return
        self._tar_shard_path = _make_shard_path(self.output_audio_tar_path, _TAR_SHARD_EXT)
        self._tar = tarfile.open(self._tar_shard_path, "w")
        logger.info(f"[{self.name}] writing audio tar shard to {self._tar_shard_path}")

    def teardown(self) -> None:
        if self._tar is not None:
            try:
                self._tar.close()
            except OSError as e:
                logger.warning(f"[{self.name}] failed to close tar shard {self._tar_shard_path}: {e}")
            finally:
                self._tar = None

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

        ``audio_filepath`` is the tar-internal basename
        ``<snippet_id>.<output_format>`` for parity with real runs --
        the tar itself is not opened in dry-run.  Snippet ``duration``
        is the planned ``end - start`` (vs. the resampled-frame-count
        duration the real path would compute -- the difference is at
        most one frame at ``target_sample_rate``).
        """
        outputs: list[AudioTask] = []
        for snippet in plan:
            start_sec = float(snippet["start"])
            end_sec = float(snippet["end"])
            snippet_id = make_snippet_id(original_id, start_sec, end_sec)
            out_path = f"{snippet_id}.{self.output_format}"
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
        member_name = f"{snippet_id}.{self.output_format}"
        try:
            buf = io.BytesIO()
            sf.write(
                buf,
                mono,
                self.target_sample_rate,
                format=self.output_format.upper(),
                subtype=_SOUNDFILE_SUBTYPES[self.output_format],
            )
            payload = buf.getvalue()
            tarinfo = tarfile.TarInfo(name=member_name)
            tarinfo.size = len(payload)
            self._tar.addfile(tarinfo, io.BytesIO(payload))
            # Flush the tar's BufferedWriter so this member's bytes hit
            # the kernel page cache. Cosmos-Xenna shuts actors down with
            # `ray.kill()` (see lines 74, 1220, 1473 and
            # cosmos_xenna/ray_utils/actor_pool.py), which does a Quick
            # exit that bypasses Python cleanup. Anything still in the
            # user-space buffer at kill time is lost. Page cache survives
            # process death — the downstream merger reads back the same
            # file and gets every fully-completed member regardless of
            # whether teardown() ever ran. Without this flush, ~50%+ of
            # snippets per shard get dropped during _merge_tar_shards's
            # Pass 2 streaming because their data sections are truncated
            # on disk.
            self._tar.fileobj.flush()
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{self.name}] failed to add {member_name} to tar shard {self._tar_shard_path}: {e}")
            return None

        return self._make_snippet_task(
            task=task,
            snippet=snippet,
            snippet_id=snippet_id,
            out_path=member_name,
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
# Stage 6: append snippet records to a JSONL manifest
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
# Stage 7: aggregate metrics across all snippets/originals
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
    * ``filtered_texts`` -- example texts of snippets dropped by the
      repetition filter; written only on the first record we see for a
      given ``id`` per replica (so the shard stays small even when many
      fan-out tasks share the same source).

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
        self._seen_ids: set[str] = set()

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
        record: dict[str, Any] = {
            "id": original_id,
            "in_segments": int(meta.get("original_seg_count", 0)),
            "in_duration_sec": float(meta.get("original_seg_duration", 0.0)),
            "dropped": {
                "empty": int(meta.get("dropped_empty", 0)),
                "overlap": int(meta.get("dropped_overlap", 0)),
                "too_long": int(meta.get("dropped_too_long", 0)),
                "too_short": int(meta.get("dropped_too_short", 0)),
                "no_text": int(meta.get("dropped_no_text", 0)),
                "repetition": int(meta.get("dropped_repetition", 0)),
            },
            "is_stub": is_stub,
            "out_segments": 0 if is_stub else len(task.data.get("segments") or []),
            "out_duration_sec": 0.0 if is_stub else float(task.data.get("duration", 0.0)),
        }
        if original_id not in self._seen_ids:
            self._seen_ids.add(original_id)
            record["filtered_texts"] = list(meta.get("filtered_repetition_texts") or [])
        with open(self._shard_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return task


# ----------------------------------------------------------------------
# Post-pipeline shard merging
# ----------------------------------------------------------------------


def prepare_audio_pretrain_outputs(
    output_manifest_path: str, metrics_path: str, output_audio_tar_path: str
) -> None:
    """Delete any pre-existing shards from prior runs.

    Call this once on the driver, BEFORE ``pipeline.run()``.  Multi-worker
    backends would race on cleanup if we did it inside a stage's
    ``setup()``, so we keep cleanup driver-only.
    """
    n_man = _delete_shards(output_manifest_path, _MANIFEST_SHARD_EXT)
    n_met = _delete_shards(metrics_path, _METRICS_SHARD_EXT)
    n_tar = _delete_shards(output_audio_tar_path, _TAR_SHARD_EXT)
    if n_man or n_met or n_tar:
        logger.info(
            f"prepare_audio_pretrain_outputs: removed {n_man} stale manifest "
            f"shard(s), {n_met} stale metrics shard(s), {n_tar} stale tar shard(s) "
            f"from prior runs"
        )


def finalize_audio_pretrain_outputs(
    output_manifest_path: str, metrics_path: str, output_audio_tar_path: str
) -> None:
    """Merge per-worker shards into the final manifest, metrics JSON, and audio tar.

    Call once on the driver, AFTER ``pipeline.run()`` returns
    successfully.  Reads all manifest + metrics + tar shards written by
    the writer / aggregator / extractor stages, concatenates / combines
    them, writes the final user-facing files at the user-provided paths,
    and removes the shards.

    After the audio tar is built, reconciles the manifest against the
    tar: any manifest row whose ``audio_filepath`` is not a valid member
    in the tar is dropped.  This guards against the Xenna failure mode
    where a worker is ``ray.kill``-ed between writing a JSONL line for
    a snippet and flushing the snippet's audio bytes to its tar shard,
    which would otherwise leave consumers (WebDataset, Energon) with a
    manifest entry pointing at a missing tar member.
    """
    _merge_manifest_shards(output_manifest_path)
    _merge_metrics_shards(metrics_path)
    _merge_tar_shards(output_audio_tar_path)
    _reconcile_manifest_with_tar(output_manifest_path, output_audio_tar_path)


def _merge_manifest_shards(output_path: str) -> None:
    shards = _glob_shards(output_path, _MANIFEST_SHARD_EXT)
    # Skip the merge when there are no shards.  This guards against silent
    # data loss on re-runs: with finalize_audio_pretrain_outputs called from
    # a try/finally, an early failure (before any worker writes a shard)
    # would otherwise truncate a previous successful run's manifest to
    # zero bytes via the "w"-mode open below.
    if not shards:
        logger.info(f"no manifest shards found for {output_path}; skipping merge")
        return
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for s in shards:
            with open(s, encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as e:
                        # A worker killed mid-write (e.g. Xenna's ray.kill)
                        # can leave a truncated final line in a shard; skip
                        # it so we don't emit invalid JSONL.
                        logger.warning(f"skipping malformed manifest shard line in {s}: {e}")
                        continue
                    out.write(line + "\n")
    for s in shards:
        try:
            os.remove(s)
        except OSError as e:
            logger.warning(f"failed to remove manifest shard {s}: {e}")
    logger.info(f"merged {len(shards)} manifest shard(s) into {output_path}")


def _merge_metrics_shards(metrics_path: str) -> None:
    shards = _glob_shards(metrics_path, _METRICS_SHARD_EXT)
    # Same re-run-safety guard as _merge_manifest_shards: skip when no
    # shards exist so an early failure on a re-run can't overwrite a
    # previous successful run's metrics summary with an all-zero JSON.
    if not shards:
        logger.info(f"no metrics shards found for {metrics_path}; skipping merge")
        return
    per_original: dict[str, dict[str, Any]] = {}
    durations: list[float] = []
    filtered_examples: list[str] = []
    for s in shards:
        with open(s, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as e:
                    # A worker killed mid-write (e.g. Xenna's ray.kill) can
                    # leave a truncated final line in a shard; skip it so
                    # finalize still merges the rest.
                    logger.warning(f"skipping malformed metrics shard line in {s}: {e}")
                    continue
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
                # Globally cap the example list. The aggregator emits
                # `filtered_texts` only on the first record per id per replica,
                # so this branch fires at most once per source under typical
                # scheduling.
                if "filtered_texts" in r and len(filtered_examples) < _MAX_FILTERED_TEXT_EXAMPLES:
                    remaining = _MAX_FILTERED_TEXT_EXAMPLES - len(filtered_examples)
                    filtered_examples.extend(r["filtered_texts"][:remaining])

    summary = _build_final_summary(per_original, durations, filtered_examples)
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


def _merge_tar_shards(output_path: str) -> None:
    """Merge per-replica audio tar shards into ``output_path``.

    Reads every ``<output_path>.shard-*.tar`` written by the extractor
    workers, copies their members into a single fresh tar at
    ``output_path`` in **lexicographic member-name order** (matches
    Energon expectations for indexed tar datasets), and removes the
    shards.  Re-write via Python ``tarfile`` instead of byte-level
    concatenation so the merger is portable, handles padding/header
    boundaries correctly, and tolerates shards left without trailing
    zero-blocks by workers that were ``ray.kill``-ed before
    ``teardown()``.
    """
    shards = _glob_shards(output_path, _TAR_SHARD_EXT)
    # Same re-run-safety guard as _merge_manifest_shards: skip when no
    # shards exist so an early failure on a re-run can't overwrite a
    # previous successful run's tar with an empty archive.
    if not shards:
        logger.info(f"no tar shards found for {output_path}; skipping merge")
        return
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    # Two-pass streaming merge.  Pass 1 builds a small in-memory index of
    # (member_name, shard_path, TarInfo) entries -- metadata only, no
    # payload bytes.  Pass 2 walks the index in sorted name order and
    # stream-copies each member from its source shard into the merged tar
    # via `extractfile() -> addfile()` (internally `copyfileobj` with 16
    # KB chunks), so peak memory is O(index_size + chunk_size) regardless
    # of total snippet count.  At 500k snippets this is ~250 MB instead
    # of ~300 GB when payloads were buffered alongside the index.
    index: list[tuple[str, str, tarfile.TarInfo]] = []
    for s in shards:
        try:
            in_tar = tarfile.open(s, "r")
        except tarfile.TarError as e:
            # An empty or non-tar file written by a worker killed before
            # the first `addfile()` -- nothing recoverable in this shard.
            logger.warning(f"skipping unreadable tar shard {s}: {e}")
            continue
        try:
            # Iterate manually and stop at the first malformed header,
            # rather than `for ti in in_tar:` which raises on truncation.
            # A worker `ray.kill`-ed mid-write can leave the trailing
            # member partially written; everything before that point is
            # still valid and recoverable.
            kept_in_shard = 0
            while True:
                try:
                    ti = in_tar.next()
                except tarfile.TarError as e:
                    logger.warning(
                        f"tar shard {s} truncated after {kept_in_shard} member(s): {e}; "
                        f"keeping the recovered members"
                    )
                    break
                if ti is None:
                    break
                if not ti.isreg():
                    continue
                index.append((ti.name, s, ti))
                kept_in_shard += 1
        finally:
            in_tar.close()
    index.sort(key=lambda e: e[0])

    # Pass 2: keep one open TarFile per source shard so we don't pay
    # reopen cost per member, then stream each member into the merged
    # tar in sorted order.
    open_shards: dict[str, tarfile.TarFile] = {}
    written = 0
    try:
        with tarfile.open(output_path, "w") as out_tar:
            for name, s, ti in index:
                in_tar = open_shards.get(s)
                if in_tar is None:
                    try:
                        in_tar = tarfile.open(s, "r")
                    except tarfile.TarError as e:
                        logger.warning(
                            f"cannot reopen tar shard {s} for streaming: {e}; skipping member {name!r}"
                        )
                        continue
                    open_shards[s] = in_tar
                try:
                    f = in_tar.extractfile(ti)
                    if f is None:
                        continue
                    out_tar.addfile(ti, f)
                except tarfile.TarError as e:
                    logger.warning(f"failed to stream member {name!r} from shard {s}: {e}; skipping")
                    continue
                written += 1
    finally:
        for in_tar in open_shards.values():
            in_tar.close()

    for s in shards:
        try:
            os.remove(s)
        except OSError as e:
            logger.warning(f"failed to remove tar shard {s}: {e}")
    logger.info(f"merged {len(shards)} tar shard(s) into {output_path} ({written} member(s))")


def _reconcile_manifest_with_tar(manifest_path: str, tar_path: str) -> None:
    """Drop manifest rows whose ``audio_filepath`` isn't in ``tar_path``.

    A no-op when the tar file doesn't exist (dry-run, or all tar shards
    were empty).  See ``finalize_audio_pretrain_outputs`` for why this
    is needed even when the pipeline reports success.
    """
    if not os.path.exists(tar_path):
        return
    if not os.path.exists(manifest_path):
        return
    try:
        with tarfile.open(tar_path, "r") as t:
            valid_members = set(t.getnames())
    except tarfile.TarError as e:
        logger.warning(f"cannot read merged tar {tar_path} for manifest reconciliation: {e}")
        return

    kept_lines: list[str] = []
    dropped = 0
    with open(manifest_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # _merge_manifest_shards already filtered these; unreachable
                # in practice, but keep the file usable if it ever happens.
                continue
            if row.get("audio_filepath") in valid_members:
                kept_lines.append(line)
            else:
                dropped += 1
    if dropped == 0:
        return
    with open(manifest_path, "w", encoding="utf-8") as f:
        for line in kept_lines:
            f.write(line + "\n")
    logger.warning(
        f"reconciled manifest {manifest_path}: dropped {dropped} row(s) whose "
        f"audio_filepath is not a valid member of {tar_path} (likely truncated "
        f"by a worker killed mid-write); {len(kept_lines)} row(s) kept"
    )


def _build_final_summary(
    per_original: dict[str, dict[str, Any]],
    durations: list[float],
    filtered_examples: list[str] | None = None,
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
        "dropped_repetition_examples": list(filtered_examples or []),
        "per_original": list(per_original.values()),
    }
