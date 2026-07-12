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

"""Segment-level planning: overlap drop, greedy packing, repetition filter.

These three stages all operate on a task's ``segments`` / ``_snippet_plan``
in memory, before the extractor reads any audio.  The pure helpers
(``filter_empty_segments``, ``find_overlapping_indices``, ``plan_snippets``,
``relativize_segments``, n-gram counters, color highlighting) are unit-
testable without Ray / soundfile / torch.
"""

from __future__ import annotations

import heapq
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from huggingface_hub import snapshot_download
from loguru import logger
from transformers import AutoTokenizer

from nemo_curator.stages.audio.alm.pretrain.utils import (
    _MAX_FILTERED_TEXT_EXAMPLES,
    _PLAN_DATA_KEY,
    _PRETRAIN_META_KEY,
    _segment_text,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata


# ----------------------------------------------------------------------
# Pure helpers (unit-testable without Ray / soundfile)
# ----------------------------------------------------------------------


_MIN_SEGMENTS_FOR_OVERLAP = 2


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

    Implementation is a sweep-line scan over segments sorted by
    ``(start, end)``.  An end-time-keyed min-heap holds the currently
    active intervals (those whose ``end`` is still beyond the cursor's
    ``start``); each new segment evicts the heap prefix it can no longer
    intersect and is then compared only against the survivors.  For
    typical diarized audio (a handful of overlapping speakers at any
    instant) this is effectively ``O(n log n)``, vs the pairwise ``O(n^2)``
    of comparing every pair; the worst case where all intervals overlap
    each other is still ``O(n^2)`` because the overlap relation itself
    is dense in that case.
    """
    n = len(segments)
    if n < _MIN_SEGMENTS_FOR_OVERLAP:
        return set()
    # Sort indirectly so we can return indices into the caller's list.
    order = sorted(
        range(n), key=lambda i: (segments[i]["start"], segments[i]["end"])
    )
    bad: set[int] = set()
    # Active interval heap, keyed by end time so the smallest-end interval
    # (the next to fall out of the active window) is always at the root.
    # Entries: (end, start, original_index).
    active: list[tuple[float, float, int]] = []
    for k in order:
        si, ei = segments[k]["start"], segments[k]["end"]
        # Evict every active interval that ends at or before our start --
        # it can't overlap us or any later (sorted-order) segment.
        while active and active[0][0] <= si:
            heapq.heappop(active)
        for ej, sj, j in active:
            # Sorted order guarantees sj <= si; a still-active interval
            # has ej > si.  The remaining no-overlap case is sj >= ei
            # (current ends at or before active's start), which can only
            # happen when sj == si and ei <= ej -- handle it explicitly.
            if sj >= ei:
                continue
            overlap = min(ei, ej) - max(si, sj)
            i_contains_j = si <= sj and ei >= ej
            j_contains_i = sj <= si and ej >= ei
            if overlap >= min_overlap_sec or i_contains_j or j_contains_i:
                bad.add(k)
                bad.add(j)
        heapq.heappush(active, (ei, si, k))
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


# ----------------------------------------------------------------------
# Stage: drop empty + overlapping segments
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
# Stage: plan snippet boundaries (no I/O)
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
# Stage: filter snippets whose joined text shows n-gram repetition
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

    ``tokenizer_path`` is either a local directory loadable by
    ``AutoTokenizer.from_pretrained`` or a HuggingFace Hub repository id
    (e.g. ``openai/whisper-large-v3``).  When it's a repo id, the
    tokenizer is fetched once per node in :meth:`setup_on_node` so
    workers in :meth:`setup` only ever read from the local cache.
    """

    tokenizer_path: str
    ngram_n: int = 10
    ngram_max_count: int = 3
    cache_dir: str | None = None
    hf_token: str | None = None

    name: str = "SnippetRepetitionFilter"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self) -> None:
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

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        # If `tokenizer_path` is a local directory, skip the Hub fetch --
        # the worker `setup()` will load straight from disk.  Otherwise
        # treat it as an HF Hub repo id and pre-download once per node so
        # workers don't race on the Hub.
        if os.path.isdir(self.tokenizer_path):
            return
        try:
            snapshot_download(
                repo_id=self.tokenizer_path,
                cache_dir=self.cache_dir,
                token=self.hf_token,
            )
        except Exception as e:
            msg = (
                f"failed to download tokenizer {self.tokenizer_path!r} from HF Hub; "
                f"pass a local directory or fix the repo id"
            )
            raise RuntimeError(msg) from e

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            use_fast=True,
            cache_dir=self.cache_dir,
            token=self.hf_token,
        )
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
