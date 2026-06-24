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

"""Shared constants and small pure helpers for the pretrain pipeline.

Anything used by two or more modules in this package lives here; keeps
``planning.py`` / ``extraction.py`` / ``io.py`` / ``finalize.py`` free
of cross-module dependencies on incidental helpers.
"""

from __future__ import annotations

import glob
import os
import uuid
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from nemo_curator.tasks import AudioTask

# soundfile encoding subtype for each output format.
_SOUNDFILE_SUBTYPES = {
    "wav": "PCM_16",
    "flac": "PCM_16",
    "ogg": "VORBIS",
}

# Per-task `_metadata` key under which the pipeline stages stash
# per-original counters that the metrics aggregator reads back out.
_PRETRAIN_META_KEY = "pretrain_long_form"
# Per-task `data` key under which the planner writes the snippet plan
# and the repetition filter mutates it.
_PLAN_DATA_KEY = "_snippet_plan"

_HISTOGRAM_BIN_WIDTH_SEC = 30.0

_MANIFEST_SHARD_EXT = "jsonl"
# Metrics shards are JSONL (one record per task processed by a replica). The
# format avoids relying on `teardown()` -- the Xenna executor never calls it
# (actors are killed with `ray.kill()`), so an in-memory-only aggregator that
# flushed in teardown would always produce an empty summary.
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


# ----------------------------------------------------------------------
# Shard path helpers
# ----------------------------------------------------------------------


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
# Tiny shared helpers
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


AUDIO_PATH_RESOLUTION_BASENAME = "basename"
AUDIO_PATH_RESOLUTION_RELATIVE = "relative"
AUDIO_PATH_RESOLUTION_AS_IS = "as_is"
_AUDIO_PATH_RESOLUTION_MODES = (
    AUDIO_PATH_RESOLUTION_BASENAME,
    AUDIO_PATH_RESOLUTION_RELATIVE,
    AUDIO_PATH_RESOLUTION_AS_IS,
)


def _resolve_audio_path(audio_dir: str, value: str, mode: str = AUDIO_PATH_RESOLUTION_BASENAME) -> str:
    """Resolve a manifest's ``audio_filepath`` against ``audio_dir``.

    The default ``"basename"`` mode is convenient for flat staging
    directories but can silently point at the wrong audio when the
    manifest preserves subdirectories or when two source rows share a
    basename across shards; ``"relative"`` and ``"as_is"`` are exposed
    so the user can opt in to subdirectory-preserving or
    fully-explicit resolution when that matters.

    Modes:

    * ``"basename"`` (default): ``audio_dir / basename(value)``.
      Manifests stay portable across hosts; pair with the duplicate-
      basename check in :class:`ReadLongFormManifestStage` to surface
      collisions explicitly.
    * ``"relative"``: ``audio_dir / value`` (preserves subdirectories).
      Absolute paths in the manifest are taken as-is (Python's
      ``os.path.join`` semantics).
    * ``"as_is"``: ``value`` returned unchanged.  The caller is
      responsible for making sure it points at the right audio file.
    """
    if mode == AUDIO_PATH_RESOLUTION_BASENAME:
        return os.path.join(audio_dir, os.path.basename(value))
    if mode == AUDIO_PATH_RESOLUTION_RELATIVE:
        return os.path.join(audio_dir, value)
    if mode == AUDIO_PATH_RESOLUTION_AS_IS:
        return value
    msg = (
        f"unknown audio_path_resolution {mode!r}; "
        f"expected one of {_AUDIO_PATH_RESOLUTION_MODES}"
    )
    raise ValueError(msg)


def _is_origin_stub(task: AudioTask) -> bool:
    """A stub task from the extractor that carries per-original metrics for an
    input that produced zero snippets.  Has no snippet_id."""
    return task.data.get("snippet_id") is None


_SNIPPET_ID_RESERVED_CHARS = (".", "/", "\\")


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

    ``original_id`` is also sanitized for the same reason: input
    manifests sometimes carry ids that include a ``.`` (e.g. a source
    filename like ``meeting.wav`` or a versioned id ``session.1.2``),
    and passing those through would re-introduce the same WebDataset
    misparse the timestamp sanitization avoids.  Path separators (``/``
    and ``\\``) are also stripped: a path-like input id such as
    ``shard1/utt001`` would otherwise turn ``<snippet_id>.<ext>`` into a
    nested tar path, breaking the "members live at the tar root"
    contract documented for the output archive.
    """
    safe_id = original_id
    for ch in _SNIPPET_ID_RESERVED_CHARS:
        safe_id = safe_id.replace(ch, "_")
    start_str = f"{start_sec:.3f}".replace(".", "_")
    end_str = f"{end_sec:.3f}".replace(".", "_")
    return f"{safe_id}-{start_str}-{end_str}"


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
