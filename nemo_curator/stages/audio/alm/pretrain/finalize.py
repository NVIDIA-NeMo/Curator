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

"""Driver-side prepare/finalize helpers and shard mergers.

Pipeline writers (manifest, metrics, audio tar) all emit one shard file
per replica.  These helpers run on the driver around ``pipeline.run()``
to clean up stale shards, merge fresh shards into the user-visible
output files, and reconcile manifest rows against the final tar.
"""

from __future__ import annotations

import json
import os
import tarfile
from collections import defaultdict
from typing import Any

import soundfile as sf
from loguru import logger

from nemo_curator.stages.audio.alm.pretrain.utils import (
    _MANIFEST_SHARD_EXT,
    _MAX_FILTERED_TEXT_EXAMPLES,
    _METRICS_SHARD_EXT,
    _TAR_SHARD_EXT,
    _delete_shards,
    _glob_shards,
    histogram_30s,
)


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
    output_manifest_path: str,
    metrics_path: str,
    output_audio_tar_path: str,
    audio_filepath_key: str = "audio_filepath",
) -> None:
    """Merge per-worker shards into the final manifest, metrics JSON, and audio tar.

    Call once on the driver, AFTER ``pipeline.run()`` returns
    successfully.  Reads all manifest + metrics + tar shards written by
    the writer / aggregator / extractor stages, concatenates / combines
    them, writes the final user-facing files at the user-provided paths,
    and removes the shards.

    After the audio tar is built, reconciles the manifest against the
    tar.  A manifest row is dropped if either:

    1. its ``audio_filepath_key`` value is not a member of the tar, or
    2. the corresponding tar member's audio header is unreadable or
       reports zero frames / zero sample rate.

    Check (1) guards against the Xenna failure mode where a worker is
    ``ray.kill``-ed between writing a JSONL line for a snippet and
    flushing the snippet's audio bytes to its tar shard.  Check (2)
    guards against truncated members (worker killed mid-payload-write
    with a header but no body) and corrupted writes that would surface
    to downstream consumers (WebDataset / Energon) as a runtime decode
    error instead of a clean filter.

    Reconcile drops are surfaced in the merged metrics under
    ``dropped.missing_audio`` (check 1) and ``dropped.corrupted_audio``
    (check 2), so the user can attribute post-pipeline integrity drops
    separately from worker-side filters (``empty``, ``overlap``, ...).
    """
    _merge_manifest_shards(output_manifest_path)
    _merge_metrics_shards(metrics_path)
    _merge_tar_shards(output_audio_tar_path)
    dropped_missing, dropped_unreadable = _reconcile_manifest_with_tar(
        output_manifest_path, output_audio_tar_path, audio_filepath_key
    )
    _patch_metrics_post_reconcile(
        metrics_path, output_manifest_path, dropped_missing, dropped_unreadable
    )


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


def _merge_metrics_shards(metrics_path: str) -> None:  # noqa: C901
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
                pid = r.get("id")
                if pid is None or (isinstance(pid, str) and not pid.strip()):
                    logger.warning(f"skipping metrics shard line in {s}: missing or empty 'id'")
                    continue
                pid = str(pid)
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


def _merge_tar_shards(output_path: str) -> None:  # noqa: C901, PLR0912, PLR0915
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
            in_tar = tarfile.open(s, "r")  # noqa: SIM115  -- closed via finally; need try/except above for unreadable shards
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
    if not index:
        logger.info(
            f"no readable tar members found in {len(shards)} tar shard(s) for {output_path}; skipping merge"
        )
        return

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
                        in_tar = tarfile.open(s, "r")  # noqa: SIM115  -- cached in open_shards and closed in finally below
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


def _reconcile_manifest_with_tar(  # noqa: C901, PLR0915
    manifest_path: str, tar_path: str, audio_filepath_key: str = "audio_filepath"
) -> tuple[int, int]:
    """Drop manifest rows whose audio path field isn't a valid, readable
    tar member with a positive duration.

    Two filters:
      1. Membership: ``audio_filepath_key`` must be a regular member of the tar.
      2. Header validity: ``soundfile.info`` must succeed on the member's
         payload and report ``frames > 0`` and ``samplerate > 0``.

    Returns ``(dropped_missing, dropped_unreadable)`` so the caller can
    surface the counts (e.g. patch them into the merged metrics summary).

    A no-op when the tar file doesn't exist (dry-run, or all tar shards
    were empty).  See ``finalize_audio_pretrain_outputs`` for why this
    is needed even when the pipeline reports success.

    The tar itself is left as-is.  Removing dropped members would
    require rewriting the whole archive; downstream consumers iterate
    the manifest and look up by ``audio_filepath_key``, so orphan members
    are harmless.
    """
    if not os.path.exists(tar_path):
        return (0, 0)
    if not os.path.exists(manifest_path):
        return (0, 0)

    try:
        tar = tarfile.open(tar_path, "r")  # noqa: SIM115  -- closed in finally below; need a try/except here for unreadable archives
    except tarfile.TarError as e:
        logger.warning(f"cannot read merged tar {tar_path} for manifest reconciliation: {e}")
        return (0, 0)

    try:
        members: dict[str, tarfile.TarInfo] = {
            ti.name: ti for ti in tar.getmembers() if ti.isreg()
        }
        # Header-validity is sticky per member name.  A name only ever
        # appears with one payload in the merged tar, but we cache anyway
        # in case a manifest row points at the same audio twice.
        header_ok: dict[str, bool] = {}

        def _audio_ok(name: str) -> bool:
            cached = header_ok.get(name)
            if cached is not None:
                return cached
            ti = members.get(name)
            if ti is None or ti.size == 0:
                header_ok[name] = False
                return False
            ok = False
            try:
                stream = tar.extractfile(ti)
                if stream is None:
                    logger.warning(
                        f"audio header unreadable for {name!r} in {tar_path}: extractfile returned None"
                    )
                else:
                    info = sf.info(stream)
                    ok = info.frames > 0 and info.samplerate > 0
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"audio header unreadable for {name!r} in {tar_path}: {exc}"
                )
            header_ok[name] = ok
            return ok

        kept_lines: list[str] = []
        dropped_missing = 0
        dropped_unreadable = 0
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
                ap = row.get(audio_filepath_key)
                if ap not in members:
                    dropped_missing += 1
                    continue
                if not _audio_ok(ap):
                    dropped_unreadable += 1
                    continue
                kept_lines.append(line)
    finally:
        tar.close()

    dropped = dropped_missing + dropped_unreadable
    if dropped == 0:
        return (0, 0)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for line in kept_lines:
            f.write(line + "\n")
    logger.warning(
        f"reconciled manifest {manifest_path}: dropped {dropped} row(s) "
        f"({dropped_missing} not in tar, {dropped_unreadable} unreadable / "
        f"zero-duration); {len(kept_lines)} row(s) kept"
    )
    return (dropped_missing, dropped_unreadable)


def _collect_reconciled_output_stats(
    manifest_path: str,
) -> tuple[dict[str, dict[str, Any]], list[float]]:
    out_per_id: dict[str, dict[str, Any]] = {}
    durations: list[float] = []
    if not os.path.exists(manifest_path):
        return out_per_id, durations

    with open(manifest_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # `_merge_manifest_shards` already filtered these; if a
                # stray line slips through, skip rather than crash the
                # finalize step.
                continue
            pid = str(row.get("id") or "")
            if not pid:
                continue
            dur = float(row.get("duration", 0.0))
            seg_count = len(row.get("segments") or [])
            entry = out_per_id.setdefault(
                pid,
                {"out_snippets": 0, "out_segments": 0, "out_duration_sec": 0.0},
            )
            entry["out_snippets"] += 1
            entry["out_segments"] += seg_count
            entry["out_duration_sec"] += dur
            durations.append(dur)
    return out_per_id, durations


def _patch_metrics_post_reconcile(
    metrics_path: str,
    manifest_path: str,
    dropped_missing: int,
    dropped_unreadable: int,
) -> None:
    """Reconcile the merged metrics summary against the post-reconcile manifest.

    Worker shards are written before ``_reconcile_manifest_with_tar`` can
    prune manifest rows whose audio is missing or unreadable, so the
    initial ``_merge_metrics_shards`` summary overcounts on the output
    side by exactly the number of rows the reconcile pass removed.  This
    helper:

    1. Increments ``dropped.missing_audio`` and ``dropped.corrupted_audio``
       with the reconcile pass's drop counts (so reconcile drops are
       attributable separately from worker-side filters like ``empty`` /
       ``overlap``).
    2. Rebuilds the output-side counters -- ``num_output_snippets``,
       ``output_total_segments``, ``output_total_duration_sec``,
       ``snippet_duration_histogram_30s``, and each
       ``per_original[*].out_*`` field -- by walking the now-authoritative
       (post-reconcile) manifest.  Input-side and worker-side dropped
       counters are left untouched.

    No-op when both reconcile counts are zero (the merged metrics already
    match the manifest) or when the metrics file doesn't exist (dry run
    / reconcile-in-isolation tests).
    """
    if dropped_missing == 0 and dropped_unreadable == 0:
        return
    if not os.path.exists(metrics_path):
        return
    try:
        with open(metrics_path, encoding="utf-8") as f:
            summary = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            f"cannot patch reconcile drops into metrics {metrics_path}: {exc}"
        )
        return

    dropped = summary.setdefault("dropped", {})
    if dropped_missing:
        dropped["missing_audio"] = (
            int(dropped.get("missing_audio", 0)) + dropped_missing
        )
    if dropped_unreadable:
        dropped["corrupted_audio"] = (
            int(dropped.get("corrupted_audio", 0)) + dropped_unreadable
        )

    # Rebuild output-side counters from the reconciled manifest.  After
    # _reconcile_manifest_with_tar drops a row the worker-emitted shard
    # record for that snippet is still summed into the pre-reconcile
    # totals, so we recompute against what survived.
    out_per_id, durations = _collect_reconciled_output_stats(manifest_path)

    total_snippets = sum(v["out_snippets"] for v in out_per_id.values())
    total_segments = sum(v["out_segments"] for v in out_per_id.values())
    total_duration = sum(v["out_duration_sec"] for v in out_per_id.values())
    summary["num_output_snippets"] = int(total_snippets)
    summary["output_total_segments"] = int(total_segments)
    summary["output_total_duration_sec"] = round(float(total_duration), 3)
    summary["snippet_duration_histogram_30s"] = histogram_30s(durations)

    for entry in summary.get("per_original", []):
        pid = entry.get("id")
        if pid is None:
            continue
        out = out_per_id.get(
            str(pid),
            {"out_snippets": 0, "out_segments": 0, "out_duration_sec": 0.0},
        )
        entry["out_snippets"] = int(out["out_snippets"])
        entry["out_segments"] = int(out["out_segments"])
        entry["out_duration_sec"] = round(float(out["out_duration_sec"]), 3)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(
        f"patched metrics {metrics_path}: missing_audio+={dropped_missing}, "
        f"corrupted_audio+={dropped_unreadable}, "
        f"rebuilt output totals from {len(out_per_id)} source(s) / {total_snippets} snippet(s)"
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
