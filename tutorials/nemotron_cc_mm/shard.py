"""Per-shard success markers for SLURM-array WARC processing.

Each array task processes one WARC and writes a JSON marker on success.
Future invocations short-circuit on the marker, so the driver pattern is:

    submit -> array of N tasks -> some succeed, some fail
    status -> show completed/missing
    retry-missing -> sbatch --array=<comma list of missing>

The marker lives at ``<output_root>/_SUCCESS/shard_NNNNN.json`` — *outside*
each shard's data dir so it survives the pipeline writer's
``--mode overwrite`` rmtree of the output directory.

Env contract:
    CURATOR_SHARD_INDEX        this task's shard index (defaults to SLURM_ARRAY_TASK_ID)
    CURATOR_NUM_SHARDS         original shard count (preferred over SLURM_ARRAY_TASK_COUNT
                               so sparse retries like ``--array=3,5,9`` keep the
                               original count for state tracking)
    CURATOR_ORIGINAL_ARRAY_SIZE  fallback set by submit_array.sh on first submission
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_MARKER_GLOB = re.compile(r"shard_(\d+)\.json$")


class Shard:
    """Static helpers for the current shard. No instance state."""

    @staticmethod
    def env() -> tuple[int, int]:
        """Return ``(shard_index, num_shards)`` from env vars.

        Prefers ``CURATOR_*`` over ``SLURM_ARRAY_*`` so sparse retry
        submissions like ``--array=3,5,9`` still report the original
        shard count.
        """
        idx_raw = os.environ.get("CURATOR_SHARD_INDEX") or os.environ.get("SLURM_ARRAY_TASK_ID") or "0"
        idx = int(idx_raw)
        n = 1
        for var in ("CURATOR_NUM_SHARDS", "CURATOR_ORIGINAL_ARRAY_SIZE", "SLURM_ARRAY_TASK_COUNT"):
            v = os.environ.get(var)
            if v:
                n = int(v)
                break
        if n < 1 or not (0 <= idx < n):
            msg = f"shard_index={idx} out of range for num_shards={n}"
            raise ValueError(msg)
        return idx, n

    @staticmethod
    def marker_path(output_root: str | Path, idx: int) -> Path:
        return Path(output_root) / "_SUCCESS" / f"shard_{idx:05d}.json"

    @staticmethod
    def has_marker(output_root: str | Path, idx: int | None = None) -> bool:
        if idx is None:
            idx = Shard.env()[0]
        return Shard.marker_path(output_root, idx).is_file()

    @staticmethod
    def completed(output_root: str | Path) -> set[int]:
        """Set of shard indices that have a success marker."""
        success_dir = Path(output_root) / "_SUCCESS"
        if not success_dir.is_dir():
            return set()
        out: set[int] = set()
        for p in success_dir.glob("shard_*.json"):
            m = _MARKER_GLOB.search(p.name)
            if m:
                out.add(int(m.group(1)))
        return out

    @staticmethod
    def missing(output_root: str | Path, num_shards: int) -> list[int]:
        done = Shard.completed(output_root)
        return [i for i in range(num_shards) if i not in done]

    @staticmethod
    def write_marker(
        output_root: str | Path,
        idx: int,
        num_shards: int,
        payload: dict[str, Any] | None = None,
    ) -> Path:
        """Write a success marker.  Returns the path."""
        path = Shard.marker_path(output_root, idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        body: dict[str, Any] = {
            "status": "success",
            "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "shard_index": idx,
            "num_shards": num_shards,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "host": os.uname().nodename,
        }
        if payload:
            body.update(payload)
        path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
        return path
