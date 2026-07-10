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

"""Check progress of the CC → Lance pipeline from the manifest file.

Usage:
  python check_progress.py --manifest-file /lustre/.../cc_manifest.jsonl
  python check_progress.py --manifest-file ... --snapshot CC-MAIN-2025-26
  python check_progress.py --manifest-file ... --show-failed
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_manifest(manifest_file: str) -> list[dict]:
    path = Path(manifest_file)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarise(events: list[dict], total_splits: int = 40, snapshot_filter: str | None = None) -> None:
    # Latest event per (snapshot_id, split)
    split_state: dict[tuple[str, int], dict] = {}
    snapshot_committed: dict[str, dict] = {}

    for e in events:
        sid = e.get("snapshot_id", "")
        if snapshot_filter and sid != snapshot_filter:
            continue
        split = e.get("split")
        event = e.get("event", "")
        if event == "snapshot_committed":
            snapshot_committed[sid] = e
        elif split is not None:
            split_state[(sid, split)] = e

    # Group by snapshot
    snapshots: dict[str, dict[int, dict]] = defaultdict(dict)
    for (sid, sp), state in split_state.items():
        snapshots[sid][sp] = state

    if not snapshots and not snapshot_committed:
        print("No events in manifest yet.")
        return

    all_snapshots = sorted(set(list(snapshots.keys()) + list(snapshot_committed.keys())))
    print(f"{'Snapshot':<25} {'Staged':>8} {'Failed':>8} {'Missing':>9} {'Committed':>10} {'Fragments':>12}")
    print("-" * 78)

    for sid in all_snapshots:
        splits = snapshots.get(sid, {})
        staged = sum(1 for v in splits.values() if v.get("event") in ("split_staged", "split_empty"))
        failed = sum(1 for v in splits.values() if v.get("event") == "split_failed")
        missing = total_splits - len(splits)
        committed = "✓" if sid in snapshot_committed else ""
        frags = snapshot_committed.get(sid, {}).get("total_fragments", "")
        frags_str = f"{frags:,}" if isinstance(frags, int) else ""
        print(f"{sid:<25} {staged:>8} {failed:>8} {missing:>9} {committed:>10} {frags_str:>12}")


def show_failed(events: list[dict], snapshot_filter: str | None = None) -> None:
    failed = [
        e
        for e in events
        if e.get("event") == "split_failed" and (not snapshot_filter or e.get("snapshot_id") == snapshot_filter)
    ]
    if not failed:
        print("No failed splits.")
        return
    print(f"{'Snapshot':<25} {'Split':>6} {'Error'}")
    print("-" * 70)
    for e in failed:
        print(f"{e.get('snapshot_id', '?'):<25} {e.get('split', '?'):>6}  {e.get('error', '')[:50]}")


def resubmit_commands(events: list[dict], total_splits: int, snapshot_filter: str | None = None) -> None:
    """Print sbatch commands to resubmit failed/missing splits."""
    split_state: dict[tuple[str, int], str] = {}
    for e in events:
        sid = e.get("snapshot_id", "")
        if snapshot_filter and sid != snapshot_filter:
            continue
        split = e.get("split")
        event = e.get("event", "")
        if split is not None and event in ("split_staged", "split_empty", "split_failed"):
            split_state[(sid, split)] = event

    all_sids = {sid for (sid, _) in split_state} | ({snapshot_filter} if snapshot_filter else set())
    for sid in sorted(all_sids):
        missing = [
            s for s in range(total_splits) if (sid, s) not in split_state or split_state[(sid, s)] == "split_failed"
        ]
        if missing:
            splits_str = ",".join(str(s) for s in missing)
            print(f"# {sid} — resubmit splits: {splits_str}")
            print(f"sbatch --array={splits_str} slurm/process_snapshot_array.sh {sid}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check CC → Lance processing progress.")
    parser.add_argument("--manifest-file", required=True)
    parser.add_argument("--snapshot", default=None, help="Filter to one snapshot.")
    parser.add_argument("--total-splits", type=int, default=40)
    parser.add_argument("--show-failed", action="store_true")
    parser.add_argument("--resubmit", action="store_true", help="Print resubmit commands.")
    args = parser.parse_args()

    events = load_manifest(args.manifest_file)
    if not events:
        print(f"No events in {args.manifest_file}")
        sys.exit(0)

    summarise(events, total_splits=args.total_splits, snapshot_filter=args.snapshot)
    if args.show_failed:
        print()
        show_failed(events, snapshot_filter=args.snapshot)
    if args.resubmit:
        print()
        resubmit_commands(events, total_splits=args.total_splits, snapshot_filter=args.snapshot)
