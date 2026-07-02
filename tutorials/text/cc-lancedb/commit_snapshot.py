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

"""Commit staged lance fragments for one snapshot.

Run after all array splits complete (via Slurm --dependency=afterok).
Reads all split_NNN.pkl files from staging/<snapshot_id>/, collects every
FragmentMetadata, and issues a single LanceDataset.commit() — one manifest
write covering the entire snapshot.

Usage:
  python commit_snapshot.py \\
      --snapshot-id CC-MAIN-2025-26 \\
      --lance-uri   s3://vjawa-cc-lance/cc_all \\
      --staging-dir /lustre/.../cc_lance_staging
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from loguru import logger  # noqa: E402

from nemo_curator.stages.text.io.writer.utils import s3_storage_options_from_env  # noqa: E402


def _append_manifest(manifest_file: str | None, entry: dict) -> None:
    if not manifest_file:
        return
    entry["ts"] = datetime.now(tz=UTC).isoformat()
    Path(manifest_file).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main(args: argparse.Namespace) -> None:
    staging_path = Path(args.staging_dir) / args.snapshot_id
    pkl_files = sorted(staging_path.glob("split_*.pkl"))

    if not pkl_files:
        logger.error(f"No staged files found at {staging_path}")
        sys.exit(1)

    logger.info(f"Collecting fragments from {len(pkl_files)} staged files for {args.snapshot_id}")

    all_fragments: list = []
    schema = None
    missing_splits: list[int] = []

    for expected_split in range(args.total_splits):
        expected_file = staging_path / f"split_{expected_split:03d}.pkl"
        if not expected_file.exists():
            missing_splits.append(expected_split)
            continue
        data = pickle.loads(expected_file.read_bytes())  # noqa: S301
        all_fragments.extend(data["fragments"])
        schema = schema or data.get("schema")

    if missing_splits:
        logger.warning(f"Missing splits: {missing_splits} — committing {len(pkl_files)}/{args.total_splits} splits")
        _append_manifest(
            args.manifest_file,
            {
                "event": "commit_partial",
                "snapshot_id": args.snapshot_id,
                "missing_splits": missing_splits,
            },
        )

    if not all_fragments or schema is None:
        logger.error("No fragments collected — aborting commit")
        sys.exit(1)

    logger.info(f"Committing {len(all_fragments):,} fragments for {args.snapshot_id}")

    # Reuse the shared commit helper (handles ValueError for new datasets + retry).
    from nemo_curator.stages.text.io.writer import lance_commit_fragments

    # Wrap fragments in a minimal LanceFragmentTask so lance_commit_fragments
    # can filter them — avoids reimplementing the commit loop here.
    from nemo_curator.stages.text.io.writer.lancedb import LanceFragmentTask

    pseudo_task = LanceFragmentTask(dataset_name=args.snapshot_id, data=all_fragments, schema=schema)
    lance_commit_fragments(
        [pseudo_task],
        uri=args.lance_uri,
        storage_options=s3_storage_options_from_env(),
        retries=10,
    )

    _append_manifest(
        args.manifest_file,
        {
            "event": "snapshot_committed",
            "snapshot_id": args.snapshot_id,
            "total_fragments": len(all_fragments),
            "splits_committed": len(pkl_files),
            "job_id": os.environ.get("SLURM_JOB_ID"),
        },
    )
    logger.info(f"Done — {args.snapshot_id} committed ({len(all_fragments):,} fragments)")

    # Clean up staging files after successful commit
    if not args.keep_staging:
        for f in pkl_files:
            f.unlink()
        logger.info(f"Cleaned staging dir {staging_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Commit staged lance fragments for one CC snapshot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--snapshot-id", required=True, help="e.g. CC-MAIN-2025-26")
    parser.add_argument("--lance-uri", required=True, help="Lance dataset URI.")
    parser.add_argument("--staging-dir", default="/lustre/fsw/portfolios/llmservice/users/vjawa/cc_lance_staging")
    parser.add_argument("--total-splits", type=int, default=40)
    parser.add_argument("--manifest-file", default=None)
    parser.add_argument(
        "--keep-staging", action="store_true", help="Keep staging pkl files after commit (default: delete)."
    )
    main(parser.parse_args())
