#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Repair glued tokens in normalized manifests: punctuation first, then frequency unglue."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from david_ai_common import load_jsonl, partition_list, run_main, write_jsonl
from david_ai_glued_words import (
    apply_repair_map_to_text,
    load_lexicon_unglue_repairs,
    repair_normalized_text_two_step,
    repair_punctuation_only,
)

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def list_manifest_paths(manifests_dir: Path, sessions: list[str] | None = None) -> list[Path]:
    paths = sorted(manifests_dir.glob("*_norm.jsonl"))
    paths = [path for path in paths if path.name != "all_norm.jsonl"]
    if sessions:
        wanted = set(sessions)
        paths = [path for path in paths if path.name.removesuffix("_norm.jsonl") in wanted]
    return paths


def shard_manifest_paths(
    paths: list[Path],
    *,
    shard_count: int,
    shard_index: int,
) -> list[Path]:
    if shard_count <= 1:
        return paths
    shards = partition_list(paths, shard_count)
    if shard_index < 0 or shard_index >= len(shards):
        return []
    return shards[shard_index]


def repair_manifest(
    path: Path,
    repair_map: dict[str, str],
    *,
    num2words_lang: str,
    repair_mode: str,
) -> tuple[int, int, int]:
    rows = load_jsonl(path)
    changed = punctuation_rows = unglue_rows = 0
    for row in rows:
        text_raw = (row.get("text_raw") or row.get("text") or "").strip()
        text_norm = (row.get("text_norm") or row.get("text") or "").strip()
        if not text_raw and not text_norm:
            continue

        if repair_mode == "punctuation":
            repaired, punctuation_changed = repair_punctuation_only(
                text_raw,
                text_norm,
                num2words_lang=num2words_lang,
            )
            unglue_changed = False
        elif repair_mode == "unglue":
            repaired = apply_repair_map_to_text(text_norm, repair_map)
            punctuation_changed = False
            unglue_changed = repaired != text_norm
        else:
            repaired, punctuation_changed, unglue_changed = repair_normalized_text_two_step(
                text_raw,
                text_norm,
                num2words_lang=num2words_lang,
                repair_map=repair_map,
            )

        if repaired != text_norm:
            row["text"] = repaired
            row["text_norm"] = repaired
            changed += 1
            punctuation_rows += int(punctuation_changed)
            unglue_rows += int(unglue_changed)
    if changed:
        write_jsonl(path, rows)
    return changed, punctuation_rows, unglue_rows


def write_shard_done_marker(lexicon_dir: Path, repair_mode: str, shard_index: int) -> None:
    shard_dir = lexicon_dir / "repair_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / f"{repair_mode}_shard_{shard_index:03d}.done").write_text("ok\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifests-dir", type=Path, required=True)
    ap.add_argument("--lexicon-dir", type=Path, required=True)
    ap.add_argument("--num2words-lang", default="en")
    ap.add_argument("--session", action="append", default=[])
    ap.add_argument(
        "--repair-mode",
        choices=("punctuation", "unglue", "all"),
        default="all",
        help="punctuation=re-normalize from text_raw; unglue=frequency map only; all=both",
    )
    ap.add_argument(
        "--shard-count",
        type=int,
        default=0,
        help="Split manifest files across N parallel jobs (0 = single job)",
    )
    ap.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Process one manifest shard (use with SLURM array)",
    )
    ap.add_argument(
        "--list-shard-files",
        action="store_true",
        help="Print the shard's manifest file names (one per line) to stdout and exit. "
        "Used by the runner to bulk-stage exactly those files to node-local disk.",
    )
    ap.add_argument(
        "--changed-list",
        type=Path,
        default=None,
        help="Write the names of files whose contents changed to this path "
        "(so the runner can copy only changed files back).",
    )
    ap.add_argument(
        "--skip-done-marker",
        action="store_true",
        help="Do not write the per-shard .done marker (runner writes it after copy-back).",
    )
    args = ap.parse_args()

    manifests_dir = args.manifests_dir.resolve()
    lexicon_dir = args.lexicon_dir.resolve()

    paths = list_manifest_paths(manifests_dir, args.session or None)
    if args.shard_index is not None:
        paths = shard_manifest_paths(
            paths,
            shard_count=max(1, args.shard_count),
            shard_index=args.shard_index,
        )

    if args.list_shard_files:
        # stdout is consumed by the runner; keep it to bare file names only.
        print("\n".join(path.name for path in paths))
        return 0

    repair_map, repairs_path = load_lexicon_unglue_repairs(lexicon_dir)
    if args.repair_mode == "unglue" and not repair_map:
        logger.warning("No unglue repairs TSV found under %s; unglue pass will be a no-op", lexicon_dir)
    elif repairs_path is not None:
        logger.info("Using unglue repairs from %s (%d entries)", repairs_path, len(repair_map))

    if args.shard_index is not None:
        logger.info(
            "Repair shard %d/%d: %d manifest files (%s)",
            args.shard_index,
            max(1, args.shard_count),
            len(paths),
            args.repair_mode,
        )

    total_rows = total_punct = total_unglue = 0
    changed_files: list[str] = []
    for path in paths:
        changed, punctuation_rows, unglue_rows = repair_manifest(
            path,
            repair_map,
            num2words_lang=args.num2words_lang,
            repair_mode=args.repair_mode,
        )
        total_rows += changed
        total_punct += punctuation_rows
        total_unglue += unglue_rows
        if changed:
            changed_files.append(path.name)
            logger.info(
                "%s: repaired %d rows (punctuation=%d frequency_unglue=%d)",
                path.name,
                changed,
                punctuation_rows,
                unglue_rows,
            )

    if args.changed_list is not None:
        args.changed_list.parent.mkdir(parents=True, exist_ok=True)
        args.changed_list.write_text(
            "".join(f"{name}\n" for name in changed_files), encoding="utf-8"
        )

    if args.shard_index is not None and not args.skip_done_marker:
        write_shard_done_marker(lexicon_dir, args.repair_mode, args.shard_index)

    logger.info(
        "Manifest repair done: %d rows updated (punctuation=%d frequency_unglue=%d) across %d files",
        total_rows,
        total_punct,
        total_unglue,
        len(paths),
    )
    return 0


if __name__ == "__main__":
    run_main(main)
