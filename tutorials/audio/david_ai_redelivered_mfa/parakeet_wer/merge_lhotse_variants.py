#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Merge per-shard Lhotse WER variants into dataset-level CutSets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_lhotse_variants import VARIANTS
from lhotse import CutSet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    merged_root = args.output_dir / "lhotse_merged"
    merged_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for variant in VARIANTS:
        paths = sorted(args.output_dir.glob(f"shards/shard_*/lhotse/{variant.name}/cuts.jsonl.gz"))
        if not paths:
            parser.error(f"no shard CutSets found for {variant.name}")
        variant_dir = merged_root / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        output_path = variant_dir / "cuts.jsonl.gz"
        CutSet.from_files(paths, shuffle_iters=False).to_file(output_path)
        cut_count = sum(1 for _ in CutSet.from_file(output_path))
        summaries.append(
            {
                "variant": variant.name,
                "shards": len(paths),
                "cuts": cut_count,
                "cut_manifest": str(output_path),
            }
        )

    summary_path = merged_root / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2, sort_keys=True))
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
