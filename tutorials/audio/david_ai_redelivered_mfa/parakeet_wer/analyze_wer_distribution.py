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

"""Merge shard WER audit files and propose one dataset-wide threshold."""

from __future__ import annotations

import argparse
import json
from array import array
from pathlib import Path

import numpy as np
from manifest import build_wer_distribution_from_values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--applied-threshold-pct", type=float, default=100.0)
    args = parser.parse_args()

    audit_paths = sorted(args.output_dir.glob("shards/shard_*/segments_with_wer.jsonl"))
    if not audit_paths:
        parser.error(f"no shard audit files found under {args.output_dir}")

    values = array("d")
    total_segments = 0
    for audit_path in audit_paths:
        with audit_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                total_segments += 1
                if row.get("wer_pct") is not None:
                    values.append(float(row["wer_pct"]))

    finite = np.frombuffer(values, dtype=np.float64)
    report = build_wer_distribution_from_values(
        finite,
        total_segments=total_segments,
        applied_threshold_pct=args.applied_threshold_pct,
    )
    report["shards"] = len(audit_paths)
    report_path = args.output_dir / "wer_distribution_merged.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
