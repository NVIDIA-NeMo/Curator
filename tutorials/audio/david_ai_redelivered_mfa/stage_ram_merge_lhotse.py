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

"""Merge per-session RAM Lhotse cuts into global aligned manifests."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from david_ai_common import finish_stage, run_main
from david_ai_ram_lhotse import merge_ram_lhotse_manifests

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lhotse-dir", type=Path, required=True)
    ap.add_argument("--prefix", default="david_ai")
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--stage-done-name", default="ram_lhotse_merge")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    lhotse_dir = args.lhotse_dir.resolve()
    aligned_path = lhotse_dir / f"{args.prefix}_aligned_cuts.jsonl.gz"
    if aligned_path.is_file() and not args.force:
        logger.info("Merged Lhotse already exists: %s", aligned_path)
        return finish_stage(args.work_dir, args.stage_done_name, 0)

    merged = merge_ram_lhotse_manifests(lhotse_dir, prefix=args.prefix)
    if merged == 0:
        logger.warning("No per-session Lhotse cuts found under %s/sessions", lhotse_dir)
        return finish_stage(args.work_dir, args.stage_done_name, 1)

    logger.info(
        "Merged %d aligned Lhotse cuts into %s (%s_recordings/supervisions/cuts/aligned_cuts)",
        merged,
        lhotse_dir,
        args.prefix,
    )
    return finish_stage(args.work_dir, args.stage_done_name, 0)


if __name__ == "__main__":
    run_main(main)
