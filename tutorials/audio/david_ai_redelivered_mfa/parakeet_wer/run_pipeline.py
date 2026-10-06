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

"""Run Parakeet segment ASR, WER analysis, and FastMSS manifest filtering."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from build_lhotse_variants import build_all_variants
from loguru import logger
from manifest import (
    SegmentTaskConfig,
    build_segment_tasks,
    build_wer_distribution,
    write_pipeline_outputs,
)
from stages import ParallelInferenceAsrNemoStage, SegmentClipExtractionStage, SegmentWERStage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--masked-audio-dir", type=Path, required=True)
    parser.add_argument("--textgrid-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sessions-file", type=Path, default=None)
    parser.add_argument("--model-name", default="nvidia/parakeet-tdt-0.6b-v2")
    parser.add_argument("--model-cache-dir", type=str, default=None)
    parser.add_argument("--asr-batch-size", type=int, default=16)
    parser.add_argument("--asr-workers", type=int, default=1)
    parser.add_argument("--wer-threshold-pct", type=float, default=100.0)
    parser.add_argument("--use-recommended-threshold", action="store_true")
    parser.add_argument("--build-lhotse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--require-fastmss-alignment",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--scratch-dir", type=Path, default=None)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index must be in [0, shard-count)")
    if args.asr_workers < 1:
        parser.error("asr-workers must be at least 1")
    for path in (args.data_root, args.masked_audio_dir, args.textgrid_dir):
        if not path.is_dir():
            parser.error(f"required directory does not exist: {path}")
    if args.sessions_file is not None and not args.sessions_file.is_file():
        parser.error(f"sessions file does not exist: {args.sessions_file}")
    return args


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir / "shards" / f"shard_{args.shard_index:05d}"
    if args.model_cache_dir:
        os.environ["NEMO_CACHE_DIR"] = str(Path(args.model_cache_dir).resolve())
    scratch_dir = args.scratch_dir or (Path(tempfile.gettempdir()) / f"david_ai_parakeet_{args.shard_index:05d}")
    shutil.rmtree(scratch_dir, ignore_errors=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    initial_tasks = build_segment_tasks(
        SegmentTaskConfig(
            data_root=args.data_root.resolve(),
            masked_audio_dir=args.masked_audio_dir.resolve(),
            textgrid_dir=args.textgrid_dir.resolve(),
            sessions_file=args.sessions_file.resolve() if args.sessions_file else None,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
        )
    )
    if not initial_tasks:
        logger.info("Shard {} contains no segments", args.shard_index)
        return 0
    logger.info("Built {} segment tasks for shard {}/{}", len(initial_tasks), args.shard_index, args.shard_count)

    from nemo_curator.backends.xenna import XennaExecutor
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.resources import Resources

    pipeline = Pipeline(
        name="david-ai-parakeet-segment-wer",
        description="Segment Parakeet ASR followed by normalized WER",
        stages=[
            SegmentClipExtractionStage(scratch_dir=str(scratch_dir)),
            ParallelInferenceAsrNemoStage(
                model_name=args.model_name,
                cache_dir=None,
                filepath_key="segment_audio_filepath",
                pred_text_key="pred_text",
                batch_size=args.asr_batch_size,
                resources=Resources(gpus=1.0),
                worker_count=args.asr_workers,
            ),
            SegmentWERStage(),
        ],
    )

    try:
        results = pipeline.run(
            executor=XennaExecutor(config={"execution_mode": "streaming"}),
            initial_tasks=initial_tasks,
        )
        result_tasks = list(results or [])
        threshold = args.wer_threshold_pct
        if args.use_recommended_threshold:
            preliminary_rows = [dict(task.data) for task in result_tasks]
            threshold = float(build_wer_distribution(preliminary_rows, threshold)["recommended_threshold_pct"])
        report = write_pipeline_outputs(
            result_tasks,
            output_dir=output_dir,
            threshold_pct=threshold,
            require_fastmss_alignment=args.require_fastmss_alignment,
        )
        if args.build_lhotse:
            summaries = build_all_variants(
                output_dir / "segments_with_wer.jsonl",
                output_dir / "lhotse",
            )
            logger.info("Lhotse variants: {}", summaries)
        logger.info("WER report: {}", report)
        logger.info("Outputs: {}", output_dir)
        return 0
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
