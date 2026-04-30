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

"""CLI runner for the long-form-audio ALM pretraining pipeline.

An example to do dry-run: 


# Export the two workarounds for this session
#    - prestart only 4 workers at a time, sidesteps issue #40131
#    - bump the state-API list cap so Xenna's monitor (limit=40000) doesn't trip
export RAY_worker_maximum_startup_concurrency=4
export RAY_MAX_LIMIT_FROM_API_SERVER=50000

.venv/bin/python -m tutorials.audio.audio_pretrain.run \
      --input-manifest test.jsonl \
      --audio-dir /tmp \
      --output-dir /tmp/dryrun_unused \
      --output-manifest /tmp/test_dryrun.jsonl \
      --metrics-path /tmp/test_metrics.json \
      --max-duration-sec 900 \
      --dry-run

"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time

from loguru import logger

from tutorials.audio.audio_pretrain.pipeline import (
    build_audio_pretrain_pipeline,
    finalize_audio_pretrain_outputs,
    prepare_audio_pretrain_outputs,
)

_EXECUTOR_FACTORIES = {
    "xenna": "nemo_curator.backends.xenna:XennaExecutor",
    "ray_data": "nemo_curator.backends.experimental.ray_data:RayDataExecutor",
    "ray_actor_pool": "nemo_curator.backends.experimental.ray_actor_pool:RayActorPoolExecutor",
}


def _create_executor(backend: str, **kwargs: object) -> object:
    if backend not in _EXECUTOR_FACTORIES:
        msg = f"Unknown backend {backend!r}; choose from {list(_EXECUTOR_FACTORIES)}"
        raise ValueError(msg)
    module_path, class_name = _EXECUTOR_FACTORIES[backend].rsplit(":", 1)
    return getattr(importlib.import_module(module_path), class_name)(**kwargs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cut long-form diarized audio into snippets for ALM pretraining.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-manifest", required=True, help="Path to input JSONL manifest")
    parser.add_argument("--audio-dir", required=True, help="Directory containing source audio files")
    parser.add_argument("--output-dir", required=True, help="Directory to write snippet audio files")
    parser.add_argument("--output-manifest", required=True, help="Path to output JSONL manifest (one row per snippet)")
    parser.add_argument("--metrics-path", required=True, help="Path to metrics summary JSON")
    parser.add_argument("--max-duration-sec", type=float, required=True, help="Maximum snippet duration in seconds")
    parser.add_argument(
        "--min-duration-sec", type=float, default=0.5, help="Minimum snippet duration in seconds (default 0.5)"
    )
    parser.add_argument(
        "--min-overlap-sec",
        type=float,
        default=0.5,
        help="Minimum intersection (sec) for two segments to be considered overlapping (default 0.5)",
    )
    parser.add_argument(
        "--max-segment-gap-in-snippet",
        type=float,
        default=30.0,
        help=(
            "Maximum silence (sec) allowed between adjacent segments inside "
            "the same snippet; larger gaps force a new snippet. Default 30s "
            "treats long silences as conversation boundaries (topic change, "
            "ad break, recording boundary) and avoids bridging them."
        ),
    )
    parser.add_argument(
        "--target-sample-rate", type=int, default=16000, help="Output snippet sample rate (default 16000)"
    )
    parser.add_argument("--output-format", choices=["wav", "flac", "ogg"], default="flac", help="Output audio format")
    parser.add_argument(
        "--audio-filepath-key",
        default="audio_filepath",
        help="JSONL field naming the source audio path (default 'audio_filepath')",
    )
    parser.add_argument("--dataset-name", default="long_form_audio", help="Tag attached to emitted AudioTasks")
    parser.add_argument(
        "--backend",
        choices=sorted(_EXECUTOR_FACTORIES),
        default="xenna",
        help="Execution backend (default 'xenna')",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["streaming", "batch"],
        default="streaming",
        help="Xenna execution mode (default 'streaming')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip audio I/O entirely -- no snippet audio files are written -- "
            "but still produce the snippet manifest and metrics summary. "
            "Useful for sizing up a real dataset before committing to a full run."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")

    for path in (args.output_dir, os.path.dirname(args.output_manifest), os.path.dirname(args.metrics_path)):
        if path:
            os.makedirs(path, exist_ok=True)

    pipeline = build_audio_pretrain_pipeline(
        input_manifest=args.input_manifest,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        output_manifest_path=args.output_manifest,
        metrics_path=args.metrics_path,
        max_duration_sec=args.max_duration_sec,
        min_duration_sec=args.min_duration_sec,
        min_overlap_sec=args.min_overlap_sec,
        max_segment_gap_in_snippet=args.max_segment_gap_in_snippet,
        target_sample_rate=args.target_sample_rate,
        output_format=args.output_format,
        audio_filepath_key=args.audio_filepath_key,
        dataset_name=args.dataset_name,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        logger.info("DRY RUN: snippet audio files will NOT be written; manifest + metrics only.")
    logger.info(pipeline.describe())

    executor_kwargs: dict[str, object] = {}
    if args.backend == "xenna":
        executor_kwargs["config"] = {"execution_mode": args.execution_mode}
    executor = _create_executor(args.backend, **executor_kwargs)

    logger.info(f"Running on backend={args.backend}")
    prepare_audio_pretrain_outputs(args.output_manifest, args.metrics_path)
    t0 = time.monotonic()
    pipeline.run(executor)
    elapsed = time.monotonic() - t0
    finalize_audio_pretrain_outputs(args.output_manifest, args.metrics_path)
    logger.info(
        f"Pipeline finished in {elapsed:.2f}s ({elapsed / 60:.2f} min). "
        f"Snippets in {args.output_dir}, manifest at {args.output_manifest}, "
        f"metrics at {args.metrics_path}"
    )


if __name__ == "__main__":
    main()
