# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Video pipeline benchmarking script.

This script runs a video pipeline benchmark reusing the pipeline and argparser from
tutorials/video/getting-started/video_split_clip_example.py with comprehensive
metrics collection and various executor support.
"""

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from loguru import logger
from utils import setup_executor, write_benchmark_results

# Add tutorials directory to path to import the pipeline creation function
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tutorials" / "video" / "getting-started"))

from video_split_clip_example import (  # noqa: E402
    create_video_splitting_argparser,
    create_video_splitting_pipeline,
)


def run_video_pipeline_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run the video pipeline benchmark and collect comprehensive metrics."""
    executor = setup_executor(args.executor)

    video_dir = Path(args.video_dir).absolute()
    output_path = Path(args.output_path).absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Video directory: {video_dir}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Video limit: {args.video_limit}")
    logger.info(f"Splitting algorithm: {args.splitting_algorithm}")
    logger.info(f"Transcode encoder: {args.transcode_encoder}")
    logger.info(f"Hardware acceleration (GPU decode): {args.transcode_use_hwaccel}")
    logger.info(f"Generate embeddings: {args.generate_embeddings}")
    logger.info(f"Motion filter: {args.motion_filter}")
    logger.info(f"Generate captions: {args.generate_captions}")
    if args.generate_embeddings:
        logger.info(f"Embedding algorithm: {args.embedding_algorithm}")
        logger.info(f"Embedding GPU memory: {args.embedding_gpu_memory_gb}GB")
    logger.debug(f"Executor: {executor}")

    # Create pipeline using the tutorial's function
    pipeline = create_video_splitting_pipeline(args)

    run_start_time = time.perf_counter()

    try:
        logger.info("Running video pipeline...")
        logger.info(f"Pipeline description:\n{pipeline.describe()}")

        output_tasks = pipeline.run(executor)
        run_time_taken = time.perf_counter() - run_start_time

        # Calculate metrics from output tasks
        # Count unique videos by their input_video path
        unique_videos = {task.data.input_video for task in output_tasks if task.data and hasattr(task.data, "input_video") and task.data.input_video}
        num_videos_processed = len(unique_videos)
        num_clips_generated = sum(len(task.data.clips) for task in output_tasks if task.data and hasattr(task.data, "clips") and task.data.clips)

        logger.success(f"Benchmark completed in {run_time_taken:.2f}s")
        logger.success(f"Processed {num_videos_processed} videos")
        logger.success(f"Generated {num_clips_generated} clips")
        success = True

    except Exception as e:  # noqa: BLE001
        error_traceback = traceback.format_exc()
        logger.error(f"Benchmark failed: {e}")
        logger.debug(f"Full traceback:\n{error_traceback}")
        output_tasks = []
        run_time_taken = time.perf_counter() - run_start_time
        num_videos_processed = 0
        num_clips_generated = 0
        success = False

    return {
        "params": {
            "executor": args.executor,
            "video_dir": str(video_dir),
            "output_path": str(output_path),
            "benchmark_results_path": str(args.benchmark_results_path),
            "video_limit": args.video_limit,
            "splitting_algorithm": args.splitting_algorithm,
            "split_duration": args.fixed_stride_split_duration,
            "transcode_encoder": args.transcode_encoder,
            "transcode_use_hwaccel": args.transcode_use_hwaccel,
            "generate_embeddings": args.generate_embeddings,
            "embedding_algorithm": args.embedding_algorithm,
            "motion_filter": args.motion_filter,
            "generate_captions": args.generate_captions,
            "model_dir": args.model_dir,
        },
        "metrics": {
            "is_success": success,
            "time_taken_s": run_time_taken,
            "num_videos_processed": num_videos_processed,
            "num_clips_generated": num_clips_generated,
            "num_output_tasks": len(output_tasks),
            "throughput_videos_per_sec": num_videos_processed / run_time_taken if run_time_taken > 0 else 0,
            "throughput_clips_per_sec": num_clips_generated / run_time_taken if run_time_taken > 0 else 0,
        },
        "tasks": output_tasks,
    }


def main() -> int:
    # Start with the tutorial's argparser
    parser = create_video_splitting_argparser()

    # Add benchmark-specific arguments
    parser.add_argument(
        "--benchmark-results-path",
        type=Path,
        required=True,
        help="Path to write benchmark results",
    )
    parser.add_argument(
        "--executor",
        default="xenna",
        choices=["xenna", "ray_data"],
        help="Executor to use for pipeline execution",
    )

    args = parser.parse_args()

    logger.info("=== Video Pipeline Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    try:
        results = run_video_pipeline_benchmark(args)

    except Exception as e:  # noqa: BLE001
        error_traceback = traceback.format_exc()
        print(f"Benchmark failed: {e}")
        logger.debug(f"Full traceback:\n{error_traceback}")
        results = {
            "params": vars(args),
            "metrics": {
                "is_success": False,
            },
            "tasks": [],
        }
    finally:
        write_benchmark_results(results, args.benchmark_results_path)

    # Return proper exit code based on success
    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
