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

"""Video pipeline benchmarking script.

This script runs a basic video pipeline benchmark (read, split, transcode, write)
with comprehensive metrics collection using various executors and logs results to configured sinks.
"""

import argparse
import time
import traceback
from pathlib import Path
from typing import Any

from loguru import logger
from utils import setup_executor, write_benchmark_results

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.video.clipping.clip_extraction_stages import ClipTranscodingStage, FixedStrideExtractorStage
from nemo_curator.stages.video.clipping.clip_frame_extraction import ClipFrameExtractionStage
from nemo_curator.stages.video.embedding.cosmos_embed1 import (
    CosmosEmbed1EmbeddingStage,
    CosmosEmbed1FrameCreationStage,
)
from nemo_curator.stages.video.io.clip_writer import ClipWriterStage
from nemo_curator.stages.video.io.video_reader import VideoReader
from nemo_curator.utils.decoder_utils import FrameExtractionPolicy, FramePurpose


def create_video_pipeline(  # noqa: PLR0913
    video_dir: Path,
    output_path: Path,
    video_limit: int | None = None,
    split_duration: float = 10.0,
    min_clip_length_s: float = 2.0,
    transcode_cpus_per_worker: float = 6.0,
    transcode_encoder: str = "libopenh264",
    transcode_use_hwaccel: bool = False,
    generate_embeddings: bool = False,
    embedding_variant: str = "224p",
    embedding_gpu_memory_gb: float = 20.0,
    embedding_target_fps: float = 2.0,
    model_dir: str = "./models",
    verbose: bool = False,
) -> Pipeline:
    """Create a video pipeline with read, split, transcode, optional embeddings, and write stages."""
    pipeline = Pipeline(name="video_pipeline_benchmark", description="Video processing pipeline benchmark")

    # Stage 1: Read videos
    pipeline.add_stage(
        VideoReader(
            input_video_path=str(video_dir),
            video_limit=video_limit,
            verbose=verbose,
        )
    )

    # Stage 2: Split videos into fixed-stride clips
    pipeline.add_stage(
        FixedStrideExtractorStage(
            clip_len_s=split_duration,
            clip_stride_s=split_duration,
            min_clip_length_s=min_clip_length_s,
            limit_clips=0,  # No limit
            verbose=verbose,
        )
    )

    # Stage 3: Transcode clips
    pipeline.add_stage(
        ClipTranscodingStage(
            num_cpus_per_worker=transcode_cpus_per_worker,
            encoder=transcode_encoder,
            encoder_threads=1,
            encode_batch_size=16,
            use_hwaccel=transcode_use_hwaccel,
            verbose=verbose,
        )
    )

    # Optional: Embedding generation stages
    if generate_embeddings:
        # Stage 4a: Extract frames from clips for embedding
        pipeline.add_stage(
            ClipFrameExtractionStage(
                extraction_policies=(FrameExtractionPolicy.sequence,),
                extract_purposes=[FramePurpose.EMBEDDINGS],
                target_res=(-1, -1),  # Use original resolution
                verbose=verbose,
            )
        )

        # Stage 4b: Prepare frames for Cosmos Embed1 model
        pipeline.add_stage(
            CosmosEmbed1FrameCreationStage(
                model_dir=model_dir,
                variant=embedding_variant,
                target_fps=embedding_target_fps,
                verbose=verbose,
            )
        )

        # Stage 4c: Generate embeddings using Cosmos Embed1
        pipeline.add_stage(
            CosmosEmbed1EmbeddingStage(
                model_dir=model_dir,
                variant=embedding_variant,
                gpu_memory_gb=embedding_gpu_memory_gb,
                verbose=verbose,
            )
        )

    # Final Stage: Write clips (and embeddings if generated)
    pipeline.add_stage(
        ClipWriterStage(
            output_path=str(output_path),
            input_path=str(video_dir),
            upload_clips=True,
            dry_run=False,
            generate_embeddings=generate_embeddings,
            generate_previews=False,
            generate_captions=False,
            embedding_algorithm=f"cosmos-embed1-{embedding_variant}" if generate_embeddings else "cosmos-embed1",
            verbose=verbose,
        )
    )

    return pipeline


def run_video_pipeline_benchmark(  # noqa: PLR0913
    video_dir: Path,
    output_path: Path,
    executor_name: str,
    benchmark_results_path: Path,
    video_limit: int | None = None,
    split_duration: float = 10.0,
    min_clip_length_s: float = 2.0,
    transcode_cpus_per_worker: float = 6.0,
    transcode_encoder: str = "libopenh264",
    transcode_use_hwaccel: bool = False,
    generate_embeddings: bool = False,
    embedding_variant: str = "224p",
    embedding_gpu_memory_gb: float = 20.0,
    embedding_target_fps: float = 2.0,
    model_dir: str = "./models",
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the video pipeline benchmark and collect comprehensive metrics."""
    executor = setup_executor(executor_name)

    video_dir = video_dir.absolute()
    output_path = output_path.absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Video directory: {video_dir}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Video limit: {video_limit}")
    logger.info(f"Split duration: {split_duration}s")
    logger.info(f"Hardware acceleration (GPU decode): {transcode_use_hwaccel}")
    logger.info(f"Generate embeddings: {generate_embeddings}")
    if generate_embeddings:
        logger.info(f"Embedding variant: cosmos-embed1-{embedding_variant}")
        logger.info(f"Embedding GPU memory: {embedding_gpu_memory_gb}GB")
        logger.info(f"Model directory: {model_dir}")
    logger.debug(f"Executor: {executor}")

    # Create pipeline
    pipeline = create_video_pipeline(
        video_dir=video_dir,
        output_path=output_path,
        video_limit=video_limit,
        split_duration=split_duration,
        min_clip_length_s=min_clip_length_s,
        transcode_cpus_per_worker=transcode_cpus_per_worker,
        transcode_encoder=transcode_encoder,
        transcode_use_hwaccel=transcode_use_hwaccel,
        generate_embeddings=generate_embeddings,
        embedding_variant=embedding_variant,
        embedding_gpu_memory_gb=embedding_gpu_memory_gb,
        embedding_target_fps=embedding_target_fps,
        model_dir=model_dir,
        verbose=verbose,
    )

    run_start_time = time.perf_counter()

    try:
        logger.info("Running video pipeline...")
        logger.info(f"Pipeline description:\n{pipeline.describe()}")

        output_tasks = pipeline.run(executor)
        run_time_taken = time.perf_counter() - run_start_time

        # Calculate metrics from output tasks
        # VideoReader is a CompositeStage that decomposes to FilePartitioningStage + VideoReaderStage
        # So _stage_perf indices: 0=FilePartitioning, 1=VideoReader, 2=FixedStride, 3=Transcode, 4=Writer
        # Note: One video can produce multiple output tasks due to clip chunking in ClipTranscodingStage
        # Count unique videos by their input_video path
        unique_videos = {task.data.input_video for task in output_tasks if task.data and task.data.input_video}
        num_videos_processed = len(unique_videos)
        num_clips_generated = sum(len(task.data.clips) for task in output_tasks if task.data and task.data.clips)

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
            "executor": executor_name,
            "video_dir": str(video_dir),
            "output_path": str(output_path),
            "benchmark_results_path": str(benchmark_results_path),
            "video_limit": video_limit,
            "split_duration": split_duration,
            "min_clip_length_s": min_clip_length_s,
            "transcode_encoder": transcode_encoder,
            "transcode_use_hwaccel": transcode_use_hwaccel,
            "generate_embeddings": generate_embeddings,
            "embedding_variant": embedding_variant,
            "embedding_gpu_memory_gb": embedding_gpu_memory_gb,
            "model_dir": model_dir,
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
    parser = argparse.ArgumentParser(description="Video pipeline benchmark")
    # Paths
    parser.add_argument("--benchmark-results-path", type=Path, required=True, help="Path to benchmark results")
    parser.add_argument("--video-dir", required=True, type=Path, help="Path to input video directory")
    parser.add_argument(
        "--output-path", default=Path("./video_pipeline_output"), type=Path, help="Output directory for results"
    )
    # Executor
    parser.add_argument("--executor", default="xenna", choices=["xenna", "ray_data"], help="Executor to use")
    # Pipeline-specific
    parser.add_argument("--video-limit", type=int, default=None, help="Limit number of videos to process")
    parser.add_argument("--split-duration", type=float, default=10.0, help="Duration of clips in seconds")
    parser.add_argument("--min-clip-length-s", type=float, default=2.0, help="Minimum clip length in seconds")
    parser.add_argument(
        "--transcode-cpus-per-worker", type=float, default=6.0, help="CPUs per worker for transcoding"
    )
    parser.add_argument(
        "--transcode-encoder",
        type=str,
        default="libopenh264",
        choices=["libopenh264", "libx264", "h264_nvenc"],
        help="Video encoder for transcoding",
    )
    parser.add_argument(
        "--transcode-use-hwaccel",
        action="store_true",
        default=False,
        help="Use GPU hardware acceleration for decoding (NVDEC). Works on A100 even without NVENC.",
    )
    # Embedding arguments
    parser.add_argument(
        "--generate-embeddings",
        action="store_true",
        default=False,
        help="Generate Cosmos Embed1 embeddings for video clips.",
    )
    parser.add_argument(
        "--embedding-variant",
        type=str,
        default="224p",
        choices=["224p", "336p", "448p"],
        help="Cosmos Embed1 model variant (resolution).",
    )
    parser.add_argument(
        "--embedding-gpu-memory-gb",
        type=float,
        default=20.0,
        help="GPU memory in GB per worker for embedding generation.",
    )
    parser.add_argument(
        "--embedding-target-fps",
        type=float,
        default=2.0,
        help="Target FPS for frame sampling in embedding generation.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models",
        help="Directory containing model weights (downloaded automatically if not present).",
    )
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose logging")

    args = parser.parse_args()

    logger.info("=== Video Pipeline Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    try:
        results = run_video_pipeline_benchmark(
            video_dir=args.video_dir,
            output_path=args.output_path,
            executor_name=args.executor,
            benchmark_results_path=args.benchmark_results_path,
            video_limit=args.video_limit,
            split_duration=args.split_duration,
            min_clip_length_s=args.min_clip_length_s,
            transcode_cpus_per_worker=args.transcode_cpus_per_worker,
            transcode_encoder=args.transcode_encoder,
            transcode_use_hwaccel=args.transcode_use_hwaccel,
            generate_embeddings=args.generate_embeddings,
            embedding_variant=args.embedding_variant,
            embedding_gpu_memory_gb=args.embedding_gpu_memory_gb,
            embedding_target_fps=args.embedding_target_fps,
            model_dir=args.model_dir,
            verbose=args.verbose,
        )

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

