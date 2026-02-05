# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Estimate GPU and time requirements for video processing pipelines.

Examples:
    # Basic estimation
    python estimate_video_resources.py --input-path /data/videos

    # With specific options
    python estimate_video_resources.py \\
        --input-path /data/videos \\
        --clip-method transnetv2 \\
        --caption \\
        --embed
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.introspect import estimate_gpu_memory, estimate_throughput

# Video file extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


def count_videos(input_path: str) -> tuple[int, float]:
    """Count videos and estimate total duration.

    Returns:
        Tuple of (video_count, estimated_total_hours)
    """
    path = Path(input_path)
    if not path.exists():
        return 0, 0.0

    video_count = 0
    total_size_gb = 0.0

    if path.is_file():
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            video_count = 1
            total_size_gb = path.stat().st_size / (1024**3)
    else:
        for ext in VIDEO_EXTENSIONS:
            for video_file in path.rglob(f"*{ext}"):
                video_count += 1
                total_size_gb += video_file.stat().st_size / (1024**3)

    # Rough estimate: 1GB of video ≈ 1 hour at typical compression
    estimated_hours = total_size_gb * 1.0

    return video_count, estimated_hours


def estimate_pipeline_resources(
    video_count: int,
    total_hours: float,
    clip_method: str,
    enable_caption: bool,
    enable_embed: bool,
    embed_model: str,
    enable_motion_filter: bool,
) -> dict:
    """Estimate resources for the pipeline."""
    # Estimate clips per video
    if clip_method == "transnetv2":
        clips_per_video = 50  # Typical scene count
    else:
        clips_per_video = int(total_hours * 3600 / 10 / max(video_count, 1))  # 10s clips

    total_clips = video_count * clips_per_video

    # Calculate GPU memory requirements
    stages_used = []
    max_gpu_memory = 0.0

    # Clipping stage
    if clip_method == "transnetv2":
        stages_used.append("TransNetV2ClipExtractionStage")
        max_gpu_memory = max(max_gpu_memory, estimate_gpu_memory("TransNetV2ClipExtractionStage"))
    else:
        stages_used.append("FixedStrideExtractorStage")

    stages_used.append("ClipTranscodingStage")

    if enable_motion_filter:
        stages_used.append("MotionFilterStage")
        max_gpu_memory = max(max_gpu_memory, estimate_gpu_memory("MotionFilterStage"))

    if enable_caption:
        stages_used.append("CaptionGenerationStage")
        max_gpu_memory = max(max_gpu_memory, estimate_gpu_memory("CaptionGenerationStage"))

    if enable_embed:
        if "cosmos" in embed_model:
            stages_used.append("CosmosEmbed1EmbeddingStage")
            max_gpu_memory = max(max_gpu_memory, estimate_gpu_memory("CosmosEmbed1EmbeddingStage"))
        else:
            stages_used.append("InternVideo2EmbeddingStage")
            max_gpu_memory = max(max_gpu_memory, estimate_gpu_memory("InternVideo2EmbeddingStage"))

    stages_used.append("ClipWriterStage")

    # Estimate processing time (bottleneck stage)
    bottleneck_stage = None
    min_throughput = float("inf")

    for stage in stages_used:
        throughput = estimate_throughput(stage)
        if throughput < min_throughput:
            min_throughput = throughput
            bottleneck_stage = stage

    # Calculate time based on bottleneck
    if (bottleneck_stage and "Video" in bottleneck_stage) or "TransNet" in str(bottleneck_stage):
        # Video-level processing
        processing_time_hours = video_count / (min_throughput * 3600)
    else:
        # Clip-level processing
        processing_time_hours = total_clips / (min_throughput * 3600)

    # Recommend workers based on GPU memory
    if max_gpu_memory >= 24:
        recommended_workers = 1  # Need full GPU
    elif max_gpu_memory >= 16:
        recommended_workers = 2
    elif max_gpu_memory >= 8:
        recommended_workers = 4
    else:
        recommended_workers = 8

    return {
        "video_count": video_count,
        "estimated_video_hours": round(total_hours, 1),
        "estimated_clips": total_clips,
        "stages_used": stages_used,
        "max_gpu_memory_gb": max_gpu_memory,
        "estimated_time_hours": round(processing_time_hours, 1),
        "bottleneck_stage": bottleneck_stage,
        "recommended_workers": recommended_workers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate GPU and time requirements for video processing"
    )
    parser.add_argument(
        "--input-path", "-i", required=True, help="Path to input videos"
    )
    parser.add_argument(
        "--clip-method",
        choices=["transnetv2", "fixed_stride"],
        default="transnetv2",
        help="Clipping method",
    )
    parser.add_argument(
        "--caption", action="store_true", help="Enable captioning"
    )
    parser.add_argument(
        "--embed", action="store_true", default=True, help="Enable embedding"
    )
    parser.add_argument(
        "--no-embed", action="store_false", dest="embed", help="Disable embedding"
    )
    parser.add_argument(
        "--embed-model",
        default="cosmos-embed1-224p",
        help="Embedding model variant",
    )
    parser.add_argument(
        "--filter-motion", action="store_true", default=True, help="Enable motion filter"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )
    args = parser.parse_args()

    # Count videos
    video_count, total_hours = count_videos(args.input_path)

    if video_count == 0:
        print(f"No videos found in {args.input_path}")
        print(f"Supported extensions: {', '.join(VIDEO_EXTENSIONS)}")
        sys.exit(1)

    # Estimate resources
    estimate = estimate_pipeline_resources(
        video_count=video_count,
        total_hours=total_hours,
        clip_method=args.clip_method,
        enable_caption=args.caption,
        enable_embed=args.embed,
        embed_model=args.embed_model,
        enable_motion_filter=args.filter_motion,
    )

    if args.json:
        import json
        print(json.dumps(estimate, indent=2))
        return

    print("=" * 60)
    print("VIDEO PIPELINE RESOURCE ESTIMATE")
    print("=" * 60)
    print()
    print(f"Input Path: {args.input_path}")
    print(f"Videos Found: {estimate['video_count']}")
    print(f"Estimated Video Hours: {estimate['estimated_video_hours']}")
    print(f"Estimated Clips: {estimate['estimated_clips']}")
    print()
    print("Pipeline Configuration:")
    print(f"  Clipping: {args.clip_method}")
    print(f"  Captioning: {'enabled' if args.caption else 'disabled'}")
    print(f"  Embedding: {'enabled' if args.embed else 'disabled'}")
    if args.embed:
        print(f"  Embed Model: {args.embed_model}")
    print(f"  Motion Filter: {'enabled' if args.filter_motion else 'disabled'}")
    print()
    print("Resource Requirements:")
    print(f"  GPU Memory: {estimate['max_gpu_memory_gb']} GB minimum")
    print(f"  Estimated Time: {estimate['estimated_time_hours']} hours")
    print(f"  Bottleneck Stage: {estimate['bottleneck_stage']}")
    print(f"  Recommended Workers: {estimate['recommended_workers']}")
    print()
    print("Stages Used:")
    for stage in estimate["stages_used"]:
        gpu = estimate_gpu_memory(stage)
        gpu_str = f" ({gpu}GB GPU)" if gpu > 0 else " (CPU)"
        print(f"  - {stage}{gpu_str}")


if __name__ == "__main__":
    main()
