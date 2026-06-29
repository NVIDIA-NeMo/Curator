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

"""Build a diverse benchmark dataset for caption quality evaluation.

This script creates a representative subset of video clips from a large video
dataset by:

  1. Randomly sampling a pool of source videos (default: 3000).
  2. Running the NeMo Curator video pipeline to split videos into fixed-stride
     clips, filter by aesthetic score, and compute CosmosEmbed1 embeddings.
  3. Applying K-means clustering (default: K=200) on the 256-dim CosmosEmbed1
     embeddings to identify diverse visual clusters.
  4. Selecting one representative clip per cluster (closest to centroid, one per
     source video) to form the final benchmark dataset.

The output directory contains symlinks to the selected clips, their embeddings,
and source videos -- ready to be used with ``caption_clipscore.py``.

Example:

    python build_benchmark_dataset.py \\
        --video-dir /path/to/video_dataset \\
        --output-dir /path/to/benchmark_200 \\
        --model-dir /path/to/models \\
        --sample-size 3000 \\
        --num-clusters 200

Then score with caption_clipscore.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.cluster import KMeans

# Import the pipeline creation function from the video tutorial
_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tutorials" / "video" / "getting-started"))

from video_split_clip_example import (  # noqa: E402
    create_video_splitting_argparser,
    create_video_splitting_pipeline,
)


def _sample_videos(video_dir: str, sample_size: int, seed: int) -> list[str]:
    """Randomly sample source videos from a directory."""
    videos = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    if len(videos) <= sample_size:
        logger.info(f"Video dir has {len(videos)} videos (<= sample_size), using all")
        return videos
    random.seed(seed)
    sampled = random.sample(videos, sample_size)
    logger.info(f"Sampled {len(sampled)} from {len(videos)} videos (seed={seed})")
    return sampled


def _run_embedding_pipeline(
    input_dir: str,
    output_dir: str,
    model_dir: str,
    split_duration: float,
    aesthetic_threshold: float,
) -> None:
    """Run the NeMo Curator video pipeline for splitting + embedding (no captioning)."""
    # Build args as if parsed from the command line
    pipeline_parser = create_video_splitting_argparser()
    pipeline_args = pipeline_parser.parse_args(
        [
            "--video-dir",
            input_dir,
            "--output-path",
            output_dir,
            "--model-dir",
            model_dir,
            "--splitting-algorithm",
            "fixed_stride",
            "--fixed-stride-split-duration",
            str(split_duration),
            "--embedding-algorithm",
            "cosmos-embed1-224p",
            "--aesthetic-threshold",
            str(aesthetic_threshold),
        ]
    )

    logger.info("Creating and running embedding pipeline ...")
    pipeline = create_video_splitting_pipeline(pipeline_args)
    pipeline.run()
    logger.info("Pipeline completed")


def _kmeans_select(
    emb_dir: str,
    meta_dir: str,
    num_clusters: int,
    seed: int,
) -> list[tuple[str, dict]]:
    """K-means cluster embeddings, select one representative per cluster."""
    emb_files = sorted(glob.glob(f"{emb_dir}/*.pickle"))
    uids = [Path(f).stem for f in emb_files]
    embeddings = []
    for f in emb_files:
        with open(f, "rb") as fh:
            arr = pickle.load(fh)  # noqa: S301
        embeddings.append(np.asarray(arr).flatten())
    emb_matrix = np.stack(embeddings).astype(np.float32)
    logger.info(f"{emb_matrix.shape[0]} clips, {emb_matrix.shape[1]}-dim embeddings")

    uid_to_meta = {}
    for uid in uids:
        meta_path = f"{meta_dir}/{uid}.json"
        if os.path.exists(meta_path):
            with open(meta_path) as fh:
                uid_to_meta[uid] = json.load(fh)

    logger.info(f"K-means K={num_clusters} ...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed)
    labels = kmeans.fit_predict(emb_matrix)
    centers = kmeans.cluster_centers_
    cluster_sizes = np.bincount(labels, minlength=num_clusters)
    logger.info(
        f"Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, median={int(np.median(cluster_sizes))}"
    )

    selected = []
    used_sources = set()
    for k in range(num_clusters):
        cluster_indices = np.where(labels == k)[0]
        dists = np.linalg.norm(emb_matrix[cluster_indices] - centers[k], axis=1)
        sorted_idx = cluster_indices[np.argsort(dists)]
        picked = False
        for idx in sorted_idx:
            uid = uids[idx]
            meta = uid_to_meta.get(uid, {})
            src = meta.get("source_video", "")
            if src and src not in used_sources:
                selected.append((uid, meta))
                used_sources.add(src)
                picked = True
                break
        if not picked:
            uid = uids[sorted_idx[0]]
            selected.append((uid, uid_to_meta.get(uid, {})))

    logger.info(f"Selected {len(selected)} clips from {len(used_sources)} unique source videos")
    return selected


def _build_output(
    selected: list[tuple[str, dict]],
    pipeline_output_dir: str,
    output_dir: str,
) -> None:
    """Create the benchmark output directory with symlinks."""
    emb_dir = f"{pipeline_output_dir}/ce1_embd"
    clip_dir = f"{pipeline_output_dir}/clips"

    if not os.path.isdir(emb_dir):
        msg = f"Pipeline did not produce embeddings at {emb_dir}"
        raise FileNotFoundError(msg)
    if not os.path.isdir(clip_dir):
        msg = f"Pipeline did not produce clips at {clip_dir}"
        raise FileNotFoundError(msg)

    os.makedirs(f"{output_dir}/ce1_embd", exist_ok=True)
    os.makedirs(f"{output_dir}/clips", exist_ok=True)
    os.makedirs(f"{output_dir}/input", exist_ok=True)

    input_videos_linked = set()
    uids = []

    for uid, meta in selected:
        uids.append(uid)

        # Symlink embedding
        src = f"{emb_dir}/{uid}.pickle"
        dst = f"{output_dir}/ce1_embd/{uid}.pickle"
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(os.path.abspath(src), dst)

        # Symlink clip
        src = f"{clip_dir}/{uid}.mp4"
        dst = f"{output_dir}/clips/{uid}.mp4"
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(os.path.abspath(src), dst)

        # Symlink source video
        src_video = meta.get("source_video", "")
        if src_video and os.path.exists(src_video):
            vid_name = os.path.basename(src_video)
            dst = f"{output_dir}/input/{vid_name}"
            if vid_name not in input_videos_linked and not os.path.lexists(dst):
                os.symlink(os.path.abspath(src_video), dst)
                input_videos_linked.add(vid_name)

    # Save selected UIDs with source video and span for cross-run resolution
    with open(f"{output_dir}/selected_uids.txt", "w") as f:
        for uid, meta in sorted(selected, key=lambda x: x[0]):
            src_name = Path(meta.get("source_video", "")).name
            span = meta.get("duration_span", [0, 0])
            f.write(f"{uid}\t{src_name}\t{span[0]}\t{span[1]}\n")

    # Summary
    total_duration = sum(
        meta.get("duration_span", [0, 0])[1] - meta.get("duration_span", [0, 0])[0] for _, meta in selected
    )
    logger.info(f"Benchmark dataset ready at: {output_dir}")
    logger.info(f"  Clips       : {len(uids)}")
    logger.info(f"  Source videos: {len(input_videos_linked)}")
    logger.info(f"  Duration    : {total_duration:.0f}s ({total_duration / 60:.1f} min)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a diverse benchmark dataset for caption quality evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--video-dir",
        required=True,
        help="Directory containing source video files (e.g. OpenVid-1M).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for the benchmark dataset.",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Root model directory for the pipeline (contains CosmosEmbed1 weights etc.).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3000,
        help="Number of videos to randomly sample from --video-dir (default: 3000).",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=200,
        help="Number of K-means clusters = number of clips in the final dataset (default: 200).",
    )
    parser.add_argument(
        "--split-duration",
        type=float,
        default=10.0,
        help="Fixed-stride split duration in seconds (default: 10.0).",
    )
    parser.add_argument(
        "--aesthetic-threshold",
        type=float,
        default=3.5,
        help="Minimum aesthetic score for clip filtering (default: 3.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and K-means (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Step 1: Sample videos and create temp input directory
    logger.info("[Step 1/4] Sampling videos ...")
    sampled_videos = _sample_videos(args.video_dir, args.sample_size, args.seed)
    sample_input_dir = f"{args.output_dir}/_sample_input"
    os.makedirs(sample_input_dir, exist_ok=True)
    for vid in sampled_videos:
        src = os.path.join(args.video_dir, vid)
        dst = os.path.join(sample_input_dir, vid)
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(src), dst)

    # Step 2: Run embedding pipeline
    logger.info("[Step 2/4] Running embedding pipeline ...")
    pipeline_output = f"{args.output_dir}/_pipeline_output"
    _run_embedding_pipeline(
        input_dir=sample_input_dir,
        output_dir=pipeline_output,
        model_dir=args.model_dir,
        split_duration=args.split_duration,
        aesthetic_threshold=args.aesthetic_threshold,
    )

    # Step 3: K-means clustering and selection
    logger.info("[Step 3/4] K-means clustering ...")
    selected = _kmeans_select(
        emb_dir=f"{pipeline_output}/ce1_embd",
        meta_dir=f"{pipeline_output}/metas/v0",
        num_clusters=args.num_clusters,
        seed=args.seed,
    )

    # Step 4: Build output directory
    logger.info("[Step 4/4] Building output directory ...")
    _build_output(selected, pipeline_output, args.output_dir)


if __name__ == "__main__":
    main()
