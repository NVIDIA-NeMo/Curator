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

"""TinyStories curation pipeline with resumability.

This tutorial demonstrates pipeline resumability using the TinyStories dataset.
The pipeline can be interrupted (e.g. Ctrl+C) and restarted with --resume to
skip already-completed partitions.

Usage:
    # First run (or full run)
    python resumable_main.py --resume

    # Interrupt with Ctrl+C, then restart — completed partitions are skipped
    python resumable_main.py --resume
"""

import argparse
import os
import time

import pandas as pd
import requests

from stages import (
    IncompleteStoryFilter,
    QuotationUnifier,
)

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.text.filters import ScoreFilter
from nemo_curator.stages.text.io.reader.jsonl import JsonlReaderStage
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modifiers import Modify


TINYSTORIES_URLS = {
    "train": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
    "valid": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
}

RECORD_SEPARATOR = "<|endoftext|>"


def download_and_partition(url: str, raw_dir: str, stories_per_file: int) -> None:
    """Download a TinyStories .txt file and split it into multiple JSONL partition files.

    Skips download and splitting if the files already exist.

    Args:
        url: URL of the TinyStories .txt file.
        raw_dir: Directory to write JSONL files into.
        stories_per_file: Number of stories per output JSONL file.
    """
    os.makedirs(raw_dir, exist_ok=True)

    txt_path = os.path.join(raw_dir, os.path.basename(url))
    if not os.path.exists(txt_path):
        print(f"Downloading {url}...")
        response = requests.get(url, timeout=120)  # noqa: S113
        response.raise_for_status()
        with open(txt_path, "wb") as f:
            f.write(response.content)
        print(f"Saved to {txt_path}")

    existing = [f for f in os.listdir(raw_dir) if f.endswith(".jsonl")]
    if existing:
        print(f"Found {len(existing)} existing JSONL partition files in {raw_dir}, skipping split.")
        return

    # Parse stories separated by <|endoftext|>
    stories = []
    current: list[str] = []
    with open(txt_path) as f:
        for line in f:
            if line.strip() == RECORD_SEPARATOR:
                if current:
                    stories.append(" ".join(current))
                    current = []
            else:
                stripped = line.strip()
                if stripped:
                    current.append(stripped)
    if current:
        stories.append(" ".join(current))

    num_files = 0
    for i, start in enumerate(range(0, len(stories), stories_per_file)):
        chunk = stories[start : start + stories_per_file]
        jsonl_path = os.path.join(raw_dir, f"tinystories_{i:04d}.jsonl")
        pd.DataFrame({"text": chunk}).to_json(jsonl_path, lines=True, orient="records")
        num_files += 1

    print(f"Split {len(stories)} stories into {num_files} JSONL files ({stories_per_file} stories each)")


def main(args: argparse.Namespace) -> None:
    ray_client = RayClient()
    ray_client.start()

    raw_dir = os.path.join(args.data_root, "raw", args.split)
    curated_dir = os.path.join(args.data_root, "curated", args.split)
    checkpoint_dir = os.path.join(args.data_root, "checkpoints", args.split)
    os.makedirs(curated_dir, exist_ok=True)

    print("Running the TinyStories curation pipeline")
    print(f"    Raw JSONL partitions : {raw_dir}")
    print(f"    Curated output       : {curated_dir}")
    if args.resume:
        print(f"    Checkpoint directory : {checkpoint_dir}")

    # Step 1: Download and split into JSONL partition files (no-op if already done)
    download_and_partition(TINYSTORIES_URLS[args.split], raw_dir, args.stories_per_file)

    # Step 2: Build the curation pipeline
    #
    # FilePartitioningStage  →  one FileGroupTask per JSONL file (resumable input)
    # JsonlReaderStage        →  reads the file group into a DocumentBatch
    # ScoreFilter             →  drops stories that don't end with punctuation
    # Modify                  →  normalises quotation marks
    # JsonlWriter             →  writes curated output and records completion (resumable output)
    pipeline = Pipeline(
        name="tinystories_curation",
        description="Curation pipeline for the TinyStories dataset with resumability.",
        stages=[
            FilePartitioningStage(
                file_paths=raw_dir,
                files_per_partition=1,
                file_extensions=[".jsonl"],
            ),
            JsonlReaderStage(),
            ScoreFilter(filter_obj=IncompleteStoryFilter()),
            Modify(modifier_fn=QuotationUnifier()),
            JsonlWriter(curated_dir),
        ],
    )

    # enable_resumability configures FilePartitioningStage (input) and JsonlWriter (output):
    #   - FilePartitioningStage skips partitions that already have a completion record
    #   - JsonlWriter writes a per-partition JSON record to checkpoint_dir/pipeline_complete/
    #     after each successful write, so restarts pick up from where they left off
    if args.resume:
        pipeline.enable_resumability(checkpoint_dir)
        print("Resumability enabled — completed partitions will be skipped on restart.")

    print("\nStarting the curation pipeline")
    start_time = time.time()
    pipeline.run(RayDataExecutor())
    print(f"\nCuration pipeline finished in {time.time() - start_time:.2f} seconds")
    print(f"Curated data written to '{curated_dir}'")

    ray_client.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyStories curation pipeline with resumability.")
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)) + "/data",
        help="Root directory for raw, curated, and checkpoint data.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "valid"],
        default="valid",
        help="Dataset split to process ('valid' ~20 MB, 'train' ~2 GB).",
    )
    parser.add_argument(
        "--stories-per-file",
        type=int,
        default=2000,
        help="Stories per JSONL partition file. Controls the number of resumable partitions.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Enable resumability. Completed partitions are skipped when the pipeline is restarted.",
    )
    args = parser.parse_args()
    main(args)
