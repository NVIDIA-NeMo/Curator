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

"""
ALM (Audio Language Model) Data Pipeline for NeMo Curator.

This script processes audio manifests to create training windows for
Audio Language Models using YAML-based configuration with Hydra.

Features:
- YAML-driven pipeline configuration using nemo_curator.config.run
- Command-line parameter overrides
- Extensible stage chain

Usage:
    # Run with sample data (from Curator repo root)
    python tutorials/audio/alm/main.py \\
        --config-path . \\
        --config-name pipeline \\
        input_manifest=tests/fixtures/audio/alm/sample_input.jsonl

    # Override parameters
    python tutorials/audio/alm/main.py \\
        --config-path . \\
        --config-name pipeline \\
        input_manifest=/data/input.jsonl \\
        output_dir=./my_output \\
        stages.0.min_speakers=3 \\
        stages.1.overlap_percentage=30
"""

import json

import hydra
from fsspec.core import url_to_fs
from loguru import logger
from omegaconf import DictConfig

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.config.run import create_pipeline_from_yaml
from nemo_curator.tasks import AudioBatch
from nemo_curator.tasks.utils import TaskPerfUtils


def load_manifest(manifest_path: str) -> list[dict]:
    """Load entries from a JSONL manifest file. Supports local and cloud paths."""
    fs, path = url_to_fs(manifest_path)
    entries = []
    with fs.open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line.strip()))
    return entries


def save_manifest(entries: list[dict], output_path: str) -> None:
    """Save entries to a JSONL manifest file. Supports local and cloud paths."""
    fs, path = url_to_fs(output_path)
    # Create parent directory if it doesn't exist
    parent_dir = "/".join(path.split("/")[:-1])
    if parent_dir:
        fs.makedirs(parent_dir, exist_ok=True)
    with fs.open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Run ALM pipeline using Hydra configuration.
    """
    # Get paths from config
    input_manifest = cfg.get("input_manifest")
    if not input_manifest:
        msg = "input_manifest must be specified in config"
        raise ValueError(msg)

    output_dir = cfg.get("output_dir", "./alm_output")
    fs, output_dir_path = url_to_fs(output_dir)
    fs.makedirs(output_dir_path, exist_ok=True)

    # Load input manifest
    logger.info(f"Loading input manifest: {input_manifest}")
    entries = load_manifest(input_manifest)
    logger.info(f"Loaded {len(entries)} entries from manifest")

    # Convert entries to AudioBatch tasks
    initial_tasks = [AudioBatch(data=[entry]) for entry in entries]
    logger.info(f"Created {len(initial_tasks)} initial AudioBatch tasks")

    # Create pipeline from YAML
    pipeline = create_pipeline_from_yaml(cfg)

    # Print pipeline description
    logger.info(pipeline.describe())
    logger.info("\n" + "=" * 50 + "\n")

    # Create executor
    executor = XennaExecutor()

    # Execute pipeline
    logger.info("Starting pipeline execution...")
    results = pipeline.run(executor, initial_tasks=initial_tasks)

    # Convert results to entries
    output_entries = []
    for task in results or []:
        output_entries.extend(task.data)

    # Pipeline-level statistics from output data
    stage1_windows = sum(len(e.get("windows", [])) for e in output_entries)
    stage2_windows = sum(len(e.get("filtered_windows", [])) for e in output_entries)
    total_filtered_dur = sum(e.get("filtered_dur", 0) for e in output_entries)
    entries_with_windows = sum(1 for e in output_entries if e.get("filtered_windows") or e.get("windows"))

    logger.info("\n" + "=" * 50)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 50)
    logger.info(f"  Input entries: {len(entries)}")
    logger.info(f"  Output entries: {len(output_entries)}")
    logger.info(f"  Entries with windows: {entries_with_windows}")
    logger.info(f"  Stage 1 (Builder) windows: {stage1_windows}")
    logger.info(f"  Stage 2 (Overlap) windows: {stage2_windows}")
    logger.info(f"  Total filtered duration: {total_filtered_dur:.2f}s ({total_filtered_dur / 60:.2f} min)")

    # Per-stage performance stats from Task._stage_perf (populated by _log_metrics)
    stage_metrics = TaskPerfUtils.collect_stage_metrics(results)
    for stage_name, metrics in stage_metrics.items():
        logger.info(f"  [{stage_name}]")
        logger.info(f"    process_time: mean={metrics['process_time'].mean():.4f}s, total={metrics['process_time'].sum():.2f}s")
        logger.info(f"    items_processed: {metrics['num_items_processed'].sum():.0f}")
        if "custom.windows_created" in metrics:
            logger.info(f"    windows_created: {metrics['custom.windows_created'].sum():.0f}")
        if "custom.output_windows" in metrics:
            logger.info(f"    output_windows (after overlap): {metrics['custom.output_windows'].sum():.0f}")

    # Save results
    output_path = f"{output_dir.rstrip('/')}/alm_output.jsonl"
    save_manifest(output_entries, output_path)
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
