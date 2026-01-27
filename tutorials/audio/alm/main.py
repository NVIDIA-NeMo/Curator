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
- YAML-driven pipeline configuration
- Command-line parameter overrides
- Extensible processor chain

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
        processors.0.min_speakers=3 \\
        processors.1.overlap_percentage=30
"""

import json
import os

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.tasks import AudioBatch


def create_pipeline_from_yaml(cfg: DictConfig) -> Pipeline:
    """Create pipeline by instantiating stages from YAML config."""
    pipeline = Pipeline(
        name="alm_yaml_pipeline",
        description="ALM Pipeline created from YAML config"
    )
    for processor_cfg in cfg.processors:
        stage = hydra.utils.instantiate(processor_cfg)
        pipeline.add_stage(stage)
    return pipeline


def load_manifest(manifest_path: str) -> list[dict]:
    """Load entries from a JSONL manifest file."""
    entries = []
    with open(manifest_path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line.strip()))
    return entries


def save_manifest(entries: list[dict], output_path: str) -> None:
    """Save entries to a JSONL manifest file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Run ALM pipeline using Hydra configuration.
    """
    logger.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    # Get paths from config
    input_manifest = cfg.get("input_manifest")
    if not input_manifest:
        raise ValueError("input_manifest must be specified in config")

    output_dir = cfg.get("output_dir", "./alm_output")
    os.makedirs(output_dir, exist_ok=True)

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

    # Calculate statistics
    # Stage 1 output: total_dur_list_window contains the original window count
    stage1_windows = sum(len(e.get('total_dur_list_window', e.get('windows', []))) for e in output_entries)
    # Stage 2 output: filtered_windows contains windows after overlap filtering
    stage2_windows = sum(len(e.get('filtered_windows', [])) for e in output_entries)
    total_filtered_dur = sum(e.get('filtered_dur', 0) for e in output_entries)
    entries_with_windows = sum(1 for e in output_entries if e.get('filtered_windows') or e.get('windows'))

    logger.info("\n" + "=" * 50)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 50)
    logger.info(f"  Input entries: {len(entries)}")
    logger.info(f"  Output entries: {len(output_entries)}")
    logger.info(f"  Entries with windows: {entries_with_windows}")
    logger.info(f"  Stage 1 (Builder) windows: {stage1_windows}")
    logger.info(f"  Stage 2 (Overlap) windows: {stage2_windows}")
    logger.info(f"  Total filtered duration: {total_filtered_dur:.2f}s ({total_filtered_dur/60:.2f} min)")

    # Save results
    output_path = os.path.join(output_dir, "alm_output.jsonl")
    save_manifest(output_entries, output_path)
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
