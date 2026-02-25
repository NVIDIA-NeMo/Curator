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

"""
ALM (Audio Language Model) Data Pipeline for NeMo Curator.

This script processes audio manifests to create training windows for
Audio Language Models using YAML-based configuration with Hydra.

The pipeline starts with ALMManifestReaderStage (reads JSONL on the worker,
not the driver) followed by configurable processing stages.

Features:
- YAML-driven pipeline configuration using nemo_curator.config.run
- Command-line parameter overrides
- Extensible stage chain
- Manifest I/O on workers via EmptyTask pattern

Usage:
    # Run with sample data (from Curator repo root)
    python tutorials/audio/alm/main.py \\
        --config-path . \\
        --config-name pipeline \\
        manifest_path=tests/fixtures/audio/alm/sample_input.jsonl

    # Override parameters
    python tutorials/audio/alm/main.py \\
        --config-path . \\
        --config-name pipeline \\
        manifest_path=/data/input.jsonl \\
        output_dir=./my_output \\
        stages.1.min_speakers=3 \\
        stages.2.overlap_percentage=30
"""

import hydra
from loguru import logger
from omegaconf import DictConfig

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.config.run import create_pipeline_from_yaml
from nemo_curator.tasks.utils import TaskPerfUtils


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    """Run ALM pipeline using Hydra configuration."""
    pipeline = create_pipeline_from_yaml(cfg)

    logger.info(pipeline.describe())
    logger.info("\n" + "=" * 50 + "\n")

    executor = XennaExecutor()

    logger.info("Starting pipeline execution...")
    results = pipeline.run(executor)

    output_files = []
    for task in results or []:
        output_files.extend(task.data)
    unique_files = sorted(set(output_files))

    logger.info("\n" + "=" * 50)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 50)
    logger.info(f"  Output files written: {len(unique_files)}")
    for fp in unique_files:
        logger.info(f"    - {fp}")

    stage_metrics = TaskPerfUtils.collect_stage_metrics(results)
    for stage_name, metrics in stage_metrics.items():
        logger.info(f"  [{stage_name}]")
        logger.info(f"    process_time: mean={metrics['process_time'].mean():.4f}s, total={metrics['process_time'].sum():.2f}s")
        logger.info(f"    items_processed: {metrics['num_items_processed'].sum():.0f}")
        if "custom.windows_created" in metrics:
            logger.info(f"    windows_created: {metrics['custom.windows_created'].sum():.0f}")
        if "custom.output_windows" in metrics:
            logger.info(f"    output_windows (after overlap): {metrics['custom.output_windows'].sum():.0f}")
        if "custom.filtered_dur" in metrics:
            logger.info(f"    filtered_audio_duration: {metrics['custom.filtered_dur'].sum():.1f}s")


if __name__ == "__main__":
    main()
