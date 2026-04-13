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
ASR Pipeline for NeMo Curator.

Processes raw audio data through VAD, and then two passes of ASR alignment to produce a manifest with ASR hypotheses using VAD segments.

Usage:
    # ASR pipeline (from Curator repo root)
    python tutorials/audio/asr/main.py \\
        --config-path . \\
        --config-name vad_asr_pipeline \\
        input_manifest=/data/input.jsonl \\
        final_manifest=/data/asr_output.jsonl \\
        hf_token=<your_hf_token>

    # Override parameters
    python tutorials/audio/asr/main.py \\
        --config-path . \\
        --config-name vad_asr_pipeline \\
        input_manifest=/data/input.jsonl \\
        final_manifest=/data/asr_alignment_output.jsonl \\
        hf_token=<your_hf_token> \\
        device=cpu \\
        max_segment_length=30 \\
        stages.3.min_len=2.0
"""

import hydra
from loguru import logger
from omegaconf import DictConfig

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.config.run import create_pipeline_from_yaml
from nemo_curator.tasks.utils import TaskPerfUtils


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    """Run VAD + ASR pipeline using Hydra configuration."""
    pipeline = create_pipeline_from_yaml(cfg)

    logger.info(pipeline.describe())
    logger.info("\n" + "=" * 50 + "\n")

    config = {"execution_mode": "batch"}
    executor = XennaExecutor(config=config)

    logger.info("Starting VAD + ASR pipeline...")
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
        logger.info(
            f"    process_time: mean={metrics['process_time'].mean():.4f}s, total={metrics['process_time'].sum():.2f}s"
        )
        logger.info(f"    items_processed: {metrics['num_items_processed'].sum():.0f}")


if __name__ == "__main__":
    main()
