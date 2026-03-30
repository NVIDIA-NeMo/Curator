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
Audio Tagging Pipeline for NeMo Curator.

Processes raw audio data through diarization, ASR alignment, text
normalization, quality metrics, and segment preparation to produce
labelled training manifests for TTS or ASR.

The pipeline is YAML-driven via Hydra and supports both TTS and ASR
modalities by switching the configuration file.

Usage:
    # TTS pipeline (from Curator repo root)
    python tutorials/audio/tagging/main.py \\
        --config-path . \\
        --config-name tts_pipeline \\
        input_manifest=/data/input.jsonl \\
        final_manifest=/data/tts_output.jsonl \\
        hf_token=<your_hf_token>

    # ASR pipeline
    python tutorials/audio/tagging/main.py \\
        --config-path . \\
        --config-name asr_pipeline \\
        input_manifest=/data/input.jsonl \\
        final_manifest=/data/asr_output.jsonl \\
        hf_token=<your_hf_token>

    # Override parameters
    python tutorials/audio/tagging/main.py \\
        --config-path . \\
        --config-name tts_pipeline \\
        input_manifest=/data/input.jsonl \\
        final_manifest=/data/output.jsonl \\
        hf_token=<your_hf_token> \\
        device=cpu \\
        max_segment_length=30 \\
        stages.10.min_duration=3
"""

import hydra
from loguru import logger
from omegaconf import DictConfig

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.config.run import create_pipeline_from_yaml
from nemo_curator.tasks.utils import TaskPerfUtils


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    """Run audio tagging pipeline using Hydra configuration."""
    pipeline = create_pipeline_from_yaml(cfg)

    logger.info(pipeline.describe())
    logger.info("\n" + "=" * 50 + "\n")

    executor = XennaExecutor()

    logger.info("Starting audio tagging pipeline...")
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
