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
Hydra-based runner for DNS Challenge Read Speech pipeline.

This script loads the pipeline configuration from YAML and executes it.

Usage:
    # Run with default config
    python run.py --config-path . --config-name pipeline.yaml \
        raw_data_dir=/path/to/read_speech

    # Override settings
    python run.py --config-path . --config-name pipeline.yaml \
        raw_data_dir=/path/to/read_speech \
        max_samples=3000 \
        enable_utmos=true
"""

import os

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline


def create_pipeline_from_yaml(cfg: DictConfig) -> Pipeline:
    """Create pipeline from Hydra config."""
    pipeline = Pipeline(
        name="readspeech_yaml_pipeline", description="DNS Challenge Read Speech pipeline created from YAML config"
    )

    for processor_cfg in cfg.processors:
        stage = hydra.utils.instantiate(processor_cfg)
        pipeline.add_stage(stage)

    return pipeline


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Run DNS Challenge Read Speech pipeline from YAML configuration.
    """
    logger.info("DNS Challenge Read Speech Audio Data Filtration Pipeline (YAML)")
    logger.info("=" * 60)
    logger.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    pipeline = create_pipeline_from_yaml(cfg)
    logger.info(pipeline.describe())
    logger.info("=" * 60)

    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    backend = cfg.get("backend", "xenna")
    if backend == "ray_data":
        executor = RayDataExecutor()
    else:
        execution_mode = cfg.get("execution_mode", "batch")
        executor = XennaExecutor(config={"execution_mode": execution_mode})
    logger.info(f"Starting pipeline execution (mode: {execution_mode})...")
    pipeline.run(executor)

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline completed!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
