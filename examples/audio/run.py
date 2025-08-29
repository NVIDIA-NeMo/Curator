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

import sys

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline

"""
How to run :

SCRIPT_DIR=/path/to/Curator/examples/audio

python ${SCRIPT_DIR}/run.py \
--config-path ${SCRIPT_DIR}/fleurs \
--config-name  pipeline.yaml \
...

"""


def create_pipeline_from_yaml(cfg: DictConfig) -> Pipeline:
    pipeline = Pipeline(name="yaml_pipeline", description="Pipeline created using yaml config file")
    for p in cfg.processors:
        stage = hydra.utils.instantiate(p)
        pipeline.add_stage(stage)
    return pipeline


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Prepare pipeline and run YAML pipeline.
    """
    logger.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")
    pipeline = create_pipeline_from_yaml(cfg)

    # Print pipeline description
    logger.info(pipeline.describe())
    logger.info("\n" + "=" * 50 + "\n")

    # Create executor
    executor = XennaExecutor()

    # Execute pipeline
    logger.info("Starting pipeline execution...")
    pipeline.run(executor)

    # Print results
    logger.info("\nPipeline completed!")


if __name__ == "__main__":
    # hacking the arguments to always disable hydra's output
    sys.argv.extend(
        ["hydra.run.dir=.", "hydra.output_subdir=null", "hydra/job_logging=none", "hydra/hydra_logging=none"]
    )
    main()
