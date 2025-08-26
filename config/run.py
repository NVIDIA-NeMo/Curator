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
How to run:

SCRIPT_DIR=/path/to/this/directory
python ${SCRIPT_DIR}/run.py --config-path ${SCRIPT_DIR} --config-name example_pipeline.yaml

"""

import sys

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from ray_curator.core.client import RayClient
from ray_curator.pipeline import Pipeline


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")

    ray_client = RayClient()
    ray_client.start()
    pipeline = Pipeline(name="yaml_pipeline", description="Create and execute a pipeline from a YAML file")

    # Add stages
    for p in cfg.stages:
        stage = hydra.utils.instantiate(p)
        pipeline.add_stage(stage)

    # Print pipeline description
    logger.info(pipeline.describe())
    logger.info("\n" + "=" * 50 + "\n")

    # Execute pipeline
    print("Starting pipeline execution...")
    results = pipeline.run()

    # Print results
    print("\nPipeline completed!")
    print(f"Total output tasks: {len(results) if results else 0}")

    if results:
        for i, task in enumerate(results):
            print(f"\nTask {i}:")
            print(task.data)

    ray_client.stop()


if __name__ == "__main__":
    # hacking the arguments to always disable hydra's output
    sys.argv.extend(
        ["hydra.run.dir=.", "hydra.output_subdir=null", "hydra/job_logging=none", "hydra/hydra_logging=none"]
    )
    main()
