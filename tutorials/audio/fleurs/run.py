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

import importlib

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline

_EXECUTOR_FACTORIES = {
    "xenna": "nemo_curator.backends.xenna:XennaExecutor",
    "ray_data": "nemo_curator.backends.ray_data:RayDataExecutor",
}


def _create_executor(backend: str) -> object:
    module_path, class_name = _EXECUTOR_FACTORIES[backend].rsplit(":", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)()


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
    ray_client = RayClient()
    ray_client.start()

    try:
        logger.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")
        pipeline = create_pipeline_from_yaml(cfg)

        logger.info(pipeline.describe())
        logger.info("\n" + "=" * 50 + "\n")

        backend = cfg.get("backend", "xenna")
        if backend not in _EXECUTOR_FACTORIES:
            msg = f"Unknown backend '{backend}'. Choose from: {list(_EXECUTOR_FACTORIES)}"
            raise ValueError(msg)
        logger.info(f"Using backend: {backend}")
        executor = _create_executor(backend)

        logger.info("Starting pipeline execution...")
        pipeline.run(executor)

        logger.info("\nPipeline completed!")
    finally:
        ray_client.stop()


if __name__ == "__main__":
    main()
