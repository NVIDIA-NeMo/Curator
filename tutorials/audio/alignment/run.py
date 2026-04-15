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
Hydra-based MFA alignment pipeline runner.

Usage::

    python run.py --config-path=. --config-name=pipeline
"""

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nemo_curator.pipeline import Pipeline

_EXECUTOR_FACTORIES = {
    "xenna": "nemo_curator.backends.xenna:XennaExecutor",
    "ray_data": "nemo_curator.backends.ray_data:RayDataExecutor",
}


def _create_executor(backend: str) -> object:
    import importlib

    module_path, class_name = _EXECUTOR_FACTORIES[backend].rsplit(":", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)()


def create_pipeline_from_yaml(cfg: DictConfig) -> Pipeline:
    pipeline = Pipeline(
        name="mfa_alignment",
        description="MFA forced alignment pipeline (YAML config)",
    )
    for p in cfg.processors:
        stage = hydra.utils.instantiate(p)
        pipeline.add_stage(stage)
    return pipeline


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")
    pipeline = create_pipeline_from_yaml(cfg)

    logger.info(pipeline.describe())
    logger.info("\n" + "=" * 50 + "\n")

    backend = cfg.get("backend", "ray_data")
    if backend not in _EXECUTOR_FACTORIES:
        msg = f"Unknown backend '{backend}'. Choose from: {list(_EXECUTOR_FACTORIES)}"
        raise ValueError(msg)
    logger.info(f"Using backend: {backend}")
    executor = _create_executor(backend)

    logger.info("Starting MFA alignment pipeline...")
    pipeline.run(executor)
    logger.info("\nPipeline completed!")


if __name__ == "__main__":
    main()
