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

from typing import Any

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.resources import Resources


def create_ray_client_from_yaml(cfg: DictConfig) -> RayClient:
    if "ray_client" in cfg:
        return hydra.utils.instantiate(cfg.ray_client)
    else:
        msg = "No Ray client defined in the YAML configuration. Using default Ray client."
        logger.warning(msg)
        return RayClient()


def _instantiate_stage(stage_cfg: DictConfig) -> Any:  # noqa: ANN401
    """Instantiate a single stage from its Hydra config.

    Extracts ``resources`` and ``batch_size`` before calling
    ``hydra.utils.instantiate`` (they are not constructor arguments) and
    re-applies them via the stage's ``.with_()`` method.
    """
    cfg_dict = OmegaConf.to_container(stage_cfg, resolve=True)

    stage_resources = cfg_dict.pop("resources", None)
    stage_batch_size = cfg_dict.pop("batch_size", None)

    stage = hydra.utils.instantiate(cfg_dict)

    if stage_resources or stage_batch_size:
        with_kwargs: dict[str, Any] = {}
        if stage_resources:
            with_kwargs["resources"] = Resources(**stage_resources)
        if stage_batch_size:
            with_kwargs["batch_size"] = stage_batch_size
        stage = stage.with_(**with_kwargs)
        logger.info(f"Applied .with_() to '{stage.name}': {with_kwargs}")

    return stage


def create_pipeline_from_yaml(cfg: DictConfig) -> Pipeline | Any:  # noqa: ANN401
    logger.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")

    if "stages" in cfg and "workflow" in cfg:
        msg = "Both stages and workflow are defined in the configuration. Please define either stages or workflow, not both."
        raise RuntimeError(msg)

    if "stages" in cfg:
        pipeline = Pipeline(name="yaml_pipeline", description="Create and execute a pipeline from a YAML file")

        for stage_cfg in cfg.stages:
            stage = _instantiate_stage(stage_cfg)
            pipeline.add_stage(stage)

        return pipeline

    elif "workflow" in cfg:
        if len(cfg.workflow) != 1:
            msg = "One workflow should be defined in the YAML configuration. Please define a single workflow."
            raise RuntimeError(msg)

        # Initialize a deduplication workflow
        return hydra.utils.instantiate(cfg.workflow[0])

    else:
        msg = "Invalid YAML configuration. Please define stages to add to a pipeline or a workflow to execute."
        raise RuntimeError(msg)


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    ray_client = create_ray_client_from_yaml(cfg)
    ray_client.start()

    pipeline = create_pipeline_from_yaml(cfg)

    # Execute pipeline
    print("Starting pipeline execution...")
    _results = pipeline.run()

    print("\nPipeline completed!")

    ray_client.stop()


if __name__ == "__main__":
    main()
