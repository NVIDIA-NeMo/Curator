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

from nemo_curator.backends.base import BaseExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import CompositeStage
from nemo_curator.stages.resources import Resources

_EXECUTOR_TARGETS = {
    "xenna": "nemo_curator.backends.xenna.XennaExecutor",
    "ray_data": "nemo_curator.backends.ray_data.RayDataExecutor",
}
_XENNA_EXECUTION_MODES = {"batch", "streaming"}


def _instantiate_resources(resources: object) -> Resources:
    """Convert a YAML resource mapping into the runtime dataclass."""
    if isinstance(resources, Resources):
        return resources
    if not isinstance(resources, dict):
        msg = f"resources must be a mapping or Resources instance, found {type(resources).__name__}"
        raise TypeError(msg)
    if "_target_" in resources:
        instantiated = hydra.utils.instantiate(resources)
        if not isinstance(instantiated, Resources):
            msg = f"resources target must instantiate Resources, found {type(instantiated).__name__}"
            raise TypeError(msg)
        return instantiated
    return Resources(**resources)


def _normalize_stage_with(stage_with: dict[str, Any]) -> dict[str, Any]:
    """Hydrate special values accepted by ``ProcessingStage.with_``."""
    normalized = dict(stage_with)
    if "resources" in normalized:
        normalized["resources"] = _instantiate_resources(normalized["resources"])
    return normalized


def create_ray_client_from_yaml(cfg: DictConfig) -> RayClient:
    if "ray_client" in cfg:
        return hydra.utils.instantiate(cfg.ray_client)
    else:
        msg = "No Ray client defined in the YAML configuration. Using default Ray client."
        logger.warning(msg)
        return RayClient()


def create_executor_from_yaml(cfg: DictConfig) -> BaseExecutor | None:
    """Create the configured pipeline executor, if executor settings are present."""
    if "backend" not in cfg and "execution_mode" not in cfg:
        return None

    backend = str(cfg.get("backend", "xenna"))
    if backend not in _EXECUTOR_TARGETS:
        choices = ", ".join(_EXECUTOR_TARGETS)
        msg = f"Unknown backend '{backend}'. Choose from: {choices}."
        raise ValueError(msg)

    executor_cls = hydra.utils.get_class(_EXECUTOR_TARGETS[backend])
    if backend == "xenna":
        execution_mode = str(cfg.get("execution_mode", "streaming"))
        if execution_mode not in _XENNA_EXECUTION_MODES:
            choices = ", ".join(sorted(_XENNA_EXECUTION_MODES))
            msg = f"Unknown Xenna execution mode '{execution_mode}'. Choose from: {choices}."
            raise ValueError(msg)
        logger.info(f"Using executor backend '{backend}' in '{execution_mode}' mode.")
        return executor_cls(config={"execution_mode": execution_mode})

    logger.info(f"Using executor backend '{backend}'.")
    return executor_cls()


def _instantiate_stage(stage_cfg: DictConfig) -> Any:  # noqa: ANN401
    """Instantiate a single stage from its Hydra config.

    Extracts ``resources`` and ``stage_with`` before calling
    ``hydra.utils.instantiate``. Processing-stage overrides are applied as
    ``stage.with_(**stage_with)``; composite-stage overrides use the nested
    stage-name mapping accepted by ``CompositeStage.with_()``. ``batch_size``
    is left in the config dict so stages declaring it as a dataclass field
    receive it during construction.
    """
    cfg_dict = OmegaConf.to_container(stage_cfg, resolve=True)

    stage_resources = cfg_dict.pop("resources", None)
    stage_with = cfg_dict.pop("stage_with", None)

    stage = hydra.utils.instantiate(cfg_dict)

    with_kwargs: dict[str, Any] = {}
    if stage_resources:
        with_kwargs["resources"] = _instantiate_resources(stage_resources)

    if stage_with:
        if not isinstance(stage_with, dict):
            msg = f"stage_with for '{stage.name}' must be a mapping"
            raise TypeError(msg)
        if isinstance(stage, CompositeStage):
            if with_kwargs:
                msg = f"Composite stage '{stage.name}' cannot use top-level resources; put them under stage_with"
                raise ValueError(msg)
            composite_with: dict[str, dict[str, Any]] = {}
            for nested_stage_name, nested_stage_with in stage_with.items():
                if not isinstance(nested_stage_with, dict):
                    msg = f"stage_with entry for '{nested_stage_name}' in '{stage.name}' must be a mapping"
                    raise TypeError(msg)
                composite_with[nested_stage_name] = _normalize_stage_with(nested_stage_with)
            stage = stage.with_(composite_with)
            logger.info(f"Applied composite .with_() to '{stage.name}': {composite_with}")
        else:
            normalized_stage_with = _normalize_stage_with(stage_with)
            if "resources" in normalized_stage_with and "resources" in with_kwargs:
                msg = f"Stage '{stage.name}' defines resources both at top level and under stage_with"
                raise ValueError(msg)
            with_kwargs.update(normalized_stage_with)

    if with_kwargs:
        stage = stage.with_(**with_kwargs)
        logger.info(f"Applied .with_() to '{stage.name}': {with_kwargs}")

    return stage


def create_pipeline_from_yaml(cfg: DictConfig, *, log_config: bool = True) -> Pipeline | Any:  # noqa: ANN401
    if log_config:
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


@hydra.main(version_base=None, config_name="pipeline")
def main(cfg: DictConfig) -> None:
    ray_client = create_ray_client_from_yaml(cfg)
    ray_client.start()

    pipeline = create_pipeline_from_yaml(cfg)
    executor = create_executor_from_yaml(cfg)

    # Execute pipeline
    print("Starting pipeline execution...")
    _results = pipeline.run() if executor is None else pipeline.run(executor=executor)

    print("\nPipeline completed!")

    ray_client.stop()


if __name__ == "__main__":
    main()
