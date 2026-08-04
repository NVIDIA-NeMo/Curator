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


def create_ray_client_from_yaml(cfg: DictConfig) -> RayClient:
    if "ray_client" in cfg:
        return hydra.utils.instantiate(cfg.ray_client)
    else:
        msg = "No Ray client defined in the YAML configuration. Using default Ray client."
        logger.warning(msg)
        return RayClient()


def create_executor_from_yaml(cfg: DictConfig) -> BaseExecutor | None:
    """Create the configured pipeline executor, if executor settings are present."""
    if "backend" not in cfg and "execution_mode" not in cfg and "executor_config" not in cfg:
        return None

    backend = str(cfg.get("backend", "xenna"))
    if backend not in _EXECUTOR_TARGETS:
        choices = ", ".join(_EXECUTOR_TARGETS)
        msg = f"Unknown backend '{backend}'. Choose from: {choices}."
        raise ValueError(msg)

    executor_cls = hydra.utils.get_class(_EXECUTOR_TARGETS[backend])
    executor_config_node = cfg.get("executor_config")
    executor_config = (
        OmegaConf.to_container(executor_config_node, resolve=True) if executor_config_node is not None else {}
    )
    executor_config = executor_config or {}
    if not isinstance(executor_config, dict):
        msg = "executor_config must be a mapping."
        raise TypeError(msg)
    if backend == "xenna":
        top_level_mode = cfg.get("execution_mode")
        nested_mode = executor_config.get("execution_mode")
        if top_level_mode is not None and nested_mode is not None and str(top_level_mode) != str(nested_mode):
            msg = (
                "Conflicting Xenna execution modes: "
                f"execution_mode={top_level_mode!s} and executor_config.execution_mode={nested_mode!s}."
            )
            raise ValueError(msg)
        execution_mode = str(top_level_mode if top_level_mode is not None else nested_mode or "streaming")
        if execution_mode not in _XENNA_EXECUTION_MODES:
            choices = ", ".join(sorted(_XENNA_EXECUTION_MODES))
            msg = f"Unknown Xenna execution mode '{execution_mode}'. Choose from: {choices}."
            raise ValueError(msg)
        executor_config["execution_mode"] = execution_mode
        logger.info(f"Using executor backend '{backend}' in '{execution_mode}' mode.")
        return executor_cls(config=executor_config)

    logger.info(f"Using executor backend '{backend}'.")
    return executor_cls(config=executor_config) if executor_config else executor_cls()


def _instantiate_stage(stage_cfg: DictConfig) -> Any:  # noqa: ANN401
    """Instantiate a single stage from its Hydra config.

    Extracts ``resources`` before calling ``hydra.utils.instantiate``
    (it is applied via ``.with_()``, not as a constructor argument) and
    re-applies it after construction. ``batch_size`` is left in the config
    dict so that stages declaring it as a dataclass field receive it
    during construction.
    """
    cfg_dict = OmegaConf.to_container(stage_cfg, resolve=True)

    stage_resources = cfg_dict.pop("resources", None)
    extended_performance_metrics = cfg_dict.pop("extended_performance_metrics", None)

    stage = hydra.utils.instantiate(cfg_dict)

    with_kwargs: dict[str, Any] = {}
    if stage_resources:
        if isinstance(stage_resources, dict) and "_target_" in stage_resources:
            resources_obj = hydra.utils.instantiate(stage_resources)
        else:
            resources_obj = Resources(**stage_resources)
        with_kwargs["resources"] = resources_obj
    if extended_performance_metrics is not None and not isinstance(stage, CompositeStage):
        with_kwargs["extended_performance_metrics"] = bool(extended_performance_metrics)
    if with_kwargs:
        stage = stage.with_(**with_kwargs)
        logger.info(f"Applied .with_() to '{stage.name}': {with_kwargs}")
    if extended_performance_metrics is not None and isinstance(stage, CompositeStage):
        stage.extended_performance_metrics = bool(extended_performance_metrics)
        logger.info(
            "Applied extended_performance_metrics={} to composite '{}'; it will propagate during decomposition.",
            stage.extended_performance_metrics,
            stage.name,
        )

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


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    ray_client = create_ray_client_from_yaml(cfg)
    ray_client.start()

    pipeline = create_pipeline_from_yaml(cfg)
    executor = create_executor_from_yaml(cfg)

    # Execute pipeline
    print("Starting pipeline execution...")
    pipeline.run() if executor is None else pipeline.run(executor=executor)

    print("\nPipeline completed!")

    ray_client.stop()


if __name__ == "__main__":
    main()
