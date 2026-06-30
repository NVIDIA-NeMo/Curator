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

import json
import os
import time
from typing import Any

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.resources import Resources


def create_ray_client_from_yaml(cfg: DictConfig) -> RayClient:
    if "ray_client" in cfg:
        return hydra.utils.instantiate(cfg.ray_client)
    else:
        msg = "No Ray client defined in the YAML configuration. Using default Ray client."
        logger.warning(msg)
        return RayClient()


def _executor_observability_config(cfg: DictConfig) -> dict[str, Any]:
    return {
        "pipeline_hardware_sampler_enabled": bool(cfg.get("pipeline_hardware_sampler_enabled", False)),
        "pipeline_hardware_sampler_interval_s": float(cfg.get("pipeline_hardware_sampler_interval_s", 0.5)),
        "pipeline_hardware_sampler_startup_timeout_s": float(
            cfg.get("pipeline_hardware_sampler_startup_timeout_s", 5.0)
        ),
        "pipeline_hardware_sampler_stop_timeout_s": float(cfg.get("pipeline_hardware_sampler_stop_timeout_s", 10.0)),
    }


def _xenna_executor_config(cfg: DictConfig) -> dict[str, Any]:
    xenna_cfg = cfg.get("xenna", {})
    if not isinstance(xenna_cfg, (dict, DictConfig)):
        xenna_cfg = {}
    return {
        **_executor_observability_config(cfg),
        "execution_mode": cfg.get("execution_mode", "streaming"),
        "autoscale_interval_s": cfg.get("autoscale_interval_s", 180),
        "logging_interval": xenna_cfg.get("logging_interval", 60),
        "actor_pool_verbosity_level": xenna_cfg.get("actor_pool_verbosity_level", "INFO"),
        "monitoring_verbosity_level": xenna_cfg.get("monitoring_verbosity_level", "INFO"),
        "autoscaler_verbosity_level": xenna_cfg.get("autoscaler_verbosity_level", "INFO"),
        "executor_verbosity_level": xenna_cfg.get("executor_verbosity_level", "INFO"),
        "log_worker_allocation_layout": xenna_cfg.get("log_worker_allocation_layout", True),
    }


def create_executor_from_yaml(cfg: DictConfig) -> Any | None:  # noqa: ANN401
    backend = cfg.get("backend")
    if backend in (None, ""):
        return None
    if backend == "ray_data":
        if cfg.get("execution_mode") not in (None, "streaming"):
            logger.info("execution_mode={} is Xenna-only and is ignored by Ray Data", cfg.get("execution_mode"))
        from nemo_curator.backends.ray_data import RayDataExecutor

        return RayDataExecutor(config=_executor_observability_config(cfg))
    if backend == "xenna":
        from nemo_curator.backends.xenna import XennaExecutor

        return XennaExecutor(config=_xenna_executor_config(cfg))
    msg = f"Unknown backend {backend!r}. Expected one of: ray_data, xenna"
    raise ValueError(msg)


def _record_pipeline_duration(output_dir: str, elapsed_s: float) -> None:
    perf_summary_path = os.path.join(output_dir, "perf_summary.json")
    if not os.path.isfile(perf_summary_path):
        return
    with open(perf_summary_path, encoding="utf-8") as perf_file:
        perf_data = json.load(perf_file)
    perf_data["pipeline_duration_s"] = elapsed_s
    with open(perf_summary_path, "w", encoding="utf-8") as perf_file:
        json.dump(perf_data, perf_file, indent=2, ensure_ascii=False)
    logger.info("Performance summary ({}):\n{}", perf_summary_path, json.dumps(perf_data, indent=2))


def apply_process_env_defaults_from_yaml(cfg: DictConfig) -> None:
    env_defaults = _as_container(cfg.get("process_env_defaults", {}))
    if not env_defaults:
        return
    if not isinstance(env_defaults, dict):
        msg = f"process_env_defaults must be a mapping, got {type(env_defaults).__name__}"
        raise TypeError(msg)
    for key, value in env_defaults.items():
        if value is not None:
            os.environ.setdefault(str(key), str(value))


def _as_container(value: Any, *, resolve: bool = True) -> Any:  # noqa: ANN401
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=resolve)
    return value


def _normalise_stage_selector(value: Any) -> set[str] | None:  # noqa: ANN401
    value = _as_container(value)
    if value is None:
        return None
    if isinstance(value, str):
        if value.lower() == "all":
            return None
        items = [item.strip() for item in value.split(",")]
    else:
        items = [str(item).strip() for item in value]
    names = {item for item in items if item}
    return None if any(item.lower() == "all" for item in names) else names


def _stage_idents(stage_id: str | None, target: str | None) -> set[str]:
    return {
        ident
        for ident in (
            stage_id,
            target,
            target.rsplit(".", 1)[-1] if target else None,
        )
        if ident
    }


def _stage_config_idents(stage_cfg: Any) -> tuple[str | None, str | None, set[str]]:  # noqa: ANN401
    raw_unresolved = _as_container(stage_cfg, resolve=False)
    if not isinstance(raw_unresolved, dict):
        msg = f"Each stage entry must be a mapping, got {type(raw_unresolved).__name__}"
        raise TypeError(msg)
    stage_id_raw = raw_unresolved.get("stage_id", raw_unresolved.get("processor_id", raw_unresolved.get("id")))
    target_raw = raw_unresolved.get("_target_")
    stage_id = str(stage_id_raw) if stage_id_raw else None
    target = str(target_raw) if target_raw else None
    return stage_id, target, _stage_idents(stage_id, target)


def _resources_from_config(stage_resources: Any) -> Resources:  # noqa: ANN401
    if isinstance(stage_resources, Resources):
        return stage_resources
    if isinstance(stage_resources, dict) and "_target_" in stage_resources:
        return hydra.utils.instantiate(stage_resources)
    return Resources(**stage_resources)


def _normalise_with_kwargs(value: Any) -> dict[str, Any]:  # noqa: ANN401
    kwargs = _as_container(value) or {}
    if not isinstance(kwargs, dict):
        msg = f"stage_with entries must be mappings, got {type(kwargs).__name__}"
        raise TypeError(msg)
    kwargs = dict(kwargs)
    if "resources" in kwargs:
        kwargs["resources"] = _resources_from_config(kwargs["resources"])
    return kwargs


_ROOT_STAGE_WITH_KEYS = {"name", "resources", "batch_size", "runtime_env", "extended_performance_metrics"}


def _is_composite_child_stage_with(value: dict[str, Any]) -> bool:
    return not bool(set(value) & _ROOT_STAGE_WITH_KEYS)


def _apply_stage_with(stage: Any, value: Any) -> Any:  # noqa: ANN401
    if value is None:
        return stage
    value = _as_container(value)
    if isinstance(stage, CompositeStage) and _is_composite_child_stage_with(value):
        return stage.with_({name: _normalise_with_kwargs(kwargs) for name, kwargs in value.items()})
    return ProcessingStage.with_(stage, **_normalise_with_kwargs(value))


def _instantiate_stage(stage_cfg: DictConfig) -> tuple[Any, str | None, set[str]]:
    """Instantiate a single stage from its Hydra config.

    Extracts ``resources`` before calling ``hydra.utils.instantiate``
    (it is applied via ``.with_()``, not as a constructor argument) and
    re-applies it after construction. ``batch_size`` is left in the config
    dict so that stages declaring it as a dataclass field receive it
    during construction.
    """
    stage_id, _target, idents = _stage_config_idents(stage_cfg)

    cfg_dict = OmegaConf.to_container(stage_cfg, resolve=True)
    cfg_dict.pop("stage_id", None)
    cfg_dict.pop("processor_id", None)
    cfg_dict.pop("id", None)
    cfg_dict.pop("enabled", None)

    stage_resources = cfg_dict.pop("resources", None)
    stage_with = cfg_dict.pop("stage_with", cfg_dict.pop("with_", None))

    stage = hydra.utils.instantiate(cfg_dict)

    if stage_resources:
        resources_obj = _resources_from_config(stage_resources)
        with_kwargs: dict[str, Any] = {"resources": resources_obj}
        stage = stage.with_(**with_kwargs)
        logger.info(f"Applied .with_() to '{stage.name}': {with_kwargs}")

    stage = _apply_stage_with(stage, stage_with)
    if stage_id:
        stage._curator_stage_id = stage_id
    return stage, stage_id, idents


def _stage_is_selected(
    stage_cfg: DictConfig,
    idents: set[str],
    run_set: set[str] | None,
    skip_set: set[str],
) -> bool:
    selected = bool(stage_cfg.get("enabled", True))
    if selected and run_set is not None:
        selected = bool(idents & run_set)
    return selected and not bool(idents & skip_set)


def _raise_unknown_stage_selectors(
    run_set: set[str] | None,
    skip_set: set[str],
    available: set[str],
) -> None:
    unknown_run = run_set - available if run_set is not None else set()
    unknown_skip = skip_set - available
    if not (unknown_run or unknown_skip):
        return
    details = []
    if unknown_run:
        details.append(f"stages_to_run={sorted(unknown_run)}")
    if unknown_skip:
        details.append(f"stages_to_skip={sorted(unknown_skip)}")
    msg = f"Unknown stage selector(s): {', '.join(details)}. Available: {sorted(available)}"
    raise ValueError(msg)


def _create_staged_pipeline(cfg: DictConfig) -> Pipeline:
    pipeline = Pipeline(
        name=cfg.get("pipeline_name", "yaml_pipeline"),
        description=cfg.get("pipeline_description", "Create and execute a pipeline from a YAML file"),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    run_set = _normalise_stage_selector(cfg.get("stages_to_run", cfg.get("processors_to_run", "all")))
    skip_set = _normalise_stage_selector(cfg.get("stages_to_skip", cfg.get("processors_to_skip", []))) or set()
    available: set[str] = set()

    for stage_cfg in cfg.stages:
        stage_id, target, idents = _stage_config_idents(stage_cfg)
        available.update(idents)
        if not _stage_is_selected(stage_cfg, idents, run_set, skip_set):
            logger.info("Skipped stage {} ({})", stage_id or str(target).rsplit(".", 1)[-1], target)
            continue

        stage, stage_id, _idents = _instantiate_stage(stage_cfg)
        pipeline.add_stage(stage)
        logger.info("Enabled stage {} ({})", stage_id or stage.name, type(stage).__name__)

    _raise_unknown_stage_selectors(run_set, skip_set, available)
    return pipeline


def create_pipeline_from_yaml(cfg: DictConfig, *, log_config: bool = True) -> Pipeline | Any:  # noqa: ANN401
    if log_config:
        logger.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")

    if "stages" in cfg and "workflow" in cfg:
        msg = "Both stages and workflow are defined in the configuration. Please define either stages or workflow, not both."
        raise RuntimeError(msg)

    if "stages" in cfg:
        return _create_staged_pipeline(cfg)

    if "workflow" in cfg:
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
    apply_process_env_defaults_from_yaml(cfg)
    ray_client = None
    if cfg.get("backend") in (None, ""):
        ray_client = create_ray_client_from_yaml(cfg)
        ray_client.start()
    try:
        pipeline = create_pipeline_from_yaml(cfg)
        executor = create_executor_from_yaml(cfg)

        print("Starting pipeline execution...")
        start_s = time.time()
        _results = pipeline.run(executor=executor)
        elapsed_s = time.time() - start_s

        output_dir = cfg.get("output_dir") or cfg.get("workspace_dir", "./output")
        _record_pipeline_duration(output_dir, elapsed_s)
        print("\nPipeline completed!")
    finally:
        if ray_client is not None:
            ray_client.stop()


if __name__ == "__main__":
    main()
