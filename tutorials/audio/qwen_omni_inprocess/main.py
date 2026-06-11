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

"""Hydra entry point for the Granary v2 Qwen-Omni in-process pipeline.

The stage graph comes from the Hydra config; stage entries may carry
``stage_with``/``with_`` metadata applied after instantiation to set resource,
batch-size, and composite worker specs. Secrets are redacted before logging.

Hugging Face credentials are NOT handled here: weights download on remote Ray
workers (``ASRStage.setup_on_node`` -> ``prefetch_weights``), so a token in this
driver would not propagate. For gated models set ``HF_TOKEN``/``HF_HOME`` in the
worker environment (cluster env or executor ``runtime_env``).
"""

import importlib
import os
import time
from typing import Any

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.resources import Resources

_EXECUTOR_FACTORIES = {
    "xenna": "nemo_curator.backends.xenna:XennaExecutor",
    "ray_data": "nemo_curator.backends.ray_data:RayDataExecutor",
}

_SECRET_KEY_NAMES = {
    "access_key",
    "access_token",
    "api_key",
    "auth_token",
    "bearer_token",
    "credential",
    "credentials",
    "password",
    "passwd",
    "secret",
    "secret_key",
    "token",
}
_SECRET_KEY_PARTS = (
    "access_key",
    "api_key",
    "auth_token",
    "bearer",
    "credential",
    "password",
    "passwd",
    "secret",
)


def _as_container(value: Any, *, resolve: bool = True) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=resolve)
    return value


def _normalise_name_set(value: Any) -> set[str] | None:
    """Return selected processor names, or ``None`` for all processors."""
    value = _as_container(value)
    if value is None:
        return None
    if isinstance(value, str):
        if value.lower() == "all":
            return None
        items = [part.strip() for part in value.split(",")]
    else:
        items = [str(item).strip() for item in value]
    names = {item for item in items if item}
    return None if any(item.lower() == "all" for item in names) else names


_ALLOWED_RESOURCES_TARGETS = frozenset(
    {
        "nemo_curator.stages.resources.Resources",
        "Resources",
    }
)
_HYDRA_RESOURCE_META_KEYS = frozenset({"_target_", "_recursive_", "_convert_"})


def _instantiate_resources(value: Any) -> Resources:
    """Build a ``Resources`` from a Hydra ``stage_with`` override.

    Never call open-ended ``hydra.utils.instantiate`` here: a malicious or
    mistyped ``_target_`` in YAML could execute arbitrary code. Only the
    canonical ``Resources`` dataclass is accepted.
    """
    if isinstance(value, Resources):
        return value
    if OmegaConf.is_config(value) or isinstance(value, dict):
        cfg = value if OmegaConf.is_config(value) else OmegaConf.create(value)
        raw = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(raw, dict):
            msg = f"Invalid resources override: {value!r}"
            raise TypeError(msg)
        target = raw.get("_target_")
        if target is not None and str(target) not in _ALLOWED_RESOURCES_TARGETS:
            msg = (
                "resources overrides may only target "
                "nemo_curator.stages.resources.Resources; "
                f"got {target!r}"
            )
            raise ValueError(msg)
        fields = {key: item for key, item in raw.items() if key not in _HYDRA_RESOURCE_META_KEYS}
        return Resources(**fields)
    msg = f"Invalid resources override: {value!r}"
    raise TypeError(msg)


def _normalise_with_kwargs(value: Any) -> dict[str, Any]:
    kwargs = _as_container(value) or {}
    if not isinstance(kwargs, dict):
        msg = f"stage_with entries must be mappings, got {type(kwargs).__name__}"
        raise TypeError(msg)
    kwargs = dict(kwargs)
    if "resources" in kwargs:
        kwargs["resources"] = _instantiate_resources(kwargs["resources"])
    return kwargs


def _apply_stage_with(stage: ProcessingStage, value: Any) -> ProcessingStage:
    """Apply optional ``stage_with`` metadata after Hydra construction."""
    if value is None:
        return stage
    value = _as_container(value)
    if isinstance(stage, CompositeStage):
        return stage.with_({name: _normalise_with_kwargs(kwargs) for name, kwargs in value.items()})
    return stage.with_(**_normalise_with_kwargs(value))


def _target_idents(stage_id: str | None, target: str | None) -> set[str]:
    return {
        ident
        for ident in (
            stage_id,
            target,
            target.rsplit(".", 1)[-1] if target else None,
        )
        if ident
    }


def _configured_stage_entries(cfg: DictConfig) -> Any:
    if "stages" in cfg and cfg.stages:
        return cfg.stages
    if "processors" in cfg and cfg.processors:
        logger.warning("Using legacy 'processors:' config key; prefer 'stages:' for Qwen-Omni.")
        return cfg.processors
    return None


def _is_secret_key(key: str) -> bool:
    key_lower = key.lower()
    return (
        key_lower in _SECRET_KEY_NAMES
        or key_lower.endswith(("_token", "_secret", "_password"))
        or any(part in key_lower for part in _SECRET_KEY_PARTS)
    )


def _redact_secret_values(value: Any) -> Any:
    value = _as_container(value)
    if isinstance(value, dict):
        return {
            key: ("<redacted>" if _is_secret_key(str(key)) and item not in (None, "") else _redact_secret_values(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_secret_values(item) for item in value]
    return value


def _safe_config_yaml(cfg: DictConfig) -> str:
    redacted = _redact_secret_values(cfg)
    return OmegaConf.to_yaml(OmegaConf.create(redacted))


def _instantiate_configured_stages(cfg: DictConfig) -> list[ProcessingStage]:
    stage_entries = _configured_stage_entries(cfg)
    if not stage_entries:
        msg = (
            "qwen_omni_inprocess requires a Hydra 'stages:' list. "
            "There is no implicit Granary-v2 stage fallback; every stage must be listed explicitly."
        )
        raise ValueError(msg)

    run_set = _normalise_name_set(cfg.get("stages_to_run", cfg.get("processors_to_run", "all")))
    skip_set = _normalise_name_set(cfg.get("stages_to_skip", cfg.get("processors_to_skip", []))) or set()
    available: set[str] = set()
    stages: list[ProcessingStage] = []

    for stage_cfg in stage_entries:
        raw_unresolved = _as_container(stage_cfg, resolve=False)
        if not isinstance(raw_unresolved, dict):
            msg = f"Each processor entry must be a mapping, got {type(raw_unresolved).__name__}"
            raise TypeError(msg)

        stage_id = raw_unresolved.get("stage_id", raw_unresolved.get("processor_id", raw_unresolved.get("id")))
        enabled = bool(stage_cfg.get("enabled", True))
        target = str(raw_unresolved.get("_target_", ""))
        idents = _target_idents(stage_id, target)
        available.update(idents)
        selected = enabled
        if selected and run_set is not None:
            selected = bool(idents & run_set)
        if selected and idents & skip_set:
            selected = False
        if selected:
            raw = _as_container(stage_cfg, resolve=True)
            raw.pop("stage_id", raw.pop("processor_id", raw.pop("id", None)))
            raw.pop("enabled", None)
            stage_with = raw.pop("stage_with", raw.pop("with_", None))
            stage = hydra.utils.instantiate(OmegaConf.create(raw))
            stage = _apply_stage_with(stage, stage_with)
            stages.append(stage)
            logger.info("Enabled stage {} ({})", stage_id or stage.name, type(stage).__name__)
        else:
            logger.info("Skipped stage {} ({})", stage_id or target.rsplit(".", 1)[-1], target)

    unknown_run = run_set - available if run_set is not None else set()
    unknown_skip = skip_set - available
    if unknown_run or unknown_skip:
        details = []
        if unknown_run:
            details.append(f"stages_to_run={sorted(unknown_run)}")
        if unknown_skip:
            details.append(f"stages_to_skip={sorted(unknown_skip)}")
        msg = f"Unknown processor selector(s): {', '.join(details)}. Available: {sorted(available)}"
        raise ValueError(msg)
    if not stages:
        raise ValueError("No stages selected; check enabled/stages_to_run/stages_to_skip")
    return stages


def build_granary_v2_pipeline(cfg: DictConfig) -> Pipeline:
    """Construct the Granary v2 stage chain from the explicit Hydra stage list."""
    return Pipeline(name="qwen_omni_inference", stages=_instantiate_configured_stages(cfg))


def _create_executor(cfg: DictConfig):  # noqa: ANN201
    backend = cfg.get("backend", "ray_data")
    if backend not in _EXECUTOR_FACTORIES:
        msg = f"Unknown backend '{backend}'. Choose from: {list(_EXECUTOR_FACTORIES)}"
        raise ValueError(msg)

    module_path, class_name = _EXECUTOR_FACTORIES[backend].rsplit(":", 1)
    executor_cls = getattr(importlib.import_module(module_path), class_name)
    logger.info(f"Using backend: {backend}")

    if backend == "xenna":
        return executor_cls(
            config={
                "execution_mode": cfg.get("execution_mode", "streaming"),
                "autoscale_interval_s": cfg.get("autoscale_interval_s", 180),
            }
        )
    if cfg.get("execution_mode") not in (None, "streaming"):
        logger.info("execution_mode={} is Xenna-only and is ignored by Ray Data", cfg.get("execution_mode"))
    return executor_cls()


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point for the Granary v2 Qwen-Omni pipeline."""
    logger.info(f"Hydra config:\n{_safe_config_yaml(cfg)}")

    pipeline = build_granary_v2_pipeline(cfg)
    logger.info(f"Pipeline: {pipeline.describe()}")

    executor = _create_executor(cfg)

    t0 = time.time()
    pipeline.run(executor=executor)
    elapsed = time.time() - t0
    output_dir = cfg.get("output_dir") or cfg.get("workspace_dir", "./output")
    logger.info(f"Pipeline finished in {elapsed / 60:.1f} min. Output: {output_dir}")

    perf_summary_path = os.path.join(output_dir, "perf_summary.json")
    if os.path.isfile(perf_summary_path):
        import json as _json

        with open(perf_summary_path) as _f:
            perf_data = _json.load(_f)
        perf_data["pipeline_duration_s"] = elapsed
        with open(perf_summary_path, "w") as _f:
            _json.dump(perf_data, _f, indent=2, ensure_ascii=False)
        logger.info(f"Performance summary ({perf_summary_path}):\n{_json.dumps(perf_data, indent=2)}")


if __name__ == "__main__":
    main()
