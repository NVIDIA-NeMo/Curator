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

This wraps the same pipeline logic as
``examples/audio/qwen_omni_inprocess/run_pipeline.py`` behind a Hydra
config so it can be invoked by NvLLMOps Kratos workflows via::

    python tutorials/audio/qwen_omni_inprocess/main.py \\
        --config-path=<path> --config-name=qwen_omni_inprocess \\
        workspace_dir=/work input_manifest=/data/config.yaml

The NvLLMOps ``run_curator()`` function calls::

    tutorials/audio/{pipeline_to_run}/main.py --config-path=... --config-name=...

with Hydra overrides ``workspace_dir``, ``input_manifest``,
``language_short``, ``max_segment_length``, ``hf_token``, and
``final_manifest``.  All of these are accepted as top-level keys in the
YAML and forwarded to the appropriate stages.
"""

import importlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.audio.inference.qwen_asr import InferenceQwenASRStage
from nemo_curator.stages.audio.inference.qwen_omni import InferenceQwenOmniStage
from nemo_curator.stages.audio.text_filtering import (
    FastTextLIDStage,
    ITNRestorationStage,
    PnCRestorationStage,
)
from nemo_curator.stages.resources import Resources

_EXECUTOR_FACTORIES = {
    "xenna": "nemo_curator.backends.xenna:XennaExecutor",
    "ray_data": "nemo_curator.backends.ray_data:RayDataExecutor",
}


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


def _instantiate_resources(value: Any) -> Resources:
    if isinstance(value, Resources):
        return value
    if OmegaConf.is_config(value) or isinstance(value, dict):
        cfg = value if OmegaConf.is_config(value) else OmegaConf.create(value)
        if "_target_" in cfg:
            return hydra.utils.instantiate(cfg)
        raw = OmegaConf.to_container(cfg, resolve=True)
        return Resources(**raw)
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


def _target_idents(processor_id: str | None, target: str | None) -> set[str]:
    return {
        ident
        for ident in (
            processor_id,
            target,
            target.rsplit(".", 1)[-1] if target else None,
        )
        if ident
    }


def _instantiate_configured_processors(cfg: DictConfig) -> list[ProcessingStage]:
    if "processors" not in cfg or not cfg.processors:
        msg = (
            "qwen_omni_inprocess requires a Hydra 'processors:' list. "
            "There is no implicit Granary-v2 stage fallback; every stage must be listed explicitly."
        )
        raise ValueError(msg)

    run_set = _normalise_name_set(cfg.get("processors_to_run", "all"))
    skip_set = _normalise_name_set(cfg.get("processors_to_skip", [])) or set()
    available: set[str] = set()
    stages: list[ProcessingStage] = []

    for processor_cfg in cfg.processors:
        raw_unresolved = _as_container(processor_cfg, resolve=False)
        if not isinstance(raw_unresolved, dict):
            msg = f"Each processor entry must be a mapping, got {type(raw_unresolved).__name__}"
            raise TypeError(msg)

        processor_id = raw_unresolved.get("processor_id", raw_unresolved.get("id"))
        enabled = bool(processor_cfg.get("enabled", True))
        target = str(raw_unresolved.get("_target_", ""))
        idents = _target_idents(processor_id, target)
        available.update(idents)
        selected = enabled
        if selected and run_set is not None:
            selected = bool(idents & run_set)
        if selected and idents & skip_set:
            selected = False
        if selected:
            raw = _as_container(processor_cfg, resolve=True)
            raw.pop("processor_id", raw.pop("id", None))
            raw.pop("enabled", None)
            stage_with = raw.pop("stage_with", raw.pop("with_", None))
            stage = hydra.utils.instantiate(OmegaConf.create(raw))
            stage = _apply_stage_with(stage, stage_with)
            stages.append(stage)
            logger.info("Enabled processor {} ({})", processor_id or stage.name, type(stage).__name__)
        else:
            logger.info("Skipped processor {} ({})", processor_id or target.rsplit(".", 1)[-1], target)

    unknown_run = run_set - available if run_set is not None else set()
    unknown_skip = skip_set - available
    if unknown_run or unknown_skip:
        details = []
        if unknown_run:
            details.append(f"processors_to_run={sorted(unknown_run)}")
        if unknown_skip:
            details.append(f"processors_to_skip={sorted(unknown_skip)}")
        msg = f"Unknown processor selector(s): {', '.join(details)}. Available: {sorted(available)}"
        raise ValueError(msg)
    if not stages:
        raise ValueError("No processors selected; check enabled/processors_to_run/processors_to_skip")
    return stages


def build_granary_v2_pipeline(cfg: DictConfig) -> Pipeline:
    """Construct the Granary v2 stage chain from the explicit Hydra processor list."""
    return Pipeline(name="qwen_omni_inference", stages=_instantiate_configured_processors(cfg))


def _prefetch_models(stages: list[ProcessingStage]) -> None:
    """Download all models in parallel before pipeline execution.

    On gpu.l40s.4 (218 CPUs, 980Gi RAM, 5680Gi disk) we have massive
    network bandwidth available. Downloading QwenOmni (15GB), PnC (18GB),
    and FastText (130MB) sequentially wastes 10-15 min. By launching all
    downloads concurrently we saturate the NIC and cut total wait to the
    duration of the single largest download (~18GB / bandwidth).
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    tasks: list[tuple[str, Callable[[], str]]] = []
    seen_snapshots: set[str] = set()
    seen_hf_files: set[str] = set()

    def add_snapshot(model_id: str) -> None:
        if model_id in seen_snapshots:
            return
        seen_snapshots.add(model_id)
        tasks.append((f"snapshot:{model_id}", lambda m=model_id: snapshot_download(m)))

    def add_hf_file(repo_id: str, filename: str = "model.bin") -> None:
        key = f"{repo_id}:{filename}"
        if key in seen_hf_files:
            return
        seen_hf_files.add(key)
        tasks.append((f"hf_hub:{repo_id}", lambda r=repo_id, f=filename: hf_hub_download(repo_id=r, filename=f)))

    for stage in stages:
        if isinstance(stage, InferenceQwenOmniStage):
            add_snapshot(stage.model_id)
        elif isinstance(stage, InferenceQwenASRStage):
            add_snapshot(stage.model_id)
        elif isinstance(stage, PnCRestorationStage):
            add_snapshot(stage.model_id)
        elif isinstance(stage, ITNRestorationStage):
            add_snapshot(stage.model_id)
        elif isinstance(stage, FastTextLIDStage):
            if "/" in stage.model_path and not os.path.isfile(stage.model_path):
                add_hf_file(stage.model_path)

    if not tasks:
        return

    logger.info(f"Pre-fetching {len(tasks)} models in parallel...")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                logger.info(f"  ✓ {name} cached ({time.time() - t0:.1f}s elapsed)")
            except Exception as exc:
                logger.warning(f"  ✗ {name} failed: {exc} (will retry in setup_on_node)")

    logger.info(f"All model pre-fetch complete in {time.time() - t0:.1f}s")


def _create_executor(cfg: DictConfig):  # noqa: ANN201
    backend = cfg.get("backend", "xenna")
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
    logger.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    hf_token = cfg.get("hf_token", "")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ.setdefault("HF_HOME", "/tmp/hf_home")

    pipeline = build_granary_v2_pipeline(cfg)
    logger.info(f"Pipeline: {pipeline.describe()}")
    _prefetch_models(pipeline.stages)

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
