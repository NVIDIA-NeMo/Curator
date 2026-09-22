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

"""Small public helpers for benchmark config orchestration.

The benchmark runner already knows how to merge YAML config fragments and
expand per-entry defaults. CI orchestration needs that same behavior before it
starts scheduler jobs, but it should not import ``benchmarking/run.py`` because
that pulls in Curator runtime dependencies. Keep this module limited to YAML and
stdlib-only planning logic so external launchers can use it from a source tree.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EntryPlan:
    """Resolved launch-relevant configuration for one benchmark entry."""

    name: str
    config: dict[str, Any]
    timeout_s: int
    ray: dict[str, Any]


@dataclass(frozen=True)
class ConfigPlan:
    """Resolved launch-relevant configuration for a merged benchmark config."""

    config: dict[str, Any]
    entries: tuple[EntryPlan, ...]
    default_timeout_s: int
    startup_timeout_s: int
    cleanup_timeout_s: int
    max_timeout_s: int
    min_timeout_s: int
    slurm_max_time_s: int


@dataclass(frozen=True)
class SlurmTimeoutPlan:
    """Timeout values an external scheduler should use for one entry job."""

    entry_timeout_s: int
    wall_time_s: int
    effective_max_timeout_s: int
    capped: bool


def update_config(config: dict[str, Any], override: dict[str, Any]) -> None:
    """Merge ``override`` into ``config`` using benchmark runner semantics."""

    for key, value in override.items():
        if key not in config:
            config[key] = copy.deepcopy(value)
        elif isinstance(config[key], dict) and isinstance(value, dict):
            update_config(config[key], value)
        elif isinstance(config[key], list) and isinstance(value, list):
            _update_list(config[key], value)
        else:
            config[key] = copy.deepcopy(value)


def _update_list(config_list: list[Any], override_list: list[Any]) -> None:
    for item in override_list:
        if not isinstance(item, dict) or not item:
            config_list.append(copy.deepcopy(item))
            continue

        match_key = "name" if "name" in item else next(iter(item.keys()))
        for config_item in config_list:
            if (
                isinstance(config_item, dict)
                and config_item
                and match_key in config_item
                and config_item[match_key] == item[match_key]
            ):
                update_config(config_item, item)
                break
        else:
            config_list.append(copy.deepcopy(item))


def merge_config_files(config_files: list[str | Path]) -> dict[str, Any]:
    """Read and merge benchmark YAML config files in command-line order."""

    config: dict[str, Any] = {}
    for config_file in config_files:
        with Path(config_file).open(encoding="utf-8") as f:
            for config_part in yaml.full_load_all(f):
                if config_part:
                    update_config(config, config_part)
    return config


def load_benchmark_config(config_files: list[str | Path], **_: object) -> dict[str, Any]:
    """Compatibility wrapper for automation code that names config loading."""

    return merge_config_files(config_files)


def build_benchmark_config_plan(config: dict[str, Any], *, enabled_only: bool = True) -> ConfigPlan:
    """Return scheduler-facing entry and timeout data from a merged config."""

    default_timeout_s = config.get("default_timeout_s", 7200)
    global_ray = config.get("ray", {})
    entries = []
    for entry in config.get("entries", []):
        if enabled_only and entry.get("enabled", True) is False:
            continue
        entries.append(
            EntryPlan(
                name=entry["name"],
                config=copy.deepcopy(entry),
                timeout_s=entry.get("timeout_s", default_timeout_s),
                ray={**global_ray, **entry.get("ray", {})},
            )
        )

    return ConfigPlan(
        config=config,
        entries=tuple(entries),
        default_timeout_s=default_timeout_s,
        startup_timeout_s=config.get("startup_timeout_s", 600),
        cleanup_timeout_s=config.get("cleanup_timeout_s", 60),
        max_timeout_s=config.get("max_timeout_s", 14340),
        min_timeout_s=config.get("min_timeout_s", 600),
        slurm_max_time_s=config.get("slurm_max_time_s", 14400),
    )


def plan_entry_slurm_timeout(entry: EntryPlan, plan: ConfigPlan) -> SlurmTimeoutPlan:
    """Return entry and scheduler wall time values without exceeding Slurm limits."""

    slurm_entry_timeout_cap = max(plan.slurm_max_time_s - plan.startup_timeout_s - plan.cleanup_timeout_s, 0)
    effective_max_timeout_s = min(plan.max_timeout_s, slurm_entry_timeout_cap)
    entry_timeout_s = min(entry.timeout_s, effective_max_timeout_s)
    wall_time_s = max(entry_timeout_s + plan.startup_timeout_s + plan.cleanup_timeout_s, plan.min_timeout_s)
    wall_time_s = min(wall_time_s, plan.slurm_max_time_s)
    return SlurmTimeoutPlan(
        entry_timeout_s=entry_timeout_s,
        wall_time_s=wall_time_s,
        effective_max_timeout_s=effective_max_timeout_s,
        capped=entry_timeout_s != entry.timeout_s,
    )


def exact_entry_config(
    config: dict[str, Any],
    entry_name: str,
    *,
    enabled_only: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Return a minimal config fragment containing only ``entry_name``."""

    entries = [
        entry
        for entry in config.get("entries", [])
        if entry.get("name") == entry_name and (not enabled_only or entry.get("enabled", True) is not False)
    ]
    if not entries:
        msg = f"benchmark config does not contain entry: {entry_name}"
        raise ValueError(msg)
    return {"entries": [copy.deepcopy(entries[0])]}


def legacy_path_config(base_config: dict[str, Any], override_config: dict[str, Any]) -> dict[str, Any]:
    """Convert modern ``paths`` overrides into the legacy path-field shape."""

    paths = {
        item["name"]: item["host_path"]
        for item in override_config.get("paths", [])
        if item.get("name") in {"results_path", "datasets_path", "model_weights_path"}
    }
    missing = sorted({"results_path", "datasets_path", "model_weights_path"} - set(paths))
    if missing:
        msg = f"override config is missing required path entries: {missing}"
        raise ValueError(msg)

    legacy = dict(paths)
    datasets = _merge_datasets(base_config.get("datasets", []), override_config.get("datasets", []))
    if datasets:
        legacy["datasets"] = datasets
    return legacy


def _merge_datasets(
    base_datasets: list[dict[str, Any]],
    override_datasets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = {dataset["name"]: copy.deepcopy(dataset) for dataset in base_datasets}
    order = [dataset["name"] for dataset in base_datasets]
    for dataset in override_datasets:
        name = dataset["name"]
        if name not in merged:
            order.append(name)
            merged[name] = copy.deepcopy(dataset)
            continue

        for fmt in dataset.get("formats", []):
            base_formats = merged[name].setdefault("formats", [])
            for index, base_fmt in enumerate(base_formats):
                if base_fmt.get("type") == fmt.get("type"):
                    base_formats[index] = copy.deepcopy(fmt)
                    break
            else:
                base_formats.append(copy.deepcopy(fmt))

    return [merged[name] for name in order]
