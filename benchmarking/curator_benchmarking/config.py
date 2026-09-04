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

"""Public helpers for reading and planning Curator benchmark configs.

These functions intentionally operate on dictionaries instead of runner
``Session`` objects. External orchestration systems often need to inspect
benchmark YAML during job generation, before a Curator runtime environment is
available and without importing sink, Ray, or benchmark execution dependencies.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runner.utils import (
    assert_valid_config_dict,
    remove_disabled_blocks,
    resolve_env_vars,
)
from runner.utils import (
    merge_config_files as _merge_config_files,
)

DEFAULT_TIMEOUT_S = 7200
DEFAULT_STARTUP_TIMEOUT_S = 600
DEFAULT_CLEANUP_TIMEOUT_S = 60
DEFAULT_MAX_TIMEOUT_S = 14340
DEFAULT_MIN_TIMEOUT_S = 600
DEFAULT_SLURM_MAX_TIME_S = 14400
LEGACY_PATH_FIELDS = ("results_path", "datasets_path", "model_weights_path")


@dataclass(frozen=True)
class BenchmarkEntryPlan:
    """Effective metadata for one benchmark or data-setup entry."""

    name: str
    config: Mapping[str, Any]
    timeout_s: int
    ray: Mapping[str, Any]
    enabled: bool


@dataclass(frozen=True)
class BenchmarkConfigPlan:
    """Effective config metadata useful to launchers and CI job generators."""

    config: Mapping[str, Any]
    entries: tuple[BenchmarkEntryPlan, ...]
    data_setups: tuple[BenchmarkEntryPlan, ...]
    default_timeout_s: int
    startup_timeout_s: int
    cleanup_timeout_s: int
    max_timeout_s: int
    min_timeout_s: int
    slurm_max_time_s: int


@dataclass(frozen=True)
class SlurmTimeoutPlan:
    """Wall-clock timeout calculation for one generated Slurm job."""

    entry_timeout_s: int
    wall_time_s: int
    effective_max_timeout_s: int
    capped: bool


def merge_config_files(config_files: Sequence[str | Path]) -> dict[str, Any]:
    """Merge benchmark YAML files using Curator's name-keyed list semantics."""
    return _merge_config_files([Path(config_file) for config_file in config_files])


def load_benchmark_config(
    config_files: Sequence[str | Path],
    *,
    resolve_environment: bool = False,
    strict_environment: bool = False,
    validate: bool = False,
    drop_disabled: bool = False,
) -> dict[str, Any]:
    """Load and optionally process one or more benchmark YAML configs.

    The defaults are deliberately conservative: callers get the raw merged YAML
    by default. Runtime-like processing can be opted into by resolving
    environment variables, validating required runner fields, and dropping
    disabled blocks.
    """
    config = merge_config_files(config_files)
    if validate:
        assert_valid_config_dict(config)
    if drop_disabled:
        config = remove_disabled_blocks(config)
    if resolve_environment:
        config = resolve_env_vars(config, strict=strict_environment)
    return config


def entry_configs(config: Mapping[str, Any], *, enabled_only: bool = True) -> list[dict[str, Any]]:
    """Return benchmark entry configs, optionally filtering disabled entries."""
    entries = config.get("entries", [])
    if not isinstance(entries, list):
        msg = "benchmark config field 'entries' must be a list"
        raise TypeError(msg)
    return _filter_named_configs(entries, enabled_only=enabled_only, field_name="entries")


def data_setup_configs(config: Mapping[str, Any], *, enabled_only: bool = True) -> list[dict[str, Any]]:
    """Return data setup configs, optionally filtering disabled setup entries."""
    data_setups = config.get("data_setups", [])
    if not isinstance(data_setups, list):
        msg = "benchmark config field 'data_setups' must be a list"
        raise TypeError(msg)
    return _filter_named_configs(data_setups, enabled_only=enabled_only, field_name="data_setups")


def entry_names(config: Mapping[str, Any], *, enabled_only: bool = True) -> list[str]:
    """Return benchmark entry names in config order."""
    return [entry["name"] for entry in entry_configs(config, enabled_only=enabled_only)]


def exact_entry_config(
    config: Mapping[str, Any],
    entry_name: str,
    *,
    enabled_only: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Return an ``entries:`` override containing only one exact entry.

    This is primarily for compatibility with older benchmark runners that only
    support substring entry selection.
    """
    entries = entry_configs(config, enabled_only=enabled_only)
    matches = [entry for entry in entries if entry.get("name") == entry_name]
    if not matches:
        msg = f"benchmark config does not contain entry: {entry_name}"
        raise ValueError(msg)
    return {"entries": [copy.deepcopy(matches[0])]}


def build_benchmark_config_plan(
    config: Mapping[str, Any],
    *,
    enabled_only: bool = True,
) -> BenchmarkConfigPlan:
    """Return effective config metadata without constructing a runner Session."""
    default_timeout_s = _int_config_value(config, "default_timeout_s", DEFAULT_TIMEOUT_S)
    startup_timeout_s = _int_config_value(config, "startup_timeout_s", DEFAULT_STARTUP_TIMEOUT_S)
    cleanup_timeout_s = _int_config_value(config, "cleanup_timeout_s", DEFAULT_CLEANUP_TIMEOUT_S)
    max_timeout_s = _int_config_value(config, "max_timeout_s", DEFAULT_MAX_TIMEOUT_S)
    min_timeout_s = _int_config_value(config, "min_timeout_s", DEFAULT_MIN_TIMEOUT_S)
    slurm_max_time_s = _int_config_value(config, "slurm_max_time_s", DEFAULT_SLURM_MAX_TIME_S)
    global_ray = _mapping_config_value(config, "ray", {})

    return BenchmarkConfigPlan(
        config=config,
        entries=tuple(
            _entry_plan(entry, default_timeout_s=default_timeout_s, global_ray=global_ray)
            for entry in entry_configs(config, enabled_only=enabled_only)
        ),
        data_setups=tuple(
            _entry_plan(entry, default_timeout_s=default_timeout_s, global_ray={})
            for entry in data_setup_configs(config, enabled_only=enabled_only)
        ),
        default_timeout_s=default_timeout_s,
        startup_timeout_s=startup_timeout_s,
        cleanup_timeout_s=cleanup_timeout_s,
        max_timeout_s=max_timeout_s,
        min_timeout_s=min_timeout_s,
        slurm_max_time_s=slurm_max_time_s,
    )


def plan_slurm_timeout(  # noqa: PLR0913
    entry_timeout_s: int,
    *,
    startup_timeout_s: int = DEFAULT_STARTUP_TIMEOUT_S,
    cleanup_timeout_s: int = DEFAULT_CLEANUP_TIMEOUT_S,
    min_timeout_s: int = DEFAULT_MIN_TIMEOUT_S,
    max_timeout_s: int = DEFAULT_MAX_TIMEOUT_S,
    slurm_max_time_s: int = DEFAULT_SLURM_MAX_TIME_S,
) -> SlurmTimeoutPlan:
    """Return the Slurm wall-clock timeout for one benchmark entry."""
    _validate_positive_int("entry_timeout_s", entry_timeout_s)
    for name, value in {
        "startup_timeout_s": startup_timeout_s,
        "cleanup_timeout_s": cleanup_timeout_s,
        "min_timeout_s": min_timeout_s,
        "max_timeout_s": max_timeout_s,
        "slurm_max_time_s": slurm_max_time_s,
    }.items():
        _validate_nonnegative_int(name, value)

    slurm_entry_timeout_cap = max(slurm_max_time_s - startup_timeout_s - cleanup_timeout_s, 0)
    effective_max_timeout_s = min(max_timeout_s, slurm_entry_timeout_cap)
    capped_entry_timeout_s = min(entry_timeout_s, effective_max_timeout_s)
    wall_time_s = max(capped_entry_timeout_s + startup_timeout_s + cleanup_timeout_s, min_timeout_s)
    wall_time_s = min(wall_time_s, slurm_max_time_s)
    return SlurmTimeoutPlan(
        entry_timeout_s=capped_entry_timeout_s,
        wall_time_s=wall_time_s,
        effective_max_timeout_s=effective_max_timeout_s,
        capped=capped_entry_timeout_s != entry_timeout_s,
    )


def plan_entry_slurm_timeout(
    entry: BenchmarkEntryPlan,
    plan: BenchmarkConfigPlan,
) -> SlurmTimeoutPlan:
    """Return the Slurm wall-clock timeout for one planned benchmark entry."""
    return plan_slurm_timeout(
        entry.timeout_s,
        startup_timeout_s=plan.startup_timeout_s,
        cleanup_timeout_s=plan.cleanup_timeout_s,
        min_timeout_s=plan.min_timeout_s,
        max_timeout_s=plan.max_timeout_s,
        slurm_max_time_s=plan.slurm_max_time_s,
    )


def legacy_path_config(
    base_config: Mapping[str, Any],
    override_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build legacy top-level path fields from a modern ``paths:`` override.

    Older benchmark runners expect top-level ``results_path``, ``datasets_path``,
    and ``model_weights_path`` fields. This helper converts a modern path
    override into that shape and carries forward dataset overrides without
    replacing the full dataset list from the base config.
    """
    legacy = _legacy_paths_from_override_config(override_config)

    base_datasets = base_config.get("datasets", [])
    override_datasets = override_config.get("datasets", [])
    merged_datasets = _merge_dataset_lists(base_datasets, override_datasets)
    if merged_datasets:
        legacy["datasets"] = merged_datasets
    return legacy


def _filter_named_configs(items: list[Any], *, enabled_only: bool, field_name: str) -> list[dict[str, Any]]:
    filtered = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            msg = f"benchmark config field '{field_name}' item at index {index} must be a mapping"
            raise TypeError(msg)
        if "name" not in item:
            msg = f"benchmark config field '{field_name}' item at index {index} is missing 'name'"
            raise ValueError(msg)
        if enabled_only and item.get("enabled", True) is False:
            continue
        filtered.append(copy.deepcopy(dict(item)))
    return filtered


def _entry_plan(
    entry: Mapping[str, Any],
    *,
    default_timeout_s: int,
    global_ray: Mapping[str, Any],
) -> BenchmarkEntryPlan:
    entry_ray = _mapping_config_value(entry, "ray", {})
    return BenchmarkEntryPlan(
        name=str(entry["name"]),
        config=copy.deepcopy(dict(entry)),
        timeout_s=_int_config_value(entry, "timeout_s", default_timeout_s),
        ray=copy.deepcopy({**global_ray, **entry_ray}),
        enabled=entry.get("enabled", True) is not False,
    )


def _int_config_value(config: Mapping[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    _validate_nonnegative_int(key, value)
    return value


def _mapping_config_value(config: Mapping[str, Any], key: str, default: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get(key, default)
    if not isinstance(value, Mapping):
        msg = f"benchmark config field '{key}' must be a mapping"
        raise TypeError(msg)
    return value


def _validate_nonnegative_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        msg = f"{name} must be a non-negative integer; got {value!r}"
        raise TypeError(msg)


def _validate_positive_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        msg = f"{name} must be a positive integer; got {value!r}"
        raise TypeError(msg)


def _legacy_paths_from_override_config(override_config: Mapping[str, Any]) -> dict[str, str]:
    paths = override_config.get("paths")
    if not isinstance(paths, list):
        msg = "override config must contain a 'paths' list"
        raise TypeError(msg)

    host_paths = {}
    for item in paths:
        if not isinstance(item, Mapping):
            msg = "each 'paths' entry must be a mapping"
            raise TypeError(msg)
        name = item.get("name")
        host_path = item.get("host_path")
        if name in LEGACY_PATH_FIELDS and host_path:
            host_paths[name] = host_path

    missing = sorted(set(LEGACY_PATH_FIELDS) - set(host_paths))
    if missing:
        msg = f"override config is missing required path entries: {missing}"
        raise ValueError(msg)

    return {field_name: host_paths[field_name] for field_name in LEGACY_PATH_FIELDS}


def _merge_format_lists(base_formats: list[Any], override_formats: list[Any]) -> list[dict[str, Any]]:
    if not isinstance(base_formats, list):
        msg = "dataset 'formats' must be a list"
        raise TypeError(msg)
    if not isinstance(override_formats, list):
        msg = "dataset 'formats' must be a list"
        raise TypeError(msg)

    merged_by_type = {}
    format_order = []
    for fmt in [*base_formats, *override_formats]:
        if not isinstance(fmt, Mapping):
            msg = "each dataset format must be a mapping"
            raise TypeError(msg)
        fmt_type = fmt.get("type")
        if fmt_type is None:
            msg = "each dataset format must contain a 'type'"
            raise ValueError(msg)
        if fmt_type not in merged_by_type:
            format_order.append(fmt_type)
        merged_by_type[fmt_type] = copy.deepcopy(dict(fmt))
    return [merged_by_type[fmt_type] for fmt_type in format_order]


def _merge_dataset(base_dataset: Mapping[str, Any], override_dataset: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base_dataset))
    for key, value in override_dataset.items():
        if key in {"name", "formats"}:
            continue
        merged[key] = copy.deepcopy(value)

    base_formats = base_dataset.get("formats", [])
    override_formats = override_dataset.get("formats", [])
    if base_formats or override_formats:
        merged["formats"] = _merge_format_lists(base_formats, override_formats)
    return merged


def _merge_dataset_lists(base_datasets: object, override_datasets: object) -> list[dict[str, Any]]:
    base_dataset_list = _validate_dataset_list(base_datasets, "base config")
    override_dataset_list = _validate_dataset_list(override_datasets, "override config")
    merged_by_name = {}
    dataset_order = []

    for dataset in base_dataset_list:
        name = _dataset_name(dataset)
        if name not in merged_by_name:
            dataset_order.append(name)
        merged_by_name[name] = copy.deepcopy(dict(dataset))

    for dataset in override_dataset_list:
        name = _dataset_name(dataset)
        if name in merged_by_name:
            merged_by_name[name] = _merge_dataset(merged_by_name[name], dataset)
        else:
            dataset_order.append(name)
            merged_by_name[name] = copy.deepcopy(dict(dataset))

    return [merged_by_name[name] for name in dataset_order]


def _validate_dataset_list(datasets: object, config_name: str) -> list[Any]:
    if not isinstance(datasets, list):
        msg = f"{config_name} 'datasets' must be a list"
        raise TypeError(msg)
    return datasets


def _dataset_name(dataset: object) -> str:
    if not isinstance(dataset, Mapping):
        msg = "each dataset entry must be a mapping"
        raise TypeError(msg)
    name = dataset.get("name")
    if name is None:
        msg = "each dataset entry must contain a 'name'"
        raise ValueError(msg)
    return str(name)
