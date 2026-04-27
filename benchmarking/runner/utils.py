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

# ruff: noqa: LOG015, ERA001

import os
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

# utils.py is also imported in scripts that run before the Curator
# environment is set up so do not assume loguru is available
try:
    from loguru import logger
except ImportError:
    import logging as logger

_env_var_pattern = re.compile(r"\$\{([^}]+)\}")  # Pattern to match ${VAR_NAME}


# TODO: This utility contains some special cases for Slack JSON messages used in the Slack sink.
# Consider moving these special cases to the Slack sink itself.
def get_obj_for_json(obj: object) -> object:
    """
    Convert common objects used in the benchmark framework to JSON-friendly primitives.
    """
    if isinstance(obj, dict):
        retval = {get_obj_for_json(k): get_obj_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        retval = [get_obj_for_json(item) for item in obj]
    elif isinstance(obj, Path):
        retval = str(obj)
    elif obj is None:  # special case for Slack: JSON null not allowed, convert to string
        retval = "null"
    elif isinstance(obj, str) and len(obj) == 0:  # special case for Slack: empty strings not allowed
        retval = " "
    else:
        retval = obj
    return retval


def _replace_env_var(match: re.Match[str]) -> str:
    env_var_name = match.group(1)
    env_value = os.getenv(env_var_name)
    if env_value is not None and env_value != "":
        return env_value
    else:
        msg = f"Environment variable {env_var_name} not found in the environment or is empty"
        raise ValueError(msg)


def remove_disabled_blocks(obj: object) -> object:
    """
    Recursively remove dictionary blocks that contain "enabled": False.
    Processes dicts and lists; other types are returned unchanged.
    """
    if isinstance(obj, dict):
        # If this block explicitly disables itself, remove it
        if obj.get("enabled", True) is False:
            return None
        # Else process all values
        result = {}
        for k, v in obj.items():
            filtered = remove_disabled_blocks(v)
            if filtered is not None:
                result[k] = filtered
        return result
    elif isinstance(obj, list):
        # Process each item; skip any that are removed
        result = []
        for item in obj:
            filtered = remove_disabled_blocks(item)
            if filtered is not None:
                result.append(filtered)
        return result
    else:
        return obj


def resolve_env_vars(data: dict | list | str | object) -> dict | list | str | object:
    """Recursively resolve environment variables in strings in/from various objects.

    Environment variables are identified in strings when specified using the ${VAR_NAME}
    syntax. If the environment variable is not found, ValueError is raised.
    """
    if isinstance(data, dict):
        return {key: resolve_env_vars(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [resolve_env_vars(item) for item in data]
    elif isinstance(data, str):
        return _env_var_pattern.sub(_replace_env_var, data)
    else:
        return data


def find_result(results: dict[str, Any], key: str, default_value: Any = None) -> Any:  # noqa: ANN401
    """Find a value in the results dictionary by key, checking both the metrics sub-dict and then the results itself."""
    if "metrics" in results:
        return results["metrics"].get(key, results.get(key, default_value))
    else:
        return results.get(key, default_value)


def get_total_memory_bytes() -> int:
    """
    Get the memory limit, respecting Docker/container constraints.
    Tries cgroup limits first, falls back to system memory.
    """

    def read_int_from_file(path: str) -> int | None:
        try:
            return int(Path(path).read_text().strip())
        except (FileNotFoundError, ValueError, PermissionError):
            return None

    # Try cgroup v2 (unified hierarchy)
    limit = read_int_from_file("/sys/fs/cgroup/memory.max")
    if limit is not None:
        return limit

    # Try cgroup v1
    limit = read_int_from_file("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if limit is not None and limit < (1 << 62):  # Check if it's not "unlimited"
        return limit

    # Fallback: get total physical memory
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def get_shm_usage() -> dict[str, int | str | None]:
    """
    Get structured /dev/shm usage data using shutil.disk_usage.

    Returns a dict with keys:
        total_bytes, used_bytes, available_bytes: int or None
        summary: human-readable string summarizing usage
    """
    result_dict: dict[str, int | str | None] = {
        "total_bytes": None,
        "used_bytes": None,
        "available_bytes": None,
        "summary": None,
    }
    try:
        usage = shutil.disk_usage("/dev/shm")  # noqa: S108
    except OSError as exc:
        logger.warning(f"Could not get /dev/shm usage: {exc}")
        return result_dict

    result_dict["total_bytes"] = usage.total
    result_dict["used_bytes"] = usage.used
    result_dict["available_bytes"] = usage.free
    result_dict["summary"] = (
        f"/dev/shm: {human_readable_bytes_repr(usage.used)} used / "  # noqa: S108
        f"{human_readable_bytes_repr(usage.total)} total "
        f"({human_readable_bytes_repr(usage.free)} available)"
    )
    return result_dict


def human_readable_bytes_repr(size: int) -> str:
    """
    Convert a size in bytes to a human readable string (e.g. "1.2 GiB").
    """
    suffixes = list(enumerate(["B", "KiB", "MiB", "GiB", "TiB", "PiB"]))
    suffixes.reverse()
    for index, suffix in suffixes:
        threshold = 1024**index
        if size >= threshold:
            value = float(size) / threshold
            if index == 0:
                return f"{int(size)} {suffix}"
            return f"{value:.2f} {suffix}"
    return "0 B"


def get_gpu_stats() -> dict:
    """
    Query GPU stats using gpustat and return memory information and process info for each available GPU.

    Returns:
        dict: Keys are GPU indices; values are dicts containing:
            - "memory_total" (int): Total GPU memory in MiB.
            - "memory_used" (int): Used GPU memory in MiB.
            - "processes" (list[dict]): List of processes using the GPU, each with keys:
                "username", "command", "gpu_memory_usage", "pid".
    """
    # utils.py is also imported in scripts that run before the Curator
    # environment is set up, so import gpustat lazily.
    import gpustat

    query = gpustat.new_query()
    query_data = {}
    for gpu in query:
        # Only include certain fields from the process data.
        process_data = [
            {k: p.get(k) for k in ["username", "command", "gpu_memory_usage", "pid"]} for p in gpu.processes
        ]
        query_data[gpu.index] = {
            "memory_total": gpu.memory_total,
            "memory_used": gpu.memory_used,
            "processes": process_data,
        }
    return query_data


def log_gpu_stats(gpu_stats: dict, warn_if_in_use: bool = False) -> None:
    """Log GPU memory usage for each GPU as a percentage of total memory.

    Args:
        gpu_stats: Dictionary as returned by get_gpu_stats().
        warn_if_in_use: If True, emit a warning for any GPU with memory_used > 0.
    """
    for gpu_id, stats in gpu_stats.items():
        pct_used = stats["memory_used"] / stats["memory_total"] * 100
        logger.info(f"GPU {gpu_id} : {pct_used:.1f}%")
        if warn_if_in_use and stats["memory_used"] > 0:
            logger.warning(
                f"GPU {gpu_id} has {stats['memory_used']} MiB ({pct_used:.1f}% of total) used before benchmark started"
            )


_LEGACY_PATH_FIELDS = ["results_path", "datasets_path", "model_weights_path"]


def assert_valid_config_dict(data: dict) -> None:
    """Assert that the configuration contains the minimum required config values."""
    has_legacy = any(k in data for k in _LEGACY_PATH_FIELDS)
    has_paths = "paths" in data

    if has_legacy and has_paths:
        msg = (
            "Configuration error: 'results_path', 'datasets_path', and 'model_weights_path' "
            "are deprecated and cannot be used together with the 'paths' section. "
            "Please remove the legacy path fields and use only the 'paths' section."
        )
        raise ValueError(msg)

    if has_legacy:
        logger.warning(
            "'results_path', 'datasets_path', and 'model_weights_path' are deprecated. "
            "Please migrate to using the 'paths' section instead."
        )
        missing = [k for k in _LEGACY_PATH_FIELDS if k not in data]
        if missing:
            msg = f"Invalid configuration: missing required legacy path fields: {missing}"
            raise ValueError(msg)
    elif not has_paths:
        msg = "Invalid configuration: missing required field: 'paths'"
        raise ValueError(msg)
    else:
        for i, path_entry in enumerate(data["paths"]):
            if not isinstance(path_entry, dict):
                msg = f"Invalid configuration: 'paths' entry at index {i} must be a dict"
                raise ValueError(msg)  # noqa: TRY004
            missing = [k for k in ("name", "host_path") if k not in path_entry]
            if missing:
                msg = f"Invalid configuration: 'paths' entry at index {i} is missing required fields: {missing}"
                raise ValueError(msg)
        path_names = {p["name"] for p in data["paths"]}
        if "results_path" not in path_names:
            msg = "Invalid configuration: 'paths' section must include an entry with name 'results_path'"
            raise ValueError(msg)

    if "entries" not in data:
        msg = "Invalid configuration: missing required field: 'entries'"
        raise ValueError(msg)


def update_config(config_dict: dict, new_dict: dict) -> None:
    """Update a config dictionary with values from another."""

    # Iterate through all key-value pairs in the dictionary to merge in
    for key, value in new_dict.items():
        if key in config_dict:
            # Recursively handle nested dicts
            if isinstance(config_dict[key], dict) and isinstance(value, dict):
                update_config(config_dict[key], value)

            # Handle list merging/updating on an item-by-item basis
            # For example, the YAML:
            #     entries:
            #      - name: domain_classification_raydata
            #        requirements:
            #          - metric: throughput_docs_per_sec
            #            min_value: 2677            # results in:
            #     config_dict['entries'] = [{'name': 'domain_classification_raydata',
            #                                'requirements': [{'metric': 'throughput_docs_per_sec', 'min_value': 2677}]
            #                              }]
            #
            # so be sure to update the config_dict list items that match the new_dict list items by matching based on the first key
            elif isinstance(config_dict[key], list) and isinstance(value, list):
                for sub_val in value:
                    # Handle dicts in the list by matching based on the first key
                    if isinstance(sub_val, dict) and sub_val:
                        first_key = next(iter(sub_val.keys()))
                        for config_sub_val in config_dict[key]:
                            if (
                                isinstance(config_sub_val, dict)
                                and config_sub_val
                                and next(iter(config_sub_val.keys())) == first_key
                                and config_sub_val[first_key] == sub_val[first_key]
                            ):
                                # If matching dict found (based on first key), recursively update it
                                update_config(config_sub_val, sub_val)
                                break
                        else:
                            # If no matching dict, append the new dict to the list
                            config_dict[key].append(sub_val)
                    else:
                        # If not a dict, append the new value to the list
                        config_dict[key].append(sub_val)
            else:
                # If types differ, or not a dict/list, replace value in config_dict
                config_dict[key] = value
        else:
            # If key doesn't exist, add it to config_dict
            config_dict[key] = value


def merge_config_files(config_files: list[Path]) -> dict:
    """Merge multiple config files into a single dictionary."""
    config_dict = {}
    for config_file in config_files:
        with open(config_file) as f:
            for new_dict in yaml.full_load_all(f):
                if new_dict is not None:
                    update_config(config_dict, new_dict)
    return config_dict
