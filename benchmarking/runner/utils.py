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

# ruff: noqa: LOG015

import os
import shutil
from pathlib import Path
from typing import Any

from nemo_curator_benchmarking.config import (
    assert_valid_config_dict,
    merge_config_files,
    remove_disabled_blocks,
    resolve_env_vars,
    update_config,
)

__all__ = [
    "assert_valid_config_dict",
    "merge_config_files",
    "remove_disabled_blocks",
    "resolve_env_vars",
    "update_config",
]

# utils.py is also imported in scripts that run before the Curator
# environment is set up so do not assume loguru is available
try:
    from loguru import logger
except ImportError:
    import logging as logger


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


def log_gpu_stats(
    gpu_stats: dict,
    warn_if_in_use: bool = False,
    warning_threshold: float | None = None,
    warning_threshold_msg: str = "still in use",
) -> list[str]:
    """Log GPU memory usage for each GPU as a percentage of total memory.

    Args:
        gpu_stats: Dictionary as returned by get_gpu_stats().
        warn_if_in_use: If True, emit a warning for any GPU that exceeds the usage threshold.
        warning_threshold: Fraction of total GPU memory (0.0-1.0) above which a warning is
            emitted. If None and warn_if_in_use is True, any usage > 0 triggers a warning.
        warning_threshold_msg: Trailing context phrase appended to each warning message
            (e.g. "used before benchmark started"). Defaults to "still in use".

    Returns:
        List of warning strings for any GPUs that triggered a warning.
    """
    warnings = []
    for gpu_id, stats in gpu_stats.items():
        pct_used = stats["memory_used"] / stats["memory_total"] * 100
        logger.info(f"GPU {gpu_id} : {pct_used:.1f}%")
        if warn_if_in_use:
            fraction_used = stats["memory_used"] / stats["memory_total"]
            threshold_exceeded = (
                fraction_used > warning_threshold if warning_threshold is not None else stats["memory_used"] > 0
            )
            if threshold_exceeded:
                msg = f"GPU {gpu_id}: {stats['memory_used']} MiB ({pct_used:.1f}% of total) {warning_threshold_msg}"
                logger.warning(msg)
                warnings.append(msg)
    return warnings
