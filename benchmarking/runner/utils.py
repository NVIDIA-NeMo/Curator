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

import os
import re
import subprocess
from pathlib import Path
from typing import Any

# utils.py is also imported in scripts that run before the Curator
# environment is set up so do not assume loguru is available
# ruff: noqa: LOG015
try:
    from loguru import logger
except ImportError:
    import logging as logger


def get_obj_for_json(obj: object) -> str | int | float | bool | list | dict:
    """
    Recursively convert objects to Python primitives for JSON serialization.
    Useful for objects like Path, sets, bytes, etc.
    """
    if isinstance(obj, dict):
        retval = {get_obj_for_json(k): get_obj_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        retval = [get_obj_for_json(item) for item in obj]
    elif hasattr(obj, "as_posix"):  # Path objects
        retval = obj.as_posix()
    elif isinstance(obj, bytes):
        retval = obj.decode("utf-8", errors="replace")
    elif hasattr(obj, "to_json") and callable(obj.to_json):
        retval = obj.to_json()
    elif hasattr(obj, "__dict__"):
        retval = get_obj_for_json(vars(obj))
    elif obj is None:
        retval = "null"
    elif isinstance(obj, str) and len(obj) == 0:  # special case for Slack, empty strings not allowed
        retval = " "
    else:
        retval = obj
    return retval


_env_var_pattern = re.compile(r"\$\{([^}]+)\}")  # Pattern to match ${VAR_NAME}


def _replace_env_var(match: re.Match[str]) -> str:
    env_var_name = match.group(1)
    env_value = os.getenv(env_var_name)
    if env_value is not None and env_value != "":
        return env_value
    else:
        msg = f"Environment variable {env_var_name} not found in the environment or is empty"
        raise ValueError(msg)


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


def run_shm_size_check(human_readable: bool = False) -> tuple[int | None, str | None]:
    """
    Run the appropriate "df" command to check the size of the system shared memory space.
    """
    command = ["df", "-h", "/dev/shm"] if human_readable else ["df", "--block-size=1", "/dev/shm"]  # noqa: S108
    command_str = " ".join(command)
    result = None
    try:
        result = subprocess.run(  # noqa: S603
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.debug(f"`{command_str}` output:\n{result.stdout}")
    except subprocess.CalledProcessError as df_exc:
        logger.warning(f"Could not run `{command_str}`: {df_exc}")

    # Extract the size from the last line of the output
    if result is not None:
        output = result.stdout
        line = output.strip().split("\n")[-1]
        try:
            size = line.split()[1]  # Size is the second column
            # Convert to a real number if not meant for simply reading by humans
            if not human_readable:
                size = int(size)
        except (ValueError, IndexError):
            logger.warning(f"Could not parse size from `{command_str}` output line: {line}")
            size = None
        return (size, output)
    else:
        return (None, None)


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
