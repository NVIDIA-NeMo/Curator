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
from typing import Any


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
