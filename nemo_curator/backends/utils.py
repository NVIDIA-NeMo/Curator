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

from copy import deepcopy
from typing import TYPE_CHECKING

import ray
from loguru import logger

if TYPE_CHECKING:
    import loguru


def _logger_custom_serializer(
    _: "loguru.Logger",
) -> None:
    return None


def _logger_custom_deserializer(
    _: None,
) -> "loguru.Logger":
    # Initialize a default logger
    return logger


def register_loguru_serializer() -> None:
    """Initialize a new local Ray cluster or connects to an existing one."""
    # Turn off serization for loguru. This is needed as loguru is not serializable in general.
    ray.util.register_serializer(
        logger.__class__,
        serializer=_logger_custom_serializer,
        deserializer=_logger_custom_deserializer,
    )


def merge_executor_configs(base_config: dict | None, override_config: dict | None) -> dict:
    """
    Recursively merge two executor configs with deep merging of nested dicts.

    Args:
        base_config: Base configuration dictionary
        override_config: Configuration to merge on top of base_config

    Returns:
        Merged configuration dictionary with all nested dicts recursively merged

    Notes:
        - Recursively merges all nested dictionaries
        - Non-dict values in override_config will overwrite base_config
        - Handles None values gracefully
        - Does not modify original inputs (uses deep copy)

    Examples:
        >>> base = {"runtime_env": {"env_vars": {"A": "1", "B": "2"}}}
        >>> override = {"runtime_env": {"env_vars": {"B": "3", "C": "4"}}}
        >>> merge_executor_configs(base, override)
        {"runtime_env": {"env_vars": {"A": "1", "B": "3", "C": "4"}}}
    """
    # Handle None cases
    if base_config is None and override_config is None:
        return {}
    if base_config is None:
        return deepcopy(override_config)
    if override_config is None:
        return deepcopy(base_config)

    # Deep copy to avoid modifying originals
    merged_config = deepcopy(base_config)

    # Recursively merge each key from override_config
    for key, value in override_config.items():
        if isinstance(value, dict):
            if key not in merged_config or not isinstance(merged_config[key], dict):
                # If key doesn't exist or isn't a dict, just use the override value
                merged_config[key] = deepcopy(value)
            else:
                # Recursively merge nested dicts
                merged_config[key] = merge_executor_configs(merged_config[key], value)
        else:
            # For non-dict values, overwrite
            merged_config[key] = value

    return merged_config


def warn_on_env_var_override(existing_config: dict | None, merged_config: dict | None) -> None:
    existing_env_vars = (existing_config or {}).get("runtime_env", {}).get("env_vars", {})
    merged_env_vars = (merged_config or {}).get("runtime_env", {}).get("env_vars", {})
    if not existing_env_vars or not merged_env_vars:
        return

    overridden_keys = sorted(
        key
        for key in existing_env_vars.keys() & merged_env_vars.keys()
        if existing_env_vars[key] != merged_env_vars[key]
    )
    if overridden_keys:
        logger.warning(
            "Merged executor configuration overrides env_vars %s from the supplied executor. "
            "Update the executor configuration before running if this is unintended.",
            overridden_keys,
        )
