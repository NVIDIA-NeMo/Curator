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
import time
from copy import deepcopy
from enum import Enum
from typing import TYPE_CHECKING, Any

import ray
from loguru import logger
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.base import ProcessingStage

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


# Global variable to cache head node ID
_HEAD_NODE_ID_CACHE = None


def is_head_node(node: dict[str, Any]) -> bool:
    """Check if a node is the head node."""
    return "node:__internal_head__" in node.get("Resources", {})


def get_head_node_id() -> str | None:
    """Get the head node ID from the Ray cluster, with lazy evaluation and caching.

    Returns:
        The head node ID if a head node exists, otherwise None.
    """
    global _HEAD_NODE_ID_CACHE  # noqa: PLW0603

    if _HEAD_NODE_ID_CACHE is not None:
        return _HEAD_NODE_ID_CACHE

    # Compute head node ID
    for node in ray.nodes():
        if is_head_node(node):
            _HEAD_NODE_ID_CACHE = node["NodeID"]
            return _HEAD_NODE_ID_CACHE

    return None


class RayStageSpecKeys(str, Enum):
    """String enum of different flags that define keys inside ray_stage_spec."""

    IS_ACTOR_STAGE = "is_actor_stage"
    IS_FANOUT_STAGE = "is_fanout_stage"
    IS_RAFT_ACTOR = "is_raft_actor"
    IS_LSH_STAGE = "is_lsh_stage"
    IS_SHUFFLE_STAGE = "is_shuffle_stage"
    MAX_CALLS_PER_WORKER = "max_calls_per_worker"
    RAY_REMOTE_ARGS = "ray_remote_args"


def get_worker_metadata_and_node_id() -> tuple[NodeInfo, WorkerMetadata]:
    """Get the worker metadata and node id from the runtime context."""
    ray_context = ray.get_runtime_context()
    return NodeInfo(node_id=ray_context.get_node_id()), WorkerMetadata(worker_id=ray_context.get_worker_id())


def get_available_cpu_gpu_resources(
    init_and_shutdown: bool = False, ignore_head_node: bool = False
) -> tuple[int, int]:
    """Get available CPU and GPU resources from Ray."""
    if init_and_shutdown:
        ray.init(ignore_reinit_error=True)
    time.sleep(0.2)  # ray.available_resources() returns might have a lag
    # available resources can be different from total resources, however curator assumes
    # entire cluster is available for use and only one pipeline is being run at a time.
    # therefore available resources should match total resources.
    available_resources = ray.available_resources()
    available_cpus = available_resources.get("CPU", 0)
    available_gpus = available_resources.get("GPU", 0)
    if ignore_head_node:
        head_node_id = get_head_node_id()
        if head_node_id is not None:
            total_resources = ray.state.total_resources_per_node().get(head_node_id, {})
            head_node_cpus = total_resources.get("CPU", 0)
            head_node_gpus = total_resources.get("GPU", 0)
            logger.info(
                f"Ignoring head node {head_node_id} with {head_node_cpus} CPUs and {head_node_gpus} GPUs for resource calculation"
            )
            available_cpus = max(0, available_cpus - head_node_cpus)
            available_gpus = max(0, available_gpus - head_node_gpus)
        else:
            logger.warning("ignore_head_node=True but no head node found in the cluster")
    if init_and_shutdown:
        ray.shutdown()
    return (available_cpus, available_gpus)


def check_total_gpu_capacity(gpus_needed: int, *, ignore_head_node: bool = False) -> None:
    """Raise if the cluster doesn't have enough GPUs to satisfy aggregate demand.

    Intended as a coarse pre-check before submitting placement groups: Ray's
    PG scheduler can hang indefinitely on ``pg.ready()`` when demand exceeds
    capacity, so a fast, explicit error with the actual numbers is friendlier
    than waiting on a timeout.
    """
    _, available_gpus = get_available_cpu_gpu_resources(ignore_head_node=ignore_head_node)
    available = int(available_gpus)
    if gpus_needed > available:
        msg = f"Need {gpus_needed} GPUs but cluster has {available} available."
        raise RuntimeError(msg)


@ray.remote
def _setup_stage_on_node(stage: ProcessingStage, node_info: NodeInfo, worker_metadata: WorkerMetadata) -> None:
    """Ray remote function to execute setup_on_node for a stage.

    This runs as a Ray remote task (not an actor).
    vLLM's auto-detection only forces the spawn multiprocessing method inside Ray actors,
    not in Ray tasks. Without this override, vLLM defaults to fork in tasks and hits
    RuntimeError: Cannot re-initialize CUDA in forked subprocess.
    We explicitly set the environment variable to spawn to prevent this.
    """
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    stage.setup_on_node(node_info, worker_metadata)


def execute_setup_on_node(stages: list[ProcessingStage], ignore_head_node: bool = False) -> None:
    """Execute setup on node for a stage."""
    head_node_id = get_head_node_id()
    ray_tasks = []
    for node in ray.nodes():
        node_id = node["NodeID"]
        node_info = NodeInfo(node_id=node_id)
        worker_metadata = WorkerMetadata(worker_id="", allocation=None)
        if ignore_head_node and node_id == head_node_id:
            logger.info(f"Ignoring setup on head node {node_id}")
            continue

        logger.info(f"Executing setup on node {node_id} for {len(stages)} stages")

        for stage in stages:
            ray_tasks.append(
                _setup_stage_on_node.options(
                    num_cpus=stage.resources.cpus if stage.resources is not None else 1,
                    num_gpus=stage.resources.gpus if stage.resources is not None else 0,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
                ).remote(stage, node_info, worker_metadata)
            )
    ray.get(ray_tasks)
