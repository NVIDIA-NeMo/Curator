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
from typing import TYPE_CHECKING

import ray
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.backends.perf_identity import build_ray_perf_identity, stamp_worker_metadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.utils.ray_utils import get_head_node_id, submit_on_each_node

if TYPE_CHECKING:
    import loguru


def _logger_custom_serializer(
    _: "loguru.Logger",
) -> None:
    return None


def _logger_custom_deserializer(
    _: None,
) -> "loguru.Logger":
    return logger


def register_loguru_serializer() -> None:
    """Register a no-op (de)serializer for loguru (not serializable in general)."""
    ray.util.register_serializer(
        logger.__class__,
        serializer=_logger_custom_serializer,
        deserializer=_logger_custom_deserializer,
    )


def merge_executor_configs(base_config: dict | None, override_config: dict | None) -> dict:
    """Recursively deep-merge two executor configs (override wins, inputs untouched).

    Args:
        base_config: Base configuration dictionary
        override_config: Configuration merged on top of base_config

    Returns:
        Merged config with nested dicts merged recursively
    """
    if base_config is None and override_config is None:
        return {}
    if base_config is None:
        return deepcopy(override_config)
    if override_config is None:
        return deepcopy(base_config)

    merged_config = deepcopy(base_config)

    for key, value in override_config.items():
        if isinstance(value, dict):
            if key not in merged_config or not isinstance(merged_config[key], dict):
                merged_config[key] = deepcopy(value)
            else:
                merged_config[key] = merge_executor_configs(merged_config[key], value)
        else:
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


class RayStageSpecKeys(str, Enum):
    """String enum of different flags that define keys inside ray_stage_spec."""

    IS_ACTOR_STAGE = "is_actor_stage"
    IS_FANOUT_STAGE = "is_fanout_stage"
    IS_RAFT_ACTOR = "is_raft_actor"
    IS_LSH_STAGE = "is_lsh_stage"
    IS_SHUFFLE_STAGE = "is_shuffle_stage"
    MAX_CALLS_PER_WORKER = "max_calls_per_worker"
    MIN_WORKERS = "min_workers"
    MAX_WORKERS = "max_workers"
    INITIAL_WORKERS = "initial_workers"
    RAY_REMOTE_ARGS = "ray_remote_args"
    RAY_NUM_CPUS = "ray_num_cpus"


def get_worker_metadata_and_node_id() -> tuple[NodeInfo, WorkerMetadata]:
    """Get the worker metadata and node id from the runtime context."""
    ray_context = ray.get_runtime_context()
    return NodeInfo(node_id=ray_context.get_node_id()), WorkerMetadata(worker_id=ray_context.get_worker_id())


def get_worker_metadata_and_node_id_with_perf(
    stage_name: str,
    *,
    requires_gpu: bool = False,
) -> tuple[NodeInfo, WorkerMetadata]:
    """Get worker metadata with opt-in Ray-resolved performance identity."""
    node_info, worker_metadata = get_worker_metadata_and_node_id()
    identity = build_ray_perf_identity(stage_name, requires_gpu=requires_gpu)
    stamp_worker_metadata(worker_metadata, identity)
    return node_info, worker_metadata


def get_available_cpu_gpu_resources(
    init_and_shutdown: bool = False, ignore_head_node: bool = False
) -> tuple[int, int]:
    """Get available CPU and GPU resources from Ray."""
    if init_and_shutdown:
        ray.init(ignore_reinit_error=True)
    time.sleep(0.2)  # ray.available_resources() can lag
    # Curator assumes the whole cluster is free (one pipeline at a time), so
    # available resources should match total resources.
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
    """Raise if the cluster lacks enough GPUs for aggregate demand.

    Coarse pre-check: Ray's placement-group scheduler can hang on ``pg.ready()``
    when demand exceeds capacity, so fail fast with the actual numbers.
    """
    _, available_gpus = get_available_cpu_gpu_resources(ignore_head_node=ignore_head_node)
    available = int(available_gpus)
    if gpus_needed > available:
        msg = f"Need {gpus_needed} GPUs but cluster has {available} available."
        raise RuntimeError(msg)


@ray.remote
def _setup_stage_on_node(stage: ProcessingStage) -> None:
    """Run ``setup_on_node`` for a stage as a Ray task.

    Force vLLM's spawn method: it auto-sets spawn only inside Ray actors, not
    tasks, so without this fork would hit "Cannot re-initialize CUDA in forked
    subprocess".
    """
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    node_id = ray.get_runtime_context().get_node_id()
    stage.setup_on_node(NodeInfo(node_id=node_id), WorkerMetadata(worker_id="", allocation=None))


def execute_setup_on_node(stages: list[ProcessingStage], ignore_head_node: bool = False) -> None:
    """Execute ``setup_on_node`` for every stage on every alive Ray node.

    All ``(stage, node)`` tasks are submitted up front and awaited with one
    ``ray.get``, so wall-clock time is bounded by the slowest stage (matters when
    setup is heavy: model downloads, weight loads).
    """
    head_node_id = get_head_node_id() if ignore_head_node else None
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        node_id = node["NodeID"]
        if ignore_head_node and node_id == head_node_id:
            continue
        logger.info(f"Executing setup on node {node_id} for {len(stages)} stages")

    refs: list = []
    for stage in stages:
        setup_resources = stage.setup_on_node_resources()
        refs.extend(
            submit_on_each_node(
                _setup_stage_on_node,
                stage,
                ignore_head_node=ignore_head_node,
                num_cpus=setup_resources.cpus,
                num_gpus=setup_resources.gpus,
            )
        )
    ray.get(refs)
