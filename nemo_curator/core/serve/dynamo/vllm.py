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

"""Dynamo vLLM worker launch helpers for aggregated serving."""

from __future__ import annotations

import json
import re
from functools import reduce
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.core.serve.base import BaseModelConfig
from nemo_curator.core.serve.dynamo.infra import build_worker_actor_name, engine_kwargs_to_cli_flags
from nemo_curator.core.serve.placement import (
    build_replica_pg,
    get_bundle_node_ip,
    get_free_port_in_bundle,
    plan_replica_bundle_shape,
)
from nemo_curator.core.serve.subprocess_mgr import ManagedSubprocess

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

    from nemo_curator.core.serve.dynamo.config import DynamoVLLMModelConfig
    from nemo_curator.core.serve.placement import ReplicaBundleSpec


# Installs ai-dynamo[vllm] to pin the exact vLLM release matching the
# installed ai-dynamo — required because ai-dynamo's CLI surface tracks
# a specific vLLM version.
DYNAMO_VLLM_RUNTIME_ENV: dict[str, Any] = {"uv": ["ai-dynamo[vllm]"]}


def model_name_to_component(name: str) -> str:
    """Sanitize *name* into a valid Dynamo component slug.

    Dynamo endpoints use ``dyn://namespace.component.endpoint`` where
    dots are delimiters, so any dotted identifier in the model name has
    to be flattened.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not slug:
        msg = f"Model name {name!r} produces an empty component slug after sanitization."
        raise ValueError(msg)
    return slug


def dynamo_endpoint(namespace: str, component: str, role: str | None = None) -> str:
    """Build the ``dyn://namespace.component.endpoint`` URI a Dynamo worker registers under."""
    suffix = f"_{role}" if role else ""
    return f"dyn://{namespace}.{component}{suffix}.generate"


def dynamo_runtime_env(model_config: DynamoVLLMModelConfig) -> dict[str, Any]:
    """Merge the user's ``runtime_env`` with the Dynamo-vLLM package pin."""
    return BaseModelConfig.merge_runtime_envs(DYNAMO_VLLM_RUNTIME_ENV, model_config.runtime_env or None)


def merge_model_runtime_envs(models: list[DynamoVLLMModelConfig]) -> dict[str, Any]:
    """Merge every model's ``runtime_env`` onto the Dynamo-vLLM pin for the shared frontend actor."""
    envs = [m.runtime_env for m in models if m.runtime_env]
    user_merged = reduce(BaseModelConfig.merge_runtime_envs, envs) if envs else None
    return BaseModelConfig.merge_runtime_envs(DYNAMO_VLLM_RUNTIME_ENV, user_merged)


def aggregated_model_uses_exact_kv_events(
    model_config: DynamoVLLMModelConfig, *, router_mode: str | None, router_kv_events: bool
) -> bool:
    """True if this aggregated model should publish ZMQ KV events."""
    if model_config.mode == "disagg":
        return False
    if router_mode != "kv":
        return False
    return router_kv_events


def build_worker_kv_events_config(
    model_config: DynamoVLLMModelConfig,
    *,
    pg: PlacementGroup,
    bundle_index: int,
    port_seed: int,
    enabled: bool,
) -> str:
    """JSON blob for ``--kv-events-config``.

    Always passed explicitly. Without this, Dynamo's ``args.py`` auto-creates
    a ``KVEventsConfig`` bound to ``tcp://*:20080`` when ``prefix_caching`` is
    enabled (vLLM >=0.16 default), causing every worker on the same node to
    fight over the same port.
    """
    template = dict(model_config.kv_events_config)

    if not enabled:
        template["enable_kv_cache_events"] = False
        template.pop("endpoint", None)
        return json.dumps(template)

    kv_events_port = get_free_port_in_bundle(pg, bundle_index, port_seed)
    template.update(
        {
            "publisher": "zmq",
            "topic": "kv-events",
            "endpoint": f"tcp://*:{kv_events_port}",
            "enable_kv_cache_events": True,
        }
    )
    return json.dumps(template)


def launch_replicas(  # noqa: PLR0913
    model_config: DynamoVLLMModelConfig,
    *,
    base_env: dict[str, str],
    namespace: str,
    request_plane: str,
    event_plane: str,
    runtime_dir: str,
    actor_name_prefix: str,
    router_mode: str | None,
    router_kv_events: bool,
    topology: list[dict[str, Any]] | None = None,
) -> tuple[list[PlacementGroup], list[ManagedSubprocess], list[dict[str, Any]]]:
    """Plan PGs and launch every worker actor for one non-disagg model.

    Returns ``(replica_pgs, worker_actors, manifest_entries)``; callers own
    the returned handles and are responsible for teardown.
    """
    tp_size = model_config.engine_kwargs.get("tensor_parallel_size", 1)
    model_name = model_config.resolved_model_name
    component = model_name_to_component(model_name)
    spec = plan_replica_bundle_shape(tp_size, _topology=topology)

    replica_pgs: list[PlacementGroup] = []
    worker_actors: list[ManagedSubprocess] = []
    entries: list[dict[str, Any]] = []

    for replica_index in range(model_config.num_replicas):
        pg_name = f"{actor_name_prefix}_pg_{component}_DP{replica_index}"
        pg = build_replica_pg(spec, name=pg_name)
        replica_pgs.append(pg)

        master_addr = get_bundle_node_ip(pg, 0) if spec.is_multi_node else None
        if spec.is_multi_node:
            logger.info(
                f"Replica {replica_index}: multi-node TP across {spec.nnodes} nodes "
                f"(total {spec.total_gpus} GPUs, master={master_addr})"
            )
        else:
            logger.info(f"Replica {replica_index}: single-node, {spec.total_gpus} GPU(s)")

        for node_rank in range(spec.nnodes):
            worker_actors.append(
                _launch_vllm_worker(
                    model_config=model_config,
                    base_env=base_env,
                    pg=pg,
                    spec=spec,
                    replica_index=replica_index,
                    node_rank=node_rank,
                    master_addr=master_addr,
                    namespace=namespace,
                    request_plane=request_plane,
                    event_plane=event_plane,
                    runtime_dir=runtime_dir,
                    actor_name_prefix=actor_name_prefix,
                    router_mode=router_mode,
                    router_kv_events=router_kv_events,
                )
            )

        entries.append(
            {
                "model": model_name,
                "replica": replica_index,
                "nnodes": spec.nnodes,
                "gpus_per_node": spec.per_node_gpus,
                "multi_node": spec.is_multi_node,
                "master_addr": master_addr,
            }
        )

    return replica_pgs, worker_actors, entries


def _launch_vllm_worker(  # noqa: PLR0913
    *,
    model_config: DynamoVLLMModelConfig,
    base_env: dict[str, str],
    pg: PlacementGroup,
    spec: ReplicaBundleSpec,
    replica_index: int,
    node_rank: int,
    master_addr: str | None,
    namespace: str,
    request_plane: str,
    event_plane: str,
    runtime_dir: str,
    actor_name_prefix: str,
    router_mode: str | None,
    router_kv_events: bool,
) -> ManagedSubprocess:
    """Spawn one ``python -m dynamo.vllm`` actor, pinned to bundle *node_rank*.

    Rank 0 is the "real" worker (model registration + scheduler + KV events
    publisher). Rank >0 is ``--headless`` — no scheduler, so KV events are
    always disabled for it even if rank 0 publishes.
    """
    model_name = model_config.resolved_model_name
    component = model_name_to_component(model_name)
    tp_size = model_config.engine_kwargs.get("tensor_parallel_size", 1)
    is_rank_zero = node_rank == 0

    kv_events_enabled = is_rank_zero and aggregated_model_uses_exact_kv_events(
        model_config, router_mode=router_mode, router_kv_events=router_kv_events
    )
    kv_events_config = build_worker_kv_events_config(
        model_config,
        pg=pg,
        bundle_index=node_rank,
        port_seed=20080 + replica_index + node_rank,
        enabled=kv_events_enabled,
    )

    python_args: list[str] = [
        "-m",
        "dynamo.vllm",
        "--model",
        model_config.model_identifier,
    ]
    if is_rank_zero:
        python_args += [
            "--served-model-name",
            model_name,
            "--endpoint",
            dynamo_endpoint(namespace, component),
            "--discovery-backend",
            "etcd",
            "--request-plane",
            request_plane,
            "--event-plane",
            event_plane,
        ]
    else:
        python_args.append("--headless")

    python_args += ["--kv-events-config", kv_events_config]

    if spec.is_multi_node:
        assert master_addr is not None, "master_addr must be set for multi-node replicas"  # noqa: S101
        python_args += [
            "--nnodes",
            str(spec.nnodes),
            "--node-rank",
            str(node_rank),
            "--master-addr",
            master_addr,
        ]

    python_args += engine_kwargs_to_cli_flags(model_config.engine_kwargs)
    python_args += engine_kwargs_to_cli_flags(model_config.dynamo_kwargs)

    label = build_worker_actor_name(model_name, replica_index, node_rank, tp_size)
    return ManagedSubprocess.spawn(
        label,
        pg,
        node_rank,
        num_gpus=spec.per_node_gpus,
        python_args=python_args,
        runtime_dir=runtime_dir,
        actor_name_prefix=actor_name_prefix,
        subprocess_env=base_env,
        runtime_env=dynamo_runtime_env(model_config),
    )
