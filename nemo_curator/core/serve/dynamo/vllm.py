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

"""Dynamo vLLM worker launch helpers for aggregated and disaggregated serving."""

from __future__ import annotations

import json
from functools import reduce
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.core.serve.base import BaseModelConfig
from nemo_curator.core.serve.dynamo.infra import (
    build_worker_actor_name,
    dynamo_endpoint,
    engine_kwargs_to_cli_flags,
    model_name_to_component,
)
from nemo_curator.core.serve.placement import (
    build_replica_pg,
    get_bundle_node_ip,
    get_free_port_in_bundle,
    plan_replica_bundle_shape,
)
from nemo_curator.core.serve.subprocess_mgr import ManagedSubprocess

if TYPE_CHECKING:
    from typing import Literal

    from ray.util.placement_group import PlacementGroup

    from nemo_curator.core.serve.dynamo.config import DynamoVLLMModelConfig
    from nemo_curator.core.serve.placement import ReplicaBundleSpec


# Installs ai-dynamo[vllm] to pin the exact vLLM release matching the
# installed ai-dynamo — required because ai-dynamo's CLI surface tracks
# a specific vLLM version.
DYNAMO_VLLM_RUNTIME_ENV: dict[str, Any] = {"uv": ["ai-dynamo[vllm]"]}

# Default KV-cache transfer configuration for disagg — NixlConnector is the
# production path; ``kv_both`` makes each worker both send and receive KV
# blocks so roles can be swapped without reconfiguring.
DEFAULT_KV_TRANSFER_CONFIG: dict[str, Any] = {"kv_connector": "NixlConnector", "kv_role": "kv_both"}

# Port seed offsets for disagg worker per-bundle port allocation. Worker index
# is added so concurrent workers on the same node don't collide.
_DISAGG_KV_EVENTS_PORT_SEED = 20081
_DISAGG_NIXL_PORT_SEED = 20097


def dynamo_runtime_env(model_config: DynamoVLLMModelConfig) -> dict[str, Any]:
    """Merge the user's ``runtime_env`` with the Dynamo-vLLM package pin."""
    return BaseModelConfig.merge_runtime_envs(DYNAMO_VLLM_RUNTIME_ENV, model_config.runtime_env or None)


def merge_model_runtime_envs(models: list[DynamoVLLMModelConfig]) -> dict[str, Any]:
    """Merge every model's ``runtime_env`` onto the Dynamo-vLLM pin for the shared frontend actor."""
    envs = [m.runtime_env for m in models if m.runtime_env]
    user_merged = reduce(BaseModelConfig.merge_runtime_envs, envs) if envs else None
    return BaseModelConfig.merge_runtime_envs(DYNAMO_VLLM_RUNTIME_ENV, user_merged)


def resolve_disagg_role_config(
    model_config: DynamoVLLMModelConfig, role: Literal["prefill", "decode"]
) -> tuple[int, dict[str, Any]]:
    """Resolve ``(num_replicas, engine_kwargs)`` for one disagg role.

    Role-level ``engine_kwargs`` merges over the model-wide
    ``engine_kwargs`` so users can override only what they need per role
    (for example a smaller TP on decode).
    """
    if model_config.mode != "disagg":
        msg = f"resolve_disagg_role_config requires mode='disagg'; got mode={model_config.mode!r}"
        raise ValueError(msg)
    role_cfg = model_config.prefill if role == "prefill" else model_config.decode
    if role_cfg is None:
        msg = f"mode='disagg' requires both prefill and decode; {role!r} is not configured"
        raise ValueError(msg)
    merged_engine_kwargs = {**model_config.engine_kwargs, **role_cfg.engine_kwargs}
    return role_cfg.num_replicas, merged_engine_kwargs


def plan_disagg_shape(
    tp_size: int,
    *,
    role: Literal["prefill", "decode"],
    worker_index: int,
    model_name: str,
    topology: list[dict[str, Any]] | None = None,
) -> ReplicaBundleSpec:
    """Plan a single-bundle PG spec for one disagg worker.

    Disagg does not support multi-node TP — each role's TP group must fit
    on one node. Raise early if ``plan_replica_bundle_shape`` hands back
    a multi-bundle (multi-node) spec.
    """
    spec = plan_replica_bundle_shape(tp_size, _topology=topology)
    if spec.is_multi_node:
        msg = (
            f"Disaggregated serving does not support multi-node TP. "
            f"Model '{model_name}' {role} worker {worker_index} requires TP={tp_size} "
            f"which cannot fit on a single node. Reduce tensor_parallel_size for this role."
        )
        raise ValueError(msg)
    return spec


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


def launch_disagg_replicas(  # noqa: PLR0913
    model_config: DynamoVLLMModelConfig,
    *,
    base_env: dict[str, str],
    namespace: str,
    request_plane: str,
    event_plane: str,
    runtime_dir: str,
    actor_name_prefix: str,
    topology: list[dict[str, Any]] | None = None,
) -> tuple[list[PlacementGroup], list[ManagedSubprocess], list[dict[str, Any]]]:
    """Plan PGs and launch every worker actor for one disagg model.

    Each role (prefill/decode) becomes its own pool of single-bundle PGs
    so roles can scale independently. Only the prefill pool publishes KV
    events (decode reads them via Nixl). KV transfer defaults to
    NixlConnector with ``kv_both`` unless the user overrides via
    ``DynamoVLLMModelConfig.kv_transfer_config``.
    """
    replica_pgs: list[PlacementGroup] = []
    worker_actors: list[ManagedSubprocess] = []
    entries: list[dict[str, Any]] = []

    kv_transfer_config = json.dumps(model_config.kv_transfer_config or DEFAULT_KV_TRANSFER_CONFIG)
    model_name = model_config.resolved_model_name
    component = model_name_to_component(model_name)

    num_prefill, prefill_ek = resolve_disagg_role_config(model_config, "prefill")
    num_decode, decode_ek = resolve_disagg_role_config(model_config, "decode")
    prefill_tp = prefill_ek.get("tensor_parallel_size", 1)
    decode_tp = decode_ek.get("tensor_parallel_size", 1)

    worker_index = 0
    # Decode first (Dynamo example convention). Only prefill publishes KV events.
    for role, num_workers, role_ek, publishes_kv_events in (
        ("decode", num_decode, decode_ek, False),
        ("prefill", num_prefill, prefill_ek, True),
    ):
        role_pgs, role_actors, role_entries, worker_index = _launch_disagg_role(
            model_config,
            base_env=base_env,
            role=role,
            num_workers=num_workers,
            engine_kwargs=role_ek,
            publishes_kv_events=publishes_kv_events,
            namespace=namespace,
            request_plane=request_plane,
            event_plane=event_plane,
            component=component,
            kv_transfer_config=kv_transfer_config,
            worker_index_start=worker_index,
            runtime_dir=runtime_dir,
            actor_name_prefix=actor_name_prefix,
            topology=topology,
        )
        replica_pgs.extend(role_pgs)
        worker_actors.extend(role_actors)
        entries.extend(role_entries)

    total_gpus = num_decode * decode_tp + num_prefill * prefill_tp
    tp_desc = f"TP={decode_tp}" if decode_tp == prefill_tp else f"prefill_TP={prefill_tp}, decode_TP={decode_tp}"
    logger.info(
        f"Disaggregated '{model_name}': {num_decode} decode + {num_prefill} prefill "
        f"workers launched ({total_gpus} GPUs total, {tp_desc})"
    )

    return replica_pgs, worker_actors, entries


def _launch_disagg_role(  # noqa: PLR0913
    model_config: DynamoVLLMModelConfig,
    *,
    base_env: dict[str, str],
    role: Literal["prefill", "decode"],
    num_workers: int,
    engine_kwargs: dict[str, Any],
    publishes_kv_events: bool,
    namespace: str,
    request_plane: str,
    event_plane: str,
    component: str,
    kv_transfer_config: str,
    worker_index_start: int,
    runtime_dir: str,
    actor_name_prefix: str,
    topology: list[dict[str, Any]] | None,
) -> tuple[list[PlacementGroup], list[ManagedSubprocess], list[dict[str, Any]], int]:
    """Launch the N workers for a single disagg role (prefill or decode)."""
    tp_size = engine_kwargs.get("tensor_parallel_size", 1)
    model_name = model_config.resolved_model_name

    replica_pgs: list[PlacementGroup] = []
    worker_actors: list[ManagedSubprocess] = []
    entries: list[dict[str, Any]] = []
    worker_index = worker_index_start

    for i in range(num_workers):
        spec = plan_disagg_shape(tp_size, role=role, worker_index=i, model_name=model_name, topology=topology)
        pg_name = f"{actor_name_prefix}_pg_{component}_{role}_{i}"
        pg = build_replica_pg(spec, name=pg_name)
        replica_pgs.append(pg)

        # Global-enough seed so concurrent workers on one node don't collide.
        nixl_port = get_free_port_in_bundle(pg, 0, _DISAGG_NIXL_PORT_SEED + worker_index)

        role_args: list[str] = []
        if publishes_kv_events:
            kv_events_config = build_worker_kv_events_config(
                model_config, pg=pg, bundle_index=0, port_seed=_DISAGG_KV_EVENTS_PORT_SEED + i, enabled=True
            )
            role_args += ["--kv-events-config", kv_events_config]

        python_args: list[str] = [
            "-m",
            "dynamo.vllm",
            "--model",
            model_config.model_identifier,
            "--served-model-name",
            model_name,
            "--endpoint",
            dynamo_endpoint(namespace, component, role=role),
            "--discovery-backend",
            "etcd",
            "--request-plane",
            request_plane,
            "--event-plane",
            event_plane,
            "--disaggregation-mode",
            role,
            "--kv-transfer-config",
            kv_transfer_config,
            *role_args,
        ]
        python_args += engine_kwargs_to_cli_flags(engine_kwargs)
        python_args += engine_kwargs_to_cli_flags(model_config.dynamo_kwargs)

        label = build_worker_actor_name(model_name, i, 0, tp_size, role=role)
        logger.info(f"Disagg {role} worker {i}: {spec.per_node_gpus} GPU(s), nixl_port={nixl_port}")
        proc = ManagedSubprocess.spawn(
            label,
            pg,
            0,
            num_gpus=spec.per_node_gpus,
            python_args=python_args,
            runtime_dir=runtime_dir,
            actor_name_prefix=actor_name_prefix,
            subprocess_env={**base_env, "VLLM_NIXL_SIDE_CHANNEL_PORT": str(nixl_port), "PYTHONHASHSEED": "0"},
            runtime_env=dynamo_runtime_env(model_config),
        )
        worker_actors.append(proc)
        entries.append(
            {
                "model": model_name,
                "replica": i,
                "mode": "disagg",
                "role": role,
                "gpus_per_node": spec.per_node_gpus,
                "nnodes": 1,
                "multi_node": False,
                "master_addr": None,
            }
        )
        worker_index += 1

    return replica_pgs, worker_actors, entries, worker_index
