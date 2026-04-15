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

from __future__ import annotations

import contextlib
import http
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.core.serve.config import BaseModelConfig, DynamoRoleConfig, DynamoServerConfig, DynamoVLLMModelConfig
from nemo_curator.core.serve.internal.base import InferenceBackend
from nemo_curator.core.serve.internal.constants import (
    DEFAULT_ETCD_PORT,
    DEFAULT_NATS_PORT,
)
from nemo_curator.core.serve.internal.errors import SubprocessError
from nemo_curator.core.serve.internal.subprocess_mgr import (
    ManagedSubprocess,
    NodeAllocation,
    ReplicaPlan,
    _check_binary,
    _engine_kwargs_to_cli_flags,
    _get_gpu_inventory,
    _ignore_head_node,
    _kill_actor,
    _resolve_node_ip,
    _wait_for_port,
    build_worker_actor_name,
    get_free_port_on_node,
    kill_orphaned_actors,
    plan_replica_placement,
    spawn_actor,
)

if TYPE_CHECKING:
    from nemo_curator.core.serve.server import InferenceServer


def _model_name_to_component(name: str) -> str:
    """Sanitize a model name into a valid Dynamo component name.

    Dynamo endpoints use ``dyn://namespace.component.endpoint`` format
    where dots are delimiters.  This replaces all non-alphanumeric
    characters with underscores to produce a safe component name.

    Differs from ``nemo_curator.stages.text.models.utils.format_name_with_suffix``
    which only takes the last path component and does not replace dots —
    both are required here to avoid collisions across HuggingFace orgs
    and to keep dots out of the ``dyn://`` URI.

    Examples:
        >>> _model_name_to_component("Qwen/Qwen3-0.6B")
        'qwen_qwen3_0_6b'
        >>> _model_name_to_component("google/gemma-3-4b-it")
        'google_gemma_3_4b_it'
        >>> _model_name_to_component("my-custom-model")
        'my_custom_model'
    """
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not slug:
        msg = f"Model name {name!r} produces an empty component slug after sanitization."
        raise ValueError(msg)
    return slug


def _dynamo_endpoint(namespace: str, component: str, role: str | None = None) -> str:
    """Build a ``dyn://namespace.component.endpoint`` URI for worker registration."""
    suffix = f"_{role}" if role else ""
    return f"dyn://{namespace}.{component}{suffix}.generate"


# Default runtime_env for Dynamo workers — installs ai-dynamo[vllm] which
# brings the exact vLLM version matching the installed ai-dynamo release.
# Uses uv (faster than pip). Ray caches this per content-hash, so the
# install cost is paid once per node.
_DYNAMO_VLLM_RUNTIME_ENV: dict[str, Any] = {
    "uv": ["ai-dynamo[vllm]"],
}


_FRONTEND_ROUTER_KEYS = (
    "router_mode",
    "router_kv_events",
    "router_kv_overlap_score_weight",
    "router_temperature",
    "router_queue_threshold",
    "router_ttl_secs",
    "router_max_tree_size",
    "router_prune_target_ratio",
    "router_reset_states",
)


class DynamoBackend(InferenceBackend):
    """NVIDIA Dynamo inference backend.

    Launches Dynamo infrastructure (etcd + NATS), vLLM workers, and an
    OpenAI-compatible frontend as subprocesses managed by detached Ray actors.

    Supports both single-node and multi-node tensor parallelism:

    - **Single-node TP**: Each replica runs on one node.  One Ray actor per
      replica with ``num_gpus=tp_size``.
    - **Multi-node TP** (TP > GPUs on any single node): Each replica spans
      multiple nodes.  Rank 0 runs the full Dynamo worker (model registration
      in etcd); rank 1+ run headless vLLM workers coordinated via
      ``torch.distributed``.  Each rank is pinned to its planned node via
      ``NodeAffinitySchedulingStrategy``.

    Supports KV-cache-aware routing via the typed server router config.
    When ``router_mode="kv"``, workers publish KV cache events over ZMQ
    (exact mode) or the frontend predicts cache state from routing decisions
    (approximate mode, ``router_kv_events=False``).  Router settings are
    resolved once across all models and validated for consistency.

    This backend reserves GPUs via ``num_gpus`` on Ray actors, so
    Ray's scheduler is aware of Dynamo's GPU usage.  Pipelines with GPU
    stages can coexist when using executors that respect Ray's resource
    accounting (e.g. ``RayDataExecutor``).
    """

    def __init__(self, server: InferenceServer) -> None:
        self._server = server
        self._runtime_dir: str | None = None
        self._head_ip: str | None = None
        self._infra_node_id: str | None = None
        self._infra_ip: str | None = None
        self._etcd_actor: ManagedSubprocess | None = None
        self._nats_actor: ManagedSubprocess | None = None
        self._worker_actors: list[ManagedSubprocess] = []
        self._frontend_actor: ManagedSubprocess | None = None
        self._etcd_port: int | None = None
        self._nats_port: int | None = None
        self._actor_name_prefix: str = ""

    @staticmethod
    def _subtract_placed_gpus(inventory: list[dict[str, Any]], plans: list[ReplicaPlan]) -> list[dict[str, Any]]:
        """Remove GPUs consumed by *plans* from *inventory*, returning what remains."""
        used: dict[str, int] = {}
        for plan in plans:
            for rank in plan.ranks:
                used[rank.node_id] = used.get(rank.node_id, 0) + rank.num_gpus
        return [
            {**n, "num_gpus": n["num_gpus"] - used.get(n["node_id"], 0)}
            for n in inventory
            if n["num_gpus"] - used.get(n["node_id"], 0) > 0
        ]

    @staticmethod
    def _plan_to_placement(
        model_name: str, plan: ReplicaPlan, *, mode: str | None = None, role: str | None = None
    ) -> dict[str, Any]:
        """Convert a ReplicaPlan to a manifest-friendly dict."""
        entry: dict[str, Any] = {
            "model": model_name,
            "replica": plan.replica_index,
            "ranks": [{"node": r.node_ip, "gpus": r.num_gpus, "node_rank": r.node_rank} for r in plan.ranks],
        }
        if mode:
            entry["mode"] = mode
        if role:
            entry["role"] = role
        return entry

    def _write_manifest(self, data: dict[str, Any], *, ready: bool) -> None:
        """Write deployment manifest to ``{runtime_dir}/manifest.json`` and log it."""
        manifest = {**data, "ready": ready, "timestamp": time.time()}

        # Log manifest to driver logs so it's accessible even in multi-node
        # clusters where the runtime dir may not be on a shared filesystem.
        logger.info(f"Deployment manifest (ready={ready}): {json.dumps(manifest, indent=2)}")

        if not self._runtime_dir:
            return
        manifest_path = os.path.join(self._runtime_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    @staticmethod
    def _resolve_frontend_router_config(server: InferenceServer) -> dict[str, Any]:
        """Resolve effective frontend router settings from the typed server config."""
        backend = server.backend
        if not isinstance(backend, DynamoServerConfig):
            msg = "DynamoBackend requires DynamoServerConfig."
            raise TypeError(msg)

        any_disagg = any(model.mode == "disagg" for model in server.models if isinstance(model, DynamoVLLMModelConfig))
        router_mode = backend.router.mode if backend.router.mode is not None else ("kv" if any_disagg else None)

        return {
            "router_mode": router_mode,
            "router_kv_events": backend.router.kv_events,
            "router_kv_overlap_score_weight": backend.router.kv_overlap_score_weight,
            "router_temperature": backend.router.temperature,
            "router_queue_threshold": backend.router.queue_threshold,
            "router_ttl_secs": backend.router.ttl_secs,
            "router_max_tree_size": backend.router.max_tree_size,
            "router_prune_target_ratio": backend.router.prune_target_ratio,
            "router_reset_states": backend.router.reset_states,
        }

    @staticmethod
    def _validate_frontend_config(_server: InferenceServer) -> None:
        """Typed frontend config lives once on the server config."""
        return

    @staticmethod
    def _validate_unique_model_names(server: InferenceServer) -> None:
        """Reject duplicate model names and component-slug collisions.

        Each model must have a unique ``model_name`` (or ``model_identifier``
        when ``model_name`` is not set).  Dynamo routes requests by model name,
        so duplicates would be unroutable.  When deploying the same
        ``model_identifier`` multiple times (e.g. with different TP), each
        config must set a distinct ``model_name``.

        Also checks that sanitized component names are unique — two
        different model names that collapse to the same slug (e.g. ``a.b``
        and ``a-b`` both become ``a_b``) would cause silent routing conflicts.
        """
        seen_names: dict[str, int] = {}
        seen_components: dict[str, tuple[int, str]] = {}
        for i, m in enumerate(server.models):
            name = m.model_name or m.model_identifier
            if name in seen_names:
                msg = (
                    f"Duplicate model name {name!r} at index {i} "
                    f"(first seen at index {seen_names[name]}). "
                    f"When deploying the same model_identifier multiple times, "
                    f"each must have a distinct model_name."
                )
                raise ValueError(msg)
            seen_names[name] = i

            comp = _model_name_to_component(name)
            if comp in seen_components:
                prev_idx, prev_name = seen_components[comp]
                msg = (
                    f"Model names {prev_name!r} (index {prev_idx}) and "
                    f"{name!r} (index {i}) both sanitize to component "
                    f"{comp!r}. Use more distinct model_name values."
                )
                raise ValueError(msg)
            seen_components[comp] = (i, name)

    @staticmethod
    def _validate_gpu_requirements(server: InferenceServer, inventory: list[dict[str, Any]]) -> None:
        """Coarse fail-fast: reject configs that obviously exceed cluster capacity.

        This is an early rejection check, not proof of valid placement.
        The authoritative placement is ``plan_replica_placement()`` against
        the shared inventory.

        Check A: disagg models require TP to fit on a single node.
        Check B: total GPU demand across all models must not exceed supply.
        """
        max_gpus_per_node = max((n["num_gpus"] for n in inventory), default=0)
        total_available = sum(n["num_gpus"] for n in inventory)
        total_needed = 0

        for model_config in server.models:
            is_disagg = model_config.mode == "disagg"
            model_name = model_config.resolved_model_name

            if is_disagg:
                (num_prefill, prefill_ek), (num_decode, decode_ek) = DynamoBackend._resolve_disagg_role_config(
                    model_config
                )
                prefill_tp = prefill_ek.get("tensor_parallel_size", 1)
                decode_tp = decode_ek.get("tensor_parallel_size", 1)

                # Check A: disagg TP must fit on a single node (check both roles)
                for role, tp in [("prefill", prefill_tp), ("decode", decode_tp)]:
                    if tp > max_gpus_per_node:
                        msg = (
                            f"Model '{model_name}' {role} requests TP={tp} in disaggregated mode, "
                            f"but max GPUs per node is {max_gpus_per_node}. "
                            f"Disaggregated mode does not support multi-node TP."
                        )
                        raise ValueError(msg)

                total_needed += num_prefill * prefill_tp + num_decode * decode_tp
            else:
                tp_size = model_config.engine_kwargs.get("tensor_parallel_size", 1)
                num_replicas = DynamoBackend._resolve_num_replicas(model_config)
                total_needed += num_replicas * tp_size

        # Check B: aggregate overcommit
        if total_needed > total_available:
            msg = (
                f"Models require {total_needed} GPUs total but only "
                f"{total_available} available across {len(inventory)} node(s)."
            )
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    def start(self) -> None:
        import ray

        server = self._server
        if not server.models:
            msg = "At least one model config is required."
            raise ValueError(msg)

        backend = server.backend
        if not isinstance(backend, DynamoServerConfig):
            msg = "DynamoBackend requires DynamoServerConfig."
            raise TypeError(msg)

        if not backend.etcd_endpoint:
            _check_binary("etcd")
        if not backend.nats_url:
            _check_binary("nats-server")

        short_id = uuid.uuid4().hex[:8]
        self._actor_name_prefix = f"dynamo_{server.name}_{short_id}"
        self._runtime_dir = tempfile.mkdtemp(prefix=f"nemo_curator_dynamo_{short_id}_")
        logger.info(f"Dynamo runtime dir: {self._runtime_dir}")

        with ray.init(ignore_reinit_error=True):
            head_node_id = ray.get_runtime_context().get_node_id()
            cluster_nodes = ray.nodes()
            self._head_ip = _resolve_node_ip(head_node_id, nodes=cluster_nodes)

            self._infra_node_id, self._infra_ip = self._resolve_infra_node(cluster_nodes, head_node_id)
            server._host = self._infra_ip

            # Kill any orphaned actors from a previous run with the same server name
            # (e.g. after a Jupyter kernel restart that bypassed atexit cleanup).
            kill_orphaned_actors(ray, f"dynamo_{server.name}_")

            # Resolve ports on the infra node (not the driver) so that
            # port availability is checked where services actually bind.
            infra_node = self._infra_node_id
            self._etcd_port = (
                int(backend.etcd_endpoint.rsplit(":", 1)[-1])
                if backend.etcd_endpoint
                else get_free_port_on_node(infra_node, DEFAULT_ETCD_PORT)
            )
            self._nats_port = (
                int(backend.nats_url.rsplit(":", 1)[-1])
                if backend.nats_url
                else get_free_port_on_node(infra_node, DEFAULT_NATS_PORT)
            )
            server.port = get_free_port_on_node(infra_node, server.port)

            try:
                self._deploy_and_healthcheck(
                    server,
                    head_node_id=head_node_id,
                    cluster_nodes=cluster_nodes,
                )
            except Exception:
                # Still connected to Ray here — kill directly without opening another context.
                for label, actor in self._collect_actors_for_shutdown():
                    _kill_actor(ray, label, actor)
                self._cleanup_runtime_dir()
                raise

    def _deploy_and_healthcheck(
        self,
        server: InferenceServer,
        *,
        head_node_id: str,
        cluster_nodes: list[dict[str, Any]],
    ) -> None:
        """Validate config, launch infra/workers/frontend, and wait for health.

        Sequence: fail-fast validations (GPU requirements, frontend config
        consistency, model name uniqueness) -> etcd/NATS -> workers for each
        model -> manifest (ready=False) -> frontend with resolved router
        config -> health check -> manifest (ready=True).
        """
        infra_node_id = self._infra_node_id

        # Build GPU inventory once; each model's placement shrinks it so
        # subsequent models are assigned to different GPUs.
        inventory = _get_gpu_inventory(head_node_id, nodes=cluster_nodes)

        # Fail-fast validations before starting any infrastructure (etcd/NATS).
        self._validate_gpu_requirements(server, inventory)
        self._validate_frontend_config(server)
        self._validate_unique_model_names(server)

        # Infrastructure -- on infra node (head or first non-head when excluded)
        backend = server.backend
        if not isinstance(backend, DynamoServerConfig):
            msg = "DynamoBackend requires DynamoServerConfig."
            raise TypeError(msg)

        if not backend.etcd_endpoint:
            self._etcd_actor = self._start_etcd(infra_node_id, self._etcd_port)
        if not backend.nats_url:
            self._nats_actor = self._start_nats(infra_node_id, self._nats_port)

        etcd_endpoint = backend.etcd_endpoint or f"http://{self._infra_ip}:{self._etcd_port}"
        nats_url = backend.nats_url or f"nats://{self._infra_ip}:{self._nats_port}"
        base_env = {"ETCD_ENDPOINTS": etcd_endpoint, "NATS_SERVER": nats_url}

        expected_models: set[str] = set()
        placements: list[dict[str, Any]] = []

        for model_config in server.models:
            namespace = backend.namespace
            request_plane = backend.request_plane
            event_plane = backend.event_plane
            is_disagg = model_config.mode == "disagg"

            model_name = model_config.resolved_model_name
            expected_models.add(model_name)

            if not is_disagg:
                tp_size = model_config.engine_kwargs.get("tensor_parallel_size", 1)
                num_replicas = self._resolve_num_replicas(model_config)
                logger.info(f"Deploying model '{model_name}' (TP={tp_size}, replicas={num_replicas})")

            if is_disagg:
                plans_with_roles, inventory = self._launch_disagg_workers(
                    model_config,
                    base_env,
                    inventory=inventory,
                    namespace=namespace,
                    request_plane=request_plane,
                    event_plane=event_plane,
                )
            else:
                plans_with_roles, inventory = self._launch_replicas(
                    model_config,
                    base_env,
                    inventory=inventory,
                    num_replicas=num_replicas,
                    namespace=namespace,
                    request_plane=request_plane,
                    event_plane=event_plane,
                )

            for plan, role in plans_with_roles:
                placements.append(
                    self._plan_to_placement(model_name, plan, mode="disagg" if role else None, role=role)
                )

        # Write initial manifest before frontend launch (ready=False).
        manifest_data = {
            "models": sorted(expected_models),
            "endpoint": server.endpoint,
            "etcd": etcd_endpoint,
            "nats": nats_url,
            "port": server.port,
            "placements": placements,
        }
        self._write_manifest(manifest_data, ready=False)

        # Frontend -- co-located with infra, auto-discovers all registered models
        frontend_router_cfg = self._resolve_frontend_router_config(server)
        frontend_runtime_env = self._merge_model_runtime_envs(server.models)
        self._frontend_actor = self._launch_frontend(
            infra_node_id,
            server.port,
            base_env,
            namespace=backend.namespace,
            request_plane=backend.request_plane,
            event_plane=backend.event_plane,
            frontend_router_cfg=frontend_router_cfg,
            runtime_env=frontend_runtime_env,
        )

        # Health check — must stay inside ray.init context so actor handles
        # (is_alive, read_log_tail) remain valid for subprocess liveness checks.
        self._wait_for_models(server, expected_models)

        # Update manifest after successful health check (ready=True).
        self._write_manifest(manifest_data, ready=True)

    def _wait_for_models(self, server: InferenceServer, expected_models: set[str]) -> None:
        """Poll ``/v1/models`` until all *expected_models* appear in the response.

        Also checks subprocess liveness via Ray actors periodically.  If any
        crash, reads their log tail and raises immediately with the output.
        """
        import urllib.request

        models_url = f"{server.endpoint}/models"
        deadline = time.monotonic() + server.health_check_timeout_s
        start_time = time.monotonic()
        attempt = 0
        last_error: str | None = None
        models_found: set[str] = set()
        while time.monotonic() < deadline:
            attempt += 1

            self._check_subprocess_health()

            try:
                resp = urllib.request.urlopen(models_url, timeout=5)  # noqa: S310
                if resp.status == http.HTTPStatus.OK:
                    body = json.loads(resp.read())
                    models_found = {m["id"] for m in body.get("data", [])}
                    if expected_models.issubset(models_found):
                        logger.info(
                            f"All Dynamo models registered after {attempt} health check(s): {sorted(expected_models)}"
                        )
                        return
                    if server.verbose:
                        missing = sorted(expected_models - models_found)
                        logger.debug(f"Models so far: {sorted(models_found)}, waiting for: {missing}")
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if server.verbose:
                    logger.debug(f"Health check attempt {attempt} failed, retrying...")
            time.sleep(2)

        # Final liveness check before giving up -- surface crash info if available
        self._check_subprocess_health()
        elapsed_s = round(time.monotonic() - start_time, 1)
        msg = (
            f"Models {sorted(expected_models)} did not all appear at {models_url} "
            f"within {server.health_check_timeout_s}s"
        )
        raise SubprocessError(
            msg,
            debug_context={
                "backend": "dynamo",
                "models_expected": sorted(expected_models),
                "models_found": sorted(models_found),
                "elapsed_s": elapsed_s,
                "last_error": last_error,
            },
        )

    def _check_subprocess_health(self) -> None:
        """Detect subprocess exits via ``ray.wait()`` on run refs.

        Every actor launched by ``spawn_actor`` has a ``run_ref`` — an
        ObjectRef that resolves when the subprocess exits (or the actor
        dies).  ``ray.wait(timeout=0)`` checks all refs in a single
        non-blocking call with no per-actor RPC overhead.

        Note: does NOT use ``with ray.init()`` — the driver must already be
        connected.
        """
        import ray

        procs: list[ManagedSubprocess] = []
        if self._frontend_actor is not None:
            procs.append(self._frontend_actor)
        procs.extend(self._worker_actors)
        if self._etcd_actor is not None:
            procs.append(self._etcd_actor)
        if self._nats_actor is not None:
            procs.append(self._nats_actor)

        ref_to_proc = {p.run_ref: p for p in procs if p.run_ref is not None}
        if not ref_to_proc:
            return

        ready, _ = ray.wait(list(ref_to_proc.keys()), num_returns=len(ref_to_proc), timeout=0)
        for ref in ready:
            proc = ref_to_proc[ref]
            log_tail = self._read_actor_log_tail(proc.actor)
            self._raise_subprocess_error(proc.label, log_tail, reason="subprocess exited unexpectedly")

    @staticmethod
    def _read_actor_log_tail(actor: Any) -> str:  # noqa: ANN401
        """Read log tail from a subprocess actor, returning empty string on failure."""
        import ray

        with contextlib.suppress(Exception):
            return ray.get(actor.read_log_tail.remote(), timeout=5)
        return ""

    @staticmethod
    def _raise_subprocess_error(label: str, log_tail: str, *, reason: str) -> None:
        """Raise ``SubprocessError`` with formatted subprocess crash info."""
        tail = "\n".join(log_tail.splitlines()[-50:]) if log_tail else "(no log output)"
        msg = f"Dynamo {label} {reason}.\n\n--- {label} log (last 50 lines) ---\n{tail}"
        raise SubprocessError(
            msg,
            debug_context={"label": label, "reason": reason, "log_tail": tail},
        )

    def _resolve_infra_node(self, cluster_nodes: list[dict[str, Any]], head_node_id: str) -> tuple[str, str]:
        """Pick the node that should host etcd, NATS, and the frontend.

        Returns ``(node_id, node_ip)``.  When ``CURATOR_IGNORE_RAY_HEAD_NODE``
        is set, selects the first non-head node (sorted by CPU count, then
        node ID for stability).  Otherwise returns the head node.
        """
        if not _ignore_head_node():
            return head_node_id, self._head_ip

        non_head = [n for n in cluster_nodes if n["NodeID"] != head_node_id and n.get("Alive", False)]
        if not non_head:
            msg = "CURATOR_IGNORE_RAY_HEAD_NODE is set but no non-head nodes are available."
            raise RuntimeError(msg)
        non_head.sort(key=lambda n: (-n.get("Resources", {}).get("CPU", 0), n["NodeID"]))
        infra = non_head[0]
        logger.info(
            f"Head-node exclusion active: infra will run on {infra['NodeManagerAddress']} (node {infra['NodeID'][:8]})"
        )
        return infra["NodeID"], infra["NodeManagerAddress"]

    @staticmethod
    def _dynamo_runtime_env(model_config: BaseModelConfig) -> dict[str, Any]:
        """Build the runtime_env for a Dynamo worker/frontend actor.

        Merges the default ``ai-dynamo[vllm]`` requirement with any
        user-provided ``runtime_env`` from the model config.
        """
        return BaseModelConfig.merge_runtime_envs(
            _DYNAMO_VLLM_RUNTIME_ENV,
            model_config.runtime_env or None,
        )

    @staticmethod
    def _merge_model_runtime_envs(models: list[BaseModelConfig]) -> dict[str, Any]:
        """Merge ``runtime_env`` dicts from all model configs for the frontend.

        Always includes ``ai-dynamo[vllm]`` as a base, then merges any
        user-provided ``runtime_env`` from model configs on top.
        """
        from functools import reduce

        envs = [m.runtime_env for m in models if m.runtime_env]
        user_merged = reduce(BaseModelConfig.merge_runtime_envs, envs) if envs else None
        return BaseModelConfig.merge_runtime_envs(
            _DYNAMO_VLLM_RUNTIME_ENV,
            user_merged,
        )

    @staticmethod
    def _resolve_disagg_role_config(
        model_config: DynamoVLLMModelConfig,
    ) -> tuple[tuple[int, dict[str, Any]], tuple[int, dict[str, Any]]]:
        """Resolve per-role replica count and engine_kwargs for disaggregated serving.

        Reads the typed ``prefill`` and ``decode`` role configs. Each role can
        override ``num_replicas`` and merge role-specific ``engine_kwargs`` on
        top of the model-level ``engine_kwargs``.

        Returns:
            ``((num_prefill, prefill_engine_kwargs), (num_decode, decode_engine_kwargs))``
        """
        base_ek = model_config.engine_kwargs

        def _resolve_role(role_config: DynamoRoleConfig | None) -> tuple[int, dict[str, Any]]:
            if role_config is None:
                return 1, dict(base_ek)
            return role_config.num_replicas, {**base_ek, **role_config.engine_kwargs}

        return _resolve_role(model_config.prefill), _resolve_role(model_config.decode)

    @staticmethod
    def _resolve_num_replicas(model_config: DynamoVLLMModelConfig) -> int:
        return max(model_config.num_replicas, 1)

    # ------------------------------------------------------------------
    # Infrastructure actors
    # ------------------------------------------------------------------

    def _start_etcd(self, infra_node_id: str, port: int) -> ManagedSubprocess:
        data_dir = os.path.join(self._runtime_dir, "etcd_data")
        os.makedirs(data_dir, exist_ok=True)

        peer_port = get_free_port_on_node(infra_node_id, port + 1)
        command = [
            "etcd",
            "--listen-client-urls",
            f"http://0.0.0.0:{port}",
            "--advertise-client-urls",
            f"http://{self._infra_ip}:{port}",
            "--listen-peer-urls",
            f"http://127.0.0.1:{peer_port}",
            "--initial-advertise-peer-urls",
            f"http://127.0.0.1:{peer_port}",
            "--initial-cluster",
            f"default=http://127.0.0.1:{peer_port}",
            "--data-dir",
            data_dir,
        ]
        node_alloc = NodeAllocation(node_id=infra_node_id, node_ip=self._infra_ip, num_gpus=0, node_rank=0)
        proc = spawn_actor(
            "Dynamo_ETCD",
            node_alloc,
            command=command,
            runtime_dir=self._runtime_dir,
            actor_name_prefix=self._actor_name_prefix,
            subprocess_env={"ALLOW_NONE_AUTHENTICATION": "yes"},
        )

        logger.info(f"Starting etcd on port {port} via {self._infra_ip}")
        _wait_for_port(self._infra_ip, port, timeout_s=30, label="etcd")
        logger.info("etcd is ready")
        return proc

    def _start_nats(self, infra_node_id: str, port: int) -> ManagedSubprocess:
        store_dir = os.path.join(self._runtime_dir, "nats_data")
        os.makedirs(store_dir, exist_ok=True)

        command = ["nats-server", "-p", str(port), "-js", "--store_dir", store_dir]
        node_alloc = NodeAllocation(node_id=infra_node_id, node_ip=self._infra_ip, num_gpus=0, node_rank=0)
        proc = spawn_actor(
            "Dynamo_NATS",
            node_alloc,
            command=command,
            runtime_dir=self._runtime_dir,
            actor_name_prefix=self._actor_name_prefix,
        )

        logger.info(f"Starting NATS on port {port} via {self._infra_ip}")
        _wait_for_port(self._infra_ip, port, timeout_s=30, label="nats")
        logger.info("NATS is ready")
        return proc

    # ------------------------------------------------------------------
    # Worker / frontend actors
    # ------------------------------------------------------------------

    def _aggregated_model_uses_exact_kv_events(self, model_config: DynamoVLLMModelConfig) -> bool:
        """Return True if an aggregated model should publish ZMQ KV events.

        True when the model opts into KV-aware routing (``router_mode="kv"``)
        with exact event tracking (``router_kv_events`` defaults to True).
        Disagg models are handled separately by ``_launch_disagg_workers``.
        """
        if model_config.mode == "disagg":
            return False
        backend = self._server.backend
        if not isinstance(backend, DynamoServerConfig):
            msg = "DynamoBackend requires DynamoServerConfig."
            raise TypeError(msg)
        if backend.router.mode != "kv":
            return False
        return backend.router.kv_events

    @staticmethod
    def _build_worker_kv_events_config(
        model_config: DynamoVLLMModelConfig,
        *,
        node_id: str,
        port_seed: int,
        enabled: bool,
    ) -> str:
        """Build a ``--kv-events-config`` JSON string for a Dynamo worker.

        When *enabled*, allocates a unique port via ``get_free_port_on_node``
        and merges the user's optional ``kv_events_config`` template (but
        always owns ``endpoint`` and ``enable_kv_cache_events``).

        When not enabled, explicitly disables KV events to prevent Dynamo's
        auto-configuration from binding a conflicting default port.
        """
        template = dict(model_config.kv_events_config)

        if not enabled:
            template["enable_kv_cache_events"] = False
            template.pop("endpoint", None)
            return json.dumps(template)

        kv_events_port = get_free_port_on_node(node_id, port_seed)
        template.update(
            {
                "publisher": "zmq",
                "topic": "kv-events",
                "endpoint": f"tcp://*:{kv_events_port}",
                "enable_kv_cache_events": True,
            }
        )
        return json.dumps(template)

    def _launch_replicas(  # noqa: PLR0913
        self,
        model_config: DynamoVLLMModelConfig,
        base_env: dict[str, str],
        *,
        inventory: list[dict[str, Any]],
        num_replicas: int,
        namespace: str,
        request_plane: str,
        event_plane: str,
    ) -> tuple[list[tuple[ReplicaPlan, str | None]], list[dict[str, Any]]]:
        """Plan placement and launch all worker actors for a non-disagg model.

        Returns:
            Tuple of (list of (plan, role) pairs, remaining GPU inventory).
            Role is always ``None`` for non-disaggregated models.
        """
        tp_size = model_config.engine_kwargs.get("tensor_parallel_size", 1)
        replica_plans = plan_replica_placement(num_replicas, tp_size, _inventory=inventory)

        for plan in replica_plans:
            if plan.is_multi_node:
                logger.info(
                    f"Replica {plan.replica_index}: multi-node TP across {plan.nnodes} nodes "
                    f"(total {plan.total_gpus} GPUs, master={plan.master_addr})"
                )
            else:
                logger.info(f"Replica {plan.replica_index}: single-node, {plan.total_gpus} GPU(s)")

            for rank in plan.ranks:
                if rank.node_rank == 0:
                    proc = self._launch_worker(
                        replica_index=plan.replica_index,
                        model_config=model_config,
                        base_env=base_env,
                        node_alloc=rank,
                        namespace=namespace,
                        request_plane=request_plane,
                        event_plane=event_plane,
                        multi_node_plan=plan if plan.is_multi_node else None,
                    )
                else:
                    proc = self._launch_headless_worker(
                        replica_index=plan.replica_index,
                        model_config=model_config,
                        base_env=base_env,
                        node_alloc=rank,
                        plan=plan,
                    )
                self._worker_actors.append(proc)

        remaining = self._subtract_placed_gpus(inventory, replica_plans)
        return [(plan, None) for plan in replica_plans], remaining

    def _launch_disagg_workers(  # noqa: PLR0913
        self,
        model_config: DynamoVLLMModelConfig,
        base_env: dict[str, str],
        *,
        inventory: list[dict[str, Any]],
        namespace: str,
        request_plane: str,
        event_plane: str,
    ) -> tuple[list[tuple[ReplicaPlan, str]], list[dict[str, Any]]]:
        """Launch separate prefill and decode worker pools for disaggregated serving.

        Each role can have its own ``num_replicas`` and ``engine_kwargs``
        (including ``tensor_parallel_size``), configured via the typed
        ``prefill`` and ``decode`` role configs. Role-level ``engine_kwargs``
        are merged on top of the model-level defaults.

        Workers are placed greedily on available GPUs.  Prefill workers
        additionally get ``--kv-events-config`` for ZMQ event publishing.

        Multi-node tensor parallelism *within* a single disagg worker is not
        yet supported — each worker's TP group must fit on one node.

        Returns:
            Tuple of (list of (plan, role) pairs, remaining GPU inventory).
            Role is ``"decode"`` or ``"prefill"``.
        """
        (num_prefill, prefill_ek), (num_decode, decode_ek) = self._resolve_disagg_role_config(model_config)
        prefill_tp = prefill_ek.get("tensor_parallel_size", 1)
        decode_tp = decode_ek.get("tensor_parallel_size", 1)

        # Plan placement separately per role (they may have different TP sizes).
        decode_plans = plan_replica_placement(num_decode, decode_tp, _inventory=inventory) if num_decode else []
        inventory = self._subtract_placed_gpus(inventory, decode_plans)

        prefill_plans = plan_replica_placement(num_prefill, prefill_tp, _inventory=inventory) if num_prefill else []

        for plan in [*decode_plans, *prefill_plans]:
            if plan.is_multi_node:
                msg = (
                    f"Disaggregated serving does not yet support multi-node tensor parallelism. "
                    f"Worker {plan.replica_index} requires {plan.total_gpus} GPUs across "
                    f"{plan.nnodes} nodes. Reduce tensor_parallel_size to fit on a single node."
                )
                raise ValueError(msg)

        kv_transfer_config = json.dumps(
            model_config.kv_transfer_config or {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
        )

        model_name = model_config.resolved_model_name
        component = _model_name_to_component(model_name)
        worker_index = 0

        # Launch decode workers first (matching Dynamo example convention)
        for i, plan in enumerate(decode_plans):
            rank0 = plan.ranks[0]
            nixl_port = get_free_port_on_node(rank0.node_id, 20097 + worker_index)

            python_args = [
                "-m",
                "dynamo.vllm",
                "--model",
                model_config.model_identifier,
                "--served-model-name",
                model_name,
                "--endpoint",
                _dynamo_endpoint(namespace, component, role="decode"),
                "--discovery-backend",
                "etcd",
                "--request-plane",
                request_plane,
                "--event-plane",
                event_plane,
                "--disaggregation-mode",
                "decode",
                "--kv-transfer-config",
                kv_transfer_config,
                *_engine_kwargs_to_cli_flags(decode_ek),
            ]

            label = build_worker_actor_name(model_name, i, 0, decode_tp, role="decode")
            logger.info(f"Disagg decode worker {i}: {rank0.num_gpus} GPU(s) on {rank0.node_ip}, nixl_port={nixl_port}")
            proc = spawn_actor(
                label,
                rank0,
                python_args=python_args,
                runtime_dir=self._runtime_dir,
                actor_name_prefix=self._actor_name_prefix,
                subprocess_env={**base_env, "VLLM_NIXL_SIDE_CHANNEL_PORT": str(nixl_port), "PYTHONHASHSEED": "0"},
                runtime_env=self._dynamo_runtime_env(model_config),
            )
            self._worker_actors.append(proc)
            worker_index += 1

        # Launch prefill workers
        for i, plan in enumerate(prefill_plans):
            rank0 = plan.ranks[0]
            nixl_port = get_free_port_on_node(rank0.node_id, 20097 + worker_index)
            kv_events_config = self._build_worker_kv_events_config(
                model_config,
                node_id=rank0.node_id,
                port_seed=20081 + i,
                enabled=True,
            )

            python_args = [
                "-m",
                "dynamo.vllm",
                "--model",
                model_config.model_identifier,
                "--served-model-name",
                model_name,
                "--endpoint",
                _dynamo_endpoint(namespace, component, role="prefill"),
                "--discovery-backend",
                "etcd",
                "--request-plane",
                request_plane,
                "--event-plane",
                event_plane,
                "--disaggregation-mode",
                "prefill",
                "--kv-transfer-config",
                kv_transfer_config,
                "--kv-events-config",
                kv_events_config,
                *_engine_kwargs_to_cli_flags(prefill_ek),
            ]

            label = build_worker_actor_name(model_name, i, 0, prefill_tp, role="prefill")
            logger.info(
                f"Disagg prefill worker {i}: {rank0.num_gpus} GPU(s) on {rank0.node_ip}, nixl_port={nixl_port}"
            )
            proc = spawn_actor(
                label,
                rank0,
                python_args=python_args,
                runtime_dir=self._runtime_dir,
                actor_name_prefix=self._actor_name_prefix,
                subprocess_env={**base_env, "VLLM_NIXL_SIDE_CHANNEL_PORT": str(nixl_port), "PYTHONHASHSEED": "0"},
                runtime_env=self._dynamo_runtime_env(model_config),
            )
            self._worker_actors.append(proc)
            worker_index += 1

        total_gpus = num_decode * decode_tp + num_prefill * prefill_tp
        tp_desc = f"TP={decode_tp}" if decode_tp == prefill_tp else f"prefill_TP={prefill_tp}, decode_TP={decode_tp}"
        logger.info(
            f"Disaggregated serving: {num_decode} decode + {num_prefill} prefill "
            f"workers launched ({total_gpus} GPUs total, {tp_desc})"
        )
        all_plans = self._subtract_placed_gpus(inventory, prefill_plans)
        plans_with_roles: list[tuple[ReplicaPlan, str]] = [(p, "decode") for p in decode_plans] + [
            (p, "prefill") for p in prefill_plans
        ]
        return plans_with_roles, all_plans

    def _launch_worker(  # noqa: PLR0913
        self,
        *,
        replica_index: int,
        model_config: DynamoVLLMModelConfig,
        base_env: dict[str, str],
        node_alloc: NodeAllocation,
        namespace: str,
        request_plane: str,
        event_plane: str,
        multi_node_plan: ReplicaPlan | None = None,
    ) -> ManagedSubprocess:
        """Launch a full Dynamo vLLM worker (rank 0).

        This worker creates a ``DistributedRuntime``, registers the model in
        etcd, and serves inference requests.  For multi-node TP, it also
        coordinates with headless workers via ``torch.distributed``.
        """
        model_name = model_config.resolved_model_name
        component = _model_name_to_component(model_name)

        # Always pass explicit --kv-events-config for aggregated workers.
        # Without this, Dynamo's args.py auto-creates KVEventsConfig binding
        # tcp://*:20080 when prefix_caching is enabled (default in vLLM >=0.16),
        # causing all workers on the same node to fight over the same port.
        kv_events_enabled = self._aggregated_model_uses_exact_kv_events(model_config)
        kv_events_config = self._build_worker_kv_events_config(
            model_config,
            node_id=node_alloc.node_id,
            port_seed=20080 + replica_index,
            enabled=kv_events_enabled,
        )

        python_args = [
            "-m",
            "dynamo.vllm",
            "--model",
            model_config.model_identifier,
            "--served-model-name",
            model_name,
            "--endpoint",
            _dynamo_endpoint(namespace, component),
            "--discovery-backend",
            "etcd",
            "--request-plane",
            request_plane,
            "--event-plane",
            event_plane,
            "--kv-events-config",
            kv_events_config,
            *_engine_kwargs_to_cli_flags(model_config.engine_kwargs),
        ]

        if multi_node_plan is not None:
            python_args.extend(
                [
                    "--nnodes",
                    str(multi_node_plan.nnodes),
                    "--node-rank",
                    "0",
                    "--master-addr",
                    multi_node_plan.master_addr,
                ]
            )

        tp_size = model_config.engine_kwargs.get("tensor_parallel_size", 1)
        label = build_worker_actor_name(model_name, replica_index, 0, tp_size)
        return spawn_actor(
            label,
            node_alloc,
            python_args=python_args,
            runtime_dir=self._runtime_dir,
            actor_name_prefix=self._actor_name_prefix,
            subprocess_env=base_env,
            runtime_env=self._dynamo_runtime_env(model_config),
        )

    def _launch_headless_worker(
        self,
        *,
        replica_index: int,
        model_config: DynamoVLLMModelConfig,
        base_env: dict[str, str],
        node_alloc: NodeAllocation,
        plan: ReplicaPlan,
    ) -> ManagedSubprocess:
        """Launch a headless vLLM worker (rank > 0, multi-node TP).

        Headless workers bypass Dynamo's ``DistributedRuntime`` entirely --
        they run only vLLM workers coordinated with rank 0 via
        ``torch.distributed`` (NCCL).  No model registration, no etcd/NATS.
        KV events are explicitly disabled to prevent Dynamo's auto-config
        from binding a conflicting default port.
        """
        # Headless workers don't run the scheduler so they won't bind a KV
        # events port.  Pass explicit disable defensively so that future
        # Dynamo/vLLM versions can't auto-configure one.
        kv_events_config = self._build_worker_kv_events_config(
            model_config,
            node_id=node_alloc.node_id,
            port_seed=20080 + replica_index + node_alloc.node_rank,
            enabled=False,
        )

        python_args = [
            "-m",
            "dynamo.vllm",
            "--model",
            model_config.model_identifier,
            "--headless",
            "--kv-events-config",
            kv_events_config,
            "--nnodes",
            str(plan.nnodes),
            "--node-rank",
            str(node_alloc.node_rank),
            "--master-addr",
            plan.master_addr,
            *_engine_kwargs_to_cli_flags(model_config.engine_kwargs),
        ]

        tp_size = model_config.engine_kwargs.get("tensor_parallel_size", 1)
        model_name = model_config.resolved_model_name
        label = build_worker_actor_name(model_name, replica_index, node_alloc.node_rank, tp_size)
        return spawn_actor(
            label,
            node_alloc,
            python_args=python_args,
            runtime_dir=self._runtime_dir,
            actor_name_prefix=self._actor_name_prefix,
            subprocess_env=base_env,
            runtime_env=self._dynamo_runtime_env(model_config),
        )

    def _launch_frontend(  # noqa: PLR0913, C901
        self,
        infra_node_id: str,
        port: int,
        base_env: dict[str, str],
        *,
        namespace: str,
        request_plane: str,
        event_plane: str,
        frontend_router_cfg: dict[str, Any],
        runtime_env: dict[str, Any] | None = None,
    ) -> ManagedSubprocess:
        """Launch the Dynamo frontend (OpenAI-compatible HTTP proxy).

        Translates *frontend_router_cfg* (from ``_resolve_frontend_router_config``)
        into ``--router-*`` CLI flags for ``dynamo.frontend``.
        """
        frontend_env = dict(base_env)
        router_mode = frontend_router_cfg.get("router_mode")
        if router_mode:
            frontend_env["PYTHONHASHSEED"] = "0"

        python_args = [
            "-m",
            "dynamo.frontend",
            "--http-port",
            str(port),
            "--namespace",
            namespace,
            "--discovery-backend",
            "etcd",
            "--request-plane",
            request_plane,
            "--event-plane",
            event_plane,
        ]

        if router_mode:
            python_args.extend(["--router-mode", router_mode])

            # Exact vs approximate KV routing
            if not frontend_router_cfg.get("router_kv_events", True):
                python_args.append("--no-router-kv-events")

            # Tuning knobs — only pass when explicitly set (non-default)
            if frontend_router_cfg.get("router_kv_overlap_score_weight") is not None:
                python_args.extend(
                    [
                        "--router-kv-overlap-score-weight",
                        str(frontend_router_cfg["router_kv_overlap_score_weight"]),
                    ]
                )
            if frontend_router_cfg.get("router_temperature") is not None:
                python_args.extend(["--router-temperature", str(frontend_router_cfg["router_temperature"])])
            if frontend_router_cfg.get("router_queue_threshold") is not None:
                python_args.extend(["--router-queue-threshold", str(frontend_router_cfg["router_queue_threshold"])])

            # Approximate-mode knobs (TTL, tree pruning) — only relevant when
            # KV events are disabled so the router predicts cache state.
            if not frontend_router_cfg.get("router_kv_events", True):
                if frontend_router_cfg.get("router_ttl_secs") is not None:
                    python_args.extend(["--router-ttl-secs", str(frontend_router_cfg["router_ttl_secs"])])
                if frontend_router_cfg.get("router_max_tree_size") is not None:
                    python_args.extend(["--router-max-tree-size", str(frontend_router_cfg["router_max_tree_size"])])
                if frontend_router_cfg.get("router_prune_target_ratio") is not None:
                    python_args.extend(
                        [
                            "--router-prune-target-ratio",
                            str(frontend_router_cfg["router_prune_target_ratio"]),
                        ]
                    )

            if frontend_router_cfg.get("router_reset_states"):
                python_args.append("--router-reset-states")

        node_alloc = NodeAllocation(node_id=infra_node_id, node_ip=self._infra_ip, num_gpus=0, node_rank=0)
        logger.info(f"Starting Dynamo frontend on port {port}")
        return spawn_actor(
            "Dynamo_Frontend",
            node_alloc,
            python_args=python_args,
            runtime_dir=self._runtime_dir,
            actor_name_prefix=self._actor_name_prefix,
            subprocess_env=frontend_env,
            runtime_env=runtime_env,
        )

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    def stop(self) -> None:
        import ray

        actors_to_kill = self._collect_actors_for_shutdown()

        try:
            with ray.init(ignore_reinit_error=True):
                for label, actor in actors_to_kill:
                    _kill_actor(ray, label, actor)
                # atexit is unreliable in Jupyter — sweep by name to catch actors
                # orphaned when the kernel was restarted without calling stop().
                kill_orphaned_actors(ray, f"dynamo_{self._server.name}_")
        except Exception:  # noqa: BLE001
            logger.debug("Could not connect to Ray during Dynamo shutdown (cluster may be gone)")

        self._cleanup_runtime_dir()
        self._server._host = "localhost"
        logger.info("Dynamo backend stopped")

    def _collect_actors_for_shutdown(self) -> list[tuple[str, Any]]:
        """Collect all actors in reverse-start order: frontend -> workers -> nats -> etcd."""
        actors: list[tuple[str, Any]] = []

        if self._frontend_actor is not None:
            actors.append((self._frontend_actor.label, self._frontend_actor.actor))
            self._frontend_actor = None

        for proc in self._worker_actors:
            actors.append((proc.label, proc.actor))
        self._worker_actors.clear()

        if self._nats_actor is not None:
            actors.append((self._nats_actor.label, self._nats_actor.actor))
            self._nats_actor = None

        if self._etcd_actor is not None:
            actors.append((self._etcd_actor.label, self._etcd_actor.actor))
            self._etcd_actor = None

        return actors

    def _cleanup_runtime_dir(self) -> None:
        if self._runtime_dir:
            try:
                shutil.rmtree(self._runtime_dir)
                logger.debug(f"Cleaned up runtime dir: {self._runtime_dir}")
            except FileNotFoundError:
                pass
            except Exception:  # noqa: BLE001
                logger.debug(f"Failed to clean up runtime dir: {self._runtime_dir}")
            self._runtime_dir = None
