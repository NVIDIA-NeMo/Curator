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

"""NVIDIA Dynamo inference backend for aggregated serving.

One detached placement group per replica carries its TP bundles; a
separate STRICT_PACK PG co-locates etcd, NATS, and the Dynamo frontend.
``mode="disagg"`` raises ``NotImplementedError`` for now.
"""

from __future__ import annotations

import contextlib
import http
import json
import os
import shutil
import tempfile
import time
import urllib.request
import uuid
from typing import TYPE_CHECKING, Any, cast

import ray
from loguru import logger

from nemo_curator.backends.utils import check_total_gpu_capacity
from nemo_curator.core.serve.base import InferenceBackend
from nemo_curator.core.serve.dynamo.config import DynamoServerConfig
from nemo_curator.core.serve.dynamo.constants import (
    DEFAULT_ETCD_PORT,
    DEFAULT_NATS_PORT,
    ETCD_ACTOR_LABEL,
    FRONTEND_ACTOR_LABEL,
    INFRA_ETCD_BUNDLE,
    INFRA_FRONTEND_BUNDLE,
    INFRA_NATS_BUNDLE,
    INFRA_NUM_BUNDLES,
    NATS_ACTOR_LABEL,
    NEMO_CURATOR_DYNAMO_NAMESPACE,
)
from nemo_curator.core.serve.dynamo.infra import build_infra_pg, engine_kwargs_to_cli_flags
from nemo_curator.core.serve.dynamo.vllm import launch_replicas, merge_model_runtime_envs
from nemo_curator.core.serve.placement import (
    _get_gpu_topology,
    get_bundle_node_ip,
    get_free_port_in_bundle,
    remove_named_pgs_with_prefix,
)
from nemo_curator.core.serve.subprocess_mgr import (
    ManagedSubprocess,
    SubprocessError,
    _check_binary,
    _wait_for_port,
    reacquire_detached_actor_handles,
    sweep_orphan_actors_by_prefix,
)
from nemo_curator.core.utils import ignore_ray_head_node

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

    from nemo_curator.core.serve.dynamo.config import DynamoVLLMModelConfig
    from nemo_curator.core.serve.server import InferenceServer


class DynamoBackend(InferenceBackend):
    """Dynamo backend for ``InferenceServer`` — aggregated serving on Ray PGs.

    - ``start()`` enters the ``nemo_curator_dynamo`` namespace, sweeps any
      leftover actors + PGs from a prior driver session, then deploys
      infra → workers → frontend and blocks on a ``/v1/models`` health check.
    - ``stop()`` re-enters the same namespace in a fresh Ray session; because
      ``ActorHandle`` objects do not survive a ``ray.shutdown()`` boundary,
      the stored handles are refreshed by name before the parallel
      SIGTERM → SIGKILL teardown runs. Replica + infra PGs are then removed.
    """

    def __init__(self, server: InferenceServer) -> None:
        if not isinstance(server.backend, DynamoServerConfig):
            msg = f"DynamoBackend requires DynamoServerConfig; got {type(server.backend).__name__}"
            raise TypeError(msg)
        self._server = server
        self._backend_cfg: DynamoServerConfig = server.backend
        # InferenceServer._validate_model_configs has already enforced that every
        # entry is a DynamoVLLMModelConfig; narrow once for the IDE.
        self._models: list[DynamoVLLMModelConfig] = cast("list[DynamoVLLMModelConfig]", server.models)
        self._runtime_dir: str | None = None
        self._infra_pg: PlacementGroup | None = None
        self._replica_pgs: list[PlacementGroup] = []
        self._infra_ip: str | None = None
        self._etcd_actor: ManagedSubprocess | None = None
        self._nats_actor: ManagedSubprocess | None = None
        self._worker_actors: list[ManagedSubprocess] = []
        self._frontend_actor: ManagedSubprocess | None = None
        self._actor_name_prefix: str = ""
        self._pg_name_prefix: str = ""

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def start(self) -> None:
        server = self._server
        backend_cfg = self._backend_cfg

        if not self._models:
            msg = "At least one DynamoVLLMModelConfig is required."
            raise ValueError(msg)

        for model_config in self._models:
            if model_config.mode == "disagg":
                msg = (
                    f"Disaggregated serving (mode='disagg') on model "
                    f"'{model_config.resolved_model_name}' is not yet supported by DynamoBackend; "
                    f"this lands in the next PR in the Inference Server stack."
                )
                raise NotImplementedError(msg)

        if not backend_cfg.etcd_endpoint:
            _check_binary("etcd")
        if not backend_cfg.nats_url:
            _check_binary("nats-server")

        short_id = uuid.uuid4().hex[:8]
        self._pg_name_prefix = f"dynamo_{server.name}_"
        self._actor_name_prefix = f"{self._pg_name_prefix}{short_id}"
        self._runtime_dir = tempfile.mkdtemp(prefix=f"nemo_curator_dynamo_{short_id}_")
        logger.info(f"Dynamo runtime dir: {self._runtime_dir}")

        with ray.init(namespace=NEMO_CURATOR_DYNAMO_NAMESPACE, ignore_reinit_error=True):
            self._sweep_orphan_actors()
            remove_named_pgs_with_prefix(self._pg_name_prefix)

            try:
                self._deploy_and_healthcheck(server, backend_cfg)
            except Exception:
                self._teardown_actors_and_pgs()
                self._cleanup_runtime_dir()
                raise

    def stop(self) -> None:
        try:
            with ray.init(namespace=NEMO_CURATOR_DYNAMO_NAMESPACE, ignore_reinit_error=True):
                self._teardown_actors_and_pgs()
                # Safety net for PGs left behind when the driver crashed
                # between start() and stop(). Guard against an empty prefix
                # -- an early-failing ``start()`` (e.g. ``mode="disagg"``)
                # leaves ``_pg_name_prefix`` unset, and
                # ``remove_named_pgs_with_prefix("")`` would match every
                # PG in the namespace.
                if self._pg_name_prefix:
                    remove_named_pgs_with_prefix(self._pg_name_prefix)
        except Exception:  # noqa: BLE001
            logger.warning("Dynamo backend shutdown hit an error (cluster may be gone)", exc_info=True)

        self._cleanup_runtime_dir()
        self._server._host = "localhost"
        logger.info("Dynamo backend stopped")

    # ------------------------------------------------------------------
    # Deployment
    # ------------------------------------------------------------------

    def _deploy_and_healthcheck(self, server: InferenceServer, backend_cfg: DynamoServerConfig) -> None:
        """Validate, create PGs, launch infra/workers/frontend, health-check."""
        topology = _get_gpu_topology()
        self._validate_gpu_requirements(self._models)

        infra_pg_name = f"{self._actor_name_prefix}_pg_infra"
        self._infra_pg = build_infra_pg(name=infra_pg_name, num_bundles=INFRA_NUM_BUNDLES)
        self._infra_ip = get_bundle_node_ip(self._infra_pg, INFRA_ETCD_BUNDLE)
        server._host = self._infra_ip

        server.port = get_free_port_in_bundle(self._infra_pg, INFRA_FRONTEND_BUNDLE, server.port)

        if backend_cfg.etcd_endpoint:
            etcd_endpoint = backend_cfg.etcd_endpoint
        else:
            etcd_port = get_free_port_in_bundle(self._infra_pg, INFRA_ETCD_BUNDLE, DEFAULT_ETCD_PORT)
            self._etcd_actor = self._start_etcd(etcd_port)
            etcd_endpoint = f"http://{self._infra_ip}:{etcd_port}"

        if backend_cfg.nats_url:
            nats_url = backend_cfg.nats_url
        else:
            nats_port = get_free_port_in_bundle(self._infra_pg, INFRA_NATS_BUNDLE, DEFAULT_NATS_PORT)
            self._nats_actor = self._start_nats(nats_port)
            nats_url = f"nats://{self._infra_ip}:{nats_port}"

        base_env = {"ETCD_ENDPOINTS": etcd_endpoint, "NATS_SERVER": nats_url}

        expected_models: set[str] = set()
        placements: list[dict[str, Any]] = []

        for model_config in self._models:
            model_name = model_config.resolved_model_name
            expected_models.add(model_name)
            tp_size = model_config.engine_kwargs.get("tensor_parallel_size", 1)
            logger.info(f"Deploying model '{model_name}' (TP={tp_size}, replicas={model_config.num_replicas})")

            pgs, actors, entries = launch_replicas(
                model_config,
                base_env=base_env,
                namespace=backend_cfg.namespace,
                request_plane=backend_cfg.request_plane,
                event_plane=backend_cfg.event_plane,
                runtime_dir=self._runtime_dir,
                actor_name_prefix=self._actor_name_prefix,
                router_mode=backend_cfg.router.mode,
                router_kv_events=backend_cfg.router.kv_events,
                topology=topology,
            )
            self._replica_pgs.extend(pgs)
            self._worker_actors.extend(actors)
            placements.extend(entries)

        manifest_data = {
            "models": sorted(expected_models),
            "endpoint": server.endpoint,
            "etcd": etcd_endpoint,
            "nats": nats_url,
            "port": server.port,
            "placements": placements,
        }
        self._write_manifest(manifest_data, ready=False)

        self._frontend_actor = self._launch_frontend(
            server.port,
            base_env,
            backend_cfg=backend_cfg,
            runtime_env=merge_model_runtime_envs(self._models),
        )

        self._wait_for_models(server, expected_models)
        self._write_manifest(manifest_data, ready=True)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_gpu_requirements(models: list[DynamoVLLMModelConfig]) -> None:
        """Coarse fail-fast on cluster-wide GPU over-commit.

        Ray's per-PG ``STRICT_PACK`` / ``STRICT_SPREAD`` is the authoritative
        admission gate; this produces a better error than the admission timeout.
        Cross-model validators (frontend-config coherence, disagg-TP-fit,
        unique model names) land in the next PR.
        """
        total_needed = 0
        for model_config in models:
            tp_size = model_config.engine_kwargs.get("tensor_parallel_size", 1)
            total_needed += model_config.num_replicas * tp_size
        check_total_gpu_capacity(total_needed, ignore_head_node=ignore_ray_head_node())

    # ------------------------------------------------------------------
    # Infra actors
    # ------------------------------------------------------------------

    def _start_infra_service(
        self,
        *,
        label: str,
        bundle_index: int,
        port: int,
        command: list[str],
        subprocess_env: dict[str, str] | None = None,
    ) -> ManagedSubprocess:
        proc = ManagedSubprocess.spawn(
            label,
            self._infra_pg,
            bundle_index,
            num_gpus=0,
            command=command,
            runtime_dir=self._runtime_dir,
            actor_name_prefix=self._actor_name_prefix,
            subprocess_env=subprocess_env,
        )
        short_label = label.rsplit("_", 1)[-1].lower()
        logger.info(f"Starting {short_label} on port {port} via {self._infra_ip}")
        _wait_for_port(self._infra_ip, port, timeout_s=30, label=short_label)
        logger.info(f"{short_label} is ready")
        return proc

    def _start_etcd(self, port: int) -> ManagedSubprocess:
        data_dir = os.path.join(self._runtime_dir, "etcd_data")
        os.makedirs(data_dir, exist_ok=True)
        peer_port = get_free_port_in_bundle(self._infra_pg, INFRA_ETCD_BUNDLE, port + 1)
        return self._start_infra_service(
            label=ETCD_ACTOR_LABEL,
            bundle_index=INFRA_ETCD_BUNDLE,
            port=port,
            command=[
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
            ],
            subprocess_env={"ALLOW_NONE_AUTHENTICATION": "yes"},
        )

    def _start_nats(self, port: int) -> ManagedSubprocess:
        store_dir = os.path.join(self._runtime_dir, "nats_data")
        os.makedirs(store_dir, exist_ok=True)
        return self._start_infra_service(
            label=NATS_ACTOR_LABEL,
            bundle_index=INFRA_NATS_BUNDLE,
            port=port,
            command=["nats-server", "-p", str(port), "-js", "--store_dir", store_dir],
        )

    # ------------------------------------------------------------------
    # Frontend
    # ------------------------------------------------------------------

    def _launch_frontend(
        self,
        port: int,
        base_env: dict[str, str],
        *,
        backend_cfg: DynamoServerConfig,
        runtime_env: dict[str, Any] | None = None,
    ) -> ManagedSubprocess:
        """Launch the Dynamo frontend bound to the infra node.

        Router wiring is minimal: ``--router-mode`` if set, followed by every
        entry in ``router_kwargs`` as ``--key value``. Full per-key router
        flag translation (kv_events, temperature, etc.) lands in the next PR.
        """
        frontend_env = dict(base_env)
        router_mode = backend_cfg.router.mode
        if router_mode:
            # Dynamo KV-aware routing depends on a stable Python hash seed so
            # prefix hashes agree across processes.
            frontend_env["PYTHONHASHSEED"] = "0"

        python_args = [
            "-m",
            "dynamo.frontend",
            "--http-port",
            str(port),
            "--namespace",
            backend_cfg.namespace,
            "--discovery-backend",
            "etcd",
            "--request-plane",
            backend_cfg.request_plane,
            "--event-plane",
            backend_cfg.event_plane,
        ]
        if router_mode:
            python_args.extend(["--router-mode", router_mode])
        python_args.extend(engine_kwargs_to_cli_flags(backend_cfg.router.router_kwargs))

        logger.info(f"Starting Dynamo frontend on port {port}")
        return ManagedSubprocess.spawn(
            FRONTEND_ACTOR_LABEL,
            self._infra_pg,
            INFRA_FRONTEND_BUNDLE,
            num_gpus=0,
            python_args=python_args,
            runtime_dir=self._runtime_dir,
            actor_name_prefix=self._actor_name_prefix,
            subprocess_env=frontend_env,
            runtime_env=runtime_env,
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def _wait_for_models(self, server: InferenceServer, expected_models: set[str]) -> None:
        """Poll ``/v1/models`` until all *expected_models* appear."""
        models_url = f"{server.endpoint}/models"
        deadline = time.monotonic() + server.health_check_timeout_s
        start_time = time.monotonic()
        attempt = 0
        last_error: str | None = None
        models_found: set[str] = set()

        # Worker set is fixed for the lifetime of this wait; freeze it once.
        monitored = self._all_actor_procs()

        while time.monotonic() < deadline:
            attempt += 1
            self._check_subprocess_health(monitored)

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

        self._check_subprocess_health(monitored)
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

    def _all_actor_procs(self) -> list[ManagedSubprocess]:
        procs: list[ManagedSubprocess] = []
        if self._frontend_actor is not None:
            procs.append(self._frontend_actor)
        procs.extend(self._worker_actors)
        if self._etcd_actor is not None:
            procs.append(self._etcd_actor)
        if self._nats_actor is not None:
            procs.append(self._nats_actor)
        return procs

    def _check_subprocess_health(self, monitored: list[ManagedSubprocess]) -> None:
        """Detect subprocess exits via ``ray.wait()`` on the cached run refs."""
        ref_to_proc = {p.run_ref: p for p in monitored if p.run_ref is not None}
        if not ref_to_proc:
            return

        ready, _ = ray.wait(list(ref_to_proc.keys()), num_returns=len(ref_to_proc), timeout=0)
        for ref in ready:
            proc = ref_to_proc[ref]
            log_tail = ""
            with contextlib.suppress(Exception):
                log_tail = proc.read_log_tail()
            self._raise_subprocess_error(proc.label, log_tail, reason="subprocess exited unexpectedly")

    @staticmethod
    def _raise_subprocess_error(label: str, log_tail: str, *, reason: str) -> None:
        tail = "\n".join(log_tail.splitlines()[-50:]) if log_tail else "(no log output)"
        msg = f"Dynamo {label} {reason}.\n\n--- {label} log (last 50 lines) ---\n{tail}"
        raise SubprocessError(
            msg,
            debug_context={"label": label, "reason": reason, "log_tail": tail},
        )

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def _teardown_actors_and_pgs(self) -> None:
        """Parallel-stop every actor, then release the placement groups.

        ``ActorHandle`` objects stored on ``self`` during ``start()`` belong to
        that session's Ray job and are invalid here (``stop()`` opened its own
        ``with ray.init()``), so the handles are refreshed by detached-actor
        name before any ``.remote()`` call is issued.
        """
        refreshed = reacquire_detached_actor_handles(
            self._all_actor_procs(),
            actor_name_prefix=self._actor_name_prefix,
            namespace=NEMO_CURATOR_DYNAMO_NAMESPACE,
        )
        ManagedSubprocess.stop_many(refreshed)

        self._frontend_actor = None
        self._worker_actors.clear()
        self._etcd_actor = None
        self._nats_actor = None

        for pg in self._replica_pgs:
            with contextlib.suppress(Exception):
                ray.util.remove_placement_group(pg)
        self._replica_pgs.clear()

        if self._infra_pg is not None:
            with contextlib.suppress(Exception):
                ray.util.remove_placement_group(self._infra_pg)
            self._infra_pg = None

    def _sweep_orphan_actors(self) -> None:
        """Reap any detached actors left behind by a prior driver session.

        ``remove_named_pgs_with_prefix`` force-kills actors scheduled into
        the reaped PGs, which would orphan the subprocess tree; sweeping
        named actors first lets ``graceful_stop_actors`` ``killpg`` each
        process group cleanly.
        """
        sweep_orphan_actors_by_prefix(
            prefix=self._pg_name_prefix,
            namespace=NEMO_CURATOR_DYNAMO_NAMESPACE,
        )

    def _cleanup_runtime_dir(self) -> None:
        if self._runtime_dir is None:
            return
        try:
            shutil.rmtree(self._runtime_dir)
            logger.debug(f"Cleaned up runtime dir: {self._runtime_dir}")
        except FileNotFoundError:
            pass
        except Exception:  # noqa: BLE001
            logger.debug(f"Failed to clean up runtime dir: {self._runtime_dir}")
        self._runtime_dir = None

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _write_manifest(self, data: dict[str, Any], *, ready: bool) -> None:
        manifest = {**data, "ready": ready, "timestamp": time.time()}
        logger.debug(f"Deployment manifest (ready={ready}): {json.dumps(manifest, indent=2)}")

        if not self._runtime_dir:
            return
        manifest_path = os.path.join(self._runtime_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
