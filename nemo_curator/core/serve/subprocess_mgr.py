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

"""Generic Ray-placement-group-based subprocess manager.

Each replica is backed by one placement group whose bundles describe its
topology. Single-node replicas use a single ``STRICT_PACK`` bundle;
multi-node (e.g. TP > per-node GPU count) replicas use ``STRICT_SPREAD``
with one bundle per node.

PGs are created with ``lifetime="detached"`` and a stable ``name=`` inside
the caller-chosen Ray namespace, so a reconnecting driver can find and
reap them across ``ray.shutdown()`` / ``ray.init()`` cycles.
``remove_placement_group`` is the primary teardown handle -- killing the
PG forcibly terminates its actors. Subprocess cleanup is explicit: actors
implement graceful shutdown (SIGTERM -> SIGKILL on the child's process
group) via ``__ray_terminate__`` + ``atexit``, so the child tree is reaped
even when Ray hard-kills the actor.

Backend-specific construction (service layout, actor naming, CLI flag
translation) lives alongside each backend (e.g.
``nemo_curator.core.serve.dynamo.infra``), not here.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import signal
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

if TYPE_CHECKING:
    import subprocess

    from ray import ObjectRef
    from ray.actor import ActorHandle
    from ray.util.placement_group import PlacementGroup

from nemo_curator.core.utils import get_free_port, ignore_ray_head_node  # noqa: F401 - re-exported

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_WORKER_NODE_LABEL = {"ray.io/node-type": "worker"}
"""Bundle label selector applied when ``CURATOR_IGNORE_RAY_HEAD_NODE=1``.

Anyscale auto-labels head/worker nodes. OSS Ray users must start worker nodes
with ``ray start --labels ray.io/node-type=worker`` for this to take effect.
"""

_SIGTERM_WAIT_S = 10
_SIGKILL_WAIT_S = 5
_PG_READY_TIMEOUT_S = 180

_NOSET_CUDA_RUNTIME_ENV: dict[str, Any] = {"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}}
"""Tells Ray not to overwrite the worker's ``CUDA_VISIBLE_DEVICES``.

We explicitly set ``CUDA_VISIBLE_DEVICES`` in ``subprocess_env`` from
``ray.get_accelerator_ids()``, so for the subprocess this flag is likely
redundant -- it's kept defensively because the canonical vLLM+Ray pattern
(vLLM issues #7890/#30016/#35848) relies on it, and Dynamo follows suit.
"""


# ---------------------------------------------------------------------------
# Replica bundle-shape planner
# ---------------------------------------------------------------------------


@dataclass
class ReplicaBundleSpec:
    """Bundle shape + strategy for a single model replica.

    Attributes:
        bundles: List of resource dicts, one per bundle (e.g.
            ``[{"CPU": 1, "GPU": 4}, {"CPU": 1, "GPU": 4}]`` for TP=8 across 2 nodes).
        strategy: ``"STRICT_PACK"`` for single-bundle replicas (irrelevant but
            explicit); ``"STRICT_SPREAD"`` for multi-bundle (forces distinct nodes).
        nnodes: Number of bundles == number of distinct nodes required.
        per_node_gpus: GPUs each bundle requires (== ``tp_size // nnodes``).
        bundle_label_selector: Per-bundle node-label constraints; ``None`` when
            head-node exclusion is off.
    """

    bundles: list[dict[str, float]]
    strategy: Literal["STRICT_PACK", "STRICT_SPREAD"]
    nnodes: int
    per_node_gpus: int
    bundle_label_selector: list[dict[str, str]] | None = None

    @property
    def is_multi_node(self) -> bool:
        return self.nnodes > 1

    @property
    def total_gpus(self) -> int:
        return self.nnodes * self.per_node_gpus


def _get_gpu_topology(
    head_node_id: str | None = None,
    nodes: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return per-node GPU topology: ``[{"node_id", "num_gpus", "is_head"}, ...]``.

    Uses total node resources, not current availability -- topology shape is a
    static property of the cluster. Ray's PG scheduler handles dynamic capacity.

    Args:
        head_node_id: Ray node ID to tag as head in output (for
            ``CURATOR_IGNORE_RAY_HEAD_NODE`` filtering).
        nodes: Pre-fetched ``ray.nodes()`` to avoid a redundant call.
    """
    import ray

    if head_node_id is None:
        head_node_id = ray.get_runtime_context().get_node_id()

    topology: list[dict[str, Any]] = []
    for node in nodes or ray.nodes():
        if not node.get("Alive", False):
            continue
        resources = node.get("Resources", {})
        num_gpus = int(resources.get("GPU", 0))
        if num_gpus == 0:
            continue
        node_id = node["NodeID"]
        topology.append({"node_id": node_id, "num_gpus": num_gpus, "is_head": node_id == head_node_id})
    return topology


def plan_replica_bundle_shape(
    tp_size: int,
    *,
    head_node_id: str | None = None,
    _topology: list[dict[str, Any]] | None = None,
    _nodes: list[dict[str, Any]] | None = None,
) -> ReplicaBundleSpec:
    """Pick the bundle shape for one replica given current cluster topology.

    Single-node: if any node has ``>= tp_size`` GPUs, return one bundle of
    size ``tp_size`` with ``STRICT_PACK``.

    Multi-node: find the smallest ``nnodes`` such that
    ``tp_size % nnodes == 0`` and at least ``nnodes`` nodes have
    ``>= tp_size / nnodes`` GPUs each. Return ``nnodes`` equal bundles with
    ``STRICT_SPREAD``. vLLM requires an even per-node split (1+3 for TP=4
    fails with a CUDA device ordinal error), so asymmetric splits are never
    considered.

    When ``CURATOR_IGNORE_RAY_HEAD_NODE`` is set, the head node is filtered
    out of topology and every bundle gets
    ``[{"ray.io/node-type": "worker"}]`` as a label selector.
    """
    topology = _topology if _topology is not None else _get_gpu_topology(head_node_id, nodes=_nodes)
    if not topology:
        msg = "No GPU nodes found in the Ray cluster."
        raise RuntimeError(msg)

    ignore_head = ignore_ray_head_node()
    if ignore_head:
        topology = [n for n in topology if not n["is_head"]]
        if not topology:
            msg = "CURATOR_IGNORE_RAY_HEAD_NODE is set but no non-head GPU nodes are available."
            raise RuntimeError(msg)

    label_selector_one = [_WORKER_NODE_LABEL] if ignore_head else None

    # Single-node preference: cheapest placement, no cross-node NCCL.
    max_per_node = max(n["num_gpus"] for n in topology)
    if max_per_node >= tp_size:
        return ReplicaBundleSpec(
            bundles=[{"CPU": 1, "GPU": tp_size}],
            strategy="STRICT_PACK",
            nnodes=1,
            per_node_gpus=tp_size,
            bundle_label_selector=label_selector_one,
        )

    # Multi-node even split: smallest nnodes dividing tp_size with sufficient
    # per-node topology. vLLM's distributed executor requires identical
    # local_world_size per node.
    for nnodes in range(2, len(topology) + 1):
        if tp_size % nnodes != 0:
            continue
        per_node = tp_size // nnodes
        eligible = sum(1 for n in topology if n["num_gpus"] >= per_node)
        if eligible >= nnodes:
            selector = [_WORKER_NODE_LABEL] * nnodes if ignore_head else None
            return ReplicaBundleSpec(
                bundles=[{"CPU": 1, "GPU": per_node}] * nnodes,
                strategy="STRICT_SPREAD",
                nnodes=nnodes,
                per_node_gpus=per_node,
                bundle_label_selector=selector,
            )

    msg = (
        f"Cannot place TP={tp_size}: need an even split across nodes but no "
        f"valid combination found across {len(topology)} eligible node(s)."
    )
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Placement group construction
# ---------------------------------------------------------------------------


def _build_pg(
    bundles: list[dict[str, float]],
    strategy: str,
    *,
    name: str,
    bundle_label_selector: list[dict[str, str]] | None,
    ready_timeout_s: float,
) -> PlacementGroup:
    """Create a detached, named PG and wait until ready; clean up on failure."""
    import ray
    from ray.util.placement_group import placement_group

    pg_kwargs: dict[str, Any] = {
        "bundles": bundles,
        "strategy": strategy,
        "name": name,
        "lifetime": "detached",
    }
    if bundle_label_selector is not None:
        pg_kwargs["bundle_label_selector"] = bundle_label_selector

    pg = placement_group(**pg_kwargs)
    try:
        ray.get(pg.ready(), timeout=ready_timeout_s)
    except Exception:
        with contextlib.suppress(Exception):
            ray.util.remove_placement_group(pg)
        raise
    return pg


def build_replica_pg(
    spec: ReplicaBundleSpec,
    *,
    name: str,
    ready_timeout_s: float = _PG_READY_TIMEOUT_S,
) -> PlacementGroup:
    """Create a detached, named PG for one replica and wait until ready.

    PG is created with ``lifetime="detached"`` so it survives driver
    disconnects between ``server.start()``, ``pipeline.run()``, and
    ``server.stop()``. The caller-supplied ``name`` is used for orphan
    cleanup via ``remove_named_pgs_with_prefix``.
    """
    return _build_pg(
        spec.bundles,
        spec.strategy,
        name=name,
        bundle_label_selector=spec.bundle_label_selector,
        ready_timeout_s=ready_timeout_s,
    )


# ---------------------------------------------------------------------------
# Subprocess actor
# ---------------------------------------------------------------------------


@dataclass
class ManagedSubprocess:
    """Track a detached Ray actor and the subprocess it owns.

    The instance methods hide the ``ray.get(self.actor.X.remote())``
    boilerplate callers would otherwise repeat for every lifecycle
    operation. Teardown is parallelised via ``stop_many`` so a whole
    replica (or the infra trio) can be reaped with one round-trip.
    """

    label: str
    actor: ActorHandle
    run_ref: ObjectRef | None = None
    log_file: str | None = None

    def is_alive(self) -> bool:
        import ray

        return ray.get(self.actor.is_alive.remote())

    def pid(self) -> int:
        import ray

        return ray.get(self.actor.pid.remote())

    def read_log_tail(self, num_bytes: int = 8192) -> str:
        import ray

        return ray.get(self.actor.read_log_tail.remote(num_bytes))

    def wait(self, timeout: float | None = None) -> int:
        """Block until the subprocess exits. Returns its exit code."""
        import ray

        if self.run_ref is None:
            msg = f"{self.label}: run_ref is None; subprocess was never started via spawn_actor"
            raise RuntimeError(msg)
        return ray.get(self.run_ref, timeout=timeout)

    def stop(self, *, timeout_s: float | None = None) -> None:
        """Best-effort graceful stop: SIGTERM the subprocess, then hard-kill the actor.

        Calls ``actor.stop.remote()`` (reaps the subprocess group) with a
        bounded wait, then falls back to ``ray.kill`` if stop times out.
        Use this before ``remove_placement_group`` so the subprocess tree
        dies before Ray hard-kills the actor.
        """
        type(self).stop_many([self], timeout_s=timeout_s)

    @classmethod
    def stop_many(cls, procs: list[ManagedSubprocess], *, timeout_s: float | None = None) -> None:
        """Stop many managed subprocesses in parallel.

        Kicks off every ``actor.stop.remote()`` first so subprocess
        teardown happens concurrently, then waits on all of them with a
        shared deadline and force-kills anything that did not drain.
        """
        graceful_stop_actors([(p.label, p.actor) for p in procs], timeout_s=timeout_s)


def _stop_subprocess(proc: subprocess.Popen, sigterm_wait: float = _SIGTERM_WAIT_S) -> int | None:
    """SIGTERM -> wait -> SIGKILL a subprocess and its entire process group.

    Subprocesses are launched with ``start_new_session=True`` so they become
    process-group leaders. Signaling the group (``os.killpg``) ensures child
    processes (e.g. vLLM torch.distributed workers) are also terminated
    rather than becoming orphans.
    """
    if proc.poll() is not None:
        return proc.returncode
    pgid: int | None = None
    with contextlib.suppress(OSError):
        pgid = os.getpgid(proc.pid)
    is_group_leader = pgid is not None and pgid == proc.pid
    if is_group_leader:
        os.killpg(pgid, signal.SIGTERM)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=sigterm_wait)
    except Exception:  # noqa: BLE001
        if is_group_leader:
            os.killpg(pgid, signal.SIGKILL)
        else:
            proc.kill()
        proc.wait(timeout=_SIGKILL_WAIT_S)
    return proc.returncode


def _define_subprocess_actor(actor_type: str = "SubprocessActor") -> type:  # noqa: C901
    """Return a Ray remote actor class named *actor_type*.

    Each call produces a class whose ``__name__`` and ``__qualname__`` are
    set to *actor_type* so the Ray dashboard shows descriptive labels
    (e.g. ``Dynamo_ETCD``, ``Dynamo_DP0_Qwen3-0.6B``).

    Lifecycle:

    1. **Create** with GPU options (``num_gpus=...``).
    2. **Discover GPUs** via ``get_assigned_gpus()`` (returns Ray-assigned IDs).
    3. **Launch subprocess** via ``initialize()``.
    4. **``run()``** blocks until exit -- returned ObjectRef resolves then.

    Graceful shutdown: we override ``__ray_terminate__`` to reap the
    subprocess group before ``ray.actor.exit_actor()``. ``atexit`` is also
    registered as a belt-and-suspenders path for unexpected actor exit. If
    Ray hard-kills the actor (``ray.kill`` or ``remove_placement_group``),
    ``atexit`` does not run, so callers should prefer
    ``actor.stop.remote()`` (or ``actor.__ray_terminate__.remote()``) before
    removing the PG.
    """
    import ray

    class _SubprocessActor:
        """Manages a subprocess on a Ray node with optional file-based logging.

        ``max_concurrency=2`` lets ``run()`` block on the subprocess while
        other methods (``is_alive``, ``read_log_tail``, ``stop``) stay
        responsive.
        """

        def __init__(self) -> None:
            self._proc: Any = None
            self._log_fh: Any = None
            self._log_file: str | None = None
            self._cleanup_registered = False

        def get_assigned_gpus(self) -> list[str]:
            """Return Ray-assigned accelerator IDs for this actor."""
            import ray as _ray

            return _ray.get_runtime_context().get_accelerator_ids().get("GPU", [])

        def get_node_ip(self) -> str:
            """Return the routable IP of the node this actor is running on."""
            import ray as _ray

            return _ray.util.get_node_ip_address()

        def initialize(
            self,
            command: list[str] | None,
            subprocess_env: dict[str, str],
            log_file: str | None = None,
            *,
            python_args: list[str] | None = None,
        ) -> dict:
            """Launch the subprocess with the actor's env + *subprocess_env* overrides.

            Pass *command* for binary subprocesses (etcd, nats) or
            *python_args* for Python module invocations (``-m dynamo.vllm``).
            With *python_args*, the actor prepends its own ``sys.executable``
            so the subprocess uses the ``runtime_env``'s Python, not the driver's.

            Returns: ``{"pid": int, "log_file": str | None}``.
            """
            if (command is None) == (python_args is None):
                msg = "Exactly one of 'command' or 'python_args' must be provided"
                raise ValueError(msg)

            import subprocess as _sp
            import sys as _sys

            if python_args is not None:
                command = [_sys.executable, *python_args]

            merged_env = {**os.environ, **subprocess_env}

            self._log_file = log_file
            if log_file:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                self._log_fh = open(log_file, "w")  # noqa: SIM115
                self._proc = _sp.Popen(  # noqa: S603
                    command, env=merged_env, stdout=self._log_fh, stderr=_sp.STDOUT, start_new_session=True
                )
            else:
                self._proc = _sp.Popen(  # noqa: S603
                    command, env=merged_env, stdout=_sp.DEVNULL, stderr=_sp.STDOUT, start_new_session=True
                )

            if not self._cleanup_registered:
                atexit.register(self._cleanup)
                self._cleanup_registered = True

            return {"pid": self._proc.pid, "log_file": log_file}

        def run(self) -> int:
            """Block until the subprocess exits. Returns the exit code."""
            if self._proc is None:
                return -1
            return self._proc.wait()

        def pid(self) -> int:
            return self._proc.pid if self._proc else -1

        def is_alive(self) -> bool:
            return self._proc is not None and self._proc.poll() is None

        def log_file(self) -> str | None:
            return self._log_file

        def read_log_tail(self, num_bytes: int = 8192) -> str:
            """Read the last *num_bytes* of the log file, flushing first."""
            if not self._log_file:
                return ""
            try:
                if self._log_fh and not self._log_fh.closed:
                    self._log_fh.flush()
                with open(self._log_file, "rb") as f:
                    f.seek(0, 2)
                    size = f.tell()
                    f.seek(max(0, size - num_bytes))
                    return f.read().decode(errors="replace")
            except Exception:  # noqa: BLE001
                return ""

        def stop(self, sigterm_wait: float = _SIGTERM_WAIT_S) -> int | None:
            """Gracefully stop the subprocess (SIGTERM group -> wait -> SIGKILL)."""
            if self._proc is None:
                return None
            rc = _stop_subprocess(self._proc, sigterm_wait)
            if self._log_fh:
                with contextlib.suppress(Exception):
                    self._log_fh.close()
            return rc

        def _cleanup(self) -> None:
            """atexit hook: last-line-of-defense subprocess reap."""
            with contextlib.suppress(Exception):
                if self._proc is not None:
                    _stop_subprocess(self._proc, sigterm_wait=3)
            with contextlib.suppress(Exception):
                if self._log_fh:
                    self._log_fh.close()

        def __ray_terminate__(self) -> None:
            """Override Ray's default termination to reap the subprocess group first.

            Ray's default implementation just calls ``ray.actor.exit_actor()``.
            We reap our subprocess tree beforehand so that a graceful
            ``actor.__ray_terminate__.remote()`` leaves no child processes behind.
            """
            self._cleanup()
            import ray as _ray

            worker = _ray._private.worker.global_worker
            if worker.mode != _ray.LOCAL_MODE:
                _ray.actor.exit_actor()

    _SubprocessActor.__name__ = actor_type
    _SubprocessActor.__qualname__ = actor_type
    return ray.remote(num_cpus=1, num_gpus=0, max_concurrency=2)(_SubprocessActor)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_binary(name: str) -> None:
    """Raise if *name* is not found on ``$PATH``."""
    import shutil as _shutil

    if _shutil.which(name) is None:
        msg = f"Required binary '{name}' not found on $PATH. Install it, then try again."
        raise FileNotFoundError(msg)


def _wait_for_port(host: str, port: int, timeout_s: float = 30, label: str = "") -> None:
    """Block until a TCP connection to *host:port* succeeds."""
    import socket as _socket

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with contextlib.suppress(OSError), _socket.create_connection((host, port), timeout=2):
            return
        time.sleep(0.5)
    tag = f" ({label})" if label else ""
    msg = f"Port {port}{tag} did not become reachable within {timeout_s}s"
    raise TimeoutError(msg)


# ---------------------------------------------------------------------------
# Remote discovery (port + IP) via PG bundles
#
# TODO(dynamo-refactor): collapse ``_run_in_bundle`` / ``get_free_port_in_bundle`` /
# ``get_bundle_node_ip`` (and possibly ``spawn_actor``'s (pg, bundle_index)
# tuple) into a ``Bundle(pg, index)`` wrapper once PR 4's DynamoBackend
# wiring shows how often the same (pg, idx) pair gets threaded through.
# Worth doing only if the repetition is real in the consumer, not upfront.
# ---------------------------------------------------------------------------


def _run_in_bundle(pg: PlacementGroup, bundle_index: int, remote_fn: Any, *args: Any) -> Any:  # noqa: ANN401
    """Schedule *remote_fn* into ``pg``'s bundle *bundle_index* and return the result."""
    import ray
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    return ray.get(
        remote_fn.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=bundle_index
            ),
        ).remote(*args),
    )


def get_free_port_in_bundle(
    pg: PlacementGroup, bundle_index: int, start_port: int, get_next_free_port: bool = True
) -> int:
    """Find a free port on the node hosting ``pg``'s bundle *bundle_index*.

    The remote task is scheduled into the target bundle via
    ``PlacementGroupSchedulingStrategy``, so port availability is checked on
    the same node where the consuming actor will bind.
    """
    import ray

    @ray.remote(num_cpus=0)
    def _remote_get_free_port(start: int, get_next: bool) -> int:
        from nemo_curator.core.utils import get_free_port as _local_get_free_port

        return _local_get_free_port(start, get_next)

    return _run_in_bundle(pg, bundle_index, _remote_get_free_port, start_port, get_next_free_port)


def get_bundle_node_ip(pg: PlacementGroup, bundle_index: int) -> str:
    """Return the routable IP of the node hosting ``pg``'s bundle *bundle_index*.

    Used to resolve the master-addr for multi-node TP after ``pg.ready()``:
    the rank-0 actor will schedule into this same bundle, so its peers can
    connect to this IP.
    """
    import ray

    @ray.remote(num_cpus=0)
    def _remote_get_node_ip() -> str:
        return ray.util.get_node_ip_address()

    return _run_in_bundle(pg, bundle_index, _remote_get_node_ip)


# ---------------------------------------------------------------------------
# Actor spawning
# ---------------------------------------------------------------------------


def spawn_actor(  # noqa: PLR0913
    label: str,
    pg: PlacementGroup,
    bundle_index: int,
    *,
    num_gpus: int,
    command: list[str] | None = None,
    python_args: list[str] | None = None,
    runtime_dir: str | None = None,
    actor_name_prefix: str = "",
    subprocess_env: dict[str, str] | None = None,
    runtime_env: dict[str, Any] | None = None,
) -> ManagedSubprocess:
    """Create a detached Ray actor bound to ``pg``'s bundle *bundle_index*.

    Pass *command* for binary subprocesses (etcd, nats) or *python_args* for
    Python module invocations (``["-m", "dynamo.vllm", ...]``). When
    *python_args* is used, the actor prepends its own ``sys.executable`` --
    which inside a ``runtime_env`` points to the isolated virtualenv's
    Python, not the driver's. This ensures subprocesses load packages from
    the runtime_env (e.g. the correct vLLM version).

    The actor class is created per-call with ``__name__`` set to *label*,
    so the Ray dashboard shows descriptive names.

    The subprocess inherits the actor's ``os.environ`` (raylet env +
    ``runtime_env`` contributions). *subprocess_env* adds targeted overrides
    on top (e.g. ``ETCD_ENDPOINTS``, ``NATS_SERVER``, ``CUDA_VISIBLE_DEVICES``).

    Args:
        label: Human-readable label (used for actor naming, class naming, logs).
        pg: The placement group that owns the bundle.
        bundle_index: Which bundle in *pg* to pin this actor to.
        num_gpus: GPUs to reserve for the actor (must match the bundle's GPU count).
        command: Full subprocess command for binary processes (mutually
            exclusive with *python_args*).
        python_args: Arguments for a Python subprocess; actor prepends
            ``sys.executable``.
        runtime_dir: Directory for log files. ``None`` discards logs.
        actor_name_prefix: Prefix for the detached actor name (used for
            orphan cleanup / dashboard grouping).
        subprocess_env: Extra env vars for the subprocess (applied as overrides).
        runtime_env: Ray runtime environment for the actor. Merged with
            ``_NOSET_CUDA_RUNTIME_ENV`` so the NOSET flag is always set.
    """
    if (command is None) == (python_args is None):
        msg = "spawn_actor requires exactly one of 'command' or 'python_args'"
        raise ValueError(msg)

    import ray
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    actor_cls = _define_subprocess_actor(label)

    log_file = os.path.join(runtime_dir, f"{label}.log") if runtime_dir else None
    actor_name = f"{actor_name_prefix}_{label}" if actor_name_prefix else label

    if runtime_env:
        merged_runtime_env = {**runtime_env}
        user_env_vars = runtime_env.get("env_vars", {})
        merged_runtime_env["env_vars"] = {**user_env_vars, **_NOSET_CUDA_RUNTIME_ENV["env_vars"]}
    else:
        merged_runtime_env = _NOSET_CUDA_RUNTIME_ENV

    actor = actor_cls.options(
        name=actor_name,
        lifetime="detached",
        num_gpus=num_gpus,
        runtime_env=merged_runtime_env,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=bundle_index,
            placement_group_capture_child_tasks=True,
        ),
    ).remote()

    actual_env = dict(subprocess_env or {})
    if num_gpus > 0:
        gpu_ids = ray.get(actor.get_assigned_gpus.remote())
        if gpu_ids:
            actual_env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

    status = ray.get(actor.initialize.remote(command, actual_env, log_file, python_args=python_args))
    run_ref = actor.run.remote()

    logger.info(f"Launching {label} on bundle {bundle_index} (actor: {actor_name}, gpus={num_gpus})")
    return ManagedSubprocess(label=label, actor=actor, run_ref=run_ref, log_file=status.get("log_file"))


# ---------------------------------------------------------------------------
# Graceful teardown + orphan cleanup
# ---------------------------------------------------------------------------


def graceful_stop_actors(
    labeled_actors: list[tuple[str, ActorHandle]],
    *,
    timeout_s: float | None = None,
) -> None:
    """Stop many detached actors in parallel.

    Prefer ``ManagedSubprocess.stop`` / ``ManagedSubprocess.stop_many``.
    This primitive is for the raw-actor case (e.g. the reconnecting
    driver looking up orphaned actors by name).

    Kicks off every ``actor.stop.remote()`` first so subprocess teardown
    happens concurrently, then waits on all of them with a shared deadline
    and force-kills anything that didn't drain in time.
    """
    if not labeled_actors:
        return
    import ray

    wait = timeout_s if timeout_s is not None else _SIGTERM_WAIT_S + _SIGKILL_WAIT_S + 5
    refs = [actor.stop.remote() for _, actor in labeled_actors]
    with contextlib.suppress(Exception):
        ray.wait(refs, num_returns=len(refs), timeout=wait)
    for (label, actor), ref in zip(labeled_actors, refs, strict=True):
        try:
            ray.get(ref, timeout=0)
        except Exception:  # noqa: BLE001 - any failure here means force-kill
            logger.debug(f"{label} actor stop() did not drain in time, force-killing")
        with contextlib.suppress(Exception):
            ray.kill(actor, no_restart=True)


def remove_named_pgs_with_prefix(prefix: str) -> int:
    """Remove all placement groups in the current namespace whose name starts with *prefix*.

    Requires a live Ray connection on the current driver. Intended for orphan
    cleanup after a driver restart: since PGs are namespace-scoped and named,
    a reconnecting driver (with matching ``namespace=``) can find and reap
    leftover state from a prior session. Removing a PG forcibly kills all
    actors scheduled into it, releasing the reserved resources.

    Returns the number of PGs removed.
    """
    import ray
    from ray.util.placement_group import placement_group_table

    try:
        table = placement_group_table()
    except Exception:  # noqa: BLE001
        logger.debug("Could not list placement groups for orphan cleanup")
        return 0

    # placement_group_table may return {pg_id: meta} or a list of metas;
    # normalize to an iterable of metas.
    metas = table.values() if isinstance(table, dict) else list(table)

    removed = 0
    for meta in metas:
        name = meta.get("name") or ""
        state = meta.get("state", "")
        if not name.startswith(prefix):
            continue
        if state == "REMOVED":
            continue
        pg_id = meta.get("placement_group_id") or meta.get("id")
        if pg_id is None:
            continue
        try:
            pg = ray.util.get_placement_group(name)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Could not look up orphaned PG {name!r}: {exc}")
            continue
        with contextlib.suppress(Exception):
            ray.util.remove_placement_group(pg)
            removed += 1
            logger.debug(f"Removed orphaned placement group: {name}")
    if removed:
        logger.info(f"Removed {removed} orphaned placement group(s) with prefix '{prefix}'")
    return removed
