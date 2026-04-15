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

"""Generic subprocess management for inference server backends.

Provides Ray-actor-based subprocess lifecycle management, GPU placement
planning, environment variable propagation, and remote port discovery.
Backend implementations compose these primitives to launch and monitor
their specific processes.
"""

from __future__ import annotations

import contextlib
import os
import signal
import time
from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger

from nemo_curator.core.utils import get_free_port  # noqa: F401 - re-exported

# ---------------------------------------------------------------------------
# GPU placement planner
# ---------------------------------------------------------------------------


@dataclass
class NodeAllocation:
    """A single node's contribution to a replica's TP group.

    Attributes:
        node_id: Ray node ID.
        node_ip: Routable IP address of the node.
        num_gpus: Number of GPUs allocated on this node.
        node_rank: Rank within the TP group (0 = leader, 1+ = headless).
    """

    node_id: str
    node_ip: str
    num_gpus: int
    node_rank: int


@dataclass
class ReplicaPlan:
    """Placement plan for one model replica (may span multiple nodes).

    For single-node TP, ``ranks`` contains a single entry.  For multi-node
    TP (TP size > GPUs on any single node), ``ranks`` contains one entry
    per participating node.  Rank 0 is always the leader that runs the full
    worker; rank 1+ run headless workers coordinated via
    ``torch.distributed``.
    """

    replica_index: int
    ranks: list[NodeAllocation]

    @property
    def is_multi_node(self) -> bool:
        return len(self.ranks) > 1

    @property
    def nnodes(self) -> int:
        return len(self.ranks)

    @property
    def master_addr(self) -> str:
        """IP of the rank-0 (leader) node."""
        return self.ranks[0].node_ip

    @property
    def total_gpus(self) -> int:
        return sum(r.num_gpus for r in self.ranks)


def _get_gpu_inventory(
    head_node_id: str | None = None,
    nodes: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return per-node GPU information from the Ray cluster.

    Each entry: ``{"node_id": str, "node_ip": str, "num_gpus": int, "is_head": bool}``.

    Args:
        head_node_id: Optional node ID to tag as head in inventory.
        nodes: Pre-fetched ``ray.nodes()`` result to avoid a redundant API call.
    """
    import ray

    inventory = []
    for node in nodes or ray.nodes():
        if not node.get("Alive", False):
            continue
        resources = node.get("Resources", {})
        num_gpus = int(resources.get("GPU", 0))
        if num_gpus == 0:
            continue
        node_id = node["NodeID"]
        inventory.append(
            {
                "node_id": node_id,
                "node_ip": node["NodeManagerAddress"],
                "num_gpus": num_gpus,
                "is_head": node_id == head_node_id,
            }
        )
    return inventory


def _resolve_node_ip(node_id: str, nodes: list[dict[str, Any]] | None = None) -> str:
    """Get the routable IP address for a Ray node by its ID.

    Args:
        node_id: The Ray node ID to resolve.
        nodes: Pre-fetched ``ray.nodes()`` result to avoid a redundant API call.
    """
    import ray

    for node in nodes or ray.nodes():
        if node["NodeID"] == node_id and node.get("Alive", False):
            return node["NodeManagerAddress"]
    msg = f"Could not resolve IP for Ray node {node_id}"
    raise RuntimeError(msg)


def _ignore_head_node() -> bool:
    """Return True if ``CURATOR_IGNORE_RAY_HEAD_NODE`` is set."""
    return os.environ.get("CURATOR_IGNORE_RAY_HEAD_NODE", "").strip().lower() in ("1", "true", "yes")


def plan_replica_placement(
    num_replicas: int,
    tp_size: int,
    head_node_id: str | None = None,
    _inventory: list[dict[str, Any]] | None = None,
    _nodes: list[dict[str, Any]] | None = None,
) -> list[ReplicaPlan]:
    """Plan GPU allocation for replicas, supporting multi-node TP.

    For each replica, GPUs are allocated greedily across nodes:
    - If ``tp_size`` fits on a single node, the replica gets one rank (rank 0).
    - If ``tp_size`` exceeds any single node, the replica spans multiple nodes.
      Rank 0 runs the full worker; rank 1+ run headless workers.

    When ``CURATOR_IGNORE_RAY_HEAD_NODE`` is set, the head node is excluded
    from worker placement entirely.  Otherwise the head node is preferred
    for rank 0 and the remaining nodes are sorted stably by ``node_id``.

    Args:
        num_replicas: Number of independent model replicas.
        tp_size: Tensor parallel degree (GPUs per replica).
        head_node_id: Optional node ID to tag as head in inventory.
        _inventory: Pre-built inventory for testing (bypasses Ray call).
        _nodes: Pre-fetched ``ray.nodes()`` result passed to ``_get_gpu_inventory``.

    Returns:
        List of ``ReplicaPlan``, one per replica.
    """
    inventory = _inventory if _inventory is not None else _get_gpu_inventory(head_node_id, nodes=_nodes)
    if not inventory:
        msg = "No GPU nodes found in the Ray cluster."
        raise RuntimeError(msg)

    ignore_head = _ignore_head_node()
    if ignore_head:
        inventory = [n for n in inventory if not n["is_head"]]
        if not inventory:
            msg = "CURATOR_IGNORE_RAY_HEAD_NODE is set but no non-head GPU nodes are available."
            raise RuntimeError(msg)
    else:
        # Head node first for deterministic rank-0, then stable by node_id
        inventory.sort(key=lambda n: (not n["is_head"], n["node_id"]))

    total_gpus = sum(n["num_gpus"] for n in inventory)
    needed = num_replicas * tp_size
    if needed > total_gpus:
        msg = (
            f"Need {needed} GPUs ({num_replicas} replicas x {tp_size} TP) "
            f"but only {total_gpus} available across {len(inventory)} node(s)."
        )
        raise RuntimeError(msg)

    available = {n["node_id"]: n["num_gpus"] for n in inventory}

    plans: list[ReplicaPlan] = []
    for replica_idx in range(num_replicas):
        ranks = _place_single_replica(tp_size, inventory, available)
        plans.append(ReplicaPlan(replica_index=replica_idx, ranks=ranks))

    return plans


def _place_single_replica(
    tp_size: int,
    inventory: list[dict[str, Any]],
    available: dict[str, int],
) -> list[NodeAllocation]:
    """Place one replica, returning a list of ``NodeAllocation`` ranks.

    Placement strategy:
    1. **Single-node**: if any node has ``>= tp_size`` GPUs, place entirely
       on the first such node (greedy, head-first).
    2. **Multi-node (even split)**: vLLM requires each node in a multi-node
       TP group to have the same number of local GPUs (``tp_size // nnodes``).
       Find the smallest set of nodes that can each contribute an equal
       share of ``tp_size``.  Nodes with leftover GPUs below the per-node
       share are skipped — they cannot participate.

    Updates *available* in-place.
    """
    # --- Strategy 1: single-node ---
    for node in inventory:
        nid = node["node_id"]
        if available[nid] >= tp_size:
            available[nid] -= tp_size
            return [NodeAllocation(node_id=nid, node_ip=node["node_ip"], num_gpus=tp_size, node_rank=0)]

    # --- Strategy 2: multi-node even split ---
    # Collect candidate nodes in inventory order so head-node preference and
    # other deterministic placement rules are preserved.
    candidates = [(node, available[node["node_id"]]) for node in inventory if available[node["node_id"]] > 0]

    # Try increasing nnodes (2, 3, ...) to find the smallest even split.
    for nnodes in range(2, len(candidates) + 1):
        if tp_size % nnodes != 0:
            continue
        per_node = tp_size // nnodes

        # Pick the first `nnodes` candidates that have >= per_node GPUs.
        chosen: list[dict[str, Any]] = []
        for node, avail in candidates:
            if avail >= per_node:
                chosen.append(node)
            if len(chosen) == nnodes:
                break

        if len(chosen) == nnodes:
            ranks: list[NodeAllocation] = []
            for node_rank, node in enumerate(chosen):
                nid = node["node_id"]
                available[nid] -= per_node
                ranks.append(
                    NodeAllocation(
                        node_id=nid,
                        node_ip=node["node_ip"],
                        num_gpus=per_node,
                        node_rank=node_rank,
                    )
                )
            return ranks

    # No valid placement found.
    total_avail = sum(available.values())
    msg = (
        f"Cannot place TP={tp_size} replica: need an even split across nodes "
        f"but no valid combination found ({total_avail} GPUs available across "
        f"{sum(1 for v in available.values() if v > 0)} node(s))."
    )
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Subprocess management
# ---------------------------------------------------------------------------

_SIGTERM_WAIT_S = 10
_SIGKILL_WAIT_S = 5


@dataclass
class ManagedSubprocess:
    """Track a detached Ray actor and the subprocess it owns."""

    label: str
    actor: Any
    run_ref: Any | None = None
    log_file: str | None = None


def _stop_subprocess(proc: Any, sigterm_wait: float = _SIGTERM_WAIT_S) -> int | None:  # noqa: ANN401
    """SIGTERM -> wait -> SIGKILL a subprocess and its entire process group.

    Subprocesses are launched with ``start_new_session=True`` so they
    become process-group leaders.  Signaling the group (``os.killpg``)
    ensures child processes (e.g. vLLM torch.distributed workers) are
    also terminated rather than becoming orphans.
    """
    if proc.poll() is not None:
        return proc.returncode
    # start_new_session=True makes the child a process-group leader (pgid == pid).
    # Kill the entire group so grandchildren die too; fall back to just the
    # child if it somehow changed its group.
    pgid = None
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


_NOSET_CUDA_RUNTIME_ENV: dict[str, Any] = {"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}}


def build_worker_actor_name(
    model_name: str,
    replica_index: int,
    node_rank: int,
    tp_size: int,
    *,
    role: Literal["decode", "prefill"] | None = None,
) -> str:
    """Build a descriptive actor name for Ray dashboard visibility.

    Format: ``Dynamo_[<role>_]DP<n>[_TP<n>]_<model>``.

    Examples::

        build_worker_actor_name("Qwen3-0.6B", 0, 0, 1)
        # -> "Dynamo_DP0_Qwen3-0.6B"

        build_worker_actor_name("Qwen3-0.6B", 1, 0, 4)
        # -> "Dynamo_DP1_TP0_Qwen3-0.6B"

        build_worker_actor_name("Qwen3-0.6B", 0, 0, 2, role="decode")
        # -> "Dynamo_decode_DP0_TP0_Qwen3-0.6B"
    """
    # Use the last path component for HF identifiers (e.g. "Qwen/Qwen3-0.6B" -> "Qwen3-0.6B")
    short_name = model_name.rsplit("/", 1)[-1]
    parts = ["Dynamo"]
    if role:
        parts.append(role)
    parts.append(f"DP{replica_index}")
    if tp_size > 1:
        parts.append(f"TP{node_rank}")
    parts.append(short_name)
    return "_".join(parts)


def _define_subprocess_actor(actor_type: str = "SubprocessActor") -> type:  # noqa: C901
    """Return a Ray remote actor class named *actor_type*.

    Each call produces a class whose ``__name__`` and ``__qualname__`` are
    set to *actor_type*, so the Ray dashboard shows descriptive labels
    (e.g. ``etcd``, ``nats``, ``Dynamo_DP0_Qwen3-0.6B``) instead of a generic
    ``_SubprocessActor``.

    A single actor class used for all backend subprocesses (etcd, NATS,
    frontend, vLLM workers, etc.).  GPU resources are configured per-instance
    via ``.options(num_gpus=...)``.

    Four-phase initialization (orchestrated by ``spawn_actor``):

    1. **Create** the actor (lightweight ``__init__``).
    2. **Discover GPUs** via ``get_assigned_gpus()`` -- returns Ray-assigned
       accelerator IDs.
    3. **Launch subprocess** via ``initialize(command, subprocess_env,
       log_file)`` or ``initialize(python_args=..., subprocess_env,
       log_file)`` -- applies *subprocess_env* as overrides on top of
       the actor's inherited ``os.environ``, then starts the process.
       When *python_args* is given, the actor prepends its own
       ``sys.executable`` so the subprocess uses the runtime_env's
       Python, not the driver's.
    4. **Start run()** -- the returned ``ObjectRef`` resolves on exit,
       enabling ``ray.wait()``-based liveness detection.

    The ``run()`` method blocks until the subprocess exits and is intended
    to be called as ``actor.run.remote()`` -- the returned ``ObjectRef``
    resolves on exit, enabling ``ray.wait()``-based liveness detection.
    """
    import ray

    class _SubprocessActor:
        """Manages a subprocess on a Ray node with optional file-based logging.

        ``max_concurrency=2`` allows ``run()`` to block waiting for the
        subprocess while other methods (``is_alive``, ``read_log_tail``,
        ``stop``) remain responsive.
        """

        def __init__(self) -> None:
            self._proc = None
            self._log_fh = None
            self._log_file = None

        def get_assigned_gpus(self) -> list[str]:
            """Return Ray-assigned accelerator IDs for this actor."""
            import ray as _ray

            return _ray.get_runtime_context().get_accelerator_ids().get("GPU", [])

        def initialize(
            self,
            command: list[str] | None,
            subprocess_env: dict[str, str],
            log_file: str | None = None,
            *,
            python_args: list[str] | None = None,
        ) -> dict:
            """Launch the subprocess with the actor's env plus *subprocess_env* overrides.

            The actor inherits its base environment from the raylet.
            *subprocess_env* adds explicit overrides on top (e.g.
            ``CUDA_VISIBLE_DEVICES``, ``ETCD_ENDPOINTS``).

            Pass *command* for binary subprocesses (etcd, nats) or
            *python_args* for Python module invocations (``-m dynamo.vllm``).
            When *python_args* is given, the actor prepends its own
            ``sys.executable`` — which inside a ``runtime_env`` points to
            the isolated virtualenv's Python, not the driver's.

            Returns:
                ``{"pid": int, "log_file": str | None}``
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

            return {"pid": self._proc.pid, "log_file": log_file}

        def run(self) -> int:
            """Block until the subprocess exits.  Returns the exit code.

            Intended to be called as ``actor.run.remote()`` -- the returned
            ``ObjectRef`` resolves when the process terminates, enabling
            ``ray.wait()``-based liveness detection.
            """
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
            """Read the last *num_bytes* of the log file.

            Flushes the write handle first so buffered output is visible.
            Called remotely by the driver for crash diagnosis.
            """
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
            if self._proc is None:
                return None
            rc = _stop_subprocess(self._proc, sigterm_wait)
            if self._log_fh:
                self._log_fh.close()
            return rc

    # Set name BEFORE applying @ray.remote so Ray captures the descriptive name.
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


def _engine_kwargs_to_cli_flags(engine_kwargs: dict[str, Any]) -> list[str]:
    """Convert engine_kwargs dict to vLLM CLI flags.

    Example: ``{"tensor_parallel_size": 4, "enforce_eager": True}``
    becomes ``["--tensor-parallel-size", "4", "--enforce-eager"]``
    """
    import json

    flags: list[str] = []
    for key, value in engine_kwargs.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                flags.append(flag)
        elif isinstance(value, (dict, list)):
            flags.extend([flag, json.dumps(value)])
        else:
            flags.extend([flag, str(value)])
    return flags


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


def _kill_actor(ray_mod: Any, label: str, actor: Any) -> None:  # noqa: ANN401
    """Best-effort stop + kill of a single detached actor."""
    try:
        ray_mod.get(actor.stop.remote(), timeout=_SIGTERM_WAIT_S + _SIGKILL_WAIT_S + 5)
    except Exception:  # noqa: BLE001
        logger.debug(f"{label} actor stop() failed, force-killing")
    with contextlib.suppress(Exception):
        ray_mod.kill(actor, no_restart=True)


# ---------------------------------------------------------------------------
# Remote port discovery
# ---------------------------------------------------------------------------


def get_free_port_on_node(node_id: str, start_port: int, get_next_free_port: bool = True) -> int:
    """Find a free port on a specific Ray node via a remote task.

    Unlike ``get_free_port`` which checks the local (driver) machine, this
    schedules a lightweight Ray task on *node_id* to verify port
    availability where the service will actually bind.

    Requires an active Ray connection (call inside ``ray.init()`` context).
    """
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    @ray.remote(num_cpus=0)
    def _remote_get_free_port(start: int, get_next: bool) -> int:
        from nemo_curator.core.utils import get_free_port as _local_get_free_port

        return _local_get_free_port(start, get_next)

    return ray.get(
        _remote_get_free_port.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
        ).remote(start_port, get_next_free_port),
    )


# ---------------------------------------------------------------------------
# Actor spawning
# ---------------------------------------------------------------------------


def spawn_actor(  # noqa: PLR0913
    label: str,
    node_alloc: NodeAllocation,
    *,
    command: list[str] | None = None,
    python_args: list[str] | None = None,
    runtime_dir: str | None = None,
    actor_name_prefix: str = "",
    subprocess_env: dict[str, str] | None = None,
    runtime_env: dict[str, Any] | None = None,
) -> ManagedSubprocess:
    """Create a detached Ray actor that runs a subprocess.

    Handles both GPU workers and non-GPU infra actors (etcd, NATS,
    frontend).  For GPU actors, discovers Ray-assigned GPU IDs and sets
    ``CUDA_VISIBLE_DEVICES`` explicitly in *subprocess_env*.

    Pass *command* for binary subprocesses (etcd, nats) or *python_args*
    for Python module invocations (``["-m", "dynamo.vllm", ...]``).
    When *python_args* is used, the **actor** prepends its own
    ``sys.executable`` — which inside a ``runtime_env`` points to the
    isolated virtualenv's Python, not the driver's.  This ensures
    subprocesses load packages from the runtime_env (e.g. the correct
    vLLM version) rather than the base environment.

    The actor class is created per-call with ``__name__`` set to *label*,
    so the Ray dashboard shows descriptive names (e.g. ``Dynamo_ETCD``,
    ``Dynamo_DP0_Qwen3-0.6B``) instead of a generic ``_SubprocessActor``.

    The subprocess inherits the actor's ``os.environ`` (which comes from
    the raylet and any ``runtime_env`` settings).  *subprocess_env* adds
    targeted overrides on top (e.g. ``ETCD_ENDPOINTS``, ``NATS_SERVER``,
    ``CUDA_VISIBLE_DEVICES``).

    Args:
        label: Human-readable label (used for actor naming, class naming,
            and logs).
        node_alloc: Node + GPU allocation for this actor.
        command: Full subprocess command for binary processes (mutually
            exclusive with *python_args*).
        python_args: Arguments for a Python subprocess (e.g.
            ``["-m", "dynamo.vllm", "--model", ...]``).  The actor
            prepends ``sys.executable`` at launch time (mutually
            exclusive with *command*).
        runtime_dir: Directory for log files.  If ``None``, logs are
            discarded.
        actor_name_prefix: Prefix for the detached actor name (used for
            orphan cleanup).
        subprocess_env: Extra env vars for the subprocess (applied as
            overrides on top of the actor's inherited ``os.environ``).
        runtime_env: Ray runtime environment for the actor (e.g. ``pip``
            packages, ``working_dir``).  Merged with the internal
            ``_NOSET_CUDA_RUNTIME_ENV`` so that
            ``RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`` is always set.
    """
    if (command is None) == (python_args is None):
        msg = "spawn_actor requires exactly one of 'command' or 'python_args'"
        raise ValueError(msg)

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    actor_cls = _define_subprocess_actor(label)

    log_file = os.path.join(runtime_dir, f"{label}.log") if runtime_dir else None
    actor_name = f"{actor_name_prefix}_{label}" if actor_name_prefix else label

    # Merge caller-provided runtime_env with _NOSET_CUDA_RUNTIME_ENV.
    # NOSET env vars always win so that Dynamo can manage GPUs explicitly.
    if runtime_env:
        merged_runtime_env = {**runtime_env}
        user_env_vars = runtime_env.get("env_vars", {})
        merged_runtime_env["env_vars"] = {**user_env_vars, **_NOSET_CUDA_RUNTIME_ENV["env_vars"]}
    else:
        merged_runtime_env = _NOSET_CUDA_RUNTIME_ENV

    # Phase 1: create actor -- lightweight __init__, no subprocess yet
    actor = actor_cls.options(
        name=actor_name,
        lifetime="detached",
        num_gpus=node_alloc.num_gpus,
        runtime_env=merged_runtime_env,
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_alloc.node_id, soft=False),
    ).remote()

    # Phase 2: discover GPUs (if any) and add to subprocess env
    actual_env = dict(subprocess_env or {})
    if node_alloc.num_gpus > 0:
        gpu_ids = ray.get(actor.get_assigned_gpus.remote())
        if gpu_ids:
            actual_env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

    # Phase 3: launch subprocess
    status = ray.get(actor.initialize.remote(command, actual_env, log_file, python_args=python_args))

    # Phase 4: start run() -- the ObjectRef resolves when the subprocess exits
    run_ref = actor.run.remote()

    logger.info(f"Launching {label} on {node_alloc.node_ip} (actor: {actor_name}, gpus={node_alloc.num_gpus})")
    return ManagedSubprocess(label=label, actor=actor, run_ref=run_ref, log_file=status.get("log_file"))


# ---------------------------------------------------------------------------
# Actor lifecycle
# ---------------------------------------------------------------------------


def kill_orphaned_actors(ray_mod: Any, prefix: str) -> None:  # noqa: ANN401
    """Kill any live actors whose name starts with *prefix*.

    Searches across all Ray namespaces so it finds actors created in
    anonymous namespaces (e.g. after a Jupyter kernel restart).
    """
    from ray.util.state import list_actors

    try:
        all_actors = list_actors(filters=[("state", "=", "ALIVE")], limit=500, timeout=10)
    except Exception:  # noqa: BLE001
        logger.debug("Could not list Ray actors for orphan cleanup")
        return

    orphans = [a for a in all_actors if a.get("name", "").startswith(prefix)]
    if not orphans:
        return

    logger.info(f"Found {len(orphans)} orphaned actor(s) with prefix '{prefix}' -- cleaning up")
    for actor_info in orphans:
        name = actor_info["name"]
        ns = actor_info.get("ray_namespace")
        with contextlib.suppress(Exception):
            handle = ray_mod.get_actor(name, namespace=ns)
            _kill_actor(ray_mod, name, handle)
            logger.debug(f"Killed orphaned actor: {name}")
