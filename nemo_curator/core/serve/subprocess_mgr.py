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

"""Ray-actor + subprocess lifecycle management.

``ManagedSubprocess`` is the primary handle: it carries the detached
Ray actor, the long-running ``run()`` ObjectRef, and the optional log
path, and exposes the lifecycle operations (``spawn``, ``stop``,
``stop_many``, ``is_alive``, ``pid``, ``read_log_tail``, ``wait``) so
callers never have to hand-write ``ray.get(actor.<something>.remote())``
plumbing.

Each spawned actor owns one subprocess launched with
``start_new_session=True`` so the child becomes a process-group leader.
Teardown is SIGTERM -> (bounded wait) -> SIGKILL on the process group
so vLLM's torch.distributed workers get reaped along with the main
process. ``__ray_terminate__`` + ``atexit`` are both wired so the
subprocess tree goes away even if Ray hard-kills the actor.

Placement-group construction, bundle-level helpers, and the orphan PG
sweep live in ``placement``.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import signal
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    import subprocess

    from ray import ObjectRef
    from ray.actor import ActorHandle
    from ray.util.placement_group import PlacementGroup

from nemo_curator.core.serve.constants import (
    NOSET_CUDA_RUNTIME_ENV,
    SIGKILL_WAIT_S,
    SIGTERM_WAIT_S,
)

# ---------------------------------------------------------------------------
# ManagedSubprocess: Ray actor + subprocess handle + lifecycle
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

    @classmethod
    def spawn(  # noqa: PLR0913
        cls,
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
        """Create a detached Ray actor bound to ``pg``'s bundle *bundle_index* and launch its subprocess.

        Pass *command* for binary subprocesses (etcd, nats) or *python_args*
        for Python module invocations (``["-m", "dynamo.vllm", ...]``). When
        *python_args* is used, the actor prepends its own ``sys.executable``
        -- which inside a ``runtime_env`` points to the isolated virtualenv's
        Python, not the driver's. This ensures subprocesses load packages
        from the runtime_env (e.g. the correct vLLM version).

        The actor class is created per-call with ``__name__`` set to *label*,
        so the Ray dashboard shows descriptive names.

        The subprocess inherits the actor's ``os.environ`` (raylet env +
        ``runtime_env`` contributions). *subprocess_env* adds targeted
        overrides on top (e.g. ``ETCD_ENDPOINTS``, ``NATS_SERVER``,
        ``CUDA_VISIBLE_DEVICES``).

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
                ``NOSET_CUDA_RUNTIME_ENV`` so the NOSET flag is always set.
        """
        if (command is None) == (python_args is None):
            msg = "ManagedSubprocess.spawn requires exactly one of 'command' or 'python_args'"
            raise ValueError(msg)

        import ray
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        actor_cls = _define_subprocess_actor(label)

        log_file = os.path.join(runtime_dir, f"{label}.log") if runtime_dir else None
        actor_name = f"{actor_name_prefix}_{label}" if actor_name_prefix else label

        if runtime_env:
            merged_runtime_env = {**runtime_env}
            user_env_vars = runtime_env.get("env_vars", {})
            merged_runtime_env["env_vars"] = {**user_env_vars, **NOSET_CUDA_RUNTIME_ENV["env_vars"]}
        else:
            merged_runtime_env = NOSET_CUDA_RUNTIME_ENV

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
        return cls(label=label, actor=actor, run_ref=run_ref, log_file=status.get("log_file"))

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
            msg = f"{self.label}: run_ref is None; subprocess was never launched via ManagedSubprocess.spawn"
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


# ---------------------------------------------------------------------------
# Subprocess reaper + Ray actor class factory
# ---------------------------------------------------------------------------


def _stop_subprocess(proc: subprocess.Popen, sigterm_wait: float = SIGTERM_WAIT_S) -> int | None:
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
        proc.wait(timeout=SIGKILL_WAIT_S)
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

        def stop(self, sigterm_wait: float = SIGTERM_WAIT_S) -> int | None:
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
# Small helpers
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
# Graceful teardown for raw actor handles
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

    wait = timeout_s if timeout_s is not None else SIGTERM_WAIT_S + SIGKILL_WAIT_S + 5
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
