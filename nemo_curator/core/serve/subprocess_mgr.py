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

Every subprocess is launched with ``start_new_session=True``, so it
becomes its own session and process-group leader with ``pgid == pid``.
For the Dynamo -> vLLM path we care about, descendants are created via
Python ``multiprocessing`` without calling ``setsid()``, so they stay in
that process group.

Teardown is driver-initiated (see ``graceful_stop_actors``): SIGTERM on
the subprocess group -> bounded wait -> escalated SIGKILL on the same
group -> ``ray.kill`` for the actor itself. ``ray.kill`` on its own
SIGKILLs the actor process but leaves the launcher subprocess (which
was Popen'd by the actor) reparented to init and alive, so the
killpg-first ordering is required; it is not about the atexit hook.

Backends that use the ``with ray.init()`` attach/detach pattern around
start/stop must refresh stored ``ActorHandle`` references between the
two sessions -- see ``reacquire_detached_actor_handles`` below. The
companion orphan sweeper ``sweep_orphan_actors_by_prefix`` matches
``remove_named_pgs_with_prefix`` in ``placement``.
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
# Naming
# ---------------------------------------------------------------------------


def _detached_actor_name(prefix: str, label: str) -> str:
    """Assemble the Ray detached-actor name for a given *prefix* and *label*.

    Both ``ManagedSubprocess.spawn`` (when naming a new actor) and
    ``reacquire_detached_actor_handles`` (when re-looking it up in a later
    session) must agree on this formula, so it lives in one place.
    """
    return f"{prefix}_{label}" if prefix else label


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SubprocessError(RuntimeError):
    """Raised when a managed subprocess fails during inference server lifecycle.

    Carries an optional ``debug_context`` dict so callers can surface
    subprocess logs, PIDs, or any other diagnostic state alongside the
    user-facing message.
    """

    def __init__(self, message: str, debug_context: dict | None = None):
        super().__init__(message)
        self.debug_context = debug_context or {}


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
        for Python module invocations (``["-m", "dynamo.vllm", ...]``).
        When *python_args* is used, the actor prepends its own ``sys.executable``
        -- which inside a ``runtime_env`` points to the isolated virtualenv's
        Python, not the driver's. This ensures subprocesses load packages
        from the runtime_env.

        The actor class is created per-call with ``__name__`` set to *label*,
        so the Ray dashboard shows descriptive names.

        The subprocess inherits the actor's ``os.environ`` (raylet env +
        ``runtime_env`` contributions). *subprocess_env* adds targeted
        overrides on top (e.g. ``CUDA_VISIBLE_DEVICES``).

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
        actor_name = _detached_actor_name(actor_name_prefix, label)

        if runtime_env:
            merged_runtime_env = {**runtime_env}
            user_env_vars = runtime_env.get("env_vars", {})
            merged_runtime_env["env_vars"] = {**user_env_vars, **NOSET_CUDA_RUNTIME_ENV["env_vars"]}
        else:
            merged_runtime_env = {**NOSET_CUDA_RUNTIME_ENV}

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

        # One round-trip: the actor injects CUDA_VISIBLE_DEVICES from its
        # Ray-assigned accelerator IDs inside initialize() itself.
        ray.get(actor.initialize.remote(command, subprocess_env or {}, log_file, python_args=python_args))
        run_ref = actor.run.remote()

        logger.info(f"Launching {label} on bundle {bundle_index} (actor: {actor_name}, gpus={num_gpus})")
        return cls(label=label, actor=actor, run_ref=run_ref)

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
        """Best-effort graceful stop: reap the subprocess group, then kill the actor."""
        type(self).stop_many([self], timeout_s=timeout_s)

    @classmethod
    def stop_many(cls, procs: list[ManagedSubprocess], *, timeout_s: float | None = None) -> None:
        """Stop many managed subprocesses in parallel.

        Kicks off every ``actor.stop.remote()`` first so subprocess
        teardown happens concurrently, then waits on all of them with a
        shared deadline before tearing down the actors.
        """
        graceful_stop_actors([(p.label, p.actor) for p in procs], timeout_s=timeout_s)


# ---------------------------------------------------------------------------
# Subprocess helpers + Ray actor class factory
# ---------------------------------------------------------------------------


def _reap_process_group(pgid: int, *, sigterm_wait: float = SIGTERM_WAIT_S) -> None:
    """SIGTERM the whole process group, wait briefly, then SIGKILL.

    Because every managed subprocess is launched with
    ``start_new_session=True``, ``pgid == pid`` for the launcher and every
    descendant that did not explicitly call ``setsid()`` remains in that
    group. The group persists while any member is alive, so this still
    works even if the launcher itself already exited.
    """
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + sigterm_wait
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)


def _stop_subprocess(proc: subprocess.Popen, sigterm_wait: float = SIGTERM_WAIT_S) -> int | None:
    """Reap the subprocess's process group and return its exit code.

    Assumes ``proc`` was launched with ``start_new_session=True``, so
    ``proc.pid`` is the process-group id to target.
    """
    _reap_process_group(proc.pid, sigterm_wait=sigterm_wait)
    with contextlib.suppress(Exception):
        proc.wait(timeout=SIGKILL_WAIT_S)
    return proc.returncode


def _define_subprocess_actor(actor_type: str = "SubprocessActor") -> type:  # noqa: C901
    """Return a Ray remote actor class named *actor_type*.

    Each call produces a class whose ``__name__`` and ``__qualname__`` are
    set to *actor_type* so the Ray dashboard shows descriptive labels
    (e.g. ``Dynamo_ETCD``, ``Dynamo_DP0_Qwen3-0.6B``).

    Lifecycle:

    1. ``initialize`` Popens the subprocess with ``start_new_session=True``
       so the whole subtree shares one process group (``pgid == popen.pid``).
    2. ``run`` blocks until the subprocess exits; the returned ObjectRef
       resolves on exit so ``ray.wait`` can be used for liveness.
    3. ``stop`` / ``force_sigkill_subprocess`` reap that process group.

    Teardown is driver-initiated via ``graceful_stop_actors`` so the
    actor can ``killpg`` its subprocess group before ``ray.kill`` hard-kills
    the actor process itself -- otherwise the Popen-launched child
    reparents to init and keeps running. A Python ``atexit`` hook is
    registered as a backup for clean actor exits (e.g. a normal raylet
    shutdown); it does not fire on ``ray.kill`` / PG removal / SIGKILL.
    """
    import ray

    class _SubprocessActor:
        """Manages a subprocess on a Ray node with optional file-based logging.

        ``max_concurrency=4`` lets ``run()`` block on the subprocess while
        other methods (``is_alive``, ``read_log_tail``, ``stop``,
        ``force_sigkill_subprocess``) stay responsive -- importantly,
        ``force_sigkill_subprocess`` must be able to preempt a stuck
        ``stop()``.
        """

        def __init__(self) -> None:
            self._proc: Any = None
            self._log_fh: Any = None
            self._log_file: str | None = None
            self._cleanup_registered = False

        def initialize(
            self,
            command: list[str] | None,
            subprocess_env: dict[str, str],
            log_file: str | None = None,
            *,
            python_args: list[str] | None = None,
        ) -> int:
            """Launch the subprocess with the actor's env + *subprocess_env* overrides.

            Pass *command* for binary subprocesses (etcd, nats) or
            *python_args* for Python module invocations (``-m dynamo.vllm``).
            With *python_args*, the actor prepends its own ``sys.executable``
            so the subprocess uses the ``runtime_env``'s Python, not the driver's.

            Injects ``CUDA_VISIBLE_DEVICES`` from the actor's Ray-assigned
            accelerator IDs so the caller (``ManagedSubprocess.spawn``)
            does not need a separate round-trip before this call.

            Returns the subprocess pid.
            """
            import subprocess as _sp
            import sys as _sys

            import ray as _ray

            if python_args is not None:
                command = [_sys.executable, *python_args]

            merged_env = {**os.environ, **subprocess_env}
            gpu_ids = _ray.get_runtime_context().get_accelerator_ids().get("GPU", [])
            if gpu_ids:
                merged_env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

            self._log_file = log_file
            if log_file:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                self._log_fh = open(log_file, "w")  # noqa: SIM115
                stdout: Any = self._log_fh
            else:
                stdout = _sp.DEVNULL

            try:
                self._proc = _sp.Popen(  # noqa: S603
                    command,
                    env=merged_env,
                    stdout=stdout,
                    stderr=_sp.STDOUT,
                    start_new_session=True,
                )
            except Exception:
                if self._log_fh is not None:
                    with contextlib.suppress(Exception):
                        self._log_fh.close()
                    self._log_fh = None
                raise

            if not self._cleanup_registered:
                atexit.register(self._cleanup)
                self._cleanup_registered = True

            return self._proc.pid

        def run(self) -> int:
            """Block until the subprocess exits. Returns the exit code."""
            if self._proc is None:
                return -1
            return self._proc.wait()

        def pid(self) -> int:
            return self._proc.pid if self._proc else -1

        def is_alive(self) -> bool:
            return self._proc is not None and self._proc.poll() is None

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
            """Gracefully stop the subprocess tree via ``killpg``."""
            rc: int | None = None
            if self._proc is not None:
                rc = _stop_subprocess(self._proc, sigterm_wait)
            if self._log_fh:
                with contextlib.suppress(Exception):
                    self._log_fh.close()
            return rc

        def force_sigkill_subprocess(self) -> None:
            """SIGKILL the whole process group without a grace period."""
            if self._proc is not None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self._proc.pid, signal.SIGKILL)

        def _cleanup(self) -> None:
            """atexit hook: last-line-of-defense subprocess reap."""
            with contextlib.suppress(Exception):
                if self._proc is not None:
                    _stop_subprocess(self._proc, sigterm_wait=3)
            with contextlib.suppress(Exception):
                if self._log_fh:
                    self._log_fh.close()

    _SubprocessActor.__name__ = actor_type
    _SubprocessActor.__qualname__ = actor_type
    return ray.remote(num_cpus=1, num_gpus=0, max_concurrency=4)(_SubprocessActor)


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

    Escalation order per actor:

    1. ``actor.stop.remote()`` -- actor-driven SIGTERM on the subprocess
       group with a bounded wait, then SIGKILL if needed.
    2. If stop did not drain in time, ``actor.force_sigkill_subprocess.remote()``
       -- host-side SIGKILL on the subprocess group. This runs on the
       actor's node (because actor methods always do), so it works
       multi-node without any driver-side connectivity assumption.
    3. ``ray.kill(actor)`` -- tear down the actor itself.

    Step 2 matters because ``ray.kill`` SIGKILLs the actor process; its
    Popen child (the launcher) reparents to init and keeps running, and
    that launcher's own children (e.g. vLLM EngineCore) stay alive too.
    Sending ``killpg`` first ensures the whole subprocess tree is gone
    before the actor itself is torn down. Each dispatch is guarded
    individually so a single invalid handle (e.g. a stale
    cross-``ray.init()``-session handle) does not abort the batch.
    """
    if not labeled_actors:
        return
    import ray

    wait = timeout_s if timeout_s is not None else SIGTERM_WAIT_S + SIGKILL_WAIT_S + 5
    refs = [_dispatch_actor_stop(label, actor) for label, actor in labeled_actors]
    valid_refs = [r for r in refs if r is not None]
    if valid_refs:
        with contextlib.suppress(Exception):
            ray.wait(valid_refs, num_returns=len(valid_refs), timeout=wait)

    for (label, actor), ref in zip(labeled_actors, refs, strict=True):
        if not _actor_stop_drained(ref):
            logger.debug(f"{label} actor.stop did not drain; escalating to host-side SIGKILL")
            with contextlib.suppress(Exception):
                ray.get(actor.force_sigkill_subprocess.remote(), timeout=SIGKILL_WAIT_S)
        with contextlib.suppress(Exception):
            ray.kill(actor, no_restart=True)


def _dispatch_actor_stop(label: str, actor: ActorHandle) -> ObjectRef | None:
    """Best-effort dispatch of ``actor.stop.remote()`` for one actor."""
    try:
        return actor.stop.remote()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"{label} actor.stop.remote() dispatch failed: {exc}")
        return None


def _actor_stop_drained(ref: ObjectRef | None) -> bool:
    """Return True iff the actor-side ``stop()`` completed successfully."""
    if ref is None:
        return False
    import ray

    try:
        ray.get(ref, timeout=0)
    except Exception:  # noqa: BLE001
        return False
    return True


# ---------------------------------------------------------------------------
# Cross-session helpers: refreshing handles + sweeping orphans by prefix
# ---------------------------------------------------------------------------


def reacquire_detached_actor_handles(
    procs: list[ManagedSubprocess],
    *,
    actor_name_prefix: str,
    namespace: str,
) -> list[ManagedSubprocess]:
    """Refresh each ``proc.actor`` in place by re-looking up its detached actor.

    ``ActorHandle`` objects are pinned to the Ray job that created them, so
    any handle stored across a ``with ray.init()`` boundary raises
    ``ActorHandleNotFoundError`` on ``.remote()``. Backends that use the
    start/stop attach-detach pattern must re-acquire their handles inside
    the stop-side ``with ray.init()`` before calling any actor method.

    Actor names follow the ``ManagedSubprocess.spawn`` convention:
    ``f"{actor_name_prefix}_{label}"``, or just ``label`` when the prefix
    is empty.

    Returns a filtered list containing only procs whose actor was
    successfully re-acquired; entries whose actor is already gone are
    dropped so callers can safely pass the result to
    ``ManagedSubprocess.stop_many``.
    """
    import ray

    refreshed: list[ManagedSubprocess] = []
    for proc in procs:
        actor_name = _detached_actor_name(actor_name_prefix, proc.label)
        try:
            proc.actor = ray.get_actor(actor_name, namespace=namespace)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Could not re-acquire detached actor {actor_name!r}: {exc}")
            continue
        refreshed.append(proc)
    return refreshed


def sweep_orphan_actors_by_prefix(*, prefix: str, namespace: str) -> int:
    """Graceful-stop any ALIVE detached actor whose name starts with *prefix*.

    Intended for orphan cleanup after a driver restart: detached actors
    outlive the driver that created them, so a reconnecting driver (in the
    same *namespace*) can list them by name, re-acquire handles, and reap
    them + their subprocess trees via ``graceful_stop_actors``. Returns the
    number of orphans reaped. Mirrors ``remove_named_pgs_with_prefix`` in
    ``placement``.
    """
    try:
        from ray.util.state import list_actors
    except ImportError:
        return 0

    import ray

    try:
        alive = list_actors(filters=[("state", "=", "ALIVE")])
    except Exception:  # noqa: BLE001
        logger.debug("Could not list actors for orphan sweep")
        return 0

    stale: list[tuple[str, ActorHandle]] = []
    for entry in alive:
        name = getattr(entry, "name", "") or ""
        if not name.startswith(prefix):
            continue
        try:
            actor = ray.get_actor(name, namespace=namespace)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Orphan-actor lookup for {name!r} failed: {exc}")
            continue
        stale.append((name, actor))

    if stale:
        logger.info(f"Sweeping {len(stale)} orphan actor(s) with prefix {prefix!r}")
        graceful_stop_actors(stale)
    return len(stale)
