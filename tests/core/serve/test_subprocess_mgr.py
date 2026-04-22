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
import os
import re
import socket
from typing import TYPE_CHECKING
from unittest import mock

import psutil
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from nemo_curator.core.serve.constants import PLACEMENT_GROUP_READY_TIMEOUT_S
from nemo_curator.core.serve.placement import (
    build_pg,
    build_replica_pg,
    get_free_port_in_bundle,
    plan_replica_bundle_shape,
)
from nemo_curator.core.serve.subprocess_mgr import (
    ManagedSubprocess,
    _check_binary,
    _define_subprocess_actor,
    _wait_for_port,
    graceful_stop_actors,
    reacquire_detached_actor_handles,
    sweep_orphan_actors_by_prefix,
)


@pytest.mark.gpu
class TestManagedSubprocess:
    """End-to-end ManagedSubprocess lifecycle against a live GPU replica.

    A class-scoped fixture spawns one GPU-pinned subprocess whose bash
    command emits env markers to the log and backgrounds a ``sleep`` in
    the same process group. Tests operate on the live ``self.proc``
    (``read_log_tail``, etc.) and the live ``self.pg``. The final test
    ``test_stop_reaps_tree`` is destructive: it calls ``stop()`` and
    asserts the whole subprocess tree was reaped. It must run last
    (pytest's definition order) because it destroys shared fixture state.
    Cross-session handle refresh is covered by ``TestCrossSessionTeardown``.
    """

    proc: ManagedSubprocess = None  # type: ignore[assignment]
    pg = None

    @pytest.fixture(scope="class", autouse=True)
    def _spawn(self, request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> None:
        import ray

        tag = os.getpid()
        namespace = f"test_managed_{tag}"
        pg_name = f"test_managed_pg_{tag}"
        actor_name_prefix = f"test_managed_{tag}"
        sentinel = f"CURATOR_SENTINEL_{tag}"

        # Own the ``ray.init`` lifecycle: ``shared_ray_client`` is
        # function-scoped and would tear the driver down between methods,
        # invalidating the class-scoped ``pg`` and actor handles.
        if ray.is_initialized():
            ray.shutdown()
        ray.init(namespace=namespace, ignore_reinit_error=True)

        spec = plan_replica_bundle_shape(tp_size=1, _topology=[{"node_id": "n", "num_gpus": 1, "is_head": False}])
        pg = build_replica_pg(spec, name=pg_name)

        tmp_path = tmp_path_factory.mktemp("managed_subprocess")
        os.environ[sentinel] = "hello_from_driver"
        try:
            proc = ManagedSubprocess.spawn(
                "replica_lifecycle",
                pg,
                0,
                num_gpus=1,
                command=[
                    "bash",
                    "-c",
                    f"echo CUDA=$CUDA_VISIBLE_DEVICES; echo PATH=$PATH; "
                    f"echo etcd=$ETCD_ENDPOINTS; echo post_init=${{{sentinel}:-MISSING}}; "
                    "sleep 300 & echo child_pid=$!; wait",
                ],
                runtime_dir=str(tmp_path),
                actor_name_prefix=actor_name_prefix,
                subprocess_env={"ETCD_ENDPOINTS": "http://10.0.0.1:2379"},
            )
            assert proc.is_alive(), "ManagedSubprocess failed to start"
        finally:
            os.environ.pop(sentinel, None)

        request.cls.proc = proc
        request.cls.pg = pg

        yield

        # Cleanup. ``test_stop_reaps_tree`` has already reaped the
        # subprocess + actor, so ``sweep`` finds 0 and we just drop the
        # PG + driver. If a test failed earlier and left state behind,
        # let the resulting cleanup error surface.
        sweep_orphan_actors_by_prefix(prefix=actor_name_prefix, namespace=namespace)
        ray.util.remove_placement_group(pg)
        ray.shutdown()

    def test_env_vars_propagated(self) -> None:
        """CUDA sourced from Ray accelerator IDs, PATH inherited from the
        raylet, ``subprocess_env`` overrides reach the subprocess, and
        driver ``os.environ`` mutations set after ``ray.init()`` do NOT
        leak into the actor's env."""
        log = self.proc.read_log_tail()
        cuda_match = re.search(r"CUDA=(\S+)", log)
        assert cuda_match is not None, f"CUDA line missing in log:\n{log}"
        for token in cuda_match.group(1).split(","):
            assert token.strip().isdigit(), f"non-numeric CUDA id: {token!r}"
        assert "PATH=/" in log, f"PATH should be inherited from raylet:\n{log}"
        assert "etcd=http://10.0.0.1:2379" in log, f"subprocess_env override missing:\n{log}"
        assert "post_init=MISSING" in log, f"driver env leak after ray.init():\n{log}"

    def test_name_qualname(self) -> None:
        """``_define_subprocess_actor`` labels the underlying class so the
        Ray dashboard shows a descriptive name (e.g. ``Dynamo_DP0_Qwen3-0.6B``)."""
        label = "Dynamo_DP0_Qwen3-0.6B"
        cls = _define_subprocess_actor(label)
        assert cls.__ray_metadata__.modified_class.__name__ == label
        assert label in repr(cls)

    def test_port(self) -> None:
        """``get_free_port_in_bundle`` resolves to a usable port inside the bundle."""
        port = get_free_port_in_bundle(self.pg, bundle_index=0, start_port=30000)
        assert 30000 <= port < 65536

    def test_ray_wait_resolves_on_actor_kill(self) -> None:
        """``DynamoBackend`` polls ``ray.wait(run_refs, timeout=0)`` to detect
        crashed subprocesses; killing an actor must make its run ref resolve.

        Uses its own throwaway actor so the shared ``self.proc`` stays alive
        for the destructive ``test_stop_reaps_tree`` below.
        """
        import ray

        actor_cls = _define_subprocess_actor()
        actor = actor_cls.options(
            name=f"test_death_{os.getpid()}",
            lifetime="detached",
            num_gpus=0,
        ).remote()
        try:
            ray.get(actor.initialize.remote(["sleep", "60"], {}, None), timeout=30)
            run_ref = actor.run.remote()
            ray.kill(actor, no_restart=True)
            ready, _ = ray.wait([run_ref], timeout=30)
            assert len(ready) == 1
        finally:
            # Idempotent: no-op if ``ray.kill`` above already ran;
            # reaps the actor if an earlier step failed.
            ray.kill(actor, no_restart=True)

    # --- destructive: must stay the last test method (definition order) ---

    def test_stop_reaps_tree(self) -> None:
        """``ManagedSubprocess.stop()`` reaps the whole process group via
        ``killpg`` -- both the bash launcher and the backgrounded ``sleep``
        (which shares the launcher's ``pgid``).

        Destroys shared fixture state; must run last in the class.
        """
        log = self.proc.read_log_tail()
        match = re.search(r"child_pid=(\d+)", log)
        assert match is not None, f"backgrounded child pid not found in log:\n{log}"
        child_pid = int(match.group(1))

        self.proc.stop(timeout_s=15)
        with contextlib.suppress(psutil.NoSuchProcess):
            psutil.Process(child_pid).wait(timeout=5)
        assert not psutil.pid_exists(child_pid), f"subprocess-tree child (pid={child_pid}) survived proc.stop()"


class TestCrossSessionTeardown:
    """``with ray.init()`` attach/detach teardown cycle.

    Exercises detached-actor survival across a driver disconnect, stale
    ``ActorHandle`` dispatch failure, ``reacquire_detached_actor_handles``
    refreshing + filtering, ``ManagedSubprocess.stop`` reaping via the
    refreshed handle, and ``sweep_orphan_actors_by_prefix`` as the
    safety-net path. CPU-only (``sleep`` subprocess on a ``num_gpus=0``
    bundle).
    """

    def test_end_to_end(self, tmp_path: Path) -> None:
        import ray

        tag = os.getpid()
        namespace = f"test_cross_session_{tag}"
        pg_name = f"test_cross_session_pg_{tag}"
        actor_name_prefix = f"test_xs_{tag}"
        live_label = "real_target"
        missing_label = "never_existed"
        subprocess_pid: int | None = None

        # ``ray.init(ignore_reinit_error=True)`` is a no-op on an already-
        # initialized driver and would silently reuse its namespace.
        # Reset so the namespace below takes effect.
        if ray.is_initialized():
            ray.shutdown()

        try:
            with ray.init(namespace=namespace, ignore_reinit_error=True):
                pg = build_pg(
                    bundles=[{"CPU": 1}],
                    strategy="STRICT_PACK",
                    name=pg_name,
                    bundle_label_selector=None,
                    ready_timeout_s=PLACEMENT_GROUP_READY_TIMEOUT_S,
                )
                stored_proc = ManagedSubprocess.spawn(
                    live_label,
                    pg,
                    0,
                    num_gpus=0,
                    command=["sleep", "3600"],
                    runtime_dir=str(tmp_path),
                    actor_name_prefix=actor_name_prefix,
                )
                subprocess_pid = stored_proc.pid()
                assert psutil.pid_exists(subprocess_pid)
                stale_actor = stored_proc.actor  # pinned to this driver's job id

            assert psutil.pid_exists(subprocess_pid), "detached actor's subprocess must survive a driver disconnect"

            with ray.init(namespace=namespace, ignore_reinit_error=True):
                # Stale handle from the previous driver session cannot dispatch.
                # Exact exception varies (``ActorHandleNotFoundError`` at dispatch
                # vs. a delivered-error ref), so the match is broad.
                with pytest.raises(Exception):  # noqa: B017, PT011
                    ray.get(stale_actor.is_alive.remote(), timeout=5)

                # A bogus entry must be filtered so callers don't have to
                # defend against ``None`` actor handles downstream.
                bogus = ManagedSubprocess(label=missing_label, actor=stale_actor)
                refreshed = reacquire_detached_actor_handles(
                    [stored_proc, bogus],
                    actor_name_prefix=actor_name_prefix,
                    namespace=namespace,
                )
                assert [p.label for p in refreshed] == [live_label]
                assert refreshed[0] is stored_proc
                assert stored_proc.actor is not stale_actor
                assert ray.get(stored_proc.actor.is_alive.remote(), timeout=10) is True

                # Refreshed handle drives the full reap path.
                stored_proc.stop(timeout_s=15)
                with contextlib.suppress(psutil.NoSuchProcess):
                    psutil.Process(subprocess_pid).wait(timeout=5)
                assert not psutil.pid_exists(subprocess_pid), (
                    "subprocess must be reaped via the refreshed actor handle"
                )

                # After a clean stop nothing should match the prefix.
                assert sweep_orphan_actors_by_prefix(prefix=actor_name_prefix, namespace=namespace) == 0
        finally:
            # The happy path's ``with ray.init()`` context has already
            # exited; re-attach to clean up the PG + drain any actors
            # left behind by an earlier assertion failure.
            if not ray.is_initialized():
                ray.init(namespace=namespace, ignore_reinit_error=True)
            sweep_orphan_actors_by_prefix(prefix=actor_name_prefix, namespace=namespace)
            # PG may be absent: test failed before creating it, or it was
            # already removed. ``ray.util.get_placement_group`` raises
            # ``ValueError`` in that case.
            with contextlib.suppress(ValueError):
                ray.util.remove_placement_group(ray.util.get_placement_group(pg_name))
            ray.shutdown()
            if subprocess_pid is not None:
                with contextlib.suppress(psutil.NoSuchProcess):
                    psutil.Process(subprocess_pid).kill()


class TestSubprocessUtils:
    """Standalone helpers in ``subprocess_mgr`` not reached by the
    ``TestManagedSubprocess`` / ``TestCrossSessionTeardown`` paths."""

    def test_check_binary(self) -> None:
        _check_binary("bash")
        with pytest.raises(FileNotFoundError, match="not found on"):
            _check_binary("definitely-not-a-real-binary-xyz-12345")

    def test_graceful_stop_actors_tolerates_dispatch_failure(self) -> None:
        import sys
        from types import ModuleType

        wait_calls: list[tuple[list[object], int, float | None]] = []
        kill_calls: list[tuple[object, bool]] = []

        fake_ray = ModuleType("ray")

        def _wait(refs: list[object], *, num_returns: int, timeout: float | None) -> tuple[list[object], list[object]]:
            refs_copy = list(refs)
            wait_calls.append((refs_copy, num_returns, timeout))
            return refs_copy, []

        def _get(ref: object, timeout: float | None = None) -> object:
            _ = timeout
            return ref

        def _kill(actor: object, *, no_restart: bool = True) -> None:
            kill_calls.append((actor, no_restart))

        fake_ray.wait = _wait
        fake_ray.get = _get
        fake_ray.kill = _kill

        healthy = mock.Mock()
        healthy.stop.remote.return_value = "healthy-stop"

        stale = mock.Mock()
        stale.stop.remote.side_effect = RuntimeError("stale handle")
        stale.force_sigkill_subprocess.remote.return_value = "stale-sigkill"

        with mock.patch.dict(sys.modules, {"ray": fake_ray}):
            graceful_stop_actors([("healthy", healthy), ("stale", stale)], timeout_s=7)

        assert wait_calls == [(["healthy-stop"], 1, 7)]
        healthy.force_sigkill_subprocess.remote.assert_not_called()
        stale.force_sigkill_subprocess.remote.assert_called_once_with()
        assert kill_calls == [(healthy, True), (stale, True)]

    def test_wait_for_port(self) -> None:
        # Timeout path: bind then close so the port is guaranteed unreachable.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            closed_port = probe.getsockname()[1]
        with pytest.raises(TimeoutError, match="did not become reachable"):
            _wait_for_port("127.0.0.1", closed_port, timeout_s=0.5, label="closed")

        # Success path: listen, then probe.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            open_port = server.getsockname()[1]
            _wait_for_port("127.0.0.1", open_port, timeout_s=2, label="open")
