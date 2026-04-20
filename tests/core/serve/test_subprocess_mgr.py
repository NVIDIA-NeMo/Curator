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
import json
import os
import re
import subprocess
import sys
import time

import psutil
import pytest

from nemo_curator.core.serve.placement import (
    build_replica_pg,
    get_bundle_node_ip,
    get_free_port_in_bundle,
    plan_replica_bundle_shape,
)
from nemo_curator.core.serve.subprocess_mgr import (
    ManagedSubprocess,
    _define_subprocess_actor,
    _stop_subprocess,
)


@pytest.mark.gpu
@pytest.mark.usefixtures("shared_ray_client")
class TestReplicaLifecycle:
    """Exercise the PG + actor + subprocess lifecycle end-to-end.

    Collapses what were previously ~10 independent GPU tests (PG readiness,
    bundle IP/port lookup, actor spawn, CUDA env propagation, subprocess env
    semantics, graceful stop) into one run that shares a single replica PG
    and subprocess actor. A per-test PG costs ~10s on this box; keeping a
    single shared lifecycle keeps the GPU slice bounded.
    """

    def test_end_to_end(self, tmp_path: os.PathLike) -> None:
        import ray

        spec = plan_replica_bundle_shape(tp_size=1, _topology=[{"node_id": "n", "num_gpus": 1, "is_head": False}])
        pg_name = f"test_replica_lifecycle_{os.getpid()}"
        pg = build_replica_pg(spec, name=pg_name)
        try:
            # 1. PG is ready and retrievable by name.
            assert ray.util.get_placement_group(pg_name) is not None

            # 2. Bundle-scoped helpers resolve against a real Ray node.
            ip = get_bundle_node_ip(pg, 0)
            assert re.match(r"^\d+\.\d+\.\d+\.\d+$", ip), f"unexpected ip: {ip!r}"
            port = get_free_port_in_bundle(pg, 0, 30000)
            assert 30000 <= port < 65536

            # 3. ManagedSubprocess.spawn: CUDA_VISIBLE_DEVICES is sourced
            #    from Ray-assigned accelerator IDs and written into the
            #    subprocess env; targeted env overrides reach the
            #    subprocess; pre-existing PATH is inherited from the raylet.
            sentinel = f"CURATOR_SENTINEL_{os.getpid()}"
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
                        "echo marker=${CURATOR_SUBPROC_MARKER:-MISSING}",
                    ],
                    runtime_dir=str(tmp_path),
                    actor_name_prefix=f"test_{os.getpid()}",
                    subprocess_env={"ETCD_ENDPOINTS": "http://10.0.0.1:2379"},
                )
                proc.wait(timeout=30)
                log = proc.read_log_tail()
            finally:
                os.environ.pop(sentinel, None)

            cuda_match = re.search(r"CUDA=(\S+)", log)
            assert cuda_match is not None, f"CUDA line missing in log:\n{log}"
            for token in cuda_match.group(1).split(","):
                assert token.strip().isdigit(), f"non-numeric CUDA id: {token!r}"
            assert "PATH=/" in log, f"PATH should be inherited from raylet:\n{log}"
            assert "etcd=http://10.0.0.1:2379" in log, f"subprocess_env override missing:\n{log}"
            assert "post_init=MISSING" in log, (
                f"driver os.environ mutations set AFTER ray.init() must NOT leak to the actor:\n{log}"
            )
            assert "marker=MISSING" in log, f"cleanup marker should not be injected into the subprocess env:\n{log}"

            # 4. Graceful stop reaps the subprocess without raising.
            proc.stop()
        finally:
            with contextlib.suppress(Exception):
                ray.util.remove_placement_group(pg)


def test_stop_subprocess_reaps_spawned_multiprocessing_children(tmp_path: os.PathLike) -> None:
    """Children launched through Python multiprocessing spawn must die with the parent group."""
    helper = os.path.join(tmp_path, "spawn_tree.py")
    helper_src = """
import json
import multiprocessing
import os
import time

def child_main():
    time.sleep(300)

if __name__ == '__main__':
    ctx = multiprocessing.get_context('spawn')
    child = ctx.Process(target=child_main, name='spawn-child')
    child.start()
    print(json.dumps({
        'outer_pid': os.getpid(),
        'outer_pgid': os.getpgid(0),
        'child_pid': child.pid,
        'child_pgid': os.getpgid(child.pid),
    }), flush=True)
    time.sleep(300)
"""
    with open(helper, "w") as f:
        f.write(helper_src)

    parent = subprocess.Popen(  # noqa: S603
        [sys.executable, helper],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert parent.stdout is not None
        info = json.loads(parent.stdout.readline())
        tracked_pids = [parent.pid, info["child_pid"]]
        assert info["child_pgid"] == info["outer_pgid"]

        _stop_subprocess(parent, sigterm_wait=2)

        for _ in range(20):
            if not any(psutil.pid_exists(pid) for pid in tracked_pids):
                break
            time.sleep(0.1)
        assert not any(psutil.pid_exists(pid) for pid in tracked_pids), "spawned child process tree was not reaped"
    finally:
        with contextlib.suppress(Exception):
            if parent.poll() is None:
                parent.kill()


def test_stop_subprocess_reaps_same_group_child_after_launcher_exit(tmp_path: os.PathLike) -> None:
    """Even if the launcher already exited, ``killpg(proc.pid)`` must reap the remaining group."""
    helper = os.path.join(tmp_path, "same_group_orphan.py")
    helper_src = """
import json
import os
import signal
import subprocess
import sys
import time

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == '--child':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(300)
elif __name__ == '__main__':
    child = subprocess.Popen([sys.executable, __file__, '--child'])
    print(json.dumps({
        'parent_pid': os.getpid(),
        'parent_pgid': os.getpgid(0),
        'child_pid': child.pid,
        'child_pgid': os.getpgid(child.pid),
    }), flush=True)
"""
    with open(helper, "w") as f:
        f.write(helper_src)

    parent = subprocess.Popen(  # noqa: S603
        [sys.executable, helper],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    child_pid: int | None = None
    try:
        assert parent.stdout is not None
        info = json.loads(parent.stdout.readline())
        child_pid = info["child_pid"]

        parent.wait(timeout=5)
        assert parent.poll() is not None
        assert info["parent_pid"] == parent.pid
        assert info["parent_pgid"] == parent.pid
        assert info["child_pgid"] == parent.pid
        assert psutil.pid_exists(child_pid), "same-process-group child should still be alive before cleanup"

        _stop_subprocess(parent, sigterm_wait=1)

        for _ in range(20):
            if not psutil.pid_exists(child_pid):
                break
            time.sleep(0.1)
        assert not psutil.pid_exists(child_pid), "same-process-group child survived after launcher exit"
    finally:
        with contextlib.suppress(Exception):
            if parent.poll() is None:
                parent.kill()
        with contextlib.suppress(Exception):
            if child_pid is not None and psutil.pid_exists(child_pid):
                psutil.Process(child_pid).kill()


@pytest.mark.gpu
@pytest.mark.usefixtures("shared_ray_client")
def test_actor_death_surfaces_via_run_ref() -> None:
    """Hard-killing the actor makes its run ref resolve in ray.wait().

    This is the signal DynamoBackend uses to detect a crashed subprocess
    (``ray.wait(run_refs, timeout=0)``).
    """
    import ray

    actor_cls = _define_subprocess_actor()
    actor_name = f"test_liveness_death_{os.getpid()}"
    actor = actor_cls.options(name=actor_name, lifetime="detached").remote()
    ray.get(actor.initialize.remote(["sleep", "3600"], {}, None), timeout=30)
    run_ref = actor.run.remote()
    proc = ManagedSubprocess(label="death", actor=actor, run_ref=run_ref)
    try:
        assert proc.is_alive()
        ray.kill(proc.actor, no_restart=True)
        ready, _ = ray.wait([proc.run_ref], timeout=30)
        assert len(ready) == 1
    except Exception:
        with contextlib.suppress(Exception):
            ray.kill(proc.actor, no_restart=True)
        raise
