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

import pytest

from nemo_curator.core.serve.placement import (
    build_replica_pg,
    get_bundle_node_ip,
    get_free_port_in_bundle,
    plan_replica_bundle_shape,
)
from nemo_curator.core.serve.subprocess_mgr import ManagedSubprocess, _define_subprocess_actor


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
                        f"echo etcd=$ETCD_ENDPOINTS; echo post_init=${{{sentinel}:-MISSING}}",
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

            # 4. Graceful stop reaps the subprocess without raising.
            proc.stop()
        finally:
            with contextlib.suppress(Exception):
                ray.util.remove_placement_group(pg)


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
