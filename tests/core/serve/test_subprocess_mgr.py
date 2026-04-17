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

from nemo_curator.core.serve.subprocess_mgr import (
    ManagedSubprocess,
    ReplicaBundleSpec,
    _define_subprocess_actor,
    build_replica_pg,
    get_bundle_node_ip,
    get_free_port_in_bundle,
    plan_replica_bundle_shape,
    remove_named_pgs_with_prefix,
    spawn_actor,
)

# ---------------------------------------------------------------------------
# Pure-logic tests (no Ray)
# ---------------------------------------------------------------------------


class TestPlanReplicaBundleShape:
    """Covers the full matrix of planner decisions against mocked topologies."""

    def test_single_node_fits_and_total_gpus(self) -> None:
        topology = [{"node_id": "n1", "num_gpus": 8, "is_head": False}]
        spec = plan_replica_bundle_shape(tp_size=4, _topology=topology)
        assert spec.strategy == "STRICT_PACK"
        assert not spec.is_multi_node
        assert spec.nnodes == 1
        assert spec.per_node_gpus == 4
        assert spec.total_gpus == 4
        assert spec.bundles == [{"CPU": 1, "GPU": 4}]
        assert spec.bundle_label_selector is None

    def test_single_node_preferred_when_possible(self) -> None:
        """5+8 cluster with TP=4 picks the 8-GPU node and stays single-node."""
        topology = [
            {"node_id": "n1", "num_gpus": 5, "is_head": False},
            {"node_id": "n2", "num_gpus": 8, "is_head": False},
        ]
        spec = plan_replica_bundle_shape(tp_size=4, _topology=topology)
        assert not spec.is_multi_node

    def test_multi_node_even_split(self) -> None:
        """TP=8 across two 4-GPU nodes spreads with equal per-node slices."""
        topology = [
            {"node_id": "n1", "num_gpus": 4, "is_head": False},
            {"node_id": "n2", "num_gpus": 4, "is_head": False},
        ]
        spec = plan_replica_bundle_shape(tp_size=8, _topology=topology)
        assert spec.strategy == "STRICT_SPREAD"
        assert spec.is_multi_node
        assert spec.nnodes == 2
        assert spec.per_node_gpus == 4
        assert spec.total_gpus == 8
        assert spec.bundles == [{"CPU": 1, "GPU": 4}, {"CPU": 1, "GPU": 4}]

    def test_multi_node_three_node_split(self) -> None:
        topology = [{"node_id": f"n{i}", "num_gpus": 4, "is_head": False} for i in range(1, 4)]
        spec = plan_replica_bundle_shape(tp_size=12, _topology=topology)
        assert spec.nnodes == 3
        assert spec.per_node_gpus == 4

    @pytest.mark.parametrize(
        ("tp_size", "topology", "match"),
        [
            # vLLM requires equal per-node local_world_size -> 1+3 is rejected.
            (
                4,
                [
                    {"node_id": "n1", "num_gpus": 1, "is_head": False},
                    {"node_id": "n2", "num_gpus": 3, "is_head": False},
                ],
                "even split",
            ),
            # TP=6 on two 2-GPU nodes: 2 does not divide 6 and 3 nodes are not available.
            (
                6,
                [
                    {"node_id": "n1", "num_gpus": 2, "is_head": False},
                    {"node_id": "n2", "num_gpus": 2, "is_head": False},
                ],
                "even split",
            ),
            # Empty topology is a hard error (no GPUs in cluster).
            (1, [], "No GPU nodes"),
        ],
    )
    def test_infeasible_shapes_raise(self, tp_size: int, topology: list[dict], match: str) -> None:
        with pytest.raises(RuntimeError, match=match):
            plan_replica_bundle_shape(tp_size=tp_size, _topology=topology)


class TestHeadNodeExclusion:
    """CURATOR_IGNORE_RAY_HEAD_NODE filters head from topology AND emits the label selector."""

    def test_selector_absent_when_unset(self) -> None:
        topology = [
            {"node_id": "head", "num_gpus": 8, "is_head": True},
            {"node_id": "worker", "num_gpus": 8, "is_head": False},
        ]
        assert plan_replica_bundle_shape(tp_size=4, _topology=topology).bundle_label_selector is None

    def test_flag_filters_head_and_emits_selector(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Head's 16 GPUs are ignored; planner must split across the two workers,
        and every bundle carries the worker-label selector."""
        monkeypatch.setenv("CURATOR_IGNORE_RAY_HEAD_NODE", "1")
        topology = [
            {"node_id": "head", "num_gpus": 16, "is_head": True},
            {"node_id": "w1", "num_gpus": 4, "is_head": False},
            {"node_id": "w2", "num_gpus": 4, "is_head": False},
        ]
        spec = plan_replica_bundle_shape(tp_size=8, _topology=topology)
        assert spec.is_multi_node
        assert spec.nnodes == 2
        assert spec.bundle_label_selector == [{"ray.io/node-type": "worker"}] * 2

    def test_only_head_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CURATOR_IGNORE_RAY_HEAD_NODE", "1")
        topology = [{"node_id": "head", "num_gpus": 8, "is_head": True}]
        with pytest.raises(RuntimeError, match="CURATOR_IGNORE_RAY_HEAD_NODE"):
            plan_replica_bundle_shape(tp_size=4, _topology=topology)


def test_replica_bundle_spec_properties() -> None:
    single = ReplicaBundleSpec(bundles=[{"CPU": 1, "GPU": 4}], strategy="STRICT_PACK", nnodes=1, per_node_gpus=4)
    multi = ReplicaBundleSpec(bundles=[{"CPU": 1, "GPU": 4}] * 2, strategy="STRICT_SPREAD", nnodes=2, per_node_gpus=4)
    assert not single.is_multi_node
    assert single.total_gpus == 4
    assert multi.is_multi_node
    assert multi.total_gpus == 8


# ---------------------------------------------------------------------------
# Real-Ray GPU integration
# ---------------------------------------------------------------------------


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

            # 3. spawn_actor: CUDA_VISIBLE_DEVICES is sourced from Ray-assigned
            #    accelerator IDs and written into the subprocess env; targeted
            #    env overrides reach the subprocess; pre-existing PATH is
            #    inherited from the raylet.
            sentinel = f"CURATOR_SENTINEL_{os.getpid()}"
            os.environ[sentinel] = "hello_from_driver"
            try:
                proc = spawn_actor(
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


@pytest.mark.gpu
@pytest.mark.usefixtures("shared_ray_client")
def test_orphan_pg_cleanup_by_prefix() -> None:
    """remove_named_pgs_with_prefix reaps matching PGs and leaves non-matches alone."""
    import ray
    from ray.util.placement_group import placement_group

    prefix = f"orphan_test_{os.getpid()}_"
    assert remove_named_pgs_with_prefix(f"no_such_prefix_{os.getpid()}_") == 0

    created = []
    try:
        for i in range(3):
            pg = placement_group(
                bundles=[{"CPU": 1}], strategy="STRICT_PACK", lifetime="detached", name=f"{prefix}{i}"
            )
            ray.get(pg.ready(), timeout=30)
            created.append(pg)

        assert remove_named_pgs_with_prefix(prefix) >= 3

        for i in range(3):
            with pytest.raises(Exception):  # noqa: B017, PT011
                ray.util.get_placement_group(f"{prefix}{i}")
    finally:
        for pg in created:
            with contextlib.suppress(Exception):
                ray.util.remove_placement_group(pg)
