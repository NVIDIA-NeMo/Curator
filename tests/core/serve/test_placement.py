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

import pytest

from nemo_curator.core.serve.placement import (
    ReplicaBundleSpec,
    plan_replica_bundle_shape,
    remove_named_pgs_with_prefix,
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

    @pytest.mark.parametrize("tp_size", [0, -1])
    def test_non_positive_tp_size_rejected(self, tp_size: int) -> None:
        topology = [{"node_id": "n1", "num_gpus": 8, "is_head": False}]
        with pytest.raises(ValueError, match=r"tp_size must be >= 1"):
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
# Real-Ray GPU integration -- orphan PG cleanup by prefix
# ---------------------------------------------------------------------------


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
