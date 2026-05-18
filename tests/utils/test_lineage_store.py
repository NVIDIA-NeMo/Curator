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
"""Tests for :mod:`nemo_curator.utils.lineage_store`.

The storage-layer tests use :class:`LineageStore` directly. The end-to-end
helper tests spawn a real :class:`LineageWriterActor` and verify that calling
:func:`assign_and_record_lineage` writes through the actor.
"""

import contextlib
from dataclasses import dataclass
from pathlib import Path

import pytest
import ray

from nemo_curator.stages.base import assign_child_lineage
from nemo_curator.tasks import Task
from nemo_curator.utils.lineage_store import (
    LINEAGE_ACTOR_NAME,
    LineageStore,
    LineageWriterActor,
    _classify,
    _path_to_udid,
    are_completed,
    mark_leaves_completed,
    record_lineage,
)


@dataclass
class _T(Task[None]):
    @property
    def num_items(self) -> int:
        return 0

    def validate(self) -> bool:
        return True


def _make_child(parent_path: str, i: int) -> _T:
    """Build a lineage-assigned task as if it were emitted by a stage."""
    t = _T(task_id=f"t{i}", dataset_name="ds", data=None)
    t._set_lineage([parent_path], i)
    return t


# --------------------------------------------------------------------------- #
# Direct LineageStore tests — no Ray required.
# --------------------------------------------------------------------------- #


@pytest.fixture
def store(tmp_path: Path) -> LineageStore:
    path = tmp_path / "lineage.mdb"
    s = LineageStore(str(path))
    try:
        yield s
    finally:
        s.close()


def test_classify_truth_table() -> None:
    assert _classify(False, False) == b"source_leaf"
    assert _classify(False, True) == b"source"
    assert _classify(True, False) == b"leaf"
    assert _classify(True, True) == b"middle"


def test_get_returns_none_for_unknown(store: LineageStore) -> None:
    assert store.get("doesnotexist" + "0" * 20) is None


def test_records_single_edge_with_types(store: LineageStore) -> None:
    parent = "p" * 32
    child = "c" * 32
    store.record_emission([parent], [child])

    p_rec = store.get(parent)
    c_rec = store.get(child)
    assert p_rec is not None
    assert c_rec is not None
    assert p_rec.children == [child]
    assert p_rec.parents == []
    assert p_rec.task_type == "source"  # provisional: no parents seen yet for `parent`
    assert c_rec.parents == [parent]
    assert c_rec.children == []
    assert c_rec.task_type == "leaf"


def test_emission_is_idempotent_under_retry(store: LineageStore) -> None:
    parent = "p" * 32
    child = "c" * 32
    store.record_emission([parent], [child])
    store.record_emission([parent], [child])  # retry

    assert store.get(parent).children == [child]  # no duplicate edge
    assert store.get(child).parents == [parent]


def test_incremental_parent_attribution(store: LineageStore) -> None:
    """Multiple calls for the same child accumulate parents."""
    p1 = "1" * 32
    p2 = "2" * 32
    child = "c" * 32
    store.record_emission([p1], [child])
    store.record_emission([p2], [child])

    rec = store.get(child)
    assert set(rec.parents) == {p1, p2}
    assert rec.task_type == "leaf"
    assert set(store.get(p1).children) == {child}
    assert set(store.get(p2).children) == {child}


def test_type_promotes_monotonically_under_reordering(store: LineageStore) -> None:
    """If a node first appears as a parent, it's provisionally `source`. When
    its own parent-edge later arrives, it must promote to `middle`."""
    grandparent = "g" * 32
    parent = "p" * 32
    child = "c" * 32

    # Out-of-order: child created from parent first; parent's own creation arrives later.
    store.record_emission([parent], [child])
    assert store.get(parent).task_type == "source"

    store.record_emission([grandparent], [parent])
    assert store.get(parent).task_type == "middle"
    assert store.get(grandparent).task_type == "source"
    assert store.get(child).task_type == "leaf"


def test_source_leaf_classification(store: LineageStore) -> None:
    """A task with no parents and no children is `source_leaf`."""
    orphan = "o" * 32
    # Emit a child with no real parents, AND no children of its own.
    store.record_emission([], [orphan])
    assert store.get(orphan).task_type == "source_leaf"


def test_completed_defaults_false_and_can_be_set(store: LineageStore) -> None:
    udid = "x" * 32
    store.record_emission([], [udid])
    assert store.get(udid).completed is False
    store.mark_completed(udid)
    assert store.is_completed(udid) is True
    assert store.get(udid).completed is True
    # Idempotent.
    store.mark_completed(udid)
    assert store.is_completed(udid) is True


def test_iter_records_returns_all(store: LineageStore) -> None:
    udids = ["a" * 32, "b" * 32, "c" * 32]
    # a → b, b → c
    store.record_emission([udids[0]], [udids[1]])
    store.record_emission([udids[1]], [udids[2]])

    all_records = dict(store.iter_records())
    assert set(all_records.keys()) == set(udids)
    assert all_records[udids[0]].task_type == "source"
    assert all_records[udids[1]].task_type == "middle"
    assert all_records[udids[2]].task_type == "leaf"


def test_get_all_parents_chain(store: LineageStore) -> None:
    """a → b → c: ``get_all_parents(c)`` returns both ``a`` and ``b``."""
    a, b, c = "a" * 32, "b" * 32, "c" * 32
    store.record_emission([a], [b])
    store.record_emission([b], [c])

    parents = store.get_all_parents(c)
    assert set(parents.keys()) == {a, b}
    assert parents[b].parents == [a]
    assert parents[a].parents == []


def test_get_all_children_chain(store: LineageStore) -> None:
    """a → b → c: ``get_all_children(a)`` returns both ``b`` and ``c``."""
    a, b, c = "a" * 32, "b" * 32, "c" * 32
    store.record_emission([a], [b])
    store.record_emission([b], [c])

    children = store.get_all_children(a)
    assert set(children.keys()) == {b, c}
    assert children[b].children == [c]
    assert children[c].children == []


def test_transitive_diamond_dedup(store: LineageStore) -> None:
    """Diamond: a → {b, c} → d. ``a`` appears once in ``get_all_parents(d)``."""
    a, b, c, d = "a" * 32, "b" * 32, "c" * 32, "d" * 32
    store.record_emission([a], [b])
    store.record_emission([a], [c])
    store.record_emission([b], [d])
    store.record_emission([c], [d])

    assert set(store.get_all_parents(d).keys()) == {a, b, c}
    assert set(store.get_all_children(a).keys()) == {b, c, d}


def test_transitive_unknown_returns_empty(store: LineageStore) -> None:
    unknown = "u" * 32
    assert store.get_all_parents(unknown) == {}
    assert store.get_all_children(unknown) == {}


def test_transitive_source_and_leaf_empty(store: LineageStore) -> None:
    a, b = "a" * 32, "b" * 32
    store.record_emission([a], [b])
    # Pure source: no ancestors.
    assert store.get_all_parents(a) == {}
    # Pure leaf: no descendants.
    assert store.get_all_children(b) == {}


def test_transitive_excludes_self(store: LineageStore) -> None:
    a, b, c = "a" * 32, "b" * 32, "c" * 32
    store.record_emission([a], [b])
    store.record_emission([b], [c])
    for udid in (a, b, c):
        assert udid not in store.get_all_parents(udid)
        assert udid not in store.get_all_children(udid)


def test_path_to_udid_matches_task_set_lineage() -> None:
    """Mirror invariant: hashing a lineage path with ``_path_to_udid`` yields the same
    ``_udid`` that ``Task._set_lineage`` would assign."""
    t = _T(task_id="t", dataset_name="ds", data=None)
    t._set_lineage(["3_0"], 7)
    assert t._lineage_path == "3_0_7"
    assert _path_to_udid("3_0_7") == t._udid


# --------------------------------------------------------------------------- #
# BFS completion-propagation tests.
# --------------------------------------------------------------------------- #


def test_propagate_linear_chain(store: LineageStore) -> None:
    """A→B→C→D: propagating from D rolls up to A."""
    a, b, c, d = "a" * 32, "b" * 32, "c" * 32, "d" * 32
    store.record_emission([a], [b])
    store.record_emission([b], [c])
    store.record_emission([c], [d])

    newly = store.mark_completed_and_propagate([d])
    assert set(newly) == {a, b, c, d}
    for udid in (a, b, c, d):
        assert store.is_completed(udid)


def test_propagate_diamond_partial(store: LineageStore) -> None:
    """A→{B,C}; propagating only from B does not mark A because C is still pending.
    A second pass that completes C then rolls up to A."""
    a, b, c = "a" * 32, "b" * 32, "c" * 32
    store.record_emission([a], [b])
    store.record_emission([a], [c])

    newly_first = store.mark_completed_and_propagate([b])
    assert newly_first == [b]
    assert store.is_completed(b)
    assert not store.is_completed(a)
    assert not store.is_completed(c)

    newly_second = store.mark_completed_and_propagate([c])
    assert set(newly_second) == {a, c}
    assert store.is_completed(a)


def test_propagate_diamond_full_batch(store: LineageStore) -> None:
    """A→{B,C}→D: batch-propagate from D marks all four; A appears once (visited dedup)."""
    a, b, c, d = "a" * 32, "b" * 32, "c" * 32, "d" * 32
    store.record_emission([a], [b])
    store.record_emission([a], [c])
    store.record_emission([b], [d])
    store.record_emission([c], [d])

    newly = store.mark_completed_and_propagate([d])
    assert set(newly) == {a, b, c, d}
    assert newly.count(a) == 1
    for udid in (a, b, c, d):
        assert store.is_completed(udid)


def test_propagate_idempotent(store: LineageStore) -> None:
    """Calling propagate twice returns empty the second time; state unchanged."""
    a, b = "a" * 32, "b" * 32
    store.record_emission([a], [b])

    newly_first = store.mark_completed_and_propagate([b])
    assert set(newly_first) == {a, b}

    newly_second = store.mark_completed_and_propagate([b])
    assert newly_second == []
    assert store.is_completed(a)
    assert store.is_completed(b)


def test_propagate_stops_at_incomplete_sibling(store: LineageStore) -> None:
    """Fan-out A→{C1,C2,C3}: completing only C1 must not mark A because C2 and
    C3 are still pending children. A second pass completing C2 still leaves A
    blocked. Only after C3 is also completed does A get rolled up."""
    a, c1, c2, c3 = "a" * 32, "1" * 32, "2" * 32, "3" * 32
    store.record_emission([a], [c1])
    store.record_emission([a], [c2])
    store.record_emission([a], [c3])

    first = store.mark_completed_and_propagate([c1])
    assert first == [c1]
    assert store.is_completed(c1)
    assert not store.is_completed(a)
    assert not store.is_completed(c2)
    assert not store.is_completed(c3)

    second = store.mark_completed_and_propagate([c2])
    assert set(second) == {c2}
    assert not store.is_completed(a)

    third = store.mark_completed_and_propagate([c3])
    assert set(third) == {a, c3}
    assert store.is_completed(a)


def test_propagate_unknown_udid_is_noop(store: LineageStore) -> None:
    """An unknown but non-empty udid is silently skipped."""
    newly = store.mark_completed_and_propagate(["u" * 32])
    assert newly == []


def test_propagate_empty_udid_raises(store: LineageStore) -> None:
    """An empty udid means the caller forgot ``assign_child_lineage`` — raise loudly.
    The known leaf in the same batch must NOT be marked, since the call aborts."""
    leaf = "x" * 32
    store.record_emission([], [leaf])
    with pytest.raises(ValueError, match="empty udid"):
        store.mark_completed_and_propagate(["", leaf])
    assert not store.is_completed(leaf)


# --------------------------------------------------------------------------- #
# Bulk are_completed tests — single read txn over many udids.
# --------------------------------------------------------------------------- #


def test_are_completed_empty_input(store: LineageStore) -> None:
    assert store.are_completed([]) == []


def test_are_completed_all_completed(store: LineageStore) -> None:
    udids = ["a" * 32, "b" * 32, "c" * 32]
    for u in udids:
        store.record_emission([], [u])
        store.mark_completed(u)
    assert store.are_completed(udids) == [True, True, True]


def test_are_completed_none_completed(store: LineageStore) -> None:
    udids = ["a" * 32, "b" * 32, "c" * 32]
    for u in udids:
        store.record_emission([], [u])
    assert store.are_completed(udids) == [False, False, False]


def test_are_completed_mixed_preserves_order(store: LineageStore) -> None:
    udids = ["a" * 32, "b" * 32, "c" * 32, "d" * 32]
    for u in udids:
        store.record_emission([], [u])
    store.mark_completed(udids[0])
    store.mark_completed(udids[2])
    assert store.are_completed(udids) == [True, False, True, False]


def test_are_completed_unknown_udids_return_false(store: LineageStore) -> None:
    """Never-recorded udids return False (no key in completed_db)."""
    assert store.are_completed(["u" * 32, "v" * 32]) == [False, False]


def test_are_completed_empty_string_returns_false(store: LineageStore) -> None:
    """Empty udid short-circuits to False without an LMDB lookup."""
    udid = "a" * 32
    store.record_emission([], [udid])
    store.mark_completed(udid)
    assert store.are_completed(["", udid]) == [False, True]


# --------------------------------------------------------------------------- #
# Actor-routed tests — verify record_lineage → LineageWriterActor → LMDB.
# --------------------------------------------------------------------------- #


def _kill_actor_if_present() -> None:
    """Make sure no leftover writer actor lingers between tests."""
    with contextlib.suppress(ValueError):
        handle = ray.get_actor(LINEAGE_ACTOR_NAME)
        ray.kill(handle)


@pytest.fixture
def actor(tmp_path: Path, shared_ray_client: None) -> tuple[object, Path]:  # noqa: ARG001
    """Spawn a real :class:`LineageWriterActor` for the duration of the test."""
    _kill_actor_if_present()
    path = tmp_path / "lineage_actor.mdb"
    handle = LineageWriterActor.options(
        name=LINEAGE_ACTOR_NAME,
        get_if_exists=True,
    ).remote(path=str(path))
    try:
        yield handle, path
    finally:
        with contextlib.suppress(Exception):
            ray.get(handle.close.remote())
        ray.kill(handle)


def test_record_lineage_filters_empty_parents(actor: tuple[object, Path]) -> None:
    """``record_lineage`` should not record EmptyTask-style empty parent udids."""
    actor_handle, _ = actor
    children = assign_child_lineage([""], [_T(task_id="c", dataset_name="ds", data=None) for _ in range(3)])
    record_lineage([""], [c._udid for c in children])
    assert [c._lineage_path for c in children] == ["0", "1", "2"]

    for i, c in enumerate(children):
        rec = ray.get(actor_handle.get.remote(c._udid))
        assert rec is not None, f"child {i} not in store"
        assert rec.parents == []
        assert rec.task_type == "source_leaf"


def test_record_lineage_propagates_lineage(actor: tuple[object, Path]) -> None:
    """End-to-end through the actor: drive two stages with separate
    :func:`assign_child_lineage` + :func:`record_lineage` calls and check the
    DAG ends up in the store with correct types."""
    actor_handle, _ = actor

    # Stage 1: produce three children from an empty root.
    parents = assign_child_lineage([""], [_T(task_id=f"p{i}", dataset_name="ds", data=None) for i in range(3)])
    record_lineage([""], [p._udid for p in parents])

    # Stage 2: each parent produces two children.
    grandchildren = []
    for p in parents:
        emitted = assign_child_lineage(
            [p._lineage_path],
            [_T(task_id="g", dataset_name="ds", data=None) for _ in range(2)],
        )
        record_lineage([p._udid], [c._udid for c in emitted])
        grandchildren.extend(emitted)

    # Sources: 3 parents, classified `source` because they now have children.
    for p in parents:
        rec = ray.get(actor_handle.get.remote(p._udid))
        assert rec is not None
        assert rec.task_type == "source"
        assert len(rec.children) == 2
        assert rec.parents == []

    # Leaves: 6 grandchildren, each with one parent.
    for g in grandchildren:
        rec = ray.get(actor_handle.get.remote(g._udid))
        assert rec is not None
        assert rec.task_type == "leaf"
        assert len(rec.parents) == 1
        assert rec.children == []


def test_record_lineage_is_noop_without_actor(shared_ray_client: None) -> None:  # noqa: ARG001
    """When no LineageWriterActor is registered, ``record_lineage`` must not raise
    and must not write anything (no actor exists to write to)."""
    _kill_actor_if_present()
    # If this returned an error path it would raise. We don't have a record store
    # to assert "no write" against directly, but the absence of an actor means
    # there's literally nowhere for it to write — successful return is the assertion.
    child = _make_child("", 0)
    record_lineage([""], [child._udid])


def test_mark_completed_and_propagate_actor_passthrough(actor: tuple[object, Path]) -> None:
    """Drive a small DAG via the actor and confirm propagation through ``ray.get``."""
    actor_handle, _ = actor
    a, b, c = "a" * 32, "b" * 32, "c" * 32
    ray.get(actor_handle.record_emission.remote([a], [b]))
    ray.get(actor_handle.record_emission.remote([b], [c]))

    newly = ray.get(actor_handle.mark_completed_and_propagate.remote([c]))
    assert set(newly) == {a, b, c}
    for udid in (a, b, c):
        assert ray.get(actor_handle.is_completed.remote(udid))


def test_mark_leaves_completed_routes_through_actor(actor: tuple[object, Path]) -> None:
    """The :func:`mark_leaves_completed` helper looks up the named actor and forwards
    to its ``mark_completed_and_propagate`` method, rolling completion up the DAG."""
    actor_handle, _ = actor
    a, b, c = "a" * 32, "b" * 32, "c" * 32
    ray.get(actor_handle.record_emission.remote([a], [b]))
    ray.get(actor_handle.record_emission.remote([b], [c]))

    mark_leaves_completed([c])

    for udid in (a, b, c):
        assert ray.get(actor_handle.is_completed.remote(udid))


def test_mark_leaves_completed_noop_without_actor(shared_ray_client: None) -> None:  # noqa: ARG001
    """When no LineageWriterActor is registered, :func:`mark_leaves_completed` must
    return silently — mirrors the :func:`record_lineage` no-op contract."""
    _kill_actor_if_present()
    # No actor, no destination — successful return is the assertion.
    mark_leaves_completed(["x" * 32])


def test_mark_leaves_completed_filters_empty_udids(actor: tuple[object, Path]) -> None:
    """Empty udids in the input list are filtered (parity with :func:`record_lineage`)
    rather than triggering the underlying ``ValueError``; non-empty udids still get
    marked."""
    actor_handle, _ = actor
    leaf = "z" * 32
    ray.get(actor_handle.record_emission.remote([], [leaf]))

    # Mixing an empty udid in must not raise; the real leaf still gets marked.
    mark_leaves_completed(["", leaf])
    assert ray.get(actor_handle.is_completed.remote(leaf))


def test_module_are_completed_without_ray() -> None:
    """No Ray initialized → bulk helper returns all False (filter no-op)."""
    if ray.is_initialized():
        pytest.skip("ray already initialized by another test in this session")
    assert are_completed(["a" * 32, "b" * 32]) == [False, False]


def test_module_are_completed_no_actor(shared_ray_client: None) -> None:  # noqa: ARG001
    """Ray up but no LineageWriterActor registered → all False."""
    _kill_actor_if_present()
    assert are_completed(["a" * 32, "b" * 32]) == [False, False]


def test_module_are_completed_with_actor(actor: tuple[object, Path]) -> None:
    """Module-level helper routes through the registered actor and preserves order."""
    actor_handle, _ = actor
    udids = ["a" * 32, "b" * 32, "c" * 32, "d" * 32]
    for u in udids:
        ray.get(actor_handle.record_emission.remote([], [u]))
    ray.get(actor_handle.mark_completed.remote(udids[1]))
    ray.get(actor_handle.mark_completed.remote(udids[3]))

    assert are_completed(udids) == [False, True, False, True]


def test_module_are_completed_empty_input(actor: tuple[object, Path]) -> None:  # noqa: ARG001
    """Empty input short-circuits before any actor call."""
    assert are_completed([]) == []
