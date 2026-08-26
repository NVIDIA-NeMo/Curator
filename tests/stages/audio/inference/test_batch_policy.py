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

"""Tests for the generic cost-bucketed batching primitives: ``BatchPolicy`` and ``run_bucketed``."""

from __future__ import annotations

import pytest

from nemo_curator.stages.audio.inference.batch_policy import BatchPolicy, BucketQueueScheduler, run_bucketed

# ----------------------------------------------------------------------
# BatchPolicy: validation + bucket math
# ----------------------------------------------------------------------


def test_batch_policy_invalid_strategy_rejected() -> None:
    with pytest.raises(ValueError, match="duration_bucketed"):
        BatchPolicy(strategy="token_bucketed")


def test_batch_policy_inconsistent_lengths_rejected() -> None:
    with pytest.raises(ValueError, match="lengths must match"):
        BatchPolicy(buckets_sec=[0, 60, 600], max_items_per_batch_by_bucket=[10, 5])


def test_batch_policy_disabled_allows_placeholder_bucket_config() -> None:
    policy = BatchPolicy(
        enabled=False,
        strategy="placeholder",
        buckets_sec=[],
        max_items_per_batch_by_bucket=[],
        max_audio_sec_per_batch=-1.0,
    )

    assert policy.enabled is False


def test_batch_policy_enabled_must_be_bool() -> None:
    with pytest.raises(TypeError, match="enabled must be a bool"):
        BatchPolicy(enabled="false")  # type: ignore[arg-type]


def test_batch_policy_prebatching_window_size_validation() -> None:
    with pytest.raises(TypeError, match="prebatching_window_size must be an int or None"):
        BatchPolicy(prebatching_window_size="8")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="prebatching_window_size must be > 0"):
        BatchPolicy(prebatching_window_size=0)

    policy = BatchPolicy(
        enabled=False,
        strategy="placeholder",
        buckets_sec=[],
        max_items_per_batch_by_bucket=[],
        max_audio_sec_per_batch=-1.0,
        prebatching_window_size=0,
    )
    assert policy.enabled is False


def test_batch_policy_numeric_field_validation() -> None:
    with pytest.raises(TypeError, match="flush_interval_ms must be an int"):
        BatchPolicy(flush_interval_ms=250.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="flush_interval_ms must be >= 0"):
        BatchPolicy(flush_interval_ms=-1)

    with pytest.raises(TypeError, match="buckets_sec entry must be numeric"):
        BatchPolicy(buckets_sec=[0, "60"], max_items_per_batch_by_bucket=[1, 1])  # type: ignore[list-item]

    with pytest.raises(TypeError, match="max_items_per_batch_by_bucket entry must be an int"):
        BatchPolicy(buckets_sec=[0, 60], max_items_per_batch_by_bucket=[1, True])  # type: ignore[list-item]

    with pytest.raises(TypeError, match="max_audio_sec_per_batch must be numeric or None"):
        BatchPolicy(max_audio_sec_per_batch=True)  # type: ignore[arg-type]


def test_batch_policy_bucket_for_clamps_above_top_edge() -> None:
    """Left-edge semantics: bucket i covers [buckets_sec[i], buckets_sec[i+1])."""
    p = BatchPolicy(buckets_sec=[0, 60, 600], max_items_per_batch_by_bucket=[10, 5, 1])
    assert p.bucket_for(0.0) == 0  # [0, 60)
    assert p.bucket_for(30.0) == 0  # [0, 60)
    assert p.bucket_for(60.0) == 1  # boundary lands in the bucket that starts at 60
    assert p.bucket_for(599.0) == 1  # [60, 600)
    assert p.bucket_for(600.0) == 2  # [600, +inf)
    assert p.bucket_for(9999.0) == 2  # clamped into top bucket


def test_bucket_queue_scheduler_flushes_on_caps_timer_and_drain() -> None:
    policy = BatchPolicy(
        buckets_sec=[0, 60],
        max_items_per_batch_by_bucket=[2, 2],
        max_audio_sec_per_batch=100.0,
        flush_interval_ms=50,
    )
    scheduler = BucketQueueScheduler(policy)

    assert scheduler.enqueue(0, "short-a", 10.0, now_ms=0.0) == []
    item_cap_batch = scheduler.enqueue(1, "short-b", 20.0, now_ms=10.0)
    assert [(batch.items, batch.total_cost, batch.flush_reason) for batch in item_cap_batch] == [
        (["short-a", "short-b"], 30.0, "item_cap")
    ]

    assert scheduler.enqueue(2, "long-a", 70.0, now_ms=20.0) == []
    cost_overflow_batch = scheduler.enqueue(3, "long-b", 80.0, now_ms=30.0)
    assert [(batch.items, batch.total_cost, batch.flush_reason) for batch in cost_overflow_batch] == [
        (["long-a"], 70.0, "capacity")
    ]
    assert [(batch.items, batch.flush_reason) for batch in scheduler.flush_all()] == [(["long-b"], "drain")]

    assert scheduler.enqueue(4, "timer-a", 5.0, now_ms=100.0) == []
    assert scheduler.flush_due(now_ms=149.0) == []
    timer_batch = scheduler.flush_due(now_ms=150.0)
    assert [(batch.items, batch.flush_reason) for batch in timer_batch] == [(["timer-a"], "timer")]


def test_bucket_queue_scheduler_can_disable_timer_checks_for_finite_planning() -> None:
    policy = BatchPolicy(
        buckets_sec=[0],
        max_items_per_batch_by_bucket=[10],
        max_audio_sec_per_batch=None,
        flush_interval_ms=1,
    )
    scheduler = BucketQueueScheduler(policy, enable_timer=False)

    assert scheduler.enqueue(0, "a", 1.0, now_ms=0.0) == []
    assert scheduler.flush_due(now_ms=10.0) == []
    assert [(batch.items, batch.flush_reason) for batch in scheduler.flush_all()] == [(["a"], "drain")]


# ----------------------------------------------------------------------
# bucketize: the finite planning form callers use before dispatching
# ----------------------------------------------------------------------


def test_bucketize_groups_by_bucket_and_orders_heaviest_first() -> None:
    policy = BatchPolicy(
        buckets_sec=[0, 60],
        max_items_per_batch_by_bucket=[4, 4],
        max_audio_sec_per_batch=None,
    )
    items = [{"d": 10.0}, {"d": 600.0}, {"d": 20.0}]

    plan = policy.bucketize_with_costs(items, cost_fn=lambda it: it["d"])

    assert [(indices, total_cost) for indices, _items, total_cost in plan] == [([1], 600.0), ([0, 2], 30.0)]
    assert policy.bucketize(items, cost_fn=lambda it: it["d"]) == [
        ([1], [{"d": 600.0}]),
        ([0, 2], [{"d": 10.0}, {"d": 20.0}]),
    ]


def test_bucketize_isolates_a_single_over_cost_item() -> None:
    """An item bigger than the whole cost cap still fires, alone."""
    policy = BatchPolicy(
        buckets_sec=[0],
        max_items_per_batch_by_bucket=[8],
        max_audio_sec_per_batch=50.0,
    )

    plan = policy.bucketize_with_costs([{"d": 80.0}, {"d": 10.0}], cost_fn=lambda it: it["d"])

    assert [(indices, total_cost) for indices, _items, total_cost in plan] == [([0], 80.0), ([1], 10.0)]


def test_bucketize_with_costs_disabled_returns_one_group() -> None:
    policy = BatchPolicy(enabled=False, buckets_sec=[], max_items_per_batch_by_bucket=[])
    items = [{"d": 1.0}, {"d": 2.0}]

    assert policy.bucketize_with_costs(items, cost_fn=lambda it: it["d"]) == [([0, 1], items, 0.0)]
    assert policy.bucketize_with_costs([], cost_fn=lambda it: it["d"]) == []


def test_dispatch_signature_covers_dispatch_constraints_only() -> None:
    """Window/timer knobs decide when to look, not whether a batch is safe to run."""
    base = BatchPolicy(
        buckets_sec=[0, 60],
        max_items_per_batch_by_bucket=[4, 2],
        max_audio_sec_per_batch=120.0,
        prebatching_window_size=8,
        flush_interval_ms=250,
    )
    wider_window = BatchPolicy(
        buckets_sec=[0, 60],
        max_items_per_batch_by_bucket=[4, 2],
        max_audio_sec_per_batch=120.0,
        prebatching_window_size=64,
        flush_interval_ms=1000,
    )
    smaller_cap = BatchPolicy(
        buckets_sec=[0, 60],
        max_items_per_batch_by_bucket=[4, 1],
        max_audio_sec_per_batch=120.0,
    )

    assert base.dispatch_signature(cost_unit="seconds") == wider_window.dispatch_signature(cost_unit="seconds")
    assert base.dispatch_signature(cost_unit="seconds") != smaller_cap.dispatch_signature(cost_unit="seconds")
    assert base.dispatch_signature(cost_unit="seconds") != base.dispatch_signature(cost_unit="tokens")


def test_dispatch_signature_requires_enabled_policy() -> None:
    policy = BatchPolicy(enabled=False, buckets_sec=[], max_items_per_batch_by_bucket=[])

    with pytest.raises(ValueError, match="must be enabled"):
        policy.dispatch_signature(cost_unit="seconds")


# ----------------------------------------------------------------------
# run_bucketed: the shared, stage-agnostic dispatch helper
# ----------------------------------------------------------------------


def test_run_bucketed_preserves_input_order_across_buckets() -> None:
    """Results realign to input order regardless of internal bucket order."""
    policy = BatchPolicy(
        buckets_sec=[0, 30, 1200],
        max_items_per_batch_by_bucket=[32, 16, 8],
        max_audio_sec_per_batch=None,
    )
    # durations: long, short, long, short -> two buckets, interleaved input.
    items = [{"d": 600.0, "v": "L0"}, {"d": 5.0, "v": "S1"}, {"d": 700.0, "v": "L2"}, {"d": 10.0, "v": "S3"}]
    calls: list[list[str]] = []

    def run_fn(sub: list[dict]) -> list[str]:
        calls.append([it["v"] for it in sub])
        return [it["v"] for it in sub]

    out = run_bucketed(items, run_fn, cost_fn=lambda it: it["d"], policy=policy)

    assert out == ["L0", "S1", "L2", "S3"]
    assert len(calls) == 2  # one per occupied bucket


def test_run_bucketed_without_policy_runs_single_call() -> None:
    items = [{"d": 1.0}, {"d": 2.0}, {"d": 3.0}]
    calls = 0

    def run_fn(sub: list[dict]) -> list[int]:
        nonlocal calls
        calls += 1
        return list(range(len(sub)))

    out = run_bucketed(items, run_fn, cost_fn=lambda it: it["d"], policy=None)

    assert calls == 1
    assert out == [0, 1, 2]


def test_run_bucketed_disabled_policy_runs_single_call() -> None:
    items = [{"d": 1.0}, {"d": 120.0}, {"d": 3.0}]
    policy = BatchPolicy(
        enabled=False,
        buckets_sec=[0, 60],
        max_items_per_batch_by_bucket=[1, 1],
        max_audio_sec_per_batch=None,
    )
    calls: list[list[float]] = []

    def run_fn(sub: list[dict]) -> list[float]:
        calls.append([it["d"] for it in sub])
        return [it["d"] for it in sub]

    out = run_bucketed(items, run_fn, cost_fn=lambda it: it["d"], policy=policy)

    assert out == [1.0, 120.0, 3.0]
    assert calls == [[1.0, 120.0, 3.0]]


def test_run_bucketed_empty_items_short_circuits() -> None:
    def run_fn(_sub: list) -> list:
        msg = "run_fn must not be called for empty items"
        raise AssertionError(msg)

    assert run_bucketed([], run_fn, cost_fn=lambda _it: 0.0) == []


def test_run_bucketed_mismatched_result_count_raises() -> None:
    def run_fn(_sub: list) -> list:
        return ["only-one"]

    with pytest.raises(RuntimeError, match=r"returned 1 results for 2 items"):
        run_bucketed([{"d": 1.0}, {"d": 2.0}], run_fn, cost_fn=lambda it: it["d"])
