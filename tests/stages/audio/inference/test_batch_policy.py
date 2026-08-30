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

"""Tests for finite, duration-only audio batch planning."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import pytest

from nemo_curator.stages.audio.inference.batch_policy import BatchPolicy


@pytest.mark.parametrize(
    ("kwargs", "expected_exception", "match"),
    [
        ({"buckets_sec": []}, ValueError, "must contain at least one edge"),
        ({"buckets_sec": [1.0]}, ValueError, "must start at 0.0"),
        (
            {"buckets_sec": [0.0, 10.0, 10.0]},
            ValueError,
            "must be strictly increasing",
        ),
        (
            {"buckets_sec": [0.0, "10"]},
            TypeError,
            "buckets_sec entry must be numeric",
        ),
        (
            {"buckets_sec": [0.0, -1.0]},
            ValueError,
            "must be finite and non-negative",
        ),
        (
            {"buckets_sec": [0.0, float("inf")]},
            ValueError,
            "must be finite and non-negative",
        ),
        ({"max_audio_sec_per_batch": True}, TypeError, "max_audio_sec_per_batch must be numeric or None"),
        ({"max_audio_sec_per_batch": 0.0}, ValueError, "max_audio_sec_per_batch must be finite and > 0"),
        ({"max_audio_sec_per_batch": float("nan")}, ValueError, "max_audio_sec_per_batch must be finite and > 0"),
    ],
)
def test_batch_policy_rejects_invalid_configuration(
    kwargs: dict[str, Any],
    expected_exception: type[Exception],
    match: str,
) -> None:
    with pytest.raises(expected_exception, match=match):
        BatchPolicy(**kwargs)


def test_batch_policy_exposes_only_audio_duration_configuration() -> None:
    assert [field.name for field in fields(BatchPolicy)] == [
        "buckets_sec",
        "max_audio_sec_per_batch",
    ]


def test_batch_policy_accepts_no_total_audio_cap() -> None:
    policy = BatchPolicy(max_audio_sec_per_batch=None)

    assert policy.max_audio_sec_per_batch is None


def test_bucket_for_uses_left_edge_boundaries_and_clamps_above_top_edge() -> None:
    policy = BatchPolicy(buckets_sec=[0.0, 60.0, 600.0])

    assert policy.bucket_for(0.0) == 0
    assert policy.bucket_for(59.999) == 0
    assert policy.bucket_for(60.0) == 1
    assert policy.bucket_for(599.999) == 1
    assert policy.bucket_for(600.0) == 2
    assert policy.bucket_for(9999.0) == 2


@pytest.mark.parametrize(
    ("item", "expected_exception", "match"),
    [
        ({}, TypeError, "must provide numeric audio_seconds"),
        ({"audio_seconds": "10"}, TypeError, "must provide numeric audio_seconds"),
        ({"audio_seconds": True}, TypeError, "must provide numeric audio_seconds"),
        ({"audio_seconds": -1.0}, ValueError, "must be finite and non-negative"),
        ({"audio_seconds": float("inf")}, ValueError, "must be finite and non-negative"),
        ({"audio_seconds": float("nan")}, ValueError, "must be finite and non-negative"),
    ],
)
def test_bucketize_rejects_invalid_item_audio_seconds(
    item: dict[str, Any],
    expected_exception: type[Exception],
    match: str,
) -> None:
    policy = BatchPolicy()

    with pytest.raises(expected_exception, match=match):
        policy.bucketize([item])


def test_bucketize_groups_by_duration_and_orders_heaviest_calls_first() -> None:
    policy = BatchPolicy(
        buckets_sec=[0.0, 60.0],
        max_audio_sec_per_batch=None,
    )
    items = [
        {"audio_seconds": 10.0, "name": "short-a"},
        {"audio_seconds": 600.0, "name": "long"},
        {"audio_seconds": 20.0, "name": "short-b"},
    ]

    assert policy.bucketize(items) == [
        ([1], [items[1]]),
        ([0, 2], [items[0], items[2]]),
    ]


def test_bucketize_keeps_one_call_per_duration_bucket_without_audio_cap() -> None:
    policy = BatchPolicy(
        buckets_sec=[0.0, 10.0],
        max_audio_sec_per_batch=None,
    )
    items = [
        {"audio_seconds": 1.0},
        {"audio_seconds": 2.0},
        {"audio_seconds": 3.0},
        {"audio_seconds": 10.0},
        {"audio_seconds": 11.0},
    ]

    assert [indices for indices, _batch in policy.bucketize(items)] == [[3, 4], [0, 1, 2]]


def test_bucketize_splits_calls_at_total_audio_seconds_cap() -> None:
    policy = BatchPolicy(
        buckets_sec=[0.0],
        max_audio_sec_per_batch=100.0,
    )
    items = [
        {"audio_seconds": 40.0},
        {"audio_seconds": 50.0},
        {"audio_seconds": 30.0},
        {"audio_seconds": 70.0},
    ]

    assert policy.bucketize(items) == [
        ([2, 3], [items[2], items[3]]),
        ([0, 1], [items[0], items[1]]),
    ]


def test_bucketize_emits_single_item_above_total_audio_seconds_cap_without_loss() -> None:
    policy = BatchPolicy(
        buckets_sec=[0.0],
        max_audio_sec_per_batch=50.0,
    )
    items = [{"audio_seconds": 80.0}, {"audio_seconds": 10.0}]

    assert policy.bucketize(items) == [
        ([0], [items[0]]),
        ([1], [items[1]]),
    ]


def test_bucketize_empty_input_returns_no_calls() -> None:
    assert BatchPolicy().bucketize([]) == []


def test_bucketize_returns_every_item_once_with_original_indices_and_relative_order() -> None:
    policy = BatchPolicy(
        buckets_sec=[0.0, 10.0, 100.0],
        max_audio_sec_per_batch=None,
    )
    items = [
        {"audio_seconds": 5.0, "name": "a"},
        {"audio_seconds": 200.0, "name": "b"},
        {"audio_seconds": 12.0, "name": "c"},
        {"audio_seconds": 8.0, "name": "d"},
        {"audio_seconds": 150.0, "name": "e"},
        {"audio_seconds": 13.0, "name": "f"},
        {"audio_seconds": 6.0, "name": "g"},
    ]

    plan = policy.bucketize(items)

    assert [indices for indices, _batch in plan] == [[1, 4], [2, 5], [0, 3, 6]]
    planned_indices = [index for indices, _batch in plan for index in indices]
    assert sorted(planned_indices) == list(range(len(items)))
    for indices, batch in plan:
        assert batch == [items[index] for index in indices]
