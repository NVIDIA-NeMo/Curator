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

from unittest.mock import MagicMock

import numpy as np
import pytest

from nemo_curator.models.asr.base import ASRResult
from nemo_curator.stages.audio.inference.asr import ASRStage
from nemo_curator.stages.audio.inference.batch_policy import BatchPolicy, run_bucketed
from nemo_curator.tasks import AudioTask

_QWEN_ADAPTER_TARGET = "nemo_curator.models.asr.qwen_omni.QwenOmniASRAdapter"
_SR = 16000


def _make_stage(*, batch_policy: BatchPolicy | None = None) -> ASRStage:
    stage = ASRStage(
        adapter_target=_QWEN_ADAPTER_TARGET,
        model_id="mock/qwen-omni",
        pred_text_key="qwen3_prediction_s1",
        batch_policy=batch_policy,
    )
    mock_adapter = MagicMock()
    mock_adapter.last_metrics = {}
    stage._adapter = mock_adapter
    return stage


def _make_task(waveform_len: int = _SR) -> AudioTask:
    return AudioTask(data={"waveform": np.zeros(waveform_len, dtype=np.float32), "sample_rate": _SR})

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


def test_batch_policy_bucket_for_clamps_above_top_edge() -> None:
    """Left-edge semantics: bucket i covers [buckets_sec[i], buckets_sec[i+1])."""
    p = BatchPolicy(buckets_sec=[0, 60, 600], max_items_per_batch_by_bucket=[10, 5, 1])
    assert p.bucket_for(0.0) == 0     # [0, 60)
    assert p.bucket_for(30.0) == 0    # [0, 60)
    assert p.bucket_for(60.0) == 1    # boundary lands in the bucket that starts at 60
    assert p.bucket_for(599.0) == 1   # [60, 600)
    assert p.bucket_for(600.0) == 2   # [600, +inf)
    assert p.bucket_for(9999.0) == 2  # clamped into top bucket


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


def test_asr_batch_policy_partitions_items_by_bucket() -> None:
    policy = BatchPolicy(
        strategy="duration_bucketed",
        buckets_sec=[0, 30, 1200, 2400],
        max_items_per_batch_by_bucket=[32, 16, 8, 4],
        max_audio_sec_per_batch=10_000.0,
        flush_interval_ms=250,
    )
    stage = _make_stage(batch_policy=policy)
    short_a = _make_task(_SR * 5)
    short_b = _make_task(_SR * 10)
    long_a = _make_task(_SR * 600)

    stage._adapter.transcribe_batch.side_effect = [
        [ASRResult(text="short-a"), ASRResult(text="short-b")],
        [ASRResult(text="long")],
    ]
    results = stage.process_batch([short_a, short_b, long_a])

    assert stage._adapter.transcribe_batch.call_count == 2
    assert results[0].data["qwen3_prediction_s1"] == "short-a"
    assert results[1].data["qwen3_prediction_s1"] == "short-b"
    assert results[2].data["qwen3_prediction_s1"] == "long"


def test_asr_batch_policy_respects_per_bucket_item_cap() -> None:
    policy = BatchPolicy(
        strategy="duration_bucketed",
        buckets_sec=[0, 60],
        max_items_per_batch_by_bucket=[2, 1],
        max_audio_sec_per_batch=None,
    )
    stage = _make_stage(batch_policy=policy)
    stage._adapter.transcribe_batch.side_effect = [
        [ASRResult(text="a"), ASRResult(text="b")],
        [ASRResult(text="c")],
    ]

    stage.process_batch([_make_task() for _ in range(3)])

    assert stage._adapter.transcribe_batch.call_count == 2


def test_asr_batch_policy_respects_audio_sec_cap() -> None:
    policy = BatchPolicy(
        strategy="duration_bucketed",
        buckets_sec=[0, 60],
        max_items_per_batch_by_bucket=[100, 100],
        max_audio_sec_per_batch=15.0,
    )
    stage = _make_stage(batch_policy=policy)
    stage._adapter.transcribe_batch.side_effect = [
        [ASRResult(text="a")],
        [ASRResult(text="b")],
    ]

    stage.process_batch([_make_task(_SR * 10), _make_task(_SR * 10)])

    assert stage._adapter.transcribe_batch.call_count == 2


def test_asr_batch_policy_none_runs_single_adapter_call() -> None:
    stage = _make_stage(batch_policy=None)
    stage._adapter.transcribe_batch.return_value = [
        ASRResult(text="a"),
        ASRResult(text="b"),
    ]

    stage.process_batch([_make_task(), _make_task()])

    assert stage._adapter.transcribe_batch.call_count == 1


def test_asr_batch_policy_disabled_runs_single_adapter_call() -> None:
    policy = BatchPolicy(
        enabled=False,
        buckets_sec=[0, 30, 1200, 2400],
        max_items_per_batch_by_bucket=[1, 1, 1, 1],
        max_audio_sec_per_batch=None,
    )
    stage = _make_stage(batch_policy=policy)
    stage._adapter.transcribe_batch.return_value = [
        ASRResult(text="short"),
        ASRResult(text="long"),
    ]

    stage.process_batch([_make_task(_SR * 5), _make_task(_SR * 600)])

    assert stage._adapter.transcribe_batch.call_count == 1
