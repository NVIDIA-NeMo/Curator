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

from nemo_curator.backends.base import (
    BaseStageAdapter,
    SchedulerReadyTaskBatch,
    assemble_scheduled_task_batch_results,
    build_scheduled_task_batch_plan,
    plan_upstream_task_batches,
    stage_uses_centralized_batching,
    stage_uses_upstream_prebatching,
    upstream_prebatching_batch_size,
)
from nemo_curator.models.asr.base import ASRResult
from nemo_curator.stages.audio.inference.asr import ASRStage
from nemo_curator.stages.audio.inference.batch_policy import BatchPolicy, BucketQueueScheduler, run_bucketed
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
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


class _GenericGpuCostStage(ProcessingStage[AudioTask, AudioTask]):
    name = "generic_gpu_cost_stage"
    resources = Resources(gpus=1.0)
    batch_size = 32

    def __init__(self, batch_policy: BatchPolicy) -> None:
        self.batch_policy = batch_policy
        self.cost_calls = 0

    def process(self, task: AudioTask) -> AudioTask:
        return task

    def batch_task_cost(self, task: AudioTask) -> float:
        self.cost_calls += 1
        return float(task.data["duration"])


class _GenericCentralizedStage(_GenericGpuCostStage):
    name = "generic_centralized_stage"

    def __init__(self, batch_policy: BatchPolicy) -> None:
        super().__init__(batch_policy)
        self.build_calls = 0
        self.process_calls = 0

    def build_prebucketed_tasks(self, tasks: list[AudioTask]) -> list[AudioTask]:
        self.build_calls += 1
        return list(tasks)

    def scheduler_task_cost(self, task: AudioTask) -> float:
        return self.batch_task_cost(task)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        self.process_calls += 1
        for task in tasks:
            task.data["processed_batch_size"] = len(tasks)
        return tasks

    def assemble_prebucketed_task_results(
        self,
        tasks: list[AudioTask],
        processed_tasks: list[AudioTask],
    ) -> list[AudioTask]:
        return processed_tasks or tasks

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
    assert p.bucket_for(0.0) == 0     # [0, 60)
    assert p.bucket_for(30.0) == 0    # [0, 60)
    assert p.bucket_for(60.0) == 1    # boundary lands in the bucket that starts at 60
    assert p.bucket_for(599.0) == 1   # [60, 600)
    assert p.bucket_for(600.0) == 2   # [600, +inf)
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


def test_base_adapter_prebuckets_asr_process_batch_inputs() -> None:
    """Backend adapters split mixed batches before ASRStage.process_batch runs."""
    policy = BatchPolicy(
        strategy="duration_bucketed",
        buckets_sec=[0, 30, 1200],
        max_items_per_batch_by_bucket=[32, 16, 8],
        max_audio_sec_per_batch=None,
    )
    stage = _make_stage(batch_policy=policy)
    long_task = _make_task(_SR * 600)
    short_a = _make_task(_SR * 5)
    short_b = _make_task(_SR * 10)

    stage._adapter.transcribe_batch.side_effect = [
        [ASRResult(text="long")],
        [ASRResult(text="short-a"), ASRResult(text="short-b")],
    ]

    results = BaseStageAdapter(stage).process_batch([long_task, short_a, short_b])

    assert stage._adapter.transcribe_batch.call_count == 2
    durations_by_call = [
        [item["audio_seconds"] for item in call.args[0]]
        for call in stage._adapter.transcribe_batch.call_args_list
    ]
    assert durations_by_call == [[600.0], [5.0, 10.0]]
    assert results[0].data["qwen3_prediction_s1"] == "long"
    assert results[1].data["qwen3_prediction_s1"] == "short-a"
    assert results[2].data["qwen3_prediction_s1"] == "short-b"


def test_base_adapter_prebuckets_asr_chunks_before_process_batch() -> None:
    """Long-row tails batch with matching chunks before ASRStage.process_batch."""
    policy = BatchPolicy(
        strategy="duration_bucketed",
        buckets_sec=[0, 600, 1200, 2400],
        max_items_per_batch_by_bucket=[32, 16, 8, 4],
        max_audio_sec_per_batch=2400.0,
    )
    stage = _make_stage(batch_policy=policy)
    long_50m = _make_task(_SR * 3000)
    ten_min = _make_task(_SR * 600)
    tiny = _make_task(_SR * 5)

    stage._adapter.transcribe_batch.side_effect = [
        [ASRResult(text="long-40m")],
        [ASRResult(text="tail"), ASRResult(text="ten")],
        [ASRResult(text="tiny")],
    ]

    results = BaseStageAdapter(stage).process_batch([long_50m, ten_min, tiny])

    durations_by_call = [
        [item["audio_seconds"] for item in call.args[0]]
        for call in stage._adapter.transcribe_batch.call_args_list
    ]
    assert durations_by_call == [[2400.0], [600.0, 600.0], [5.0]]
    assert results[0].data["qwen3_prediction_s1"] == "long-40m tail"
    assert results[1].data["qwen3_prediction_s1"] == "ten"
    assert results[2].data["qwen3_prediction_s1"] == "tiny"


def test_upstream_prebatching_uses_enabled_policy_as_opt_in() -> None:
    policy = BatchPolicy(
        enabled=False,
        buckets_sec=[0, 30, 1200],
        max_items_per_batch_by_bucket=[32, 16, 8],
        max_audio_sec_per_batch=None,
    )
    stage = _GenericGpuCostStage(policy)

    assert stage_uses_upstream_prebatching(stage) is False
    assert upstream_prebatching_batch_size(stage, 32) == 32

    policy.enabled = True

    assert stage_uses_upstream_prebatching(stage) is True
    assert stage_uses_centralized_batching(stage) is False
    assert upstream_prebatching_batch_size(stage, 32) == 56

    policy.prebatching_window_size = 12
    assert upstream_prebatching_batch_size(stage, 32) == 12


def test_upstream_prebatching_generic_cost_stage_batches_before_workers() -> None:
    policy = BatchPolicy(
        buckets_sec=[0, 30, 1200],
        max_items_per_batch_by_bucket=[32, 16, 8],
        max_audio_sec_per_batch=None,
    )
    stage = _GenericGpuCostStage(policy)
    short_a = AudioTask(data={"duration": 5.0})
    short_b = AudioTask(data={"duration": 10.0})
    long = AudioTask(data={"duration": 600.0})

    planned = plan_upstream_task_batches(stage, [long, short_a, short_b])

    assert planned == [[long], [short_a, short_b]]
    assert stage.cost_calls == 3


def test_asr_centralized_planner_materializes_dispatch_chunks() -> None:
    policy = BatchPolicy(
        strategy="duration_bucketed",
        buckets_sec=[0, 600, 1200, 2400],
        max_items_per_batch_by_bucket=[32, 16, 8, 4],
        max_audio_sec_per_batch=2400.0,
    )
    stage = _make_stage(batch_policy=policy)
    long_50m = _make_task(_SR * 3000)
    ten_min = _make_task(_SR * 600)

    planned = plan_upstream_task_batches(stage, [long_50m, ten_min])

    assert stage_uses_centralized_batching(stage) is True
    assert [[task.data["_curator_asr_chunk_cost"] for task in batch] for batch in planned] == [
        [2400.0],
        [600.0, 600.0],
    ]
    assert all("_curator_asr_parent_idx" in task.data for batch in planned for task in batch)


def test_asr_prebucket_chunk_task_uses_minimal_data() -> None:
    stage = _make_stage(batch_policy=None)
    waveform = np.zeros(_SR, dtype=np.float32)
    task = AudioTask(
        task_id="parent",
        dataset_name="dataset",
        data={
            "waveform": waveform,
            "sample_rate": _SR,
            "source_lang": "en",
            "large_extra_column": object(),
        },
    )

    chunk_task = stage._make_prebucket_chunk_task(task, waveform, _SR, 0, 1)

    assert set(chunk_task.data.keys()) == {
        "waveform",
        "sample_rate",
        "source_lang",
        "_curator_asr_chunk_idx",
        "_curator_asr_chunk_count",
    }
    assert "large_extra_column" not in chunk_task.data


def test_asr_upstream_planner_splits_fanout_parents_across_bucket_batches() -> None:
    policy = BatchPolicy(
        strategy="duration_bucketed",
        buckets_sec=[0, 600, 1200, 2400],
        max_items_per_batch_by_bucket=[32, 16, 8, 4],
        max_audio_sec_per_batch=2400.0,
    )
    stage = _make_stage(batch_policy=policy)
    long_50m = _make_task(_SR * 3000)
    ten_min = _make_task(_SR * 600)
    tiny = _make_task(_SR * 5)

    planned = plan_upstream_task_batches(stage, [long_50m, ten_min, tiny])

    planned_costs = [[task.data["_curator_asr_chunk_cost"] for task in batch] for batch in planned]
    assert planned_costs == [[2400.0], [600.0, 600.0], [5.0]]


def test_asr_centralized_plan_assembles_out_of_order_chunk_results() -> None:
    policy = BatchPolicy(
        strategy="duration_bucketed",
        buckets_sec=[0, 600, 1200, 2400],
        max_items_per_batch_by_bucket=[32, 16, 8, 4],
        max_audio_sec_per_batch=2400.0,
    )
    stage = _make_stage(batch_policy=policy)
    long_50m = _make_task(_SR * 3000)
    ten_min = _make_task(_SR * 600)

    plan = build_scheduled_task_batch_plan(stage, [long_50m, ten_min])
    assert plan is not None

    processed_chunks = [task for batch in reversed(plan.task_batches) for task in batch]
    for task in processed_chunks:
        task.data["qwen3_prediction_s1"] = (
            f"chunk-{task.data['_curator_asr_parent_idx']}-{task.data['_curator_asr_chunk_idx']}"
        )

    results = assemble_scheduled_task_batch_results(stage, plan, processed_chunks)

    assert results[0].data["qwen3_prediction_s1"] == "chunk-0-0 chunk-0-1"
    assert results[1].data["qwen3_prediction_s1"] == "chunk-1-0"


def test_scheduler_ready_batch_bypasses_recursive_planning_for_generic_stage() -> None:
    policy = BatchPolicy(
        buckets_sec=[0, 30, 1200],
        max_items_per_batch_by_bucket=[2, 1, 1],
        max_audio_sec_per_batch=None,
    )
    stage = _GenericCentralizedStage(policy)
    first = AudioTask(data={"duration": 5.0})
    second = AudioTask(data={"duration": 10.0})

    adapter = BaseStageAdapter(stage)
    results = adapter.process_scheduler_ready_batch(SchedulerReadyTaskBatch(tasks=[first, second]))

    assert results == [first, second]
    assert stage.build_calls == 0
    assert stage.process_calls == 1
    assert [task.data["processed_batch_size"] for task in results] == [2, 2]


def test_asr_centralized_assembly_rejects_duplicate_chunk_index() -> None:
    policy = BatchPolicy(
        strategy="duration_bucketed",
        buckets_sec=[0, 600, 1200, 2400],
        max_items_per_batch_by_bucket=[32, 16, 8, 4],
        max_audio_sec_per_batch=2400.0,
    )
    stage = _make_stage(batch_policy=policy)
    long_50m = _make_task(_SR * 3000)
    plan = build_scheduled_task_batch_plan(stage, [long_50m])
    assert plan is not None
    first_chunk = plan.task_batches[0][0]

    with pytest.raises(RuntimeError, match="duplicate chunk index"):
        assemble_scheduled_task_batch_results(stage, plan, [first_chunk, first_chunk])


def test_asr_centralized_assembly_rejects_missing_chunk_index() -> None:
    policy = BatchPolicy(
        strategy="duration_bucketed",
        buckets_sec=[0, 600, 1200, 2400],
        max_items_per_batch_by_bucket=[32, 16, 8, 4],
        max_audio_sec_per_batch=2400.0,
    )
    stage = _make_stage(batch_policy=policy)
    long_50m = _make_task(_SR * 3000)
    plan = build_scheduled_task_batch_plan(stage, [long_50m])
    assert plan is not None
    first_chunk = plan.task_batches[0][0]

    with pytest.raises(RuntimeError, match="exact chunk results"):
        assemble_scheduled_task_batch_results(stage, plan, [first_chunk])


def test_base_adapter_centralizes_asr_chunking_and_bucketing(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = BatchPolicy(
        strategy="duration_bucketed",
        buckets_sec=[0, 600, 1200, 2400],
        max_items_per_batch_by_bucket=[32, 16, 8, 4],
        max_audio_sec_per_batch=2400.0,
    )
    stage = _make_stage(batch_policy=policy)
    long_50m = _make_task(_SR * 3000)
    ten_min = _make_task(_SR * 600)
    tiny = _make_task(_SR * 5)

    chunk_waveform_calls = 0
    original_chunk_waveform = stage._chunk_waveform

    def counting_chunk_waveform(waveform: np.ndarray, sample_rate: int, max_seconds: float) -> list[np.ndarray]:
        nonlocal chunk_waveform_calls
        chunk_waveform_calls += 1
        return original_chunk_waveform(waveform, sample_rate, max_seconds)

    monkeypatch.setattr(stage, "_chunk_waveform", counting_chunk_waveform)
    stage._adapter.transcribe_batch.side_effect = [
        [ASRResult(text="long-40m")],
        [ASRResult(text="tail"), ASRResult(text="ten")],
        [ASRResult(text="tiny")],
    ]

    results = BaseStageAdapter(stage).process_batch([long_50m, ten_min, tiny])

    durations_by_call = [
        [item["audio_seconds"] for item in call.args[0]]
        for call in stage._adapter.transcribe_batch.call_args_list
    ]
    assert durations_by_call == [[2400.0], [600.0, 600.0], [5.0]]
    assert chunk_waveform_calls == 3
    assert results[0].data["qwen3_prediction_s1"] == "long-40m tail"
    assert results[1].data["qwen3_prediction_s1"] == "ten"
    assert results[2].data["qwen3_prediction_s1"] == "tiny"


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
        [ASRResult(text="long")],
        [ASRResult(text="short-a"), ASRResult(text="short-b")],
    ]
    results = BaseStageAdapter(stage).process_batch([short_a, short_b, long_a])

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

    BaseStageAdapter(stage).process_batch([_make_task() for _ in range(3)])

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

    BaseStageAdapter(stage).process_batch([_make_task(_SR * 10), _make_task(_SR * 10)])

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
