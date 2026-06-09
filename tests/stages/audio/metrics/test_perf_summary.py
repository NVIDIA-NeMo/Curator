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

"""Tests for identity-driven perf summary and per-GPU scheduling breakdown.

Covers, with plain ``StagePerfStats`` records (no GPU / no Ray):
    * ``serialize_stage_perf`` carries identity labels when present, omits them
      when blank;
    * ``_fingerprint_perf`` distinguishes records that differ ONLY by identity
      (so two actors with byte-equal timings are not collapsed by dedup);
    * ``AudioPerformanceSummary`` emits the scheduling-half ``per_gpu`` block +
      ``gpu_ids`` / ``gpu_count`` / ``actor_count`` and the top-level
      ``pipeline_throughput`` rollup;
    * stages without resolved identity (CPU / non-Ray) emit none of the above.

The GPU *hardware* fields (uuid/device_name/util/mem/gpu_hours) are the
separate NVML/DCGM proposal and are intentionally NOT asserted here.
"""

from __future__ import annotations

from nemo_curator.stages.audio.metrics.performance import (
    AudioPerformanceSummary,
    serialize_stage_perf,
)
from nemo_curator.utils.performance_utils import StagePerfStats


def _perf(
    *,
    gpu_id: str = "",
    actor_id: str = "",
    idle: float = 0.0,
    items: int = 32,
    audio_s: float = 100.0,
) -> StagePerfStats:
    # node_id is the host half of the "<node>:<idx>" gpu label (blank for CPU).
    node_id = gpu_id.split(":", 1)[0] if gpu_id else ""
    return StagePerfStats(
        stage_name="QwenOmni_inference",
        process_time=1.0,
        actor_idle_time=idle,
        num_items_processed=items,
        custom_metrics={"audio_duration_s": audio_s, "utterances_input": float(items)},
        actor_id=actor_id,
        node_id=node_id,
        gpu_id=gpu_id,
    )


# ----------------------------------------------------------------------
# Serialization + fingerprint
# ----------------------------------------------------------------------


def test_serialize_stage_perf_carries_identity_when_present() -> None:
    [entry] = serialize_stage_perf([_perf(gpu_id="node-1:2", actor_id="S:actor-ab")])
    assert entry["gpu_id"] == "node-1:2"
    assert entry["actor_id"] == "S:actor-ab"
    assert entry["node_id"] == "node-1"  # derived from the gpu label host half


def test_serialize_stage_perf_omits_blank_identity() -> None:
    [entry] = serialize_stage_perf([_perf()])  # no identity resolved (CPU / non-Ray)
    assert "gpu_id" not in entry
    assert "actor_id" not in entry
    assert "node_id" not in entry


def test_fingerprint_distinguishes_actors_with_equal_timings() -> None:
    """Two records byte-identical except for identity must NOT dedup to one."""
    summary = AudioPerformanceSummary(duration_key="duration")
    a = _perf(gpu_id="node-0:0", actor_id="S:actor-a")
    b = _perf(gpu_id="node-0:1", actor_id="S:actor-b")
    summary.record_stage_perf([a, b])
    stage = summary.build_stage_summaries()["QwenOmni_inference"]
    assert stage["invocation_count"] == 2.0  # both counted, not collapsed by dedup


# ----------------------------------------------------------------------
# Per-GPU scheduling breakdown + topology
# ----------------------------------------------------------------------


def test_per_gpu_scheduling_breakdown_and_topology() -> None:
    summary = AudioPerformanceSummary(duration_key="duration")
    # GPU 0: two invocations (actor-a); GPU 1: two invocations (actor-b).
    records = [
        _perf(gpu_id="node-0:0", actor_id="S:actor-a", idle=0.10, items=32, audio_s=100.0),
        _perf(gpu_id="node-0:0", actor_id="S:actor-a", idle=0.30, items=32, audio_s=120.0),
        _perf(gpu_id="node-0:1", actor_id="S:actor-b", idle=0.05, items=16, audio_s=50.0),
        _perf(gpu_id="node-0:1", actor_id="S:actor-b", idle=0.20, items=16, audio_s=70.0),
    ]
    summary.record_stage_perf(records)
    stage = summary.build_stage_summaries()["QwenOmni_inference"]

    # Topology
    assert stage["gpu_ids"] == ["node-0:0", "node-0:1"]
    assert stage["gpu_count"] == 2.0
    assert stage["actor_count"] == 2.0

    per_gpu = stage["per_gpu"]
    assert set(per_gpu) == {"node-0:0", "node-0:1"}

    g0 = per_gpu["node-0:0"]
    assert g0["actor_id"] == "S:actor-a"
    assert g0["items_processed"] == 64.0  # 32 + 32
    assert g0["audio_hours_in"] == (100.0 + 120.0) / 3600.0
    assert "batch_size_p50" in g0
    assert "queue_wait_s_p50" in g0
    assert "queue_wait_s_p95" in g0

    g1 = per_gpu["node-0:1"]
    assert g1["actor_id"] == "S:actor-b"
    assert g1["items_processed"] == 32.0  # 16 + 16
    assert g1["audio_hours_in"] == (50.0 + 70.0) / 3600.0


def test_pipeline_throughput_rollup_unions_gpu_ids() -> None:
    summary = AudioPerformanceSummary(duration_key="duration")
    summary.record_stage_perf([
        _perf(gpu_id="node-0:0", actor_id="S:actor-a", audio_s=3600.0),
        _perf(gpu_id="node-1:0", actor_id="S:actor-b", audio_s=3600.0),
    ])
    # total_audio_seconds is normally driven by record_task; set it directly here.
    summary._total_audio_seconds = 7200.0  # 2 audio-hours
    out = summary.build_summary(wall_time_s=3600.0)  # 1 wall-hour

    pt = out["pipeline_throughput"]
    assert pt["gpu_ids"] == ["node-0:0", "node-1:0"]
    assert pt["gpu_count"] == 2.0
    assert pt["audio_hours_per_wallclock_hour"] == 2.0  # 2 audio-h / 1 wall-h


# ----------------------------------------------------------------------
# Graceful absence when identity is unresolved (CPU / non-Ray)
# ----------------------------------------------------------------------


def test_no_identity_emits_no_per_gpu_or_topology() -> None:
    summary = AudioPerformanceSummary(duration_key="duration")
    summary.record_stage_perf([_perf(), _perf(idle=0.5)])  # blank gpu_id/actor_id
    stage = summary.build_stage_summaries()["QwenOmni_inference"]
    assert "per_gpu" not in stage
    assert "gpu_ids" not in stage
    assert "gpu_count" not in stage
    assert "actor_count" not in stage

    out = summary.build_summary(wall_time_s=10.0)
    assert "gpu_ids" not in out.get("pipeline_throughput", {})


# ----------------------------------------------------------------------
# Mixed pipeline: GPU stages and CPU stages coexist in one summary
# ----------------------------------------------------------------------


def _cpu_perf(stage_name: str, actor_id: str) -> StagePerfStats:
    """A CPU-stage record as a real Ray run produces it: actor + node resolved,
    but no GPU assigned (``ray.get_gpu_ids()`` is empty) so ``gpu_id`` is blank."""
    return StagePerfStats(
        stage_name=stage_name,
        process_time=0.5,
        num_items_processed=64,
        custom_metrics={"writer_process_calls": 2.0},
        actor_id=actor_id,
        node_id="node-0",
        gpu_id="",  # CPU stage: no GPU
    )


def test_cpu_stage_gets_actor_count_but_no_gpu_fields() -> None:
    """A CPU stage (actor resolved, gpu_id blank) gets actor_count + its scalar
    totals, but NO gpu_ids / gpu_count / per_gpu."""
    summary = AudioPerformanceSummary(duration_key="duration")
    summary.record_stage_perf([
        _cpu_perf("ShardedManifestWriter", "ShardedManifestWriter:actor-cpu01"),
    ])
    stage = summary.build_stage_summaries()["ShardedManifestWriter"]
    assert stage["actor_count"] == 1.0
    assert stage["total_items_processed"] == 64.0
    assert "gpu_ids" not in stage
    assert "gpu_count" not in stage
    assert "per_gpu" not in stage


def test_mixed_gpu_and_cpu_stages_in_one_pipeline() -> None:
    """The realistic case: a GPU inference stage and a CPU writer stage in the
    same summary. Only the GPU stage gets the per-GPU breakdown; the CPU stage
    gets actor_count only. The pipeline rollup unions just the GPU ids."""
    summary = AudioPerformanceSummary(duration_key="duration")
    summary.record_stage_perf([
        _perf(gpu_id="node-0:0", actor_id="QwenOmni_inference:actor-a", items=32, audio_s=100.0),
        _cpu_perf("ShardedManifestWriter", "ShardedManifestWriter:actor-cpu01"),
    ])
    stages = summary.build_stage_summaries()

    gpu_stage = stages["QwenOmni_inference"]
    assert gpu_stage["gpu_ids"] == ["node-0:0"]
    assert gpu_stage["actor_count"] == 1.0
    assert set(gpu_stage["per_gpu"]) == {"node-0:0"}

    cpu_stage = stages["ShardedManifestWriter"]
    assert cpu_stage["actor_count"] == 1.0
    assert "gpu_ids" not in cpu_stage
    assert "per_gpu" not in cpu_stage

    # Pipeline rollup unions ONLY the GPU ids (the CPU stage contributes none).
    summary._total_audio_seconds = 100.0  # exercise the throughput branch
    pt = summary.build_summary(wall_time_s=50.0)["pipeline_throughput"]
    assert pt["gpu_ids"] == ["node-0:0"]
    assert pt["gpu_count"] == 1.0
