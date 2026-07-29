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

"""Tests for the identity-driven perf summary and per-actor (GPU/CPU) scheduling breakdown."""

from __future__ import annotations

from dataclasses import dataclass, field

from nemo_curator.stages.audio.metrics.performance import (
    AudioPerformanceSummary,
    serialize_stage_perf,
)
from nemo_curator.tasks import AudioTask


@dataclass
class _PerfRecord:
    """Duck-typed record proving summaries do not require the telemetry PR."""

    stage_name: str
    process_time: float = 0.0
    actor_idle_time: float = 0.0
    input_data_size_mb: float = 0.0
    num_items_processed: int = 0
    custom_metrics: dict[str, float] = field(default_factory=dict)
    stage_id: str = ""
    invocation_id: str = ""
    actor_id: str = ""
    node_id: str = ""
    gpu_id: str = ""
    physical_address: str = ""
    pod_ip: str = ""
    hostname: str = ""
    gpu_indices: list[int] = field(default_factory=list)
    gpu_uuids: list[str] = field(default_factory=list)


def _perf(
    *,
    addr: str = "",
    actor_id: str = "",
    idle: float = 0.0,
    items: int = 32,
    audio_s: float = 100.0,
) -> _PerfRecord:
    """A GPU-stage record keyed by physical address ``<host>:<idx[,idx]>`` (blank addr -> CPU stage)."""
    node_id = ""
    gpu_indices: list[int] = []
    if addr:
        host, _, idx_part = addr.rpartition(":")
        node_id = host
        gpu_indices = [int(x) for x in idx_part.split(",") if x.strip()]
    return _PerfRecord(
        stage_name="FastConformer_inference",
        process_time=1.0,
        actor_idle_time=idle,
        num_items_processed=items,
        custom_metrics={"audio_duration_s": audio_s, "utterances_input": float(items)},
        actor_id=actor_id,
        node_id=node_id,
        physical_address=addr,
        gpu_indices=gpu_indices,
    )


# ----------------------------------------------------------------------
# Serialization + fingerprint
# ----------------------------------------------------------------------


def test_serialize_stage_perf_carries_identity_when_present() -> None:
    [entry] = serialize_stage_perf(
        [
            _PerfRecord(
                stage_name="FastConformer_inference",
                process_time=1.0,
                actor_id="S:actor-ab",
                node_id="node-1",
                gpu_id="node-1:2",
                physical_address="10.0.0.5:2",
            )
        ]
    )
    assert entry["physical_address"] == "10.0.0.5:2"  # canonical address
    assert entry["gpu_id"] == "node-1:2"  # legacy label still carried
    assert entry["actor_id"] == "S:actor-ab"
    assert entry["node_id"] == "node-1"


def test_serialize_stage_perf_omits_blank_identity() -> None:
    [entry] = serialize_stage_perf(
        [_PerfRecord(stage_name="FastConformer_inference", process_time=1.0)]
    )  # no identity resolved (CPU / non-Ray)
    assert "physical_address" not in entry
    assert "gpu_id" not in entry
    assert "actor_id" not in entry
    assert "node_id" not in entry


def test_fingerprint_distinguishes_actors_with_equal_timings() -> None:
    """Two records byte-identical except for identity must NOT dedup to one."""
    summary = AudioPerformanceSummary(duration_key="duration")
    a = _perf(addr="10.0.0.5:0", actor_id="S:actor-a")
    b = _perf(addr="10.0.0.5:1", actor_id="S:actor-b")
    summary.record_stage_perf([a, b])
    stage = summary.build_stage_summaries()["FastConformer_inference"]
    assert stage["invocation_count"] == 2.0  # both counted, not collapsed by dedup


def test_same_named_stages_are_keyed_by_stable_stage_id() -> None:
    summary = AudioPerformanceSummary()
    summary.record_stage_perf(
        [
            _PerfRecord(
                stage_name="duplicate",
                stage_id="0001:duplicate",
                invocation_id="one",
                process_time=1.0,
            ),
            _PerfRecord(
                stage_name="duplicate",
                stage_id="0002:duplicate",
                invocation_id="two",
                process_time=2.0,
            ),
        ]
    )

    stages = summary.build_stage_summaries()
    assert list(stages) == ["0001:duplicate", "0002:duplicate"]
    assert stages["0001:duplicate"]["stage_name"] == "duplicate"
    assert stages["0001:duplicate"]["total_process_time_s"] == 1.0
    assert stages["0002:duplicate"]["total_process_time_s"] == 2.0


def test_stage_summary_exposes_adapter_inference_call_count() -> None:
    summary = AudioPerformanceSummary(duration_key="duration")
    summary.record_stage_perf(
        [
            _PerfRecord(
                stage_name="FastConformer_inference",
                process_time=10.0,
                num_items_processed=3,
                custom_metrics={
                    "audio_duration_s": 120.0,
                    "adapter_inference_calls": 7.0,
                    "adapter_inference_items": 9.0,
                },
            )
        ]
    )

    stage = summary.build_stage_summaries()["FastConformer_inference"]
    assert stage["invocation_count"] == 1.0
    assert stage["adapter_inference_call_count"] == 7.0
    assert stage["adapter_inference_items"] == 9.0
    assert stage["avg_adapter_inference_batch_size"] == 9.0 / 7.0
    assert stage["avg_audio_s_per_adapter_inference_call"] == 120.0 / 7.0
    assert stage["adapter_inference_calls_per_stage_invocation"] == 7.0


def test_stage_summary_reports_idle_sum_and_inference_compute_fraction() -> None:
    summary = AudioPerformanceSummary(duration_key="duration")
    summary.record_stage_perf(
        [
            _PerfRecord(
                stage_name="FastConformer_inference",
                process_time=10.0,
                actor_idle_time=2.0,
                custom_metrics={"inference_time_s": 8.0},
                invocation_id="invocation-1",
            ),
            _PerfRecord(
                stage_name="FastConformer_inference",
                process_time=20.0,
                actor_idle_time=3.0,
                custom_metrics={"inference_time_s": 12.0},
                invocation_id="invocation-2",
            ),
        ]
    )

    stage = summary.build_stage_summaries()["FastConformer_inference"]
    assert stage["total_actor_idle_time_s"] == 5.0
    assert stage["inference_compute_fraction"] == 20.0 / 30.0
    assert "wallclock_s" not in stage


def test_stage_summary_sums_model_finish_reason_counts() -> None:
    summary = AudioPerformanceSummary(duration_key="duration")
    summary.record_stage_perf(
        [
            _PerfRecord(
                stage_name="FastConformer_inference",
                process_time=1.0,
                custom_metrics={
                    "model_finish_reason_stop_count": 2.0,
                    "model_finish_reason_length_count": 1.0,
                    "model_turn1_finish_reason_length_count": 1.0,
                },
            ),
            _PerfRecord(
                stage_name="FastConformer_inference",
                process_time=1.0,
                custom_metrics={
                    "model_finish_reason_stop_count": 3.0,
                    "model_finish_reason_length_count": 2.0,
                    "model_turn1_finish_reason_length_count": 2.0,
                },
            ),
        ]
    )

    custom = summary.build_stage_summaries()["FastConformer_inference"]["custom_metrics_sum"]
    assert custom["model_finish_reason_stop_count"] == 5.0
    assert custom["model_finish_reason_length_count"] == 3.0
    assert custom["model_turn1_finish_reason_length_count"] == 3.0


# ----------------------------------------------------------------------
# Per-GPU scheduling breakdown + topology
# ----------------------------------------------------------------------


def test_per_actor_carries_physical_address_and_topology() -> None:
    """A tensor-parallel actor on 2 GPUs is one address but counts as 2 devices."""
    summary = AudioPerformanceSummary()
    summary.record_stage_perf(
        [
            _PerfRecord(
                stage_name="FastConformer_inference",
                process_time=1.0,
                actor_idle_time=0.1,
                num_items_processed=32,
                custom_metrics={"audio_duration_s": 100.0, "utterances_input": 32.0},
                actor_id="S:actor-a",
                node_id="node-0",
                gpu_id="node-0:0",
                physical_address="10.244.181.136:0,1",
                pod_ip="10.244.181.136",
                hostname="worker-0",
                gpu_indices=[0, 1],
            ),
        ]
    )
    stage = summary.build_stage_summaries()["FastConformer_inference"]
    # Topology: 1 per-actor address, but 2 distinct physical devices.
    assert stage["gpu_addresses"] == ["10.244.181.136:0,1"]
    assert stage["gpu_count"] == 2.0

    per_actor = stage["per_actor"]["S:actor-a"]  # keyed by actor_id
    assert per_actor["physical_address"] == "10.244.181.136:0,1"  # canonical GPU id, as a field
    assert per_actor["node_id"] == "node-0"
    assert per_actor["pod_ip"] == "10.244.181.136"
    assert per_actor["hostname"] == "worker-0"
    assert per_actor["gpu_indices"] == [0, 1]


def test_gpu_efficiency_uses_physical_gpu_count_not_actor_count() -> None:
    summary = AudioPerformanceSummary()
    summary.record_stage_perf(
        [
            _PerfRecord(
                stage_name="QwenOmni",
                process_time=10.0,
                num_items_processed=1,
                custom_metrics={"audio_duration_s": 20.0},
                actor_id="actor-a",
                physical_address="host:0,1",
                gpu_indices=[0, 1],
            )
        ]
    )

    stage = summary.build_stage_summaries()["QwenOmni"]
    assert stage["actor_count"] == 1.0
    assert stage["gpu_count"] == 2.0
    assert stage["audio_hours_per_gpu_hour"] == 1.0


def test_per_actor_scheduling_breakdown_and_topology() -> None:
    summary = AudioPerformanceSummary(duration_key="duration")
    # actor-a on GPU 0 (two invocations); actor-b on GPU 1 (two invocations).
    records = [
        _perf(addr="10.0.0.5:0", actor_id="S:actor-a", idle=0.10, items=32, audio_s=100.0),
        _perf(addr="10.0.0.5:0", actor_id="S:actor-a", idle=0.30, items=32, audio_s=120.0),
        _perf(addr="10.0.0.5:1", actor_id="S:actor-b", idle=0.05, items=16, audio_s=50.0),
        _perf(addr="10.0.0.5:1", actor_id="S:actor-b", idle=0.20, items=16, audio_s=70.0),
    ]
    summary.record_stage_perf(records)
    stage = summary.build_stage_summaries()["FastConformer_inference"]

    # Topology
    assert stage["gpu_addresses"] == ["10.0.0.5:0", "10.0.0.5:1"]
    assert stage["gpu_count"] == 2.0
    assert stage["actor_count"] == 2.0

    per_actor = stage["per_actor"]
    assert set(per_actor) == {"S:actor-a", "S:actor-b"}

    a = per_actor["S:actor-a"]
    assert a["physical_address"] == "10.0.0.5:0"
    assert a["node_id"] == "10.0.0.5"
    assert a["items_processed"] == 64.0  # 32 + 32
    assert a["audio_hours_in"] == (100.0 + 120.0) / 3600.0
    assert "batch_size_p50" in a
    assert "queue_wait_s_p50" in a
    assert "queue_wait_s_p95" in a

    b = per_actor["S:actor-b"]
    assert b["physical_address"] == "10.0.0.5:1"
    assert b["items_processed"] == 32.0  # 16 + 16
    assert b["audio_hours_in"] == (50.0 + 70.0) / 3600.0


def test_per_actor_gpus_block_is_keyed_per_physical_device() -> None:
    """A tensor-parallel actor reports ONE address but a nested per-device (``<host>:<idx>``) GPU map."""
    summary = AudioPerformanceSummary()
    # Actor on 2 GPUs; the sampler namespaces each device's util by UUID (``gpu_util_pct::<uuid>``).
    summary.record_stage_perf(
        [
            _PerfRecord(
                stage_name="FastConformer_inference",
                process_time=1.0,
                num_items_processed=32,
                custom_metrics={
                    "audio_duration_s": 100.0,
                    "gpu_util_pct::aaa": 90.0,
                    "gpu_mem_used_pct::aaa": 70.0,
                    "gpu_util_pct::bbb": 40.0,
                    "gpu_mem_used_pct::bbb": 30.0,
                },
                actor_id="FastConformer_inference:actor-a",
                node_id="node-0",
                physical_address="10.0.0.5:0,1",
                gpu_indices=[0, 1],
                gpu_uuids=["GPU-aaa", "GPU-bbb"],
            ),
        ]
    )
    stage = summary.build_stage_summaries()["FastConformer_inference"]
    gpus = stage["per_actor"]["FastConformer_inference:actor-a"]["gpus"]
    # One entry per physical device, keyed by <host>:<idx> -- not averaged across the actor.
    assert set(gpus) == {"10.0.0.5:0", "10.0.0.5:1"}
    assert gpus["10.0.0.5:0"]["gpu_index"] == 0
    assert gpus["10.0.0.5:0"]["gpu_uuid"] == "GPU-aaa"
    assert gpus["10.0.0.5:0"]["gpu_util_pct_p50"] == 90.0
    assert gpus["10.0.0.5:0"]["gpu_mem_used_pct_p50"] == 70.0
    assert gpus["10.0.0.5:1"]["gpu_index"] == 1
    assert gpus["10.0.0.5:1"]["gpu_util_pct_p50"] == 40.0
    # The per-GPU util is NOT summed into the stage's scalar custom-metric totals.
    assert "gpu_util_pct::aaa" not in stage.get("custom_metrics_sum", {})


def test_pipeline_throughput_rollup_unions_gpu_addresses() -> None:
    summary = AudioPerformanceSummary(duration_key="duration")
    summary.record_stage_perf(
        [
            _perf(addr="10.0.0.5:0", actor_id="S:actor-a", audio_s=3600.0),
            _perf(addr="10.0.0.6:0", actor_id="S:actor-b", audio_s=3600.0),
        ]
    )
    out = summary.build_summary(
        wall_time_s=3600.0,
        extra_stage_summaries={
            "sink": {
                "custom_metrics_sum": {
                    "pipeline_output_rows": 2.0,
                    "pipeline_output_audio_s": 7200.0,
                }
            }
        },
    )

    pt = out["pipeline_throughput"]
    assert pt["gpu_addresses"] == ["10.0.0.5:0", "10.0.0.6:0"]
    assert pt["gpu_count"] == 2.0
    assert pt["audio_hours_per_wallclock_hour"] == 2.0  # 2 audio-h / 1 wall-h


def test_summary_uses_first_input_boundary_and_last_output_boundary() -> None:
    summary = AudioPerformanceSummary(duration_key="duration")
    summary.record_stage_perf(
        [
            _PerfRecord(
                stage_name="nemo_tar_shard_discovery",
                process_time=0.1,
                custom_metrics={"input_tasks": 1.0, "shards_emitted": 8.0},
            ),
            _PerfRecord(
                stage_name="nemo_tar_shard_reader",
                process_time=1.0,
                custom_metrics={
                    "pipeline_input_rows": 123.0,
                    "pipeline_input_audio_s": 3600.0,
                    "audio_duration_s": 3600.0,
                },
            ),
            _PerfRecord(
                stage_name="intermediate",
                process_time=1.0,
                custom_metrics={
                    "pipeline_input_rows": 999.0,
                    "pipeline_input_audio_s": 999.0,
                    "pipeline_output_rows": 999.0,
                    "pipeline_output_audio_s": 999.0,
                },
            ),
        ]
    )

    out = summary.build_summary(
        wall_time_s=10.0,
        run_id="run-123",
        executor="RayDataExecutor",
        extra_stage_summaries={
            "terminal_writer": {
                "custom_metrics_sum": {
                    "pipeline_output_rows": 100.0,
                    "pipeline_output_audio_s": 1800.0,
                }
            }
        },
    )

    assert out["rows_in"] == 123.0
    assert out["input_hours"] == 1.0
    assert out["rows_out"] == 100.0
    assert out["output_hours"] == 0.5
    assert out["run_id"] == "run-123"
    assert out["executor"] == "RayDataExecutor"
    for removed in (
        "total_utterances",
        "pipeline_audio_s_per_wall_s",
        "pipeline_utterances_per_wall_s",
        "shards",
    ):
        assert removed not in out
    assert all("wallclock_s" not in stage for stage in out["stages"].values())


def test_summary_marks_missing_input_boundary_as_unavailable() -> None:
    summary = AudioPerformanceSummary(duration_key="duration")
    out = summary.build_summary(
        wall_time_s=1.0,
        extra_stage_summaries={
            "terminal_writer": {
                "custom_metrics_sum": {
                    "pipeline_output_rows": 0.0,
                    "pipeline_output_audio_s": 0.0,
                }
            }
        },
    )

    assert out["rows_in"] is None
    assert out["input_hours"] is None
    assert out["rows_out"] == 0.0
    assert out["output_hours"] == 0.0


def test_summary_marks_partial_input_audio_boundary_as_unavailable() -> None:
    summary = AudioPerformanceSummary(duration_key="duration")
    summary.record_stage_perf(
        [
            _PerfRecord(
                stage_name="manifest_reader",
                custom_metrics={
                    "pipeline_input_rows": 3.0,
                    "pipeline_input_audio_rows": 2.0,
                    "pipeline_input_audio_s": 4.0,
                },
            )
        ]
    )
    out = summary.build_summary(
        wall_time_s=1.0,
        extra_stage_summaries={
            "terminal_writer": {
                "custom_metrics_sum": {
                    "pipeline_output_rows": 3.0,
                    "pipeline_output_audio_s": 4.0,
                }
            }
        },
    )

    assert out["rows_in"] == 3.0
    assert out["input_hours"] is None


def test_summary_describes_pipeline_dataset_and_output_columns() -> None:
    summary = AudioPerformanceSummary(duration_key="duration")
    summary.record_task(
        AudioTask(
            dataset_name="wer-ready-600h",
            data={
                "audio_filepath": "audio.opus",
                "duration": 60.0,
                "text": "reference transcript",
                "fastconformer_prediction": "prediction",
            },
        )
    )
    pipeline_metadata = {
        "schema_version": 1,
        "backend": "ray_data",
        "execution_mode": None,
    }

    out = summary.build_summary(
        pipeline_metadata=pipeline_metadata,
        extra_stage_summaries={
            "writer": {
                "custom_metrics_sum": {
                    "pipeline_output_rows": 1.0,
                    "pipeline_output_audio_s": 60.0,
                }
            }
        },
    )

    assert out["pipeline"] == pipeline_metadata
    assert out["dataset_names"] == ["wer-ready-600h"]
    assert out["output_schema"]["columns"] == [
        "audio_filepath",
        "duration",
        "fastconformer_prediction",
        "text",
    ]
    assert out["output_schema"]["column_row_counts"]["text"] == 1


# ----------------------------------------------------------------------
# Graceful absence when identity is unresolved (CPU / non-Ray)
# ----------------------------------------------------------------------


def test_no_identity_emits_no_per_actor_or_topology() -> None:
    summary = AudioPerformanceSummary(duration_key="duration")
    summary.record_stage_perf([_perf(), _perf(idle=0.5)])  # blank address/actor
    stage = summary.build_stage_summaries()["FastConformer_inference"]
    assert "per_actor" not in stage
    assert "gpu_addresses" not in stage
    assert "gpu_count" not in stage
    assert "actor_count" not in stage

    out = summary.build_summary(wall_time_s=10.0)
    assert "gpu_addresses" not in out.get("pipeline_throughput", {})


# ----------------------------------------------------------------------
# Mixed pipeline: GPU stages and CPU stages coexist in one summary
# ----------------------------------------------------------------------


def _cpu_perf(stage_name: str, actor_id: str) -> _PerfRecord:
    """A CPU-stage record: actor + node resolved, but no GPU (no ``physical_address`` / ``gpu_id``)."""
    return _PerfRecord(
        stage_name=stage_name,
        process_time=0.5,
        num_items_processed=64,
        custom_metrics={"writer_process_calls": 2.0},
        actor_id=actor_id,
        node_id="node-0",
    )


def test_cpu_stage_gets_per_actor_but_no_gpu_fields() -> None:
    """A CPU stage gets actor_count + per_actor, but no gpu_addresses / gpu_count / GPU fields."""
    summary = AudioPerformanceSummary(duration_key="duration")
    summary.record_stage_perf(
        [
            _cpu_perf("ManifestWriter", "ManifestWriter:actor-cpu01"),
        ]
    )
    stage = summary.build_stage_summaries()["ManifestWriter"]
    assert stage["actor_count"] == 1.0
    assert stage["total_items_processed"] == 64.0
    assert "gpu_addresses" not in stage
    assert "gpu_count" not in stage

    per_actor = stage["per_actor"]["ManifestWriter:actor-cpu01"]
    assert per_actor["items_processed"] == 64.0
    assert per_actor["node_id"] == "node-0"
    assert "physical_address" not in per_actor  # CPU actor: no GPU
    assert "gpu_indices" not in per_actor


def test_mixed_gpu_and_cpu_stages_in_one_pipeline() -> None:
    """GPU and CPU stages coexist: both get per_actor, only the GPU stage gets topology; rollup unions GPU addresses."""
    summary = AudioPerformanceSummary(duration_key="duration")
    summary.record_stage_perf(
        [
            _perf(addr="10.0.0.5:0", actor_id="FastConformer_inference:actor-a", items=32, audio_s=100.0),
            _cpu_perf("ManifestWriter", "ManifestWriter:actor-cpu01"),
        ]
    )
    stages = summary.build_stage_summaries()

    gpu_stage = stages["FastConformer_inference"]
    assert gpu_stage["gpu_addresses"] == ["10.0.0.5:0"]
    assert gpu_stage["actor_count"] == 1.0
    assert set(gpu_stage["per_actor"]) == {"FastConformer_inference:actor-a"}
    assert gpu_stage["per_actor"]["FastConformer_inference:actor-a"]["physical_address"] == "10.0.0.5:0"

    cpu_stage = stages["ManifestWriter"]
    assert cpu_stage["actor_count"] == 1.0
    assert set(cpu_stage["per_actor"]) == {"ManifestWriter:actor-cpu01"}  # CPU gets per_actor too
    assert "gpu_addresses" not in cpu_stage

    # Pipeline rollup unions ONLY the GPU addresses (the CPU stage contributes none).
    summary._total_audio_seconds = 100.0  # exercise the throughput branch
    pt = summary.build_summary(wall_time_s=50.0)["pipeline_throughput"]
    assert pt["gpu_addresses"] == ["10.0.0.5:0"]
    assert pt["gpu_count"] == 1.0
