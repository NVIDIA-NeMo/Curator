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

from nemo_curator.stages.audio.metrics.performance import AudioPerformanceSummary, serialize_stage_perf
from nemo_curator.tasks import AudioTask
from nemo_curator.utils.performance_utils import StagePerfStats


def test_serialize_stage_perf_preserves_builtin_and_custom_metrics() -> None:
    perf = StagePerfStats(
        stage_name="QwenOmni_inference",
        process_time=1.25,
        actor_idle_time=0.5,
        input_data_size_mb=3.0,
        num_items_processed=2,
        custom_metrics={"audio_duration_s": 10.0},
        invocation_id="batch-1",
    )

    assert serialize_stage_perf([perf]) == [
        {
            "invocation_id": "batch-1",
            "stage_name": "QwenOmni_inference",
            "process_time": 1.25,
            "actor_idle_time": 0.5,
            "input_data_size_mb": 3.0,
            "num_items_processed": 2,
            "custom_metrics": {"audio_duration_s": 10.0},
        }
    ]


def test_audio_performance_summary_deduplicates_invocations_and_derives_throughput() -> None:
    perf = StagePerfStats(
        stage_name="QwenOmni_inference",
        process_time=2.0,
        num_items_processed=2,
        custom_metrics={
            "utterances_input": 2.0,
            "utterances_processed": 2.0,
            "audio_duration_s": 10.0,
            "inference_time_s": 1.0,
            "output_tokens": 50.0,
            "output_chars": 100.0,
            "waveform_bytes": 1024.0 * 1024.0,
        },
        invocation_id="same-batch",
    )
    tasks = [
        AudioTask(
            task_id="utt-1",
            data={"duration": 4.0},
            _metadata={"_shard_key": "corpus/shard_0"},
            _stage_perf=[perf],
        ),
        AudioTask(
            task_id="utt-2",
            data={"duration": 6.0},
            _metadata={"_shard_key": "corpus/shard_0"},
            _stage_perf=[perf],
        ),
    ]

    summary_builder = AudioPerformanceSummary(duration_key="duration")
    for task in tasks:
        summary_builder.record_task(task, shard_key=task._metadata["_shard_key"])

    summary = summary_builder.build_summary(wall_time_s=5.0)
    stage_summary = summary["stages"]["QwenOmni_inference"]

    assert summary["total_utterances"] == 2
    assert summary["total_audio_seconds"] == 10.0
    assert summary["total_audio_hours"] == 10.0 / 3600.0
    assert summary["pipeline_audio_s_per_wall_s"] == 2.0
    assert summary["pipeline_utterances_per_wall_s"] == 0.4
    assert summary["perf_invocations_counted"] == 1
    assert summary["shards"]["corpus/shard_0"]["utterances"] == 2
    assert summary["shards"]["corpus/shard_0"]["audio_seconds"] == 10.0

    assert stage_summary["invocation_count"] == 1
    assert stage_summary["total_items_processed"] == 2
    assert stage_summary["throughput_items_per_s"] == 1.0
    assert stage_summary["throughput_audio_s_per_process_s"] == 5.0
    assert stage_summary["throughput_audio_s_per_inference_s"] == 10.0
    assert stage_summary["throughput_output_tokens_per_process_s"] == 25.0
    assert stage_summary["throughput_output_tokens_per_inference_s"] == 50.0
    assert stage_summary["throughput_output_chars_per_process_s"] == 50.0
    assert stage_summary["throughput_output_chars_per_inference_s"] == 100.0
    assert stage_summary["throughput_waveform_mb_per_process_s"] == 0.5
    assert stage_summary["utterances_processed_per_input_utterance"] == 1.0


def test_audio_performance_summary_derives_fanout_ratios_from_stage_metrics() -> None:
    perf = StagePerfStats(
        stage_name="nemo_tar_shard_reader",
        process_time=4.0,
        num_items_processed=1,
        custom_metrics={
            "input_shards": 1.0,
            "utterances_emitted": 8.0,
            "output_tasks": 8.0,
        },
        invocation_id="reader-shard-0",
    )

    summary_builder = AudioPerformanceSummary()
    summary_builder.record_stage_perf([perf])

    stage_summary = summary_builder.build_summary()["stages"]["nemo_tar_shard_reader"]

    assert stage_summary["throughput_items_per_s"] == 0.25
    assert stage_summary["utterances_emitted_per_input_shard"] == 8.0
    assert stage_summary["custom_metrics_sum"]["output_tasks"] == 8.0


def test_audio_performance_summary_accepts_extra_terminal_stage_summaries() -> None:
    summary_builder = AudioPerformanceSummary()

    summary = summary_builder.build_summary(
        extra_stage_summaries={
            "sharded_manifest_writer": {
                "total_process_time_s": 0.5,
                "total_items_processed": 2.0,
                "invocation_count": 2.0,
            }
        }
    )

    assert summary["stages"]["sharded_manifest_writer"]["total_process_time_s"] == 0.5
