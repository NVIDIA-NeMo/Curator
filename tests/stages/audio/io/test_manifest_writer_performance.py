# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nemo_curator.stages.audio.common import ManifestReaderStage, ManifestWriterStage
from nemo_curator.tasks import AudioTask, FileGroupTask
from nemo_curator.utils.performance_utils import StagePerfStats


def test_manifest_writer_emits_fastconformer_performance_summary(tmp_path: Path) -> None:
    output_path = tmp_path / "output.jsonl"
    perf_path = tmp_path / "perf.json"
    writer = ManifestWriterStage(
        output_path=str(output_path),
        write_perf_stats=True,
        perf_summary_path=str(perf_path),
        perf_run_id="run-123",
        perf_executor="RayDataExecutor",
        perf_pipeline_metadata={"pipeline_name": "granary-v2", "backend": "ray_data"},
    )
    writer.setup()
    task = AudioTask(
        dataset_name="fastconformer-proof",
        data={"audio_filepath": "audio.wav", "duration": 4.0, "text": "hello"},
        _stage_perf=[
            StagePerfStats(
                stage_name="FastConformer_inference",
                process_time=2.0,
                num_items_processed=1,
                custom_metrics={"audio_duration_s": 4.0, "utterances_processed": 1.0},
            )
        ],
    )

    writer.process(task)
    assert not perf_path.exists()
    writer.teardown()

    summary = json.loads(perf_path.read_text())
    assert summary["rows_out"] == 1.0
    assert summary["output_hours"] == 4.0 / 3600.0
    assert summary["run_id"] == "run-123"
    assert summary["executor"] == "RayDataExecutor"
    assert summary["pipeline"] == {"pipeline_name": "granary-v2", "backend": "ray_data"}
    assert summary["stages"]["FastConformer_inference"]["throughput_audio_s_per_process_s"] == 2.0
    assert summary["stages"]["manifest_writer"]["custom_metrics_sum"]["pipeline_output_rows"] == 1.0


def test_manifest_writer_writes_one_summary_for_many_rows(tmp_path: Path) -> None:
    writer = ManifestWriterStage(
        output_path=str(tmp_path / "output.jsonl"),
        write_perf_stats=True,
        perf_summary_path=str(tmp_path / "perf.json"),
    )
    writer.setup()
    task = AudioTask(dataset_name="test", data={"duration": 1.0})

    with mock.patch.object(writer, "_write_perf_summary", wraps=writer._write_perf_summary) as write_summary:
        for _ in range(3):
            writer.process(task)
        write_summary.assert_not_called()
        writer.teardown()

    write_summary.assert_called_once_with()


def test_manifest_writer_setup_resets_all_run_scoped_metrics(tmp_path: Path) -> None:
    perf_path = tmp_path / "perf.json"
    writer = ManifestWriterStage(
        output_path=str(tmp_path / "output.jsonl"),
        write_perf_stats=True,
        perf_summary_path=str(perf_path),
    )

    writer.setup()
    writer.process(
        AudioTask(
            dataset_name="first-run",
            data={"duration": 10.0},
            _stage_perf=[StagePerfStats(stage_name="first-stage", process_time=1.0)],
        )
    )
    writer.teardown()

    writer.setup()
    writer.process(AudioTask(dataset_name="second-run", data={"duration": 2.0}))
    writer.teardown()

    summary = json.loads(perf_path.read_text())
    assert summary["rows_out"] == 1.0
    assert summary["total_audio_seconds"] == 2.0
    assert summary["dataset_names"] == ["second-run"]
    assert summary["perf_invocations_counted"] == 0
    assert "first-stage" not in summary["stages"]


def test_manifest_writer_disabled_mode_does_not_create_summary(tmp_path: Path) -> None:
    perf_path = tmp_path / "perf.json"
    writer = ManifestWriterStage(
        output_path=str(tmp_path / "output.jsonl"),
        write_perf_stats=False,
        perf_summary_path=str(perf_path),
    )
    writer.setup()
    writer.process(AudioTask(dataset_name="test", data={"duration": 1.0}))
    writer.teardown()

    assert not perf_path.exists()


def test_manifest_writer_uses_custom_duration_key(tmp_path: Path) -> None:
    perf_path = tmp_path / "perf.json"
    writer = ManifestWriterStage(
        output_path=str(tmp_path / "output.jsonl"),
        write_perf_stats=True,
        duration_key="clip_duration_s",
        perf_summary_path=str(perf_path),
    )
    writer.setup()
    writer.process(AudioTask(dataset_name="test", data={"clip_duration_s": 7.5}))
    writer.teardown()

    summary = json.loads(perf_path.read_text())
    assert summary["total_audio_seconds"] == 7.5
    assert summary["output_hours"] == 7.5 / 3600.0


def test_manifest_writer_finalizes_enabled_empty_summary(tmp_path: Path) -> None:
    perf_path = tmp_path / "perf.json"
    writer = ManifestWriterStage(
        output_path=str(tmp_path / "output.jsonl"),
        write_perf_stats=True,
        perf_summary_path=str(perf_path),
    )
    writer.setup()

    writer.teardown()

    summary = json.loads(perf_path.read_text())
    assert summary["rows_out"] == 0.0
    assert summary["stages"]["manifest_writer"]["total_items_processed"] == 0.0


def test_manifest_writer_merges_authoritative_zero_output_invocation(tmp_path: Path) -> None:
    perf_path = tmp_path / "perf.json"
    writer = ManifestWriterStage(
        output_path=str(tmp_path / "output.jsonl"),
        write_perf_stats=True,
        perf_summary_path=str(perf_path),
    )
    writer.setup()
    writer.teardown()
    perf = SimpleNamespace(
        stage_name="filter",
        stage_id="0001:filter",
        invocation_id="filtered-invocation",
        process_time=2.0,
        actor_idle_time=0.0,
        num_items_processed=4,
        custom_metrics={"utterances_filtered": 4.0},
        actor_id="",
    )

    assert writer.record_external_stage_perfs([perf]) is True

    summary = json.loads(perf_path.read_text())
    stage = summary["stages"]["0001:filter"]
    assert stage["stage_name"] == "filter"
    assert stage["invocation_count"] == 1.0
    assert stage["custom_metrics_sum"]["utterances_filtered"] == 4.0


def test_manifest_writer_external_merge_keeps_exact_total_invocation_count(tmp_path: Path) -> None:
    perf_path = tmp_path / "perf.json"
    writer = ManifestWriterStage(
        output_path=str(tmp_path / "output.jsonl"),
        write_perf_stats=True,
        perf_summary_path=str(perf_path),
    )
    writer.setup()
    writer.process(
        AudioTask(
            dataset_name="test",
            data={"duration": 1.0},
            _stage_perf=[
                SimpleNamespace(
                    stage_name="ordinary",
                    stage_id="0001:ordinary",
                    invocation_id="ordinary-1",
                    process_time=1.0,
                    actor_idle_time=0.0,
                    num_items_processed=1,
                    custom_metrics={},
                    actor_id="",
                ),
                SimpleNamespace(
                    stage_name="extended",
                    stage_id="0002:extended",
                    invocation_id="extended-1",
                    process_time=1.0,
                    actor_idle_time=0.0,
                    num_items_processed=1,
                    custom_metrics={},
                    actor_id="",
                ),
            ],
        )
    )
    writer.teardown()

    authoritative = [
        SimpleNamespace(
            stage_name="extended",
            stage_id="0002:extended",
            invocation_id=invocation_id,
            process_time=1.0,
            actor_idle_time=0.0,
            num_items_processed=1,
            custom_metrics={},
            actor_id="",
        )
        for invocation_id in ("extended-1", "extended-zero-output")
    ]
    assert writer.record_external_stage_perfs(authoritative) is True

    summary = json.loads(perf_path.read_text())
    assert summary["perf_invocations_counted"] == 3
    assert summary["stages"]["0001:ordinary"]["invocation_count"] == 1.0
    assert summary["stages"]["0002:extended"]["invocation_count"] == 2.0


def test_manifest_reader_emits_real_input_boundary_metrics(tmp_path: Path) -> None:
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"audio_filepath": "a.wav", "duration": 1.25}),
                json.dumps({"audio_filepath": "b.wav", "duration": 2.75}),
                json.dumps({"audio_filepath": "c.wav"}),
            ]
        )
    )
    reader = ManifestReaderStage()

    rows = reader.process(FileGroupTask(dataset_name="test", data=[str(manifest)]))
    metrics = reader._consume_custom_metrics()

    assert len(rows) == 3
    assert metrics["pipeline_input_rows"] == 3.0
    assert metrics["pipeline_input_audio_rows"] == 2.0
    assert metrics["pipeline_input_audio_s"] == 4.0
