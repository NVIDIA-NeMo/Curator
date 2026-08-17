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
from unittest import mock

import pytest

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.common import (
    ManifestReader,
    ManifestWriterStage,
    PreserveByValueStage,
)
from nemo_curator.tasks import AudioTask
from nemo_curator.utils.performance_utils import StagePerfStats


def _writer(tmp_path: Path, **kwargs: object) -> ManifestWriterStage:
    return ManifestWriterStage(
        output_path=str(tmp_path / "output.jsonl"),
        perf_summary_path=str(tmp_path / "perf.json"),
        **kwargs,
    )


class _InlineExecutor:
    """Small executor for exercising the driver-owned terminal lifecycle."""

    def execute(
        self,
        stages: list[ManifestWriterStage],
        initial_tasks: list[AudioTask] | None = None,
    ) -> list[AudioTask]:
        adapter = BaseStageAdapter(stages[-1])
        adapter.setup()
        results = adapter.process_batch(initial_tasks or [])
        adapter.teardown()
        return results


class _FilteredInlineExecutor(_InlineExecutor):
    def execute(
        self,
        stages: list[ManifestWriterStage],
        initial_tasks: list[AudioTask] | None = None,
    ) -> list[AudioTask]:
        results = super().execute(stages, initial_tasks)
        accepted = stages[-1].record_external_stage_perfs(
            [
                StagePerfStats(
                    stage_name="filter",
                    stage_id="001:filter",
                    invocation_id="filtered-1",
                    process_time=0.5,
                    num_items_processed=2,
                    custom_metrics={
                        "input_count": 2.0,
                        "output_count": 0.0,
                        "filtered_count": 2.0,
                    },
                )
            ]
        )
        assert accepted is True
        return results


def test_manifest_writer_emits_fastconformer_performance_summary(tmp_path: Path) -> None:
    perf_path = tmp_path / "perf.json"
    writer = _writer(
        tmp_path,
        write_perf_stats=True,
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


def test_persisted_writer_summary_omits_unmeasurable_summary_write_time(tmp_path: Path) -> None:
    perf_path = tmp_path / "perf.json"
    writer = _writer(tmp_path, write_perf_stats=True)
    writer.setup()
    writer.process(AudioTask(dataset_name="test", data={"duration": 1.0}))
    writer.teardown()

    writer_summary = json.loads(perf_path.read_text())["stages"]["manifest_writer"]
    assert "perf_write_time_s" not in writer_summary["custom_metrics_sum"]
    assert writer_summary["total_process_time_s"] == writer_summary["custom_metrics_sum"]["manifest_write_time_s"]


def test_manifest_writer_defers_summary_until_teardown(tmp_path: Path) -> None:
    perf_path = tmp_path / "perf.json"
    writer = _writer(tmp_path, write_perf_stats=True)
    writer.setup()

    with mock.patch.object(writer, "_write_perf_summary", wraps=writer._write_perf_summary) as write_summary:
        for index in range(3):
            writer.process(
                AudioTask(
                    dataset_name="test",
                    data={"audio_filepath": f"{index}.wav", "duration": 1.0},
                )
            )
            assert not perf_path.exists()
        write_summary.assert_not_called()
        writer.teardown()
        write_summary.assert_called_once_with()

    assert json.loads(perf_path.read_text())["rows_out"] == 3.0


def test_manifest_writer_setup_resets_all_run_scoped_metrics(tmp_path: Path) -> None:
    perf_path = tmp_path / "perf.json"
    writer = _writer(tmp_path, write_perf_stats=True)

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
    writer = _writer(tmp_path, write_perf_stats=False)
    writer.setup()
    writer.process(AudioTask(dataset_name="test", data={"duration": 1.0}))
    writer.teardown()

    assert not perf_path.exists()


def test_manifest_writer_uses_custom_duration_key(tmp_path: Path) -> None:
    perf_path = tmp_path / "perf.json"
    writer = _writer(
        tmp_path,
        write_perf_stats=True,
        duration_key="clip_duration_s",
    )
    writer.setup()
    writer.process(AudioTask(dataset_name="test", data={"clip_duration_s": 7.5}))
    writer.teardown()

    summary = json.loads(perf_path.read_text())
    assert summary["total_audio_seconds"] == 7.5
    assert summary["output_hours"] == 7.5 / 3600.0


def test_manifest_writer_finalizes_enabled_empty_summary(tmp_path: Path) -> None:
    perf_path = tmp_path / "perf.json"
    writer = _writer(tmp_path, write_perf_stats=True)
    writer.setup()
    writer.teardown()

    summary = json.loads(perf_path.read_text())
    assert summary["rows_out"] == 0.0
    assert summary["stages"]["manifest_writer"]["total_items_processed"] == 0.0


def test_pipeline_finalizes_one_empty_summary_with_auto_context(tmp_path: Path) -> None:
    perf_path = tmp_path / "perf.json"
    writer = _writer(tmp_path, write_perf_stats=True)

    with mock.patch.object(writer, "_write_perf_summary", wraps=writer._write_perf_summary) as write_summary:
        result = Pipeline(name="empty-writer", stages=[writer]).run(
            executor=_InlineExecutor(),  # type: ignore[arg-type]
            initial_tasks=[],
        )

    assert result == []
    write_summary.assert_called_once()
    summary = json.loads(perf_path.read_text())
    assert summary["status"] == "completed"
    assert summary["rows_out"] == 0.0
    assert summary["output_hours"] == 0.0
    assert summary["run_id"]
    assert summary["executor"] == "_InlineExecutor"
    assert summary["pipeline"]["pipeline_name"] == "empty-writer"
    assert summary["pipeline_wall_time_s"] >= 0.0
    assert summary["stages"]["000:manifest_writer"]["stage_name"] == "manifest_writer"


def test_pipeline_preserves_executor_reported_all_filtered_invocation(tmp_path: Path) -> None:
    perf_path = tmp_path / "perf.json"
    writer = _writer(tmp_path, write_perf_stats=True)
    Pipeline(name="filtered", stages=[writer]).run(
        executor=_FilteredInlineExecutor(),  # type: ignore[arg-type]
        initial_tasks=[],
    )

    summary = json.loads(perf_path.read_text())
    assert summary["rows_out"] == 0.0
    assert summary["stages"]["001:filter"]["items_filtered"] == 2.0


@pytest.mark.parametrize(
    ("executor_type", "config"),
    [
        pytest.param(RayDataExecutor, {}, id="ray-data"),
        pytest.param(XennaExecutor, {"execution_mode": "batch"}, id="xenna-batch"),
    ],
)
def test_real_executor_preserves_all_filtered_invocation(
    tmp_path: Path,
    executor_type: type[RayDataExecutor] | type[XennaExecutor],
    config: dict[str, str],
) -> None:
    perf_path = tmp_path / "perf.json"
    pipeline = Pipeline(
        name="all-filtered",
        stages=[
            PreserveByValueStage(input_value_key="keep", target_value=1),
            _writer(tmp_path, write_perf_stats=True),
        ],
    )

    result = pipeline.run(
        executor=executor_type(config=config),
        initial_tasks=[
            AudioTask(dataset_name="test", data={"keep": 0, "duration": 1.0}),
            AudioTask(dataset_name="test", data={"keep": 0, "duration": 2.0}),
        ],
    )

    assert result == []
    summary = json.loads(perf_path.read_text())
    filter_summary = summary["stages"]["000:PreserveByValueStage"]
    assert filter_summary["total_items_processed"] == 2.0
    assert filter_summary["items_filtered"] == 2.0
    assert filter_summary["custom_metrics_sum"]["input_count"] == 2.0
    assert summary["rows_out"] == 0.0


def test_pipeline_uses_public_reader_and_writer_custom_duration_key(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"audio_filepath": "a.wav", "seconds": 2.5}),
                json.dumps({"audio_filepath": "b.wav", "seconds": 1.5}),
            ]
        )
        + "\n"
    )
    perf_path = tmp_path / "perf.json"
    pipeline = Pipeline(
        name="custom-duration",
        stages=[
            ManifestReader(manifest_path=str(input_path), duration_key="seconds"),
            _writer(
                tmp_path,
                duration_key="seconds",
                write_perf_stats=True,
            ),
        ],
    )

    result = pipeline.run(executor=RayDataExecutor())

    assert len(result or []) == 2
    summary = json.loads(perf_path.read_text())
    assert summary["rows_in"] == 2.0
    assert summary["rows_out"] == 2.0
    assert summary["input_hours"] == 4.0 / 3600.0
    assert summary["output_hours"] == 4.0 / 3600.0


def test_missing_duration_is_unavailable_not_measured_zero(tmp_path: Path) -> None:
    perf_path = tmp_path / "perf.json"
    writer = _writer(
        tmp_path,
        write_perf_stats=True,
        duration_key="seconds",
    )
    writer.setup()
    writer.process(AudioTask(dataset_name="test", data={"audio_filepath": "a.wav"}))
    writer.teardown()

    summary = json.loads(perf_path.read_text())
    assert summary["rows_out"] == 1.0
    assert summary["output_duration_rows"] == 0.0
    assert summary["output_hours"] is None
    assert summary["total_audio_seconds"] is None


def test_final_write_preserves_executor_owned_external_stage(tmp_path: Path) -> None:
    perf_path = tmp_path / "perf.json"
    writer = _writer(tmp_path, write_perf_stats=True)
    writer.setup()
    accepted = writer.record_external_stage_perf(
        StagePerfStats(
            stage_name="pipeline_hardware_sampler",
            process_time=1.0,
            invocation_id="hardware-1",
            custom_metrics={"gpu_util_pct": 75.0},
        )
    )
    writer.process(AudioTask(dataset_name="test", data={"duration": 1.0}))
    writer.teardown()

    assert accepted is True
    summary = json.loads(perf_path.read_text())
    assert "pipeline_hardware_sampler" in summary["stages"]
    assert summary["rows_out"] == 1.0
