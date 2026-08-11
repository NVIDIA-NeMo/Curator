# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import pytest

from nemo_curator.backends.base import BaseExecutor
from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.backends.slurm_array import (
    SLURM_ARRAY_ENABLED_ENV_VAR,
    SLURM_ARRAY_MINIMUM_SHARD_INDEX_ENV_VAR,
    SLURM_ARRAY_SHARD_INDEX_ENV_VAR,
    SLURM_ARRAY_TOTAL_SHARDS_ENV_VAR,
)
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.common import ManifestWriterStage
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, Task
from nemo_curator.utils.stage_perf_collector import PerformanceRecordStore
from tests.utils.performance_record_store import make_performance_record_store


class _TerminalConsumer(ProcessingStage[Task, Task]):
    name = "duplicate"

    def __init__(self) -> None:
        self.finalized: tuple[PerformanceRecordStore, float, dict[str, object]] | None = None

    def process(self, task: Task) -> Task:
        return task

    def requests_performance_records(self) -> bool:
        return True

    def finalize_performance_report(
        self,
        *,
        performance_records: PerformanceRecordStore,
        wall_time_s: float,
        report_context: dict[str, object],
    ) -> None:
        self.finalized = (performance_records, wall_time_s, report_context)


class _CardinalityConsumer(_TerminalConsumer):
    name = "cardinality"

    def __init__(self, *, fan_out: bool) -> None:
        super().__init__()
        self.fan_out = fan_out

    def process(self, task: Task) -> list[Task] | None:
        self._log_metric("cardinality_metric", 3.5)
        return [task, AudioTask(dataset_name="test", data={"text": "second"})] if self.fan_out else None


class _InvalidConsumer(ProcessingStage[Task, Task]):
    name = "invalid-consumer"

    def process(self, task: Task) -> Task:
        return task

    def requests_performance_records(self) -> bool:
        return True


class _Executor:
    def __init__(self) -> None:
        self.records: PerformanceRecordStore | None = None
        self.consume_calls = 0
        self.collection_requests: list[bool] = []

    def _set_stage_perf_collection_requested(self, requested: bool) -> None:
        self.collection_requests.append(requested)

    def execute(self, stages: list[ProcessingStage], initial_tasks: list[Task] | None = None) -> list[Task]:
        terminal_stage = stages[-1]
        self.records = make_performance_record_store(
            [
                {
                    "stage_name": terminal_stage.name,
                    "stage_id": terminal_stage._curator_stage_id,
                    "invocation_id": "invocation-1",
                    "process_time": 1.0,
                    "window_start_s": 10.0,
                    "window_end_s": 11.0,
                }
            ]
        )
        return list(initial_tasks or [])

    def consume_external_perf_records(self) -> PerformanceRecordStore:
        self.consume_calls += 1
        records, self.records = self.records, None
        assert records is not None
        return records


class _UnsupportedExecutor(BaseExecutor):
    def execute(self, stages: list[ProcessingStage], initial_tasks: list[Task] | None = None) -> list[Task]:
        return list(initial_tasks or [])


def test_pipeline_assigns_stable_ids_and_drains_records_once() -> None:
    first = _TerminalConsumer()
    terminal = _TerminalConsumer()
    executor = _Executor()
    pipeline = Pipeline(name="performance", stages=[first, terminal])

    result = pipeline.run(
        executor=executor,  # type: ignore[arg-type]
        initial_tasks=[],
    )

    assert result == []
    assert first._curator_stage_id == "000:duplicate"
    assert terminal._curator_stage_id == "001:duplicate"
    assert first.finalized is None
    assert terminal.finalized is not None
    records, wall_time_s, report_context = terminal.finalized
    assert records is pipeline.performance_records
    assert len(records) == 1
    assert wall_time_s >= 0.0
    assert report_context["pipeline_name"] == "performance"
    assert len(str(report_context["run_id"])) == 32
    assert report_context["executor"] == "_Executor"
    assert report_context["slurm_array"] is None
    assert [stage["stage_id"] for stage in report_context["pipeline"]["stages"]] == [
        "000:duplicate",
        "001:duplicate",
    ]
    assert [stage["name"] for stage in report_context["pipeline"]["stages"]] == ["duplicate", "duplicate"]
    assert executor.consume_calls == 1
    assert executor.records is None
    assert executor.collection_requests == [True, False]


def test_disabled_report_skips_context_and_record_transfer(tmp_path: Path) -> None:
    writer = ManifestWriterStage(
        output_path=str(tmp_path / "manifest.jsonl"),
        performance_report_path=None,
    )
    executor = _Executor()
    pipeline = Pipeline(name="disabled", stages=[writer])

    pipeline.run(executor=executor, initial_tasks=[])

    assert pipeline.performance_records is None
    assert executor.consume_calls == 0
    assert executor.records is not None
    assert executor.collection_requests == [False, False]
    executor.records.cleanup()


def test_requesting_stage_without_finalizer_is_rejected() -> None:
    pipeline = Pipeline(name="invalid", stages=[_InvalidConsumer()])

    with pytest.raises(TypeError, match="must implement finalize_performance_report"):
        pipeline.run(executor=_Executor(), initial_tasks=[])


def test_pipeline_rejects_required_collection_on_unsupported_executor() -> None:
    pipeline = Pipeline(name="unsupported", stages=[_TerminalConsumer()])

    with pytest.raises(NotImplementedError, match="does not support run-scoped stage performance collection"):
        pipeline.run(executor=_UnsupportedExecutor(), initial_tasks=[])


def test_slurm_environment_reaches_sharded_terminal_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(SLURM_ARRAY_ENABLED_ENV_VAR, raising=False)
    monkeypatch.delenv(SLURM_ARRAY_SHARD_INDEX_ENV_VAR, raising=False)
    monkeypatch.delenv(SLURM_ARRAY_TOTAL_SHARDS_ENV_VAR, raising=False)
    monkeypatch.delenv(SLURM_ARRAY_MINIMUM_SHARD_INDEX_ENV_VAR, raising=False)
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
    monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "11")

    report_path = tmp_path / "performance.json"
    writer = ManifestWriterStage(
        output_path=str(tmp_path / "manifest.jsonl"),
        performance_report_path=str(report_path),
    )
    pipeline = Pipeline(name="slurm-report", stages=[writer])

    pipeline.run(executor=_Executor(), initial_tasks=[])

    sharded_report_path = tmp_path / "performance.shard-00007-of-00011.json"
    assert not report_path.exists()
    assert sharded_report_path.is_file()
    report = json.loads(sharded_report_path.read_text(encoding="utf-8"))
    assert report["slurm_array"] == {"shard_index": 7, "total_shards": 11}
    assert report["pipeline_name"] == "slurm-report"
    pipeline.performance_records.cleanup()


@pytest.mark.parametrize(
    "executor",
    [
        pytest.param(RayDataExecutor(), id="ray-data"),
        pytest.param(XennaExecutor(config={"execution_mode": "batch"}), id="xenna"),
    ],
)
@pytest.mark.usefixtures("shared_ray_client")
def test_report_path_alone_collects_records_end_to_end(executor: object, tmp_path: Path) -> None:
    report_path = tmp_path / "performance.json"
    writer = ManifestWriterStage(
        output_path=str(tmp_path / "manifest.jsonl"),
        performance_report_path=str(report_path),
    )
    pipeline = Pipeline(name="path-only", stages=[writer])

    pipeline.run(
        executor=executor,  # type: ignore[arg-type]
        initial_tasks=[AudioTask(dataset_name="test", data={"text": "payload"})],
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["pipeline_name"] == "path-only"
    assert report["executor"] == type(executor).__name__
    assert len(report["run_id"]) == 32
    assert report["wall_time_s"] >= 0.0
    assert report["record_count"] == 1
    assert report["pipeline"]["pipeline_name"] == "path-only"
    assert [stage["stage_id"] for stage in report["pipeline"]["stages"]] == ["000:manifest_writer"]
    [record] = report["records"]
    assert record["stage_id"] == "000:manifest_writer"
    assert record["invocation_id"]
    assert record["window_end_s"] >= record["window_start_s"] > 0
    assert record["num_items_processed"] == 1
    assert "input_data_size_mb" not in record


@pytest.mark.parametrize(
    "executor",
    [
        pytest.param(RayDataExecutor(), id="ray-data"),
        pytest.param(XennaExecutor(config={"execution_mode": "batch"}), id="xenna"),
    ],
)
@pytest.mark.parametrize("fan_out", [pytest.param(False, id="zero-output"), pytest.param(True, id="fan-out")])
@pytest.mark.usefixtures("shared_ray_client")
def test_cardinality_invocation_is_collected_once_end_to_end(executor: object, fan_out: bool) -> None:
    consumer = _CardinalityConsumer(fan_out=fan_out)
    pipeline = Pipeline(name="cardinality", stages=[consumer])

    results = pipeline.run(
        executor=executor,  # type: ignore[arg-type]
        initial_tasks=[AudioTask(dataset_name="test", data={"text": "first"})],
    )

    assert len(results) == (2 if fan_out else 0)
    assert consumer.finalized is not None
    records, _, _ = consumer.finalized
    assert len(records) == 1
    [record] = list(records)
    assert record["stage_id"] == "000:cardinality"
    assert record["invocation_id"]
    assert record["custom_metrics"] == {"cardinality_metric": 3.5}


@pytest.mark.parametrize(
    "executor",
    [
        pytest.param(RayDataExecutor(), id="ray-data"),
        pytest.param(XennaExecutor(config={"execution_mode": "batch"}), id="xenna"),
    ],
)
@pytest.mark.usefixtures("shared_ray_client")
def test_disabled_report_uses_task_attached_metrics_without_extended_report(executor: object, tmp_path: Path) -> None:
    manifest_path = tmp_path / "disabled-manifest.jsonl"
    writer = ManifestWriterStage(output_path=str(manifest_path), performance_report_path=None)
    pipeline = Pipeline(name="disabled", stages=[writer])

    results = pipeline.run(
        executor=executor,  # type: ignore[arg-type]
        initial_tasks=[AudioTask(dataset_name="test", data={"text": "payload"})],
    )

    assert results is not None
    assert len(results) == 1
    assert len(results[0]._stage_perf) == 1
    assert results[0]._stage_perf[0].stage_name == "manifest_writer"
    assert not hasattr(results[0]._stage_perf[0], "invocation_id")
    assert executor._external_perf_records is None  # type: ignore[attr-defined]
    assert {path.name for path in tmp_path.iterdir()} == {manifest_path.name}
