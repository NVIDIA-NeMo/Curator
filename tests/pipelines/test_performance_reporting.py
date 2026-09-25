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
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.common import ManifestWriterStage
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, Task
from nemo_curator.utils.stage_perf_collector import PerformanceRecordStore
from tests.utils.performance_record_store import make_performance_record_store


class _Consumer(ProcessingStage[Task, Task]):
    name = "consumer"

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


class _InvalidConsumer(ProcessingStage[Task, Task]):
    name = "invalid-consumer"

    def process(self, task: Task) -> Task:
        return task

    def requests_performance_records(self) -> bool:
        return True


class _Executor:
    def __init__(self) -> None:
        self.records: PerformanceRecordStore | None = None
        self.collection_requests: list[bool] = []

    def _set_stage_perf_collection_requested(self, requested: bool) -> None:
        self.collection_requests.append(requested)

    def execute(self, stages: list[ProcessingStage], initial_tasks: list[Task] | None = None) -> list[Task]:
        terminal = stages[-1]
        self.records = make_performance_record_store(
            [{"stage_id": terminal._curator_stage_id, "invocation_id": "invocation-1"}]
        )
        return list(initial_tasks or [])

    def consume_external_perf_records(self) -> PerformanceRecordStore:
        records, self.records = self.records, None
        assert records is not None
        return records


class _UnsupportedExecutor(BaseExecutor):
    def execute(self, stages: list[ProcessingStage], initial_tasks: list[Task] | None = None) -> list[Task]:
        return list(initial_tasks or [])


def test_pipeline_routes_one_run_scoped_store_to_the_last_consumer() -> None:
    first = _Consumer()
    terminal = _Consumer()
    executor = _Executor()
    pipeline = Pipeline(name="performance", stages=[first, terminal])

    assert pipeline.run(executor=executor, initial_tasks=[]) == []  # type: ignore[arg-type]

    assert first.finalized is None
    assert terminal.finalized is not None
    records, wall_time_s, context = terminal.finalized
    assert records is pipeline.performance_records
    assert list(records) == [{"stage_id": "001:consumer", "invocation_id": "invocation-1"}]
    assert wall_time_s >= 0.0
    assert len(str(context["run_id"])) == 32
    assert context["executor"] == "_Executor"
    assert context["slurm_array"] is None
    assert [stage["stage_id"] for stage in context["pipeline"]["stages"]] == [  # type: ignore[index]
        "000:consumer",
        "001:consumer",
    ]
    assert executor.collection_requests == [True, False]
    records.cleanup()


def test_pipeline_rejects_invalid_consumer_and_unsupported_executor() -> None:
    with pytest.raises(TypeError, match="must implement finalize_performance_report"):
        Pipeline(name="invalid", stages=[_InvalidConsumer()]).run(executor=_Executor(), initial_tasks=[])  # type: ignore[arg-type]

    with pytest.raises(NotImplementedError, match="does not support run-scoped stage performance collection"):
        Pipeline(name="unsupported", stages=[_Consumer()]).run(executor=_UnsupportedExecutor(), initial_tasks=[])


@pytest.mark.parametrize(
    "executor",
    [
        pytest.param(RayDataExecutor(), id="ray-data"),
        pytest.param(XennaExecutor(config={"execution_mode": "batch"}), id="xenna"),
    ],
)
@pytest.mark.usefixtures("shared_ray_client")
def test_supported_executors_write_one_complete_report(executor: BaseExecutor, tmp_path: Path) -> None:
    report_path = tmp_path / "performance.json"
    writer = ManifestWriterStage(
        output_path=str(tmp_path / "manifest.jsonl"),
        performance_report_path=str(report_path),
    )
    pipeline = Pipeline(name="performance", stages=[writer])

    pipeline.run(
        executor=executor,
        initial_tasks=[AudioTask(dataset_name="test", data={"text": "payload"})],
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["executor"] == type(executor).__name__
    assert report["record_count"] == 1
    assert report["pipeline"]["stages"][0]["stage_id"] == "000:manifest_writer"
    [record] = report["records"]
    assert record["stage_id"] == "000:manifest_writer"
    assert record["invocation_id"]
    assert record["num_items_processed"] == 1
    assert "input_data_size_mb" not in record
    assert pipeline.performance_records is not None
    pipeline.performance_records.cleanup()
