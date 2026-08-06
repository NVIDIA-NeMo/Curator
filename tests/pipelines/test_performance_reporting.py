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

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.common import ManifestWriterStage
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, Task
from nemo_curator.utils.performance_utils import StagePerfStats
from nemo_curator.utils.stage_perf_collector import PerformanceRecordStore


class _TerminalConsumer(ProcessingStage[Task, Task]):
    name = "duplicate"

    def __init__(self) -> None:
        self.prepared = False
        self.finalized: tuple[list[StagePerfStats], float, dict[str, object]] | None = None

    def process(self, task: Task) -> Task:
        return task

    def prepare_performance_report(self) -> None:
        self.prepared = True

    def finalize_performance_report(
        self,
        *,
        performance_records: list[StagePerfStats],
        wall_time_s: float,
        report_context: dict[str, object],
    ) -> None:
        self.finalized = (performance_records, wall_time_s, report_context)


class _Executor:
    def __init__(self) -> None:
        self.records: PerformanceRecordStore | None = PerformanceRecordStore.from_records(
            [
                StagePerfStats(
                    stage_name="duplicate",
                    invocation_id="invocation-1",
                    process_time=1.0,
                )
            ]
        )

    def execute(self, _stages: list[ProcessingStage], initial_tasks: list[Task] | None = None) -> list[Task]:
        return list(initial_tasks or [])

    def consume_external_perf_records(self) -> PerformanceRecordStore:
        records, self.records = self.records, None
        assert records is not None
        return records


def test_pipeline_assigns_stable_ids_and_fans_out_one_record_drain() -> None:
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
    assert first.prepared is False
    assert terminal.prepared is True
    assert terminal.finalized is not None
    records, wall_time_s, report_context = terminal.finalized
    assert records is pipeline.performance_records
    assert len(records) == 1
    assert wall_time_s >= 0.0
    assert report_context["pipeline_name"] == "performance"
    assert report_context["executor"] == "_Executor"
    assert report_context["slurm_array"] is None
    assert executor.records is None


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
    assert report["record_count"] >= 1
    assert report["records"]
