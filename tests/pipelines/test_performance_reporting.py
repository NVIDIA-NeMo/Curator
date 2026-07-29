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

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import Task
from nemo_curator.utils.performance_utils import StagePerfStats


class _TerminalConsumer(ProcessingStage[Task, Task]):
    name = "duplicate"

    def __init__(self) -> None:
        self.prepared = False
        self.finalized: tuple[list[Task], list[StagePerfStats], float] | None = None

    def process(self, task: Task) -> Task:
        return task

    def prepare_performance_report(self) -> None:
        self.prepared = True

    def finalize_performance_report(
        self,
        tasks: list[Task],
        *,
        performance_records: list[StagePerfStats],
        wall_time_s: float,
    ) -> None:
        self.finalized = (tasks, performance_records, wall_time_s)


class _Executor:
    def __init__(self) -> None:
        self.records = [
            StagePerfStats(
                stage_name="duplicate",
                invocation_id="invocation-1",
                process_time=1.0,
            )
        ]

    def execute(self, _stages: list[ProcessingStage], initial_tasks: list[Task] | None = None) -> list[Task]:
        return list(initial_tasks or [])

    def consume_external_perf_records(self) -> list[StagePerfStats]:
        records, self.records = self.records, []
        return records


def test_pipeline_assigns_stable_ids_and_fans_out_one_record_drain(tmp_path: Path) -> None:
    first = _TerminalConsumer()
    terminal = _TerminalConsumer()
    executor = _Executor()
    report_path = tmp_path / "raw-performance.json"
    pipeline = Pipeline(name="performance", stages=[first, terminal])

    result = pipeline.run(
        executor=executor,  # type: ignore[arg-type]
        initial_tasks=[],
        performance_report_path=report_path,
    )

    assert result == []
    assert first._curator_stage_id == "000:duplicate"
    assert terminal._curator_stage_id == "001:duplicate"
    assert first.prepared is False
    assert terminal.prepared is True
    assert terminal.finalized is not None
    tasks, records, wall_time_s = terminal.finalized
    assert tasks == []
    assert records is pipeline.performance_records
    assert len(records) == 1
    assert wall_time_s >= 0.0
    assert executor.records == []

    report = json.loads(report_path.read_text())
    assert report["schema_version"] == 1
    assert report["record_count"] == 1
    assert report["records"][0]["invocation_id"] == "invocation-1"
