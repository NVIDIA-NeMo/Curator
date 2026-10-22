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

import pickle
import sys
from pathlib import Path
from unittest.mock import Mock

import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking/scripts"))

# Importing the benchmark requires the optional exact-dedup dependencies, but these tests do not run GPU work.
pytest.importorskip("nemo_curator.stages.deduplication.exact.workflow")

import exact_dedup_identification_benchmark as benchmark
from utils import write_benchmark_results

from benchmarking.run import get_entry_script_persisted_data
from nemo_curator.pipeline.workflow import WorkflowRunResult
from nemo_curator.tasks import DocumentBatch
from nemo_curator.tasks.utils import TaskPerfUtils
from nemo_curator.utils.performance_utils import StagePerfStats


@pytest.mark.parametrize("save_tasks", [False, True])
@pytest.mark.parametrize("failure_stage", ["construction", "execution"])
def test_failed_workflow_results_can_be_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, save_tasks: bool, failure_stage: str
) -> None:
    workflow = Mock()
    if failure_stage == "construction":
        workflow.side_effect = RuntimeError("construction failed")
    else:
        workflow.return_value.run.side_effect = RuntimeError("execution failed")
    monkeypatch.setattr(benchmark, "ExactDeduplicationWorkflow", workflow)

    result = benchmark.run_exact_duplicate_identification_benchmark("input", str(tmp_path / "output"))
    assert result["tasks"] == []
    assert result["metrics"]["is_success"] is False
    assert result["metrics"]["time_taken_s"] >= 0
    for name in ("num_duplicates", "identification_time_s", "input_filegroups_time_s"):
        assert result["metrics"][name] == 0
    write_benchmark_results(result, tmp_path, save_tasks=save_tasks)
    assert get_entry_script_persisted_data(tmp_path)["metrics"] == result["metrics"]
    assert (tmp_path / "tasks.pkl").exists() is save_tasks
    if save_tasks:
        assert pickle.loads((tmp_path / "tasks.pkl").read_bytes()) == []  # noqa: S301


@pytest.mark.parametrize("save_tasks", [False, True])
def test_successful_workflow_preserves_tasks_and_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, save_tasks: bool
) -> None:
    task = DocumentBatch(dataset_name="test", data=pa.table({"text": ["payload"]}))
    task.add_stage_perf(StagePerfStats(stage_name="identify", process_time=2.0, num_items_processed=1))
    completed = WorkflowRunResult(
        workflow_name="exact",
        pipeline_tasks={"identify": [task]},
        metadata={"num_duplicates": 1, "identification_time": 2.0, "input_filegroups_time": 0.5},
    )
    workflow = Mock()
    workflow.return_value.run.return_value = completed
    monkeypatch.setattr(benchmark, "ExactDeduplicationWorkflow", workflow)

    result = benchmark.run_exact_duplicate_identification_benchmark("input", str(tmp_path / "output"))
    assert result["tasks"] is completed
    assert result["metrics"]["is_success"] is True
    assert result["metrics"]["num_duplicates"] == 1
    assert result["metrics"]["identification_time_s"] == 2.0
    assert result["metrics"]["input_filegroups_time_s"] == 0.5
    write_benchmark_results(result, tmp_path, save_tasks=save_tasks)
    expected = {**result["metrics"], **TaskPerfUtils.aggregate_task_metrics(completed, prefix="task")}
    assert get_entry_script_persisted_data(tmp_path)["metrics"] == expected
    assert (tmp_path / "tasks.pkl").exists() is save_tasks
    if save_tasks:
        restored = pickle.loads((tmp_path / "tasks.pkl").read_bytes())  # noqa: S301
        assert isinstance(restored, WorkflowRunResult)
        assert restored.pipeline_tasks["identify"][0].data.equals(task.data)
