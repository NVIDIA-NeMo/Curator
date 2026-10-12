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

import json
import pickle
import sys
from pathlib import Path

import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking" / "scripts"))

from utils import write_benchmark_results

from nemo_curator.tasks import DocumentBatch
from nemo_curator.tasks.utils import TaskPerfUtils
from nemo_curator.utils.performance_utils import StagePerfStats


@pytest.mark.parametrize("save_tasks", [None, False, True])
def test_write_results_preserves_metrics_with_optional_tasks(tmp_path: Path, save_tasks: bool | None) -> None:
    task = DocumentBatch(dataset_name="test", data=pa.table({"text": ["payload"]}))
    task.add_stage_perf(
        StagePerfStats(stage_name="test_stage", process_time=2.0, num_items_processed=1, custom_metrics={"bytes": 7.0})
    )
    tasks = [task]
    results = {"params": {"executor": "test"}, "metrics": {"time_taken_s": 3.0}, "tasks": tasks}
    if save_tasks is None:
        write_benchmark_results(results, tmp_path)
    else:
        write_benchmark_results(results, tmp_path, save_tasks=save_tasks)

    assert json.loads((tmp_path / "metrics.json").read_text()) == {
        "time_taken_s": 3.0,
        **TaskPerfUtils.aggregate_task_metrics(tasks, prefix="task"),
    }
    assert json.loads((tmp_path / "params.json").read_text()) == results["params"]
    assert results["metrics"] == {"time_taken_s": 3.0}
    tasks_path = tmp_path / "tasks.pkl"
    if save_tasks is None or save_tasks:
        restored = pickle.loads(tasks_path.read_bytes())  # noqa: S301
        assert restored[0].data.equals(task.data)
        assert TaskPerfUtils.aggregate_task_metrics(restored) == TaskPerfUtils.aggregate_task_metrics(tasks)
    else:
        assert not tasks_path.exists()


def test_disabling_task_retention_removes_stale_pickle(tmp_path: Path) -> None:
    write_benchmark_results({"tasks": []}, tmp_path)
    assert (tmp_path / "tasks.pkl").exists()

    write_benchmark_results({"tasks": []}, tmp_path, save_tasks=False)
    assert not (tmp_path / "tasks.pkl").exists()
