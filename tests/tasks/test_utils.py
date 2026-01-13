# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np

from nemo_curator.pipeline.workflow import WorkflowRunResult
from nemo_curator.tasks import _EmptyTask
from nemo_curator.tasks.utils import TaskPerfUtils
from nemo_curator.utils.performance_utils import StagePerfStats


def make_dummy_task(stage_name: str, process_time: float, custom: float = 0.0) -> _EmptyTask:
    perf = StagePerfStats(stage_name=stage_name, process_time=process_time, custom_metrics={"io": custom})
    return _EmptyTask(task_id=f"{stage_name}_{process_time}", dataset_name="test", data=None, _stage_perf=[perf])


class TestTaskPerfUtils:
    """Test cases for TaskPerfUtils class."""

    def test_collect_stage_metrics_from_workflow_result(self) -> None:
        """Test collecting stage metrics from WorkflowRunResult."""
        workflow_result = WorkflowRunResult(workflow_name="unit")
        workflow_result.add_pipeline_tasks("pipe_a", [make_dummy_task("StageA", 1.0, custom=5.0)])
        workflow_result.add_pipeline_tasks("pipe_b", [make_dummy_task("StageB", 2.0, custom=7.0)])

        metrics = TaskPerfUtils.collect_stage_metrics(workflow_result)

        assert set(metrics.keys()) == {"StageA", "StageB"}
        assert np.allclose(metrics["StageA"]["process_time"], np.array([1.0]))
        assert np.allclose(metrics["StageB"]["custom.io"], np.array([7.0]))

    def test_collect_stage_metrics_from_pipeline_tasks(self) -> None:
        """Test collecting stage metrics from pipeline tasks dictionary."""
        pipeline_tasks = {
            "pipe_a": [make_dummy_task("StageA", 1.0, custom=5.0)],
            "pipe_b": [make_dummy_task("StageB", 2.0, custom=7.0)],
        }

        metrics = TaskPerfUtils.collect_stage_metrics(pipeline_tasks)

        assert set(metrics.keys()) == {"StageA", "StageB"}
        assert np.allclose(metrics["StageA"]["process_time"], np.array([1.0]))
        assert np.allclose(metrics["StageA"]["custom.io"], np.array([5.0]))
        assert np.allclose(metrics["StageB"]["process_time"], np.array([2.0]))
        assert np.allclose(metrics["StageB"]["custom.io"], np.array([7.0]))

    def test_aggregate_task_metrics_with_pipeline_prefixes(self) -> None:
        """Test aggregating task metrics with pipeline prefixes."""
        pipeline_tasks = {
            "pipe_a": [make_dummy_task("StageShared", 1.0)],
            "pipe_b": [make_dummy_task("StageShared", 3.0)],
        }

        metrics = TaskPerfUtils.aggregate_task_metrics(pipeline_tasks)

        assert "pipe_a_StageShared_process_time_sum" in metrics
        assert metrics["pipe_a_StageShared_process_time_sum"] == 1.0
        assert metrics["pipe_b_StageShared_process_time_sum"] == 3.0

    def test_aggregate_task_metrics_with_custom_prefix(self) -> None:
        """Test aggregating task metrics with custom prefix."""
        pipeline_tasks = {
            "pipe_a": [make_dummy_task("StageShared", 1.0)],
            "pipe_b": [make_dummy_task("StageShared", 3.0)],
        }

        prefixed_metrics = TaskPerfUtils.aggregate_task_metrics(pipeline_tasks, prefix="workflow")
        assert "workflow_pipe_a_StageShared_process_time_sum" in prefixed_metrics
        assert "workflow_pipe_b_StageShared_process_time_sum" in prefixed_metrics

    def test_aggregate_task_metrics_from_workflow_result(self) -> None:
        """Test aggregating task metrics from WorkflowRunResult."""
        workflow_result = WorkflowRunResult(workflow_name="unit")
        workflow_result.add_pipeline_tasks("pipe_a", [make_dummy_task("StageA", 1.0, custom=5.0)])
        workflow_result.add_pipeline_tasks("pipe_b", [make_dummy_task("StageB", 2.0, custom=7.0)])

        metrics = TaskPerfUtils.aggregate_task_metrics(workflow_result)

        assert "pipe_a_StageA_process_time_sum" in metrics
        assert "pipe_b_StageB_process_time_sum" in metrics
        assert metrics["pipe_a_StageA_process_time_sum"] == 1.0
        assert metrics["pipe_b_StageB_process_time_sum"] == 2.0
