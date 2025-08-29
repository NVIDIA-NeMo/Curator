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

import json
import math
import os
import re
from pathlib import Path
from typing import Any

import pytest
import ray
from loguru import logger

from nemo_curator.backends.base import BaseExecutor
from nemo_curator.backends.experimental.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.backends.experimental.ray_data import RayDataExecutor
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.tasks import FileGroupTask
from nemo_curator.tasks.utils import TaskPerfUtils

from .utils import (
    EXPECTED_NUM_STAGES,
    FILES_PER_PARTITION,
    TOTAL_DOCUMENTS,
    capture_logs,
    create_test_data,
    create_test_pipeline,
)


@pytest.mark.parametrize(
    "backend_config",
    [
        pytest.param((RayDataExecutor, {}), id="ray_data"),
        pytest.param((XennaExecutor, {"execution_mode": "batch"}), id="xenna_batch"),
        pytest.param((XennaExecutor, {"execution_mode": "streaming"}), id="xenna_streaming"),
        pytest.param((RayActorPoolExecutor, {}), id="ray_actor_pool"),
    ],
    indirect=True,
)
class TestBackendIntegrations:
    NUM_TEST_FILES = 15
    EXPECTED_OUTPUT_TASKS = EXPECTED_OUTPUT_FILES = TOTAL_DOCUMENTS  # After split_into_rows stage

    # Class attributes for shared test data
    # These are set by the backend_config fixture
    backend_cls: BaseExecutor | None = None
    config: dict[str, Any] | None = None
    input_dir: Path | None = None
    output_dir: Path | None = None
    output_tasks: list[FileGroupTask] | None = None
    all_logs: str = ""

    @pytest.fixture(scope="class", autouse=True)
    def backend_config(self, request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory):
        """up test environment with backend-specific configuration and execute pipeline.."""
        # Get the backend class and config from the parametrized values
        backend_cls, config = request.param

        # Store as class attributes using request.cls (proper way for class-scoped fixtures)
        request.cls.backend_cls = backend_cls  # type: ignore[reportOptionalMemberAccess]
        request.cls.config = config  # type: ignore[reportOptionalMemberAccess]

        # Create fresh directories using tmp_path_factory for class-scoped fixture
        tmp_path = tmp_path_factory.mktemp("test_data")
        request.cls.input_dir = tmp_path / "input"  # type: ignore[reportOptionalMemberAccess]
        request.cls.output_dir = tmp_path / "output"  # type: ignore[reportOptionalMemberAccess]

        create_test_data(request.cls.input_dir, num_files=self.NUM_TEST_FILES)  # type: ignore[reportOptionalMemberAccess]
        pipeline = create_test_pipeline(request.cls.input_dir, request.cls.output_dir)  # type: ignore[reportOptionalMemberAccess]

        # Execute pipeline with comprehensive logging capture
        executor = backend_cls(config)
        with capture_logs() as log_buffer:
            request.cls.output_tasks = pipeline.run(executor)  # type: ignore[reportOptionalMemberAccess]
            # Store logs for backend-specific tests
            request.cls.all_logs = log_buffer.getvalue()  # type: ignore[reportOptionalMemberAccess]

        yield
        ray.kill(ray.get_actor("stage_call_counter", namespace="stage_call_counter"))
        logger.info(f"Ran pipeline for {request.cls.__name__}")

    def test_output_files(self):
        """Test that the correct number of output files are created with expected content."""
        assert self.output_dir is not None, "Output directory should be set by fixture"

        # Check file count
        output_files = list(self.output_dir.glob("*.jsonl"))
        assert len(output_files) == self.EXPECTED_OUTPUT_FILES, "Mismatch in number of output files"

        # Check file contents
        for file in output_files:
            with open(file) as f:
                lines = f.readlines()
                # Because of split_into_rows, each file should have 1 line
                assert len(lines) == 1, f"Expected 1 line per file but got {len(lines)}"
                data = json.loads(lines[0])
                assert set(data.keys()) == {
                    "id",
                    "text",
                    "doc_length_1",
                    "doc_length_2",
                    "node_id",
                    "random_string",
                }, "Mismatch in output file contents"

    def test_output_tasks(self):
        """Test that output tasks have the correct count, types, and properties."""
        assert self.output_tasks is not None, "Expected output tasks"

        # Check task count
        assert len(self.output_tasks) == self.EXPECTED_OUTPUT_TASKS, "Mismatch in number of output tasks"

        # Check all tasks are of type FileGroupTask
        assert all(isinstance(task, FileGroupTask) for task in self.output_tasks), "Mismatch in task types"

        # Check all task_ids are unique
        assert len({task.task_id for task in self.output_tasks}) == self.EXPECTED_OUTPUT_TASKS, (
            "Mismatch in number of task ids"
        )

        # Check all dataset names are the same
        assert all(task.dataset_name == self.output_tasks[0].dataset_name for task in self.output_tasks), (
            "Mismatch in dataset names"
        )

    def test_perf_stats(self):
        """Test that performance statistics are correctly recorded for all stages."""
        # Check content of stage perf stats
        assert self.output_tasks is not None, "Expected output tasks"
        expected_stage_names = [
            "jsonl_reader",
            "add_length",
            "split_into_rows",
            "add_length",
            "stage_with_setup",
            "jsonl_writer",
        ]
        for task_idx, task in enumerate(self.output_tasks):
            assert len(task._stage_perf) == EXPECTED_NUM_STAGES, "Mismatch in number of stage perf stats"
            # Make sure stage names match
            for stage_idx, perf_stats in enumerate(task._stage_perf):
                assert perf_stats.stage_name == expected_stage_names[stage_idx], (
                    f"Mismatch in stage name for stage {stage_idx} within task {task_idx}"
                )
                # Process time should be greater than idle time
                assert perf_stats.process_time > 0, "Process time should be non-zero for all stages"

            # We expect the first add_length and split_into_rows to have the same number of items processed
            assert task._stage_perf[1].num_items_processed == task._stage_perf[2].num_items_processed, (
                "Mismatch in number of items processed by firstadd_length and split_into_rows"
            )
            # Because we split df into a single row each, each stage after split_into_rows should have 1 item processed
            assert (
                task._stage_perf[3].num_items_processed
                == task._stage_perf[4].num_items_processed
                == task._stage_perf[5].num_items_processed
                == 1
            ), "Mismatch in number of items processed by stages after split_into_rows"

    def test_perf_stats_combined(self):
        """Test that the performance statistics are correctly combined."""
        # Also check custom metrics aggregation with TaskPerfUtils
        stage_metrics = TaskPerfUtils.collect_stage_metrics(self.output_tasks)
        # Non-zero checks for core metrics
        for m in stage_metrics.values():
            import numpy as np

            assert isinstance(m["process_time"], np.ndarray)
            assert isinstance(m["num_items_processed"], np.ndarray)
            assert m["process_time"].all() > 0.0
            assert m["num_items_processed"].all() > 0.0

        # Non-zero check for custom metrics we injected
        assert stage_metrics["add_length"]["custom.counter_actor_increment_s"].all() > 0.0
        assert stage_metrics["add_length"]["custom.compute_len_s"].all() > 0.0
        assert stage_metrics["split_into_rows"]["custom.split_into_rows_time_s"].all() > 0.0

    def test_ray_data_execution_plan(self):
        """Test that Ray Data creates the expected execution plan with correct stage organization."""
        if self.backend_cls != RayDataExecutor:
            pytest.skip("Execution plan test only applies to RayDataExecutor")

        # Look for execution plan in logs with multiple possible patterns
        matches = re.findall(r"Execution plan of Dataset.*?:\s*(.+)", self.all_logs, re.MULTILINE)
        # Take the last execution plan (most recent)
        execution_plan = matches[-1]
        # Split by " -> " to get individual stages
        stages = execution_plan.split(" -> ")
        execution_plan_stages = [stage.strip() for stage in stages]
        # Tasks can get fused with Actors, but Actors can't get fused with Tasks or Actors
        # StreamingRepartition should never get fused
        expected_stages = [
            "InputDataBuffer[Input]",
            "TaskPoolMapOperator[MapBatches(FilePartitioningStageTask)]",
            "TaskPoolMapOperator[StreamingRepartition]",
            "ActorPoolMapOperator[MapBatches(JsonlReaderStageTask)->MapBatches(AddLengthStageActor)]",
            "ActorPoolMapOperator[MapBatches(SplitIntoRowsStageActor)]",
            "TaskPoolMapOperator[StreamingRepartition]",
            "ActorPoolMapOperator[MapBatches(AddLengthStageActor)]",
            "ActorPoolMapOperator[MapBatches(StageWithSetupActor)]",
            "TaskPoolMapOperator[MapBatches(JsonlWriterTask)]",
        ]

        assert execution_plan_stages == expected_stages, (
            f"Expected execution plan stages: {expected_stages}, got: {execution_plan_stages}"
        )

    def test_stage_call_counts(self):
        """Test that the stage call counts are correctly recorded for all stages."""
        # Since they actor is killed (because each executor calls ray.shutdown())
        # we need to read the call_counters.json file
        with open(self.output_dir / "call_counters.json") as f:
            stage_call_counts = json.load(f)
        logger.info(f"Stage call counts: {stage_call_counts}")
        assert stage_call_counts == {
            "add_length_doc_length_1": math.ceil(self.NUM_TEST_FILES / FILES_PER_PARTITION),
            "add_length_doc_length_2": TOTAL_DOCUMENTS,
        }


class TestEnvVars:
    def test_max_limit_env_vars(self, shared_ray_client: None):  # noqa: ARG002
        """We set these env vars in __init__.py of the package


        # TODO: Once GPU is added we can test the env var for RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
            So that whoever starts a ray cluster, these env vars are set
            This allows Xenna to run without failure.
        """

        @ray.remote
        def get_env_vars() -> str:
            return {k: v for k, v in os.environ.items() if k.startswith("RAY_")}

        env_vars = ray.get(get_env_vars.remote())

        for env_var_name in ["RAY_MAX_LIMIT_FROM_API_SERVER", "RAY_MAX_LIMIT_FROM_DATA_SOURCE"]:
            assert os.environ[env_var_name] == str(40000), f"{env_var_name} is not correctly set on driver"
            assert env_vars[env_var_name] == str(40000), f"{env_var_name} is not correctly set on ray cluster"
