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

import math
import os
import re
import tempfile

import pandas as pd
import pytest
import ray
from loguru import logger

from nemo_curator.backends.ray_data.executor import RayDataExecutor
from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.core.client import RayClient
from nemo_curator.stages.base import ProcessingStage, Resources
from nemo_curator.tasks import DocumentBatch, EmptyTask
from tests.backends.utils import capture_logs


@pytest.fixture(scope="module")
def single_cpu_ray_cluster():
    """Start an isolated 1-CPU Ray cluster for deterministic PID testing.

    Uses a standalone Ray cluster instead of the session-scoped conftest cluster
    to ensure a single-CPU cluster for testing.
    Uses tempfile.mkdtemp for a short path to avoid hitting the Unix socket
    path length limit (108 chars) that pytest's tmp_path_factory can exceed.
    """
    original_ray_address = os.environ.pop("RAY_ADDRESS", None)

    temp_dir = tempfile.mkdtemp(prefix="ray1cpu_")
    ray_client = RayClient(
        num_cpus=1,
        num_gpus=0,
        object_store_memory=2 * (1024**3),
        ray_temp_dir=str(temp_dir),
        include_dashboard=False,
    )
    ray_client.start()

    ray_address = os.environ["RAY_ADDRESS"]

    try:
        yield ray_address
    finally:
        ray_client.stop()
        if original_ray_address is not None:
            os.environ["RAY_ADDRESS"] = original_ray_address
        elif "RAY_ADDRESS" in os.environ:
            del os.environ["RAY_ADDRESS"]


@pytest.fixture
def single_cpu_ray_client(single_cpu_ray_cluster: str) -> None:
    """Initialize Ray client for tests that need Ray API access."""
    ray.init(
        address=single_cpu_ray_cluster,
        ignore_reinit_error=True,
        log_to_driver=True,
        local_mode=False,
    )

    try:
        yield
    finally:
        logger.info("Shutting down Ray client")
        ray.shutdown()


class PidRecorderStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Test stage that records the worker PID in the output data."""

    name = "pid_recorder"

    def __init__(self, max_calls_per_worker: int | None = None):
        self._max_calls_per_worker = max_calls_per_worker

    def ray_stage_spec(self) -> dict:
        spec = {}
        if self._max_calls_per_worker is not None:
            spec[RayStageSpecKeys.MAX_CALLS_PER_WORKER] = self._max_calls_per_worker
        return spec

    def process(self, task: DocumentBatch) -> DocumentBatch:
        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=pd.DataFrame({"worker_pid": [os.getpid()]}),
        )


class PassthroughActorStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Actor stage (has setup()) that passes data through unchanged.

    Overriding setup() causes is_actor_stage() to return True, so Ray Data
    will execute this as an actor-based map_batches call.
    """

    name = "passthrough_actor"

    def ray_stage_spec(self) -> dict:
        return {
            RayStageSpecKeys.IS_ACTOR_STAGE: True,
        }

    def process(self, task: DocumentBatch) -> DocumentBatch:
        return task


class PassthroughTaskStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    name = "passthrough_task"

    def process(self, task: DocumentBatch) -> DocumentBatch:
        return task


@pytest.mark.parametrize("max_calls_per_worker", [2, None])
@pytest.mark.usefixtures("single_cpu_ray_client")
def test_pid_recycling(max_calls_per_worker: int | None):
    tasks = [EmptyTask] * 8

    stage = PidRecorderStage(max_calls_per_worker=max_calls_per_worker)
    executor = RayDataExecutor()
    results = executor.execute(stages=[stage], initial_tasks=tasks)

    pids = [r.data["worker_pid"].iloc[0] for r in results]
    expected_unique_pids = math.ceil(len(tasks) / max_calls_per_worker) if max_calls_per_worker else 1

    assert len(set(pids)) == expected_unique_pids, (
        f"Expected {expected_unique_pids} unique PIDs with max_calls={max_calls_per_worker}, "
        f"got {len(set(pids))}: {pids}"
    )


@pytest.mark.parametrize("max_calls_per_worker", [2, None])
@pytest.mark.usefixtures("single_cpu_ray_client")
def test_max_calls_not_fused_with_actor_stage(max_calls_per_worker: int | None):
    """Verify that a task stage with max_calls is not fused with a following actor stage.

    Ray Data's operator fusion optimization can merge consecutive map_batches
    operations. If PidRecorderStage (task-based, max_calls=1) were fused with
    PassthroughActorStage (actor-based), the max_calls setting would be lost
    and we'd see only 1 unique PID instead of one per task.

    By chaining the two stages and asserting PID recycling still occurs, we
    confirm that Ray Data keeps them as separate operators. We also verify
    the execution plan directly to ensure no fusion occurred.
    """
    num_tasks = 4
    tasks = [EmptyTask] * num_tasks

    pid_stage = PidRecorderStage(max_calls_per_worker=max_calls_per_worker).with_(resources=Resources(cpus=0.5))
    actor_stage = PassthroughActorStage().with_(resources=Resources(cpus=0.5))

    executor = RayDataExecutor()
    with capture_logs() as log_buffer:
        results = executor.execute(stages=[pid_stage, actor_stage], initial_tasks=tasks)
        all_logs = log_buffer.getvalue()

    # Verify PID recycling
    pids = [r.data["worker_pid"].iloc[0] for r in results]
    expected_unique_pids = math.ceil(num_tasks / max_calls_per_worker) if max_calls_per_worker else 1

    assert len(set(pids)) == expected_unique_pids, (
        f"Expected {expected_unique_pids} unique PIDs (max_calls={max_calls_per_worker}, "
        f"{num_tasks} tasks), got {len(set(pids))}: {pids}. "
        f"Stages may have been fused by Ray Data, defeating max_calls."
    )

    # Verify execution plan fusion behavior
    matches = re.findall(r"Execution plan of Dataset.*?:\s*(.+)", all_logs, re.MULTILINE)
    assert matches, f"No execution plan found in Ray Data logs. Full logs:\n{all_logs}"
    plan_stages = [s.strip() for s in matches[-1].split(" -> ")]
    map_batches_stages = [s for s in plan_stages if "MapBatches" in s]

    if max_calls_per_worker is not None:
        # When max_calls is set, stages must NOT be fused — each should be a separate operator.
        assert len(map_batches_stages) == 2, (
            f"Expected 2 separate MapBatches operators, got {len(map_batches_stages)}: {map_batches_stages}. "
            f"Full execution plan: {matches[-1]}"
        )
        for stage in map_batches_stages:
            assert stage.count("MapBatches") == 1, (
                f"Stages were fused into a single operator: {stage}. Full execution plan: {matches[-1]}"
            )
    else:
        # When max_calls is None, Ray Data is free to fuse — confirm fusion happened.
        assert len(map_batches_stages) == 1, (
            f"Expected 1 fused MapBatches operator, got {len(map_batches_stages)}: {map_batches_stages}. "
            f"Full execution plan: {matches[-1]}"
        )
        assert map_batches_stages[0].count("MapBatches") == 2, (
            f"Expected fused operator with 2 MapBatches, got: {map_batches_stages[0]}. "
            f"Full execution plan: {matches[-1]}"
        )


@pytest.mark.parametrize("max_calls_per_worker", [2, None])
@pytest.mark.usefixtures("single_cpu_ray_client")
def test_max_calls_not_fused_with_task_stage(max_calls_per_worker: int | None):
    """Verify that a task stage with max_calls is not fused with a following task stage.

    Ray Data's operator fusion optimization can merge consecutive map_batches
    operations. If PidRecorderStage (task-based, max_calls=1) were fused with
    PassthroughTaskStage (task-based), the max_calls setting would be lost
    and we'd see only 1 unique PID instead of one per task.

    We also verify the execution plan directly to ensure no fusion occurred.
    """
    num_tasks = 4
    tasks = [EmptyTask] * num_tasks

    pid_stage = PidRecorderStage(max_calls_per_worker=max_calls_per_worker).with_(resources=Resources(cpus=0.5))
    task_stage = PassthroughTaskStage().with_(resources=Resources(cpus=0.5))

    executor = RayDataExecutor()
    with capture_logs() as log_buffer:
        results = executor.execute(stages=[task_stage, pid_stage], initial_tasks=tasks)
        all_logs = log_buffer.getvalue()

    # Verify PID recycling
    pids = [r.data["worker_pid"].iloc[0] for r in results]
    expected_unique_pids = math.ceil(num_tasks / max_calls_per_worker) if max_calls_per_worker else 1

    assert len(set(pids)) == expected_unique_pids, (
        f"Expected {expected_unique_pids} unique PIDs (max_calls={max_calls_per_worker}, "
        f"{num_tasks} tasks), got {len(set(pids))}: {pids}. "
        f"Stages may have been fused by Ray Data, defeating max_calls."
    )

    # Verify execution plan fusion behavior
    matches = re.findall(r"Execution plan of Dataset.*?:\s*(.+)", all_logs, re.MULTILINE)
    assert matches, f"No execution plan found in Ray Data logs. Full logs:\n{all_logs}"
    plan_stages = [s.strip() for s in matches[-1].split(" -> ")]
    map_batches_stages = [s for s in plan_stages if "MapBatches" in s]

    if max_calls_per_worker is not None:
        # When max_calls is set, stages must NOT be fused — each should be a separate operator.
        assert len(map_batches_stages) == 2, (
            f"Expected 2 separate MapBatches operators, got {len(map_batches_stages)}: {map_batches_stages}. "
            f"Full execution plan: {matches[-1]}"
        )
        for stage in map_batches_stages:
            assert stage.count("MapBatches") == 1, (
                f"Stages were fused into a single operator: {stage}. Full execution plan: {matches[-1]}"
            )
    else:
        # When max_calls is None, Ray Data is free to fuse — confirm fusion happened.
        assert len(map_batches_stages) == 1, (
            f"Expected 1 fused MapBatches operator, got {len(map_batches_stages)}: {map_batches_stages}. "
            f"Full execution plan: {matches[-1]}"
        )
        assert map_batches_stages[0].count("MapBatches") == 2, (
            f"Expected fused operator with 2 MapBatches, got: {map_batches_stages[0]}. "
            f"Full execution plan: {matches[-1]}"
        )
