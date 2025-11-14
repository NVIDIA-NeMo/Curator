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

import pytest

from nemo_curator.stages.base import ProcessingStage, get_stage_class
from nemo_curator.stages.function_decorators import processing_stage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import Task


class MockTask(Task[int]):
    """Simple Task subclass for testing the decorator."""

    def __init__(self, value: int = 0):
        super().__init__(task_id="mock", dataset_name="test", data=value)

    @property
    def num_items(self) -> int:
        return 1

    def validate(self) -> bool:
        return True


# -----------------------------------------------------------------------------
# Helper stages created via the decorator
# -----------------------------------------------------------------------------

# A stage that increments the task's integer payload
resources_inc = Resources(cpus=1.5)


@processing_stage(name="IncrementStage", resources=resources_inc, batch_size=4)
def increment_stage(task: MockTask) -> MockTask:
    task.data += 1
    return task


# A stage that duplicates the task (fan-out style)
resources_dup = Resources(cpus=0.5)


@processing_stage(name="DuplicateStage", resources=resources_dup, batch_size=2)
def duplicate_stage(task: MockTask) -> list[MockTask]:
    return [task, task]


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


class TestProcessingStageDecorator:
    """Unit tests for the *processing_stage* decorator."""

    def test_instance_properties(self) -> None:
        """The decorator should turn the function into a ProcessingStage instance
        with the supplied configuration values.
        """

        stage = increment_stage  # Decorator replaces the function with an instance
        assert isinstance(stage, ProcessingStage)
        assert stage.name == "IncrementStage"
        assert stage.resources == resources_inc
        assert stage.batch_size == 4

    def test_process_single_task(self) -> None:
        """process() should delegate to the wrapped function and return a task."""

        stage = increment_stage
        task = MockTask(value=0)
        result = stage.process(task)
        assert isinstance(result, MockTask)
        # The function increments the payload by 1
        assert result.data == 1

    def test_process_batch_single_task(self) -> None:
        """Ensure that process_batch is inherited."""

        stage = increment_stage
        task = MockTask(value=0)
        result = stage.process_batch([task])

        # Check process_batch output
        assert isinstance(result, list)
        assert len(result) == 1

        # Check process output
        assert isinstance(result[0], MockTask)
        # The function increments the payload by 1
        assert result[0].data == 1

    @pytest.mark.parametrize("process_batch", [True, False])
    def test_process_list_output(self, process_batch: bool) -> None:
        """Stage should support functions that return lists of tasks."""

        stage = duplicate_stage
        task = MockTask(value=42)

        result = stage.process_batch([task]) if process_batch else stage.process(task)

        assert isinstance(result, list)
        assert len(result) == 2
        # All returned objects should be MockTask instances pointing at the same task
        assert all(isinstance(t, MockTask) for t in result)
        assert all(t is task for t in result)

    def test_invalid_signature_raises(self) -> None:
        """Functions with an invalid signature should raise *ValueError* when
        decorated.
        """

        with pytest.raises(ValueError):  # noqa: PT011

            @processing_stage(name="BadStage")
            def bad_stage(task: MockTask, _: int):  # type: ignore[valid-type]  # noqa: ANN202
                return task

    def test_stage_registry(self) -> None:
        """Uses get_stage_class to ensure that stage names are in the _STAGE_REGISTRY."""
        assert get_stage_class("IncrementStage") is not None
        assert get_stage_class("DuplicateStage") is not None
