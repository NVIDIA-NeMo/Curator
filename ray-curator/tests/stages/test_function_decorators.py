import pytest

from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.function_decorators import processing_stage
from ray_curator.stages.resources import Resources
from ray_curator.tasks import Task

# TODO: Add tests for ray pipelines too


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
def _increment(task: MockTask) -> MockTask:
    task.data += 1
    return task


# A stage that duplicates the task (fan-out style)
resources_dup = Resources(cpus=0.5)


@processing_stage(name="DuplicateStage", resources=resources_dup, batch_size=2)
def _duplicate(task: MockTask) -> list[MockTask]:
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

        stage = _increment  # Decorator replaces the function with an instance
        assert isinstance(stage, ProcessingStage)
        assert stage.name == "IncrementStage"
        assert stage.resources == resources_inc
        assert stage.batch_size == 4

    def test_process_single_task(self) -> None:
        """process() should delegate to the wrapped function and return a task."""

        stage = _increment
        task = MockTask(value=0)
        result = stage.process(task)
        assert isinstance(result, MockTask)
        # The function increments the payload by 1
        assert result.data == 1

    def test_process_list_output(self) -> None:
        """Stage should support functions that return lists of tasks."""

        stage = _duplicate
        task = MockTask(value=42)
        result = stage.process(task)
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
            def _bad(task: MockTask, _: int):  # type: ignore[valid-type]  # noqa: ANN202
                return task
