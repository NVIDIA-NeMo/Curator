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

from dataclasses import dataclass
from unittest.mock import Mock, patch

import pytest

from nemo_curator.pipeline.pipeline import Pipeline, assign_root_task_ids
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import EmptyTask, Task, _EmptyTask


@dataclass
class _NoopStage(ProcessingStage[Task, Task]):
    name: str = "noop"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: Task) -> Task:
        return task


@dataclass
class _SimpleTask(Task[list[int]]):
    @property
    def num_items(self) -> int:
        return len(self.data) if self.data is not None else 0

    def validate(self) -> bool:
        return True


def test_pipeline_uses_xenna_executor_by_default():
    mock_xenna_instance = Mock()

    with patch("nemo_curator.backends.xenna.XennaExecutor") as mock_xenna_class:
        mock_xenna_class.return_value = mock_xenna_instance

        pipeline = Pipeline(name="test")
        pipeline.add_stage(Mock(spec=ProcessingStage))

        pipeline.run()

        mock_xenna_class.assert_called_once_with()
        mock_xenna_instance.execute.assert_called_once()


def test_logs_info_when_ray_serve_active_with_gpu_stages_non_xenna() -> None:
    """Non-Xenna executors log an info message when Serve is active with GPU stages."""
    gpu_stage = Mock(spec=ProcessingStage)
    gpu_stage.name = "EmbeddingStage"
    gpu_stage.resources = Resources(gpus=1.0)

    with patch("nemo_curator.core.serve.is_inference_server_active", return_value=True):
        mock_executor = Mock()
        pipeline = Pipeline(name="test", stages=[gpu_stage])

        with patch("nemo_curator.pipeline.pipeline.logger") as mock_logger:
            pipeline.run(executor=mock_executor)

            mock_logger.info.assert_called()
            info_msgs = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("Ray Serve is active" in msg for msg in info_msgs)
            assert any("EmbeddingStage" in msg for msg in info_msgs)


def test_raises_when_ray_serve_active_with_xenna_and_gpu_stages() -> None:
    """XennaExecutor raises RuntimeError when Serve is active with GPU stages."""
    from nemo_curator.backends.xenna import XennaExecutor

    gpu_stage = Mock(spec=ProcessingStage)
    gpu_stage.name = "EmbeddingStage"
    gpu_stage.resources = Resources(gpus=1.0)

    with patch("nemo_curator.core.serve.is_inference_server_active", return_value=True):
        mock_executor = Mock(spec=XennaExecutor)
        pipeline = Pipeline(name="test", stages=[gpu_stage])

        with pytest.raises(RuntimeError, match="Cannot run XennaExecutor"):
            pipeline.run(executor=mock_executor)


class TestPipelineBuild:
    """Source/sink role assignment performed by ``Pipeline.build``."""

    def test_default_first_source_last_sink_stage(self) -> None:
        """With no explicit marks, the first stage is the source and the
        last is the sink; a lone stage is both."""
        s0, s1, s2 = _NoopStage(name="s0"), _NoopStage(name="s1"), _NoopStage(name="s2")
        Pipeline(name="t", stages=[s0, s1, s2]).build()
        assert [s.is_source_stage for s in (s0, s1, s2)] == [True, False, False]
        assert [s.is_sink_stage for s in (s0, s1, s2)] == [False, False, True]

        lone = _NoopStage(name="lone")
        Pipeline(name="t", stages=[lone]).build()
        assert lone.is_source_stage is True
        assert lone.is_sink_stage is True

    def test_explicit_marks_override_defaults(self) -> None:
        s0, s1, s2 = _NoopStage(name="s0"), _NoopStage(name="s1"), _NoopStage(name="s2")
        s1.is_source_stage = True
        s1.is_sink_stage = True
        Pipeline(name="t", stages=[s0, s1, s2]).build()
        # Explicit source/sink win; defaults are not applied elsewhere.
        assert [s.is_source_stage for s in (s0, s1, s2)] == [False, True, False]
        assert [s.is_sink_stage for s in (s0, s1, s2)] == [False, True, False]

    def test_multiple_explicit_marks_raise(self) -> None:
        s0, s1 = _NoopStage(name="s0"), _NoopStage(name="s1")
        s0.is_source_stage = True
        s1.is_source_stage = True
        with pytest.raises(ValueError, match="multiple source stages marked"):
            Pipeline(name="t", stages=[s0, s1]).build()

        t0, t1 = _NoopStage(name="t0"), _NoopStage(name="t1")
        t0.is_sink_stage = True
        t1.is_sink_stage = True
        with pytest.raises(ValueError, match="multiple sink stages marked"):
            Pipeline(name="t", stages=[t0, t1]).build()


class TestRootTaskIds:
    """``assign_root_task_ids`` roots user-provided initial tasks under the
    implicit ``_EmptyTask`` root id ``"0"``."""

    def test_empty_task_id_is_zero(self) -> None:
        assert EmptyTask.task_id == "0"
        assert _EmptyTask(dataset_name="d", data=None).task_id == "0"

    def test_roots_user_tasks_at_zero(self) -> None:
        tasks = [_SimpleTask(dataset_name="d", data=[1]) for _ in range(3)]
        assign_root_task_ids(tasks)
        # User-provided initial tasks are children of root "0", by position.
        assert [t.task_id for t in tasks] == ["0_0", "0_1", "0_2"]

    def test_skips_empty_tasks(self) -> None:
        et = _EmptyTask(dataset_name="d", data=None)
        real = _SimpleTask(dataset_name="d", data=[1])
        assign_root_task_ids([et, real])
        # EmptyTask stays "0"; the real task is rooted by its position.
        assert et.task_id == "0"
        assert real.task_id == "0_1"
