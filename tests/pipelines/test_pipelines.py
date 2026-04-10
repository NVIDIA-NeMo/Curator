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

from unittest.mock import Mock, patch

import pytest

from nemo_curator.core import serve as serve_module
from nemo_curator.pipeline.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources


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

    serve_module._active_servers.add("default")
    try:
        mock_executor = Mock()
        pipeline = Pipeline(name="test", stages=[gpu_stage])

        with patch("nemo_curator.pipeline.pipeline.logger") as mock_logger:
            pipeline.run(executor=mock_executor)

            mock_logger.info.assert_called()
            info_msgs = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("Ray Serve is active" in msg for msg in info_msgs)
            assert any("EmbeddingStage" in msg for msg in info_msgs)
    finally:
        serve_module._active_servers.clear()


def test_raises_when_ray_serve_active_with_xenna_and_gpu_stages() -> None:
    """XennaExecutor raises RuntimeError when Serve is active with GPU stages."""
    from nemo_curator.backends.xenna import XennaExecutor

    gpu_stage = Mock(spec=ProcessingStage)
    gpu_stage.name = "EmbeddingStage"
    gpu_stage.resources = Resources(gpus=1.0)

    serve_module._active_servers.add("default")
    try:
        mock_executor = Mock(spec=XennaExecutor)
        pipeline = Pipeline(name="test", stages=[gpu_stage])

        with pytest.raises(RuntimeError, match="Cannot run XennaExecutor"):
            pipeline.run(executor=mock_executor)
    finally:
        serve_module._active_servers.clear()
