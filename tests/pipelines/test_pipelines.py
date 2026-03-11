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

from nemo_curator.pipeline.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage


@pytest.fixture
def some_stage() -> ProcessingStage:
    return Mock(spec=ProcessingStage)


def test_pipeline_uses_xenna_executor_by_default(some_stage: ProcessingStage):
    # Create a mock executor instance
    mock_xenna_instance = Mock()

    with patch("nemo_curator.backends.xenna.XennaExecutor") as mock_xenna_class:
        mock_xenna_class.return_value = mock_xenna_instance

        pipeline = Pipeline(name="test")
        pipeline.add_stage(some_stage)

        # Call run without executor
        pipeline.run()

        # Verify XennaExecutor was instantiated
        mock_xenna_class.assert_called_once_with()

        # Verify execute was called on the XennaExecutor instance
        mock_xenna_instance.execute.assert_called_once()
