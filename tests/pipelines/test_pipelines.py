from unittest.mock import Mock, patch

import pytest

from ray_curator.pipeline.pipeline import Pipeline
from ray_curator.stages.base import ProcessingStage


@pytest.fixture
def some_stage() -> ProcessingStage:
    return Mock(spec=ProcessingStage)


def test_pipeline_uses_xenna_executor_by_default(some_stage: ProcessingStage):
    # Create a mock executor instance
    mock_xenna_instance = Mock()

    with patch("ray_curator.backends.xenna.XennaExecutor") as mock_xenna_class:
        mock_xenna_class.return_value = mock_xenna_instance

        pipeline = Pipeline(name="test")
        pipeline.add_stage(some_stage)

        # Call run without executor
        pipeline.run()

        # Verify XennaExecutor was instantiated
        mock_xenna_class.assert_called_once_with()

        # Verify execute was called on the XennaExecutor instance
        mock_xenna_instance.execute.assert_called_once()
