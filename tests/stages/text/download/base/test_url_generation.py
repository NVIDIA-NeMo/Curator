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

from unittest import mock

import pytest

from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.download.base.url_generation import URLGenerationStage, URLGenerator
from nemo_curator.tasks import FileGroupTask, _EmptyTask


class MockURLGenerator(URLGenerator):
    """Mock implementation of URLGenerator for testing."""

    def __init__(self, urls: list[str] | None = None, fail_on_generate: bool = False):
        if urls is None:
            self.urls = [
                "http://example.com/file1.txt",
                "http://example.com/file2.txt",
                "http://example.com/file3.txt",
                "http://example.com/file4.txt",
                "http://example.com/file5.txt",
            ]
        else:
            self.urls = urls
        self.fail_on_generate = fail_on_generate

    def generate_urls(self) -> list[str]:
        """Mock URL generation implementation."""
        if self.fail_on_generate:
            msg = "Failed to generate URLs"
            raise RuntimeError(msg)
        return self.urls.copy()


class TestBaseURLGenerator:
    """Base test class for URLGenerator functionality."""

    def test_url_generator_basic_functionality(self) -> None:
        """Test basic URL generation functionality."""
        urls = ["http://test.com/file1.txt", "http://test.com/file2.txt"]
        generator = MockURLGenerator(urls=urls)

        result = generator.generate_urls()

        assert result == urls
        assert result is not urls  # Should return a copy

    def test_url_generator_empty_urls(self) -> None:
        """Test URL generator with empty URL list."""
        generator = MockURLGenerator(urls=[])

        result = generator.generate_urls()
        assert result == []

    def test_url_generator_with_error(self) -> None:
        """Test URL generator behavior when generation fails."""
        generator = MockURLGenerator(fail_on_generate=True)

        with pytest.raises(RuntimeError, match="Failed to generate URLs"):
            generator.generate_urls()

    def test_url_generator_default_urls(self) -> None:
        """Test URL generator with default URLs."""
        generator = MockURLGenerator()

        result = generator.generate_urls()
        assert len(result) == 5
        assert all(url.startswith("http://example.com/") for url in result)


class TestURLGenerationStage:
    """Test class for URLGenerationStage functionality."""

    def test_stage_properties(self) -> None:
        """Test that stage properties are correctly defined."""
        generator = MockURLGenerator()
        stage = URLGenerationStage(url_generator=generator)

        # Test stage name
        assert stage.name == "url_generation_mockurlgenerator"

        # Test inputs and outputs
        assert stage.inputs() == ([], [])
        assert stage.outputs() == (["data"], [])

        # Test resources
        assert stage.resources == Resources(cpus=0.5)

        # Test ray stage spec
        assert stage.ray_stage_spec() == {"is_fanout_stage": True}

    def test_stage_properties_with_limit(self) -> None:
        """Test stage properties when limit is set."""
        generator = MockURLGenerator()
        stage = URLGenerationStage(url_generator=generator, limit=3)

        assert stage.limit == 3

    def test_process_successful_generation(self) -> None:
        """Test successful URL generation and task creation."""
        urls = ["http://example.com/file1.txt", "http://example.com/file2.txt", "http://example.com/file3.txt"]
        generator = MockURLGenerator(urls=urls)
        stage = URLGenerationStage(url_generator=generator)

        # Create input task
        input_task = _EmptyTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=None,
            _metadata={"source": "test"},
        )

        result = stage.process(input_task)

        # Verify result structure
        assert isinstance(result, list)
        assert len(result) == 3

        # Check each generated task
        for i, task in enumerate(result):
            assert isinstance(task, FileGroupTask)
            assert task.task_id == f"test_task_{i}"
            assert task.dataset_name == "test_dataset"
            assert task.data == [urls[i]]
            assert task._metadata == {"source_url": urls[i]}

    def test_process_with_limit(self) -> None:
        """Test URL generation with limit applied."""
        urls = [
            "http://example.com/file1.txt",
            "http://example.com/file2.txt",
            "http://example.com/file3.txt",
            "http://example.com/file4.txt",
            "http://example.com/file5.txt",
        ]
        generator = MockURLGenerator(urls=urls)
        stage = URLGenerationStage(url_generator=generator, limit=3)

        input_task = _EmptyTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=None,
            _metadata={},
        )

        result = stage.process(input_task)

        # Should only have 3 tasks due to limit
        assert len(result) == 3

        # Check that correct URLs were used (first 3)
        for i, task in enumerate(result):
            assert task.data == [urls[i]]

    def test_process_empty_url_list(self) -> None:
        """Test processing when generator returns empty URL list."""
        generator = MockURLGenerator(urls=[])
        stage = URLGenerationStage(url_generator=generator)

        input_task = _EmptyTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=None,
            _metadata={},
        )

        result = stage.process(input_task)

        # Should return empty list
        assert result == []

    def test_process_limit_larger_than_urls(self) -> None:
        """Test when limit is larger than available URLs."""
        urls = ["http://example.com/file1.txt", "http://example.com/file2.txt"]
        generator = MockURLGenerator(urls=urls)
        stage = URLGenerationStage(url_generator=generator, limit=10)

        input_task = _EmptyTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=None,
            _metadata={},
        )

        result = stage.process(input_task)

        # Should return all available URLs (not fail due to limit)
        assert len(result) == 2

    def test_process_limit_zero(self) -> None:
        """Test when limit is set to zero."""
        generator = MockURLGenerator()
        stage = URLGenerationStage(url_generator=generator, limit=0)

        input_task = _EmptyTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=None,
            _metadata={},
        )

        result = stage.process(input_task)

        # Should return empty list when limit is 0
        assert result == []

    @mock.patch.object(MockURLGenerator, "generate_urls")
    def test_process_generation_failure(self, mock_generate: mock.Mock) -> None:
        """Test handling when URL generation fails."""
        mock_generate.side_effect = RuntimeError("Generation failed")

        generator = MockURLGenerator()
        stage = URLGenerationStage(url_generator=generator)

        input_task = _EmptyTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=None,
            _metadata={},
        )

        # Should propagate the exception
        with pytest.raises(RuntimeError, match="Generation failed"):
            stage.process(input_task)

    def test_process_task_metadata_propagation(self) -> None:
        """Test that original task metadata is preserved in stage performance tracking."""
        urls = ["http://example.com/file1.txt"]
        generator = MockURLGenerator(urls=urls)
        stage = URLGenerationStage(url_generator=generator)

        input_task = _EmptyTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=None,
            _metadata={"original": "metadata"},
        )

        result = stage.process(input_task)

        # Check that stage performance metadata is propagated
        assert len(result) == 1
        task = result[0]
        assert task._stage_perf == input_task._stage_perf

    def test_process_single_url_per_task(self) -> None:
        """Test that each URL gets its own task for parallelization."""
        urls = ["http://example.com/file1.txt", "http://example.com/file2.txt"]
        generator = MockURLGenerator(urls=urls)
        stage = URLGenerationStage(url_generator=generator)

        input_task = _EmptyTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=None,
            _metadata={},
        )

        result = stage.process(input_task)

        # Each task should contain exactly one URL
        assert len(result) == 2
        for task in result:
            assert len(task.data) == 1

        # URLs should be distributed across tasks
        all_urls = [task.data[0] for task in result]
        assert set(all_urls) == set(urls)

    def test_process_task_id_generation(self) -> None:
        """Test that task IDs are correctly generated."""
        urls = ["http://example.com/file1.txt", "http://example.com/file2.txt"]
        generator = MockURLGenerator(urls=urls)
        stage = URLGenerationStage(url_generator=generator)

        input_task = _EmptyTask(
            task_id="parent_task",
            dataset_name="test_dataset",
            data=None,
            _metadata={},
        )

        result = stage.process(input_task)

        # Check task ID generation
        assert result[0].task_id == "parent_task_0"
        assert result[1].task_id == "parent_task_1"

    def test_process_metadata_per_task(self) -> None:
        """Test that each task gets correct source URL metadata."""
        urls = ["http://example.com/file1.txt", "http://example.com/file2.txt"]
        generator = MockURLGenerator(urls=urls)
        stage = URLGenerationStage(url_generator=generator)

        input_task = _EmptyTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=None,
            _metadata={},
        )

        result = stage.process(input_task)

        # Check metadata for each task
        assert result[0]._metadata == {"source_url": urls[0]}
        assert result[1]._metadata == {"source_url": urls[1]}

    @mock.patch.object(MockURLGenerator, "generate_urls", return_value=[])
    def test_process_with_no_urls_generated(self, mock_generate: mock.Mock) -> None:
        """Test processing when generator returns no URLs."""
        generator = MockURLGenerator()
        stage = URLGenerationStage(url_generator=generator)

        input_task = _EmptyTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=None,
            _metadata={},
        )

        result = stage.process(input_task)

        assert result == []
        mock_generate.assert_called_once()
