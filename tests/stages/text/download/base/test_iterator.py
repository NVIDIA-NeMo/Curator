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

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.download.base.iterator import DocumentIterateStage, DocumentIterator
from nemo_curator.tasks import DocumentBatch, FileGroupTask


class MockDocumentIterator(DocumentIterator):
    """Mock implementation of DocumentIterator for testing."""

    def __init__(self, records_per_file: int = 3, fail_on_file: str | None = None):
        self.records_per_file = records_per_file
        self.fail_on_file = fail_on_file

    def iterate(self, file_path: str) -> Iterator[dict[str, Any]]:
        """Mock iteration implementation - will be patched in some tests."""
        filename = Path(file_path).name

        # Simulate failure for specific files
        if self.fail_on_file and filename == self.fail_on_file:
            msg = f"Mock error processing {filename}"
            raise ValueError(msg)

        # Generate mock records
        for i in range(self.records_per_file):
            yield {
                "id": f"{filename}_record_{i}",
                "content": f"Content from {filename} record {i}",
                "metadata": f"meta_{i}",
            }

    def output_columns(self) -> list[str]:
        """Define output columns for testing."""
        return ["id", "content", "metadata"]


class TestBaseDocumentIterator:
    """Base test class for DocumentIterator functionality."""

    def test_iterator_basic_functionality(self, tmp_path: Path) -> None:
        """Test basic iteration over a file."""
        iterator = MockDocumentIterator(records_per_file=2)

        # Create a test file
        test_file = tmp_path / "test_data.txt"
        test_file.write_text("test content")

        records = list(iterator.iterate(str(test_file)))

        assert len(records) == 2
        assert records[0]["id"] == "test_data.txt_record_0"
        assert records[0]["content"] == "Content from test_data.txt record 0"
        assert records[1]["id"] == "test_data.txt_record_1"
        assert records[1]["content"] == "Content from test_data.txt record 1"

    def test_iterator_output_columns(self) -> None:
        """Test that iterator defines correct output columns."""
        iterator = MockDocumentIterator()
        columns = iterator.output_columns()

        assert columns == ["id", "content", "metadata"]

    def test_iterator_with_error(self, tmp_path: Path) -> None:
        """Test iterator behavior when processing fails."""
        iterator = MockDocumentIterator(fail_on_file="error_file.txt")

        # Create a test file that will cause an error
        error_file = tmp_path / "error_file.txt"
        error_file.write_text("test content")

        with pytest.raises(ValueError, match="Mock error processing error_file.txt"):
            list(iterator.iterate(str(error_file)))

    def test_iterator_empty_results(self) -> None:
        """Test iterator with no records."""
        iterator = MockDocumentIterator(records_per_file=0)

        records = list(iterator.iterate("any_file.txt"))
        assert len(records) == 0


class TestDocumentIterateStage:
    """Test class for DocumentIterateStage functionality."""

    def test_stage_properties(self) -> None:
        """Test that stage properties are correctly defined."""
        iterator = MockDocumentIterator()
        stage = DocumentIterateStage(iterator=iterator)

        # Test stage name
        assert stage.name == "iterate_mockdocumentiterator"

        # Test inputs and outputs
        assert stage.inputs() == (["data"], [])
        assert stage.outputs() == (["data"], ["id", "content", "metadata", "file_name"])

        # Test resources (should use default)
        assert isinstance(stage.resources, Resources)

    def test_stage_properties_without_filename_column(self) -> None:
        """Test stage properties when filename column is disabled."""
        iterator = MockDocumentIterator()
        stage = DocumentIterateStage(iterator=iterator, add_filename_column=False)

        assert stage.outputs() == (["data"], ["id", "content", "metadata"])

    def test_stage_properties_custom_filename_column(self) -> None:
        """Test stage properties with custom filename column name."""
        iterator = MockDocumentIterator()
        stage = DocumentIterateStage(iterator=iterator, add_filename_column="source_file")

        assert stage.outputs() == (["data"], ["id", "content", "metadata", "source_file"])

    def test_process_successful_iteration(self, tmp_path: Path) -> None:
        """Test successful iteration of multiple files."""
        iterator = MockDocumentIterator(records_per_file=2)
        stage = DocumentIterateStage(iterator=iterator)

        # Create test files
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content1")
        file2.write_text("content2")

        # Create input task
        input_task = FileGroupTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=[str(file1), str(file2)],
            _metadata={"source": "test"},
        )

        result = stage.process(input_task)

        # Verify result structure
        assert isinstance(result, DocumentBatch)
        assert result.task_id == "test_task"
        assert result.dataset_name == "test_dataset"
        assert result._metadata == {"source": "test"}

        # Verify DataFrame content
        df = result.data
        assert len(df) == 4  # 2 records per file, 2 files

        # Check records from first file
        file1_records = df[df["file_name"] == "file1.txt"]
        assert len(file1_records) == 2
        assert "file1.txt_record_0" in file1_records["id"].tolist()
        assert "file1.txt_record_1" in file1_records["id"].tolist()

        # Check records from second file
        file2_records = df[df["file_name"] == "file2.txt"]
        assert len(file2_records) == 2
        assert "file2.txt_record_0" in file2_records["id"].tolist()
        assert "file2.txt_record_1" in file2_records["id"].tolist()

    def test_process_with_record_limit(self, tmp_path: Path) -> None:
        """Test iteration with record limit."""
        iterator = MockDocumentIterator(records_per_file=5)
        stage = DocumentIterateStage(iterator=iterator, record_limit=3)

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        input_task = FileGroupTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=[str(test_file)],
            _metadata={},
        )

        result = stage.process(input_task)
        df = result.data

        # Should only have 3 records due to limit
        assert len(df) == 3
        assert df["id"].tolist() == ["test.txt_record_0", "test.txt_record_1", "test.txt_record_2"]

    def test_process_without_filename_column(self, tmp_path: Path) -> None:
        """Test processing without adding filename column."""
        iterator = MockDocumentIterator(records_per_file=1)
        stage = DocumentIterateStage(iterator=iterator, add_filename_column=False)

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        input_task = FileGroupTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=[str(test_file)],
            _metadata={},
        )

        result = stage.process(input_task)
        df = result.data

        # Should not have filename column
        assert "file_name" not in df.columns
        assert list(df.columns) == ["id", "content", "metadata"]

    def test_process_with_custom_filename_column(self, tmp_path: Path) -> None:
        """Test processing with custom filename column name."""
        iterator = MockDocumentIterator(records_per_file=1)
        stage = DocumentIterateStage(iterator=iterator, add_filename_column="source_file")

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        input_task = FileGroupTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=[str(test_file)],
            _metadata={},
        )

        result = stage.process(input_task)
        df = result.data

        # Should have custom filename column
        assert "source_file" in df.columns
        assert df["source_file"].iloc[0] == "test.txt"

    def test_process_with_file_errors(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Test handling when some files fail to process."""
        caplog.set_level("ERROR")

        iterator = MockDocumentIterator(records_per_file=2, fail_on_file="error_file.txt")
        stage = DocumentIterateStage(iterator=iterator)

        # Create files - one will succeed, one will fail
        good_file = tmp_path / "good_file.txt"
        error_file = tmp_path / "error_file.txt"
        good_file.write_text("content")
        error_file.write_text("content")

        input_task = FileGroupTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=[str(good_file), str(error_file)],
            _metadata={},
        )

        result = stage.process(input_task)
        df = result.data

        # Should only have records from successful file
        assert len(df) == 2
        assert all(filename == "good_file.txt" for filename in df["file_name"])

        # Check that error was logged
        assert "Error iterating" in caplog.text
        assert "error_file.txt" in caplog.text

    def test_process_empty_file_group(self) -> None:
        """Test processing an empty file group task."""
        iterator = MockDocumentIterator()
        stage = DocumentIterateStage(iterator=iterator)

        input_task = FileGroupTask(
            task_id="empty_task",
            dataset_name="test_dataset",
            data=[],
            _metadata={"source": "test"},
        )

        result = stage.process(input_task)

        assert isinstance(result, DocumentBatch)
        assert result.task_id == "empty_task"
        assert result.dataset_name == "test_dataset"
        assert len(result.data) == 0
        assert result._metadata == {"source": "test"}

    @mock.patch.object(MockDocumentIterator, "iterate", return_value=None)
    def test_process_iterator_returns_none(self, mock_iterate: mock.Mock, tmp_path: Path) -> None:
        """Test handling when iterator returns None."""
        iterator = MockDocumentIterator()
        stage = DocumentIterateStage(iterator=iterator)

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        input_task = FileGroupTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=[str(test_file)],
            _metadata={},
        )

        result = stage.process(input_task)
        df = result.data

        # Should return empty DataFrame when iterator returns None
        assert len(df) == 0
        mock_iterate.assert_called_once()

    def test_process_all_files_fail(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Test when all files fail to process."""
        caplog.set_level("ERROR")

        iterator = MockDocumentIterator(fail_on_file="error_file.txt")
        stage = DocumentIterateStage(iterator=iterator)

        # Create file that will fail
        error_file = tmp_path / "error_file.txt"
        error_file.write_text("content")

        input_task = FileGroupTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=[str(error_file)],
            _metadata={},
        )

        result = stage.process(input_task)

        # Should return empty DataFrame when all files fail
        assert len(result.data) == 0
        assert result.task_id == "test_task"
        assert result.dataset_name == "test_dataset"

        # Check that error was logged
        assert "Error iterating" in caplog.text
