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

from pathlib import Path
from typing import Any
from unittest import mock

import pandas as pd
import pytest

from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.download.base.download import DocumentDownloadStage
from nemo_curator.stages.text.download.base.stage import DocumentDownloadExtractStage, DocumentIterateExtractStage
from nemo_curator.stages.text.download.base.url_generation import URLGenerationStage
from nemo_curator.tasks import DocumentBatch, FileGroupTask

from .test_download import MockDocumentDownloader
from .test_extract import MockDocumentExtractor
from .test_iterator import MockDocumentIterator
from .test_url_generation import MockURLGenerator


class TestDocumentIterateExtractStage:
    """Test class for DocumentIterateExtractStage functionality."""

    def test_stage_properties_without_extractor(self) -> None:
        """Test that stage properties are correctly defined."""
        iterator = MockDocumentIterator()
        stage = DocumentIterateExtractStage(iterator=iterator)

        # Test stage name
        assert stage.name == "iterate_mockdocumentiterator"

        # Test inputs and outputs
        assert stage.inputs() == (["data"], [])
        assert stage.outputs() == (["data"], ["id", "content", "metadata", "file_name"])

        # Test resources (should use default)
        assert isinstance(stage.resources, Resources)

    def test_stage_properties_with_extractor(self) -> None:
        """Test that stage properties are correctly defined."""
        iterator = MockDocumentIterator()
        extractor = MockDocumentExtractor()
        stage = DocumentIterateExtractStage(iterator=iterator, extractor=extractor)

        # Test stage name
        assert stage.name == "iterate_extract_mockdocumentiterator_mockdocumentextractor"

        # Test inputs and outputs
        assert stage.inputs() == (["data"], [])
        assert stage.outputs() == (["data"], ["id", "processed_text", "language", "char_count", "file_name"])

        # Test resources (should use default)
        assert isinstance(stage.resources, Resources)

    def test_stage_properties_without_filename_column(self) -> None:
        """Test stage properties when filename column is disabled."""
        iterator = MockDocumentIterator()
        extractor = MockDocumentExtractor()
        stage = DocumentIterateExtractStage(iterator=iterator, extractor=extractor, add_filename_column=False)

        assert stage.outputs() == (["data"], ["id", "processed_text", "language", "char_count"])

    def test_stage_properties_custom_filename_column(self) -> None:
        """Test stage properties with custom filename column name."""
        iterator = MockDocumentIterator()
        extractor = MockDocumentExtractor()
        stage = DocumentIterateExtractStage(iterator=iterator, extractor=extractor, add_filename_column="source_file")

        assert stage.outputs() == (["data"], ["id", "processed_text", "language", "char_count", "source_file"])

    def test_process_successful_iteration(self, tmp_path: Path) -> None:
        """Test successful iteration of multiple files."""
        iterator = MockDocumentIterator(records_per_file=2)
        stage = DocumentIterateExtractStage(iterator=iterator)

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

    def test_process_successful_extraction(self, tmp_path: Path) -> None:
        """Test successful extraction of multiple records."""
        iterator = MockDocumentIterator(records_per_file=1)
        extractor = MockDocumentExtractor()
        stage = DocumentIterateExtractStage(iterator=iterator, extractor=extractor)

        # Create test files
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file3 = tmp_path / "file3.txt"
        file1.write_text("hello world")
        file2.write_text("foo bar")
        file3.write_text("test content")

        # Create input task
        input_task = FileGroupTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=[str(file1), str(file2), str(file3)],
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
        assert len(df) == 3

        # Check transformation
        assert df.loc[0, "processed_text"] == "HELLO WORLD"
        assert df.loc[1, "processed_text"] == "FOO BAR"
        assert df.loc[2, "processed_text"] == "TEST CONTENT"

        # Check other columns
        assert all(df["language"] == "en")
        assert df.loc[0, "char_count"] == 11
        assert df.loc[1, "char_count"] == 7
        assert df.loc[2, "char_count"] == 12

        # Check filename preservation
        assert df.loc[0, "file_name"] == "file1.txt"
        assert df.loc[1, "file_name"] == "file2.txt"
        assert df.loc[2, "file_name"] == "file3.txt"

    def test_process_with_record_limit(self, tmp_path: Path) -> None:
        """Test iteration with record limit."""
        iterator = MockDocumentIterator(records_per_file=5)
        stage = DocumentIterateExtractStage(iterator=iterator, record_limit=3)

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

    # TODO: Fix this test
    def test_process_with_filtered_records(self) -> None:
        """Test extraction with some records filtered out."""
        iterator = MockDocumentIterator(records_per_file=1)
        extractor = MockDocumentExtractor()
        stage = DocumentIterateExtractStage(iterator=iterator, extractor=extractor)

        # Create input with records that will be filtered
        input_data = pd.DataFrame(
            [
                {"id": "record_1", "content": "hello world", "file_name": "file1.txt"},
                {"id": "record_2_skip", "content": "this will be skipped", "file_name": "file1.txt"},
                {"id": "record_3", "content": "test content", "file_name": "file2.txt"},
            ]
        )

        input_task = DocumentBatch(
            task_id="test_task",
            dataset_name="test_dataset",
            data=input_data,
            _metadata={},
        )

        result = stage.process(input_task)
        df = result.data

        # Should only have 2 records (filtered out the one with "_skip")
        assert len(df) == 2
        assert "record_1" in df["id"].tolist()
        assert "record_3" in df["id"].tolist()
        assert "record_2_skip" not in df["id"].tolist()

    def test_process_iterate_without_filename_column(self, tmp_path: Path) -> None:
        """Test processing without adding filename column."""
        iterator = MockDocumentIterator(records_per_file=1)
        stage = DocumentIterateExtractStage(iterator=iterator, add_filename_column=False)

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

    def test_process_extract_without_filename_column(self, tmp_path: Path) -> None:
        """Test processing without filename column."""
        iterator = MockDocumentIterator(records_per_file=1)
        extractor = MockDocumentExtractor()
        stage = DocumentIterateExtractStage(iterator=iterator, extractor=extractor, add_filename_column=False)

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
        expected_columns = ["id", "processed_text", "language", "char_count"]
        assert list(df.columns) == expected_columns

    def test_process_iterate_with_custom_filename_column(self, tmp_path: Path) -> None:
        """Test processing with custom filename column name."""
        iterator = MockDocumentIterator(records_per_file=1)
        stage = DocumentIterateExtractStage(iterator=iterator, add_filename_column="source_file")

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

    def test_process_extract_with_custom_filename_column(self, tmp_path: Path) -> None:
        """Test processing with custom filename column name."""
        iterator = MockDocumentIterator(records_per_file=1)
        extractor = MockDocumentExtractor()
        stage = DocumentIterateExtractStage(iterator=iterator, extractor=extractor, add_filename_column="source_file")

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

        # Should preserve custom filename column
        assert "source_file" in df.columns
        assert df["source_file"].iloc[0] == "test.txt"

    def test_process_with_file_errors(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Test handling when some files fail to process."""
        caplog.set_level("ERROR")

        iterator = MockDocumentIterator(records_per_file=2, fail_on_file="error_file.txt")
        stage = DocumentIterateExtractStage(iterator=iterator)

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
        stage = DocumentIterateExtractStage(iterator=iterator)

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

    # TODO: Fix this test
    def test_process_empty_batch(self) -> None:
        """Test processing an empty document batch."""
        iterator = MockDocumentIterator()
        extractor = MockDocumentExtractor()
        stage = DocumentIterateExtractStage(iterator=iterator, extractor=extractor)

        input_data = pd.DataFrame()
        input_task = DocumentBatch(
            task_id="empty_task",
            dataset_name="test_dataset",
            data=input_data,
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
        stage = DocumentIterateExtractStage(iterator=iterator)

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

    # TODO: Fix this test
    def test_process_all_records_filtered(self) -> None:
        """Test when all records are filtered out."""
        iterator = MockDocumentIterator()
        extractor = MockDocumentExtractor()
        stage = DocumentIterateExtractStage(iterator=iterator, extractor=extractor)

        # All records will be filtered (end with "_skip")
        input_data = pd.DataFrame(
            [
                {"id": "record_1_skip", "content": "hello", "file_name": "file1.txt"},
                {"id": "record_2_skip", "content": "world", "file_name": "file2.txt"},
            ]
        )

        input_task = DocumentBatch(
            task_id="test_task",
            dataset_name="test_dataset",
            data=input_data,
            _metadata={},
        )

        result = stage.process(input_task)

        # Should return empty DataFrame when all records are filtered
        assert len(result.data) == 0
        assert result.task_id == "test_task"
        assert result.dataset_name == "test_dataset"

    def test_process_all_files_fail(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Test when all files fail to process."""
        caplog.set_level("ERROR")

        iterator = MockDocumentIterator(fail_on_file="error_file.txt")
        stage = DocumentIterateExtractStage(iterator=iterator)

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

    # TODO: Fix this test
    @mock.patch.object(MockDocumentExtractor, "extract")
    def test_process_with_extraction_errors(self, mock_extract: mock.Mock) -> None:
        """Test handling when extraction fails for some records."""

        # Mock extract to raise exception for certain records
        def side_effect(record: dict[str, str]) -> dict[str, Any] | None:
            if record.get("id") == "error_record":
                msg = "Extraction failed"
                raise ValueError(msg)
            return {
                "id": record["id"],
                "processed_text": record["content"].upper(),
                "language": "en",
                "char_count": len(record["content"]),
            }

        mock_extract.side_effect = side_effect

        iterator = MockDocumentIterator()
        extractor = MockDocumentExtractor()
        stage = DocumentIterateExtractStage(iterator=iterator, extractor=extractor)

        input_data = pd.DataFrame(
            [
                {"id": "good_record", "content": "hello", "file_name": "file1.txt"},
                {"id": "error_record", "content": "world", "file_name": "file2.txt"},
            ]
        )

        input_task = DocumentBatch(
            task_id="test_task",
            dataset_name="test_dataset",
            data=input_data,
            _metadata={},
        )

        # Should raise the exception since it's not caught in the current implementation
        with pytest.raises(ValueError, match="Extraction failed"):
            stage.process(input_task)


class TestDocumentDownloadExtractStage:
    """Test class for DocumentDownloadExtractStage composite functionality."""

    def test_stage_initialization_with_extractor(self, tmp_path: Path) -> None:
        """Test that composite stage initializes correctly with extractor."""
        url_generator = MockURLGenerator()
        downloader = MockDocumentDownloader(str(tmp_path))
        iterator = MockDocumentIterator()
        extractor = MockDocumentExtractor()

        stage = DocumentDownloadExtractStage(
            url_generator=url_generator,
            downloader=downloader,
            iterator=iterator,
            extractor=extractor,
            url_limit=5,
            record_limit=10,
            add_filename_column=True,
        )

        # Check that all components are stored
        assert stage.url_generator is url_generator
        assert stage.downloader is downloader
        assert stage.iterator is iterator
        assert stage.extractor is extractor
        assert stage.url_limit == 5
        assert stage.record_limit == 10
        assert stage.add_filename_column is True

    def test_stage_initialization_without_extractor(self, tmp_path: Path) -> None:
        """Test that composite stage initializes correctly without extractor."""
        url_generator = MockURLGenerator()
        downloader = MockDocumentDownloader(str(tmp_path))
        iterator = MockDocumentIterator()

        stage = DocumentDownloadExtractStage(
            url_generator=url_generator,
            downloader=downloader,
            iterator=iterator,
            extractor=None,
            url_limit=None,
            record_limit=None,
            add_filename_column="source_file",
        )

        # Check that components are stored correctly
        assert stage.url_generator is url_generator
        assert stage.downloader is downloader
        assert stage.iterator is iterator
        assert stage.extractor is None
        assert stage.url_limit is None
        assert stage.record_limit is None
        assert stage.add_filename_column == "source_file"

    def test_stage_properties(self, tmp_path: Path) -> None:
        """Test that stage properties are correctly defined."""
        url_generator = MockURLGenerator()
        downloader = MockDocumentDownloader(str(tmp_path))
        iterator = MockDocumentIterator()
        extractor = MockDocumentExtractor()

        stage = DocumentDownloadExtractStage(
            url_generator=url_generator,
            downloader=downloader,
            iterator=iterator,
            extractor=extractor,
        )

        # Test stage name
        expected_name = "document_download_extract_mockurlgenerator_mockdocumentdownloader_composite"
        assert stage.name == expected_name

        # Test inputs and outputs (from first and last stages)
        assert stage.inputs() == ([], [])  # From URL generation stage
        assert stage.outputs() == (
            ["data"],
            ["id", "processed_text", "language", "char_count", "file_name"],
        )  # From extract stage

    def test_stage_properties_without_extractor(self, tmp_path: Path) -> None:
        """Test stage properties when no extractor is provided."""
        url_generator = MockURLGenerator()
        downloader = MockDocumentDownloader(str(tmp_path))
        iterator = MockDocumentIterator()

        stage = DocumentDownloadExtractStage(
            url_generator=url_generator,
            downloader=downloader,
            iterator=iterator,
            extractor=None,
        )

        # Should use iterator outputs when no extractor
        assert stage.outputs() == (["data"], ["id", "content", "metadata", "file_name"])

    def test_decompose_with_extractor(self, tmp_path: Path) -> None:
        """Test decomposition into constituent stages with extractor."""
        url_generator = MockURLGenerator()
        downloader = MockDocumentDownloader(str(tmp_path))
        iterator = MockDocumentIterator()
        extractor = MockDocumentExtractor()

        stage = DocumentDownloadExtractStage(
            url_generator=url_generator,
            downloader=downloader,
            iterator=iterator,
            extractor=extractor,
            url_limit=5,
            record_limit=10,
        )

        stages = stage.decompose()

        # Should have 3 stages: URL generation, download, iterate-extract
        assert len(stages) == 3

        # Check stage types and order
        assert isinstance(stages[0], URLGenerationStage)
        assert isinstance(stages[1], DocumentDownloadStage)
        assert isinstance(stages[2], DocumentIterateExtractStage)

        # Check that parameters are propagated correctly
        url_stage = stages[0]
        assert url_stage.url_generator is url_generator
        assert url_stage.limit == 5

        download_stage = stages[1]
        assert download_stage.downloader is downloader

        iterate_extract_stage = stages[2]
        assert iterate_extract_stage.iterator is iterator
        assert iterate_extract_stage.extractor is extractor
        assert iterate_extract_stage.record_limit == 10

    def test_decompose_without_extractor(self, tmp_path: Path) -> None:
        """Test decomposition into constituent stages without extractor."""
        url_generator = MockURLGenerator()
        downloader = MockDocumentDownloader(str(tmp_path))
        iterator = MockDocumentIterator()

        stage = DocumentDownloadExtractStage(
            url_generator=url_generator,
            downloader=downloader,
            iterator=iterator,
            extractor=None,
        )

        stages = stage.decompose()

        # Should have 3 stages: URL generation, download, iterate-extract
        assert len(stages) == 3

        # Check stage types and order
        assert isinstance(stages[0], URLGenerationStage)
        assert isinstance(stages[1], DocumentDownloadStage)
        assert isinstance(stages[2], DocumentIterateExtractStage)

        # Check that parameters are propagated correctly for iterate-extract stage
        iterate_extract_stage = stages[2]
        assert iterate_extract_stage.iterator is iterator
        assert iterate_extract_stage.extractor is None

    def test_get_description(self, tmp_path: Path) -> None:
        """Test that stage description is correctly generated."""
        url_generator = MockURLGenerator()
        downloader = MockDocumentDownloader(str(tmp_path))
        iterator = MockDocumentIterator()
        extractor = MockDocumentExtractor()

        stage = DocumentDownloadExtractStage(
            url_generator=url_generator,
            downloader=downloader,
            iterator=iterator,
            extractor=extractor,
        )

        description = stage.get_description()
        expected = "URL-Download-Iterate-Extract pipeline using MockURLGenerator and MockDocumentDownloader"
        assert description == expected

    def test_stage_parameter_propagation(self, tmp_path: Path) -> None:
        """Test that parameters are correctly propagated to constituent stages."""
        url_generator = MockURLGenerator()
        downloader = MockDocumentDownloader(str(tmp_path))
        iterator = MockDocumentIterator()
        extractor = MockDocumentExtractor()

        stage = DocumentDownloadExtractStage(
            url_generator=url_generator,
            downloader=downloader,
            iterator=iterator,
            extractor=extractor,
            url_limit=3,
            record_limit=5,
            add_filename_column="custom_file",
        )

        stages = stage.decompose()

        # Check URL generation stage
        url_stage = stages[0]
        assert isinstance(url_stage, URLGenerationStage)
        assert url_stage.limit == 3

        # Check iterate-extract stage
        iterate_stage = stages[2]
        assert isinstance(iterate_stage, DocumentIterateExtractStage)
        assert iterate_stage.record_limit == 5
        assert iterate_stage.filename_col == "custom_file"

    def test_stage_resources(self, tmp_path: Path) -> None:
        """Test that stage has appropriate resource requirements."""
        url_generator = MockURLGenerator()
        downloader = MockDocumentDownloader(str(tmp_path))
        iterator = MockDocumentIterator()

        stage = DocumentDownloadExtractStage(
            url_generator=url_generator,
            downloader=downloader,
            iterator=iterator,
            extractor=None,
        )

        # Should have default resources from base class
        assert isinstance(stage.resources, Resources)

    def test_stage_different_filename_column_types(self, tmp_path: Path) -> None:
        """Test stage behavior with different filename column configurations."""
        url_generator = MockURLGenerator()
        downloader = MockDocumentDownloader(str(tmp_path))
        iterator = MockDocumentIterator()
        extractor = MockDocumentExtractor()

        # Test with boolean True
        stage_bool = DocumentDownloadExtractStage(
            url_generator=url_generator,
            downloader=downloader,
            iterator=iterator,
            extractor=extractor,
            add_filename_column=True,
        )

        stages = stage_bool.decompose()
        iterate_extract_stage = stages[2]
        assert isinstance(iterate_extract_stage, DocumentIterateExtractStage)
        assert iterate_extract_stage.filename_col == "file_name"  # Default name

        # Test with boolean False
        stage_false = DocumentDownloadExtractStage(
            url_generator=url_generator,
            downloader=downloader,
            iterator=iterator,
            extractor=extractor,
            add_filename_column=False,
        )

        stages = stage_false.decompose()
        iterate_extract_stage = stages[2]
        assert isinstance(iterate_extract_stage, DocumentIterateExtractStage)
        assert iterate_extract_stage.add_filename_column is False

        # Test with custom string
        stage_custom = DocumentDownloadExtractStage(
            url_generator=url_generator,
            downloader=downloader,
            iterator=iterator,
            extractor=extractor,
            add_filename_column="source_path",
        )

        stages = stage_custom.decompose()
        iterate_extract_stage = stages[2]
        assert isinstance(iterate_extract_stage, DocumentIterateExtractStage)
        assert iterate_extract_stage.filename_col == "source_path"

    def test_stage_edge_cases(self, tmp_path: Path) -> None:
        """Test edge cases for the composite stage."""
        url_generator = MockURLGenerator(urls=[])  # Empty URLs
        downloader = MockDocumentDownloader(str(tmp_path))
        iterator = MockDocumentIterator(records_per_file=0)  # No records

        stage = DocumentDownloadExtractStage(
            url_generator=url_generator,
            downloader=downloader,
            iterator=iterator,
            extractor=None,
            url_limit=0,  # Zero limit
            record_limit=0,  # Zero limit
        )

        stages = stage.decompose()

        # Check that limits are correctly set
        url_stage = stages[0]
        iterate_extract_stage = stages[2]
        assert isinstance(url_stage, URLGenerationStage)
        assert isinstance(iterate_extract_stage, DocumentIterateExtractStage)
        assert url_stage.limit == 0
        assert iterate_extract_stage.record_limit == 0

        # Should still have all required stages
        assert len(stages) == 3

    @mock.patch("nemo_curator.stages.text.download.base.stage.URLGenerationStage")
    @mock.patch("nemo_curator.stages.text.download.base.stage.DocumentDownloadStage")
    @mock.patch("nemo_curator.stages.text.download.base.stage.DocumentIterateExtractStage")
    def test_stage_initialization_mocking(
        self,
        mock_iterate_extract_stage: mock.Mock,
        mock_download_stage: mock.Mock,
        mock_url_stage: mock.Mock,
        tmp_path: Path,
    ) -> None:
        """Test that stage initialization creates the correct stage instances."""
        url_generator = MockURLGenerator()
        downloader = MockDocumentDownloader(str(tmp_path))
        iterator = MockDocumentIterator()
        extractor = MockDocumentExtractor()

        _ = DocumentDownloadExtractStage(
            url_generator=url_generator,
            downloader=downloader,
            iterator=iterator,
            extractor=extractor,
            url_limit=5,
            record_limit=10,
            add_filename_column="test_file",
        )

        # Verify that each stage type was instantiated with correct parameters
        mock_url_stage.assert_called_once_with(
            url_generator=url_generator,
            limit=5,
        )

        mock_download_stage.assert_called_once_with(
            downloader=downloader,
        )

        mock_iterate_extract_stage.assert_called_once_with(
            iterator=iterator,
            extractor=extractor,
            record_limit=10,
            add_filename_column="test_file",
        )
