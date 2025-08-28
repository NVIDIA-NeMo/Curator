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

from typing import Any
from unittest import mock

import pandas as pd
import pytest

from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.download.base.extract import DocumentExtractor, DocumentExtractStage
from nemo_curator.tasks import DocumentBatch


class MockDocumentExtractor(DocumentExtractor):
    """Mock implementation of DocumentExtractor for testing."""

    def __init__(self, fail_on_record: str | None = None, transform_text: bool = True):
        self.fail_on_record = fail_on_record
        self.transform_text = transform_text

    def extract(self, record: dict[str, str]) -> dict[str, Any] | None:
        """Mock extraction implementation - will be patched in some tests."""
        record_id = record.get("id", "")

        # Simulate failure for specific records
        if self.fail_on_record and record_id == self.fail_on_record:
            msg = f"Mock error processing record {record_id}"
            raise ValueError(msg)

        # Simulate filtering out certain records
        if record_id.endswith("_skip"):
            return None

        # Transform the record
        return {
            "id": record_id,
            "processed_text": record.get("content", "").upper() if self.transform_text else record.get("content", ""),
            "language": "en",
            "char_count": len(record.get("content", "")),
        }

    def input_columns(self) -> list[str]:
        """Define input columns expected."""
        return ["id", "content"]

    def output_columns(self) -> list[str]:
        """Define output columns produced."""
        return ["id", "processed_text", "language", "char_count"]


class TestBaseDocumentExtractor:
    """Base test class for DocumentExtractor functionality."""

    def test_extractor_basic_functionality(self) -> None:
        """Test basic extraction functionality."""
        extractor = MockDocumentExtractor()

        record = {"id": "test_record", "content": "hello world", "metadata": "extra_info"}

        result = extractor.extract(record)

        assert result is not None
        assert result["id"] == "test_record"
        assert result["processed_text"] == "HELLO WORLD"
        assert result["language"] == "en"
        assert result["char_count"] == 11

    def test_extractor_filtering(self) -> None:
        """Test that extractor can filter out records."""
        extractor = MockDocumentExtractor()

        record = {
            "id": "test_record_skip",
            "content": "this should be skipped",
        }

        result = extractor.extract(record)
        assert result is None

    def test_extractor_with_error(self) -> None:
        """Test extractor behavior when processing fails."""
        extractor = MockDocumentExtractor(fail_on_record="error_record")

        record = {
            "id": "error_record",
            "content": "test content",
        }

        with pytest.raises(ValueError, match="Mock error processing record error_record"):
            extractor.extract(record)

    def test_extractor_column_definitions(self) -> None:
        """Test that extractor defines correct input/output columns."""
        extractor = MockDocumentExtractor()

        assert extractor.input_columns() == ["id", "content"]
        assert extractor.output_columns() == ["id", "processed_text", "language", "char_count"]

    def test_extractor_without_transformation(self) -> None:
        """Test extractor with transformation disabled."""
        extractor = MockDocumentExtractor(transform_text=False)

        record = {
            "id": "test_record",
            "content": "hello world",
        }

        result = extractor.extract(record)
        assert result is not None
        assert result["processed_text"] == "hello world"  # Not transformed to uppercase


class TestDocumentExtractStage:
    """Test class for DocumentExtractStage functionality."""

    def test_stage_properties(self) -> None:
        """Test that stage properties are correctly defined."""
        extractor = MockDocumentExtractor()
        stage = DocumentExtractStage(extractor=extractor)

        # Test stage name
        assert stage.name == "extract_mockdocumentextractor"

        # Test inputs and outputs
        assert stage.inputs() == (["data"], ["id", "content", "file_name"])
        assert stage.outputs() == (["data"], ["id", "processed_text", "language", "char_count", "file_name"])

        # Test resources (should use default)
        assert isinstance(stage.resources, Resources)

    def test_stage_properties_without_filename_column(self) -> None:
        """Test stage properties when filename column is disabled."""
        extractor = MockDocumentExtractor()
        stage = DocumentExtractStage(extractor=extractor, add_filename_column=False)

        assert stage.inputs() == (["data"], ["id", "content"])
        assert stage.outputs() == (["data"], ["id", "processed_text", "language", "char_count"])

    def test_stage_properties_custom_filename_column(self) -> None:
        """Test stage properties with custom filename column name."""
        extractor = MockDocumentExtractor()
        stage = DocumentExtractStage(extractor=extractor, add_filename_column="source_file")

        assert stage.inputs() == (["data"], ["id", "content", "source_file"])
        assert stage.outputs() == (["data"], ["id", "processed_text", "language", "char_count", "source_file"])

    def test_process_successful_extraction(self) -> None:
        """Test successful extraction of multiple records."""
        extractor = MockDocumentExtractor()
        stage = DocumentExtractStage(extractor=extractor)

        # Create input DataFrame
        input_data = pd.DataFrame(
            [
                {"id": "record_1", "content": "hello world", "file_name": "file1.txt"},
                {"id": "record_2", "content": "foo bar", "file_name": "file1.txt"},
                {"id": "record_3", "content": "test content", "file_name": "file2.txt"},
            ]
        )

        input_task = DocumentBatch(
            task_id="test_task",
            dataset_name="test_dataset",
            data=input_data,
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

        # Check filename preservation
        assert df.loc[0, "file_name"] == "file1.txt"
        assert df.loc[2, "file_name"] == "file2.txt"

    def test_process_with_filtered_records(self) -> None:
        """Test extraction with some records filtered out."""
        extractor = MockDocumentExtractor()
        stage = DocumentExtractStage(extractor=extractor)

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

    def test_process_without_filename_column(self) -> None:
        """Test processing without filename column."""
        extractor = MockDocumentExtractor()
        stage = DocumentExtractStage(extractor=extractor, add_filename_column=False)

        input_data = pd.DataFrame(
            [
                {"id": "record_1", "content": "hello world"},
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

        # Should not have filename column
        assert "file_name" not in df.columns
        expected_columns = ["id", "processed_text", "language", "char_count"]
        assert list(df.columns) == expected_columns

    def test_process_with_custom_filename_column(self) -> None:
        """Test processing with custom filename column name."""
        extractor = MockDocumentExtractor()
        stage = DocumentExtractStage(extractor=extractor, add_filename_column="source_file")

        input_data = pd.DataFrame(
            [
                {"id": "record_1", "content": "hello world", "source_file": "test.txt"},
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

        # Should preserve custom filename column
        assert "source_file" in df.columns
        assert df["source_file"].iloc[0] == "test.txt"

    def test_process_filename_column_overwrite_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test warning when extractor tries to overwrite filename column."""

        # Create a custom extractor that includes file_name in output
        class ExtractorWithFilename(MockDocumentExtractor):
            def extract(self, record: dict[str, str]) -> dict[str, Any] | None:
                result = super().extract(record)
                if result:
                    result["file_name"] = "extracted_filename"
                return result

        extractor = ExtractorWithFilename()
        stage = DocumentExtractStage(extractor=extractor, add_filename_column=True)

        input_data = pd.DataFrame(
            [
                {"id": "record_1", "content": "hello", "file_name": "original.txt"},
            ]
        )

        input_task = DocumentBatch(
            task_id="test_task",
            dataset_name="test_dataset",
            data=input_data,
            _metadata={},
        )

        caplog.set_level("WARNING")
        result = stage.process(input_task)

        # Should warn about overwriting filename column
        assert "we'll overwrite (file_name) from the input data" in caplog.text

        # Original filename should be preserved
        assert result.data["file_name"].iloc[0] == "original.txt"

    def test_process_empty_batch(self) -> None:
        """Test processing an empty document batch."""
        extractor = MockDocumentExtractor()
        stage = DocumentExtractStage(extractor=extractor)

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

    def test_process_all_records_filtered(self) -> None:
        """Test when all records are filtered out."""
        extractor = MockDocumentExtractor()
        stage = DocumentExtractStage(extractor=extractor)

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

        extractor = MockDocumentExtractor()
        stage = DocumentExtractStage(extractor=extractor)

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
