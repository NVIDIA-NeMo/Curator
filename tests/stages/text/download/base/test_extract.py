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

from typing import Any

import pytest

from nemo_curator.stages.text.download.base.extract import DocumentExtractor


class MockDocumentExtractor(DocumentExtractor):
    """Mock implementation of DocumentExtractor for testing."""

    def __init__(self, fail_on_record: str | None = None, transform_text: bool = True):
        self.fail_on_record = fail_on_record
        self.transform_text = transform_text

    def extract(self, record: dict[str, str]) -> dict[str, Any] | None:
        """Mock extraction implementation - will be patched in some tests."""
        record_id = record.get("id", "")
        record_content = record.get("content", "")

        # Simulate failure for specific records
        if self.fail_on_record and record_content == self.fail_on_record:
            msg = f"Mock error processing record {record_id}"
            raise ValueError(msg)

        # Simulate filtering out certain records
        if "skip" in record_content.lower():
            return None

        # Transform the record
        return {
            "id": record_id,
            "processed_text": record_content.upper() if self.transform_text else record_content,
            "language": "en",
            "char_count": len(record_content),
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
            "content": "this should be skipped_skip",
        }

        result = extractor.extract(record)
        assert result is None

    def test_extractor_with_error(self) -> None:
        """Test extractor behavior when processing fails."""
        extractor = MockDocumentExtractor(fail_on_record="test_content_error")

        record = {
            "id": "error_record",
            "content": "test_content_error",
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
