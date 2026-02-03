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

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from nemo_curator.stages.text.download.base.iterator import DocumentIterator


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
