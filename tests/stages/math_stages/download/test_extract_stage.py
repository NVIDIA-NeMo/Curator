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

from nemo_curator.stages.math.download.extract import MathContentExtractor
from nemo_curator.stages.text.download.base.extract import DocumentExtractor, DocumentExtractStage
from nemo_curator.tasks import DocumentBatch


class MockMathExtractor(DocumentExtractor):
    """Mock implementation of MathContentExtractor for testing."""

    def __init__(self, fail_on_url: str | None = None):
        self.fail_on_url = fail_on_url

    def extract(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """Mock extraction that returns predictable results."""
        url = record.get("url", "")

        # Simulate failure for specific URLs
        if self.fail_on_url and url == self.fail_on_url:
            return None

        # Simple routing based on URL
        if "html" in url:
            return {"text": "Extracted HTML text", "url": url, "type": "html", "magic_mime_type": "text/html"}
        elif "ipynb" in url:
            return {
                "text": "Extracted notebook text",
                "url": url,
                "type": "notebook",
                "magic_mime_type": "application/json",
            }
        else:
            return {"text": "Plain text content", "url": url, "type": "text", "magic_mime_type": "text/plain"}

    def input_columns(self) -> list[str]:
        return ["binary_content", "url", "mime_type"]

    def output_columns(self) -> list[str]:
        return ["text", "url", "type", "magic_mime_type"]


class TestMathContentExtractorStage:
    """Tests for MathContentExtractor with DocumentExtractStage."""

    def test_stage_with_math_extractor_properties(self) -> None:
        """Test DocumentExtractStage properties with MathContentExtractor."""
        extractor = MathContentExtractor()
        stage = DocumentExtractStage(extractor=extractor, add_filename_column=False)

        # Test stage name
        assert stage.name == "extract_mathcontentextractor"

        # Test inputs and outputs
        assert stage.inputs() == (["data"], ["binary_content", "url", "mime_type"])
        assert stage.outputs() == (["data"], ["text", "url", "type", "magic_mime_type"])

    def test_stage_with_custom_column_names(self) -> None:
        """Test DocumentExtractStage with custom column names."""
        extractor = MathContentExtractor(
            binary_column="custom_binary", url_column="custom_url", mime_type_column="custom_mime"
        )
        stage = DocumentExtractStage(extractor=extractor, add_filename_column=False)

        assert stage.inputs() == (["data"], ["custom_binary", "custom_url", "custom_mime"])
        assert stage.outputs() == (["data"], ["text", "custom_url", "type", "magic_mime_type"])

    @pytest.mark.parametrize(
        ("url", "expected_type", "expected_text"),
        [
            ("http://example.com/page.html", "html", "Extracted HTML text"),
            ("http://example.com/notebook.ipynb", "notebook", "Extracted notebook text"),
            ("http://example.com/file.txt", "text", "Plain text content"),
        ],
    )
    def test_process_content_types(self, url: str, expected_type: str, expected_text: str) -> None:
        """Test processing different content types using mock extractor."""
        extractor = MockMathExtractor()
        stage = DocumentExtractStage(extractor=extractor, add_filename_column=False)

        # Create input DataFrame with single record
        input_data = pd.DataFrame([{"binary_content": b"test content", "url": url, "mime_type": "test/type"}])

        input_task = DocumentBatch(
            task_id="test_content_type", dataset_name="test_dataset", data=input_data, _metadata={}
        )

        result = stage.process(input_task)

        # Verify result structure
        assert isinstance(result, DocumentBatch)
        assert len(result.data) == 1

        row = result.data.iloc[0]
        assert row["type"] == expected_type
        assert row["text"] == expected_text
        assert row["url"] == url

    def test_process_with_extraction_failures(self) -> None:
        """Test processing when some records fail extraction."""
        # Use mock extractor that fails on specific URL
        extractor = MockMathExtractor(fail_on_url="http://example.com/bad.txt")
        stage = DocumentExtractStage(extractor=extractor, add_filename_column=False)

        # Create input DataFrame with some failing records
        input_data = pd.DataFrame(
            [
                {"binary_content": b"good content", "url": "http://example.com/good.txt", "mime_type": "text/plain"},
                {"binary_content": b"fail content", "url": "http://example.com/bad.txt", "mime_type": "text/plain"},
                {
                    "binary_content": b"another good content",
                    "url": "http://example.com/good2.txt",
                    "mime_type": "text/plain",
                },
            ]
        )

        input_task = DocumentBatch(
            task_id="test_extraction_failures", dataset_name="test_dataset", data=input_data, _metadata={}
        )

        result = stage.process(input_task)

        # Should only have 2 records (failed extraction records are filtered out)
        df = result.data
        assert len(df) == 2

        # Check that only successful records remain
        urls = df["url"].tolist()
        assert "http://example.com/good.txt" in urls
        assert "http://example.com/good2.txt" in urls
        assert "http://example.com/bad.txt" not in urls

    @mock.patch("magic.Magic")
    def test_process_with_magic_failures(self, mock_magic_class: mock.Mock) -> None:
        """Test processing when magic MIME detection fails for some records."""
        # Setup magic to fail for some records based on content
        mock_magic_instance = mock.Mock()

        class MagicDetectionError(Exception):
            """Custom exception for magic detection failures in tests."""

        def magic_side_effect(binary_content: bytes) -> str:
            if b"magic_fail" in binary_content:
                error_msg = "Magic detection failed"
                raise MagicDetectionError(error_msg)
            else:
                return "text/plain"

        mock_magic_instance.from_buffer.side_effect = magic_side_effect
        mock_magic_class.return_value = mock_magic_instance

        extractor = MathContentExtractor()
        stage = DocumentExtractStage(extractor=extractor, add_filename_column=False)

        # Use real binary content that can be decoded naturally
        input_data = pd.DataFrame(
            [
                {
                    "binary_content": b"good content",  # Will decode successfully
                    "url": "http://example.com/good.txt",
                    "mime_type": "text/plain",
                },
                {
                    "binary_content": b"magic_fail content",  # Will cause magic to fail
                    "url": "http://example.com/magic_fail.txt",
                    "mime_type": "text/plain",
                },
            ]
        )

        input_task = DocumentBatch(
            task_id="test_magic_failures", dataset_name="test_dataset", data=input_data, _metadata={}
        )

        result = stage.process(input_task)

        # Both records should be processed (magic failure doesn't prevent processing)
        df = result.data
        assert len(df) == 2

        # Check that magic_mime_type is None for failed record
        magic_fail_row = df[df["url"] == "http://example.com/magic_fail.txt"].iloc[0]
        assert magic_fail_row["magic_mime_type"] is None
        assert magic_fail_row["text"] == "magic_fail content"  # Content should still be decoded

        # Check that magic_mime_type is set for successful record
        good_row = df[df["url"] == "http://example.com/good.txt"].iloc[0]
        assert good_row["magic_mime_type"] == "text/plain"
        assert good_row["text"] == "good content"

    def test_process_empty_batch(self) -> None:
        """Test processing an empty document batch."""
        extractor = MockMathExtractor()
        stage = DocumentExtractStage(extractor=extractor, add_filename_column=False)

        input_data = pd.DataFrame()
        input_task = DocumentBatch(
            task_id="empty_task", dataset_name="test_dataset", data=input_data, _metadata={"source": "test"}
        )

        result = stage.process(input_task)

        assert isinstance(result, DocumentBatch)
        assert result.task_id == "empty_task"
        assert result.dataset_name == "test_dataset"
        assert len(result.data) == 0
        assert result._metadata == {"source": "test"}

    def test_stage_with_mock_extractor_smoke(self) -> None:
        """Smoke test for stage processing using mock extractor."""
        extractor = MockMathExtractor()
        stage = DocumentExtractStage(extractor=extractor, add_filename_column=False)

        # Test basic stage properties
        assert stage.name == "extract_mockmathextractor"
        assert stage.inputs() == (["data"], ["binary_content", "url", "mime_type"])
        assert stage.outputs() == (["data"], ["text", "url", "type", "magic_mime_type"])

        # Test processing
        input_data = pd.DataFrame(
            [{"binary_content": b"test", "url": "http://example.com/test.html", "mime_type": "text/html"}]
        )

        input_task = DocumentBatch(task_id="smoke_test", dataset_name="test", data=input_data, _metadata={})

        result = stage.process(input_task)

        assert isinstance(result, DocumentBatch)
        assert len(result.data) == 1
        assert result.data.iloc[0]["type"] == "html"

    def test_process_with_filename_column(self) -> None:
        """Test processing with filename column enabled."""
        extractor = MathContentExtractor()
        stage = DocumentExtractStage(extractor=extractor, add_filename_column=True)

        # Use real binary content that can be decoded naturally
        test_content = "Test content"
        binary_content = test_content.encode("utf-8")

        input_data = pd.DataFrame(
            [
                {
                    "binary_content": binary_content,
                    "url": "http://example.com/test.txt",
                    "mime_type": "text/plain",
                    "file_name": "test_file.txt",
                }
            ]
        )

        input_task = DocumentBatch(
            task_id="test_with_filename", dataset_name="test_dataset", data=input_data, _metadata={}
        )

        # Mock only external system boundaries
        with mock.patch("magic.Magic") as mock_magic_class:
            mock_magic_instance = mock.Mock()
            mock_magic_instance.from_buffer.return_value = "text/plain"
            mock_magic_class.return_value = mock_magic_instance

            result = stage.process(input_task)

        # Should preserve filename column
        df = result.data
        assert len(df) == 1
        assert "file_name" in df.columns
        assert df["file_name"].iloc[0] == "test_file.txt"
        assert df["text"].iloc[0] == test_content

    def test_process_real_notebook_content(self, complex_notebook_json: str) -> None:
        """Test processing with realistic notebook content - integration style."""
        extractor = MathContentExtractor()
        stage = DocumentExtractStage(extractor=extractor, add_filename_column=False)

        # Use real binary content that can be decoded naturally
        binary_content = complex_notebook_json.encode("utf-8")
        input_data = pd.DataFrame(
            [
                {
                    "binary_content": binary_content,
                    "url": "http://example.com/math_analysis.ipynb",
                    "mime_type": "application/json",
                }
            ]
        )

        input_task = DocumentBatch(
            task_id="test_real_notebook", dataset_name="test_dataset", data=input_data, _metadata={}
        )

        # Only mock external system boundaries, not internal methods
        with mock.patch("magic.Magic") as mock_magic_class:
            mock_magic_instance = mock.Mock()
            mock_magic_instance.from_buffer.return_value = "application/json"
            mock_magic_class.return_value = mock_magic_instance

            result = stage.process(input_task)

        # Verify notebook processing
        df = result.data
        assert len(df) == 1

        row = df.iloc[0]
        assert row["type"] == "notebook"
        assert row["magic_mime_type"] == "application/json"

        # Check that notebook content was extracted
        extracted_text = row["text"]
        assert "Mathematical Analysis" in extracted_text
        assert "import numpy as np" in extracted_text
        assert "Mean: 3.0" in extracted_text
        assert "1.4142135623730951" in extracted_text

    def test_process_real_html_with_math(self, math_html: str) -> None:
        """Test processing with realistic HTML containing mathematical content."""
        extractor = MathContentExtractor()
        stage = DocumentExtractStage(extractor=extractor, add_filename_column=False)

        # Use real binary content that can be decoded naturally
        binary_content = math_html.encode("utf-8")
        input_data = pd.DataFrame(
            [
                {
                    "binary_content": binary_content,
                    "url": "http://example.com/quadratic_formula.html",
                    "mime_type": "text/html",
                }
            ]
        )

        input_task = DocumentBatch(
            task_id="test_html_math", dataset_name="test_dataset", data=input_data, _metadata={}
        )

        # Mock external systems only - lynx and magic
        mock_lynx = mock.Mock()
        mock_lynx.extract_text.return_value = """Mathematical Formulas

Quadratic Formula

The quadratic formula is:

x = (-b ± √(b² - 4ac)) / 2a

Where a, b, and c are coefficients.

Example

For the equation x² + 5x + 6 = 0:

• a = 1
• b = 5
• c = 6"""

        with (
            mock.patch("nemo_curator.stages.math.download.extract.LynxExtractor", return_value=mock_lynx),
            mock.patch("magic.Magic") as mock_magic_class,
        ):
            mock_magic_instance = mock.Mock()
            mock_magic_instance.from_buffer.return_value = "text/html"
            mock_magic_class.return_value = mock_magic_instance

            result = stage.process(input_task)

        # Verify HTML processing
        df = result.data
        assert len(df) == 1

        row = df.iloc[0]
        assert row["type"] == "html"
        assert row["magic_mime_type"] == "text/html"

        # Check that mathematical content was extracted by lynx
        extracted_text = row["text"]
        assert "Quadratic Formula" in extracted_text
        assert "coefficients" in extracted_text
        assert "a = 1" in extracted_text

        # Verify lynx was called with the decoded HTML content
        mock_lynx.extract_text.assert_called_once_with(math_html)
