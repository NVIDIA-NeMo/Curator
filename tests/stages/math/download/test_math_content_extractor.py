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

from nemo_curator.stages.math.download.extract import MathContentExtractor


class TestMathContentExtractor:
    """Test the MathContentExtractor class."""

    def test_extractor_init_default_values(self) -> None:
        """Test extractor initialization with default values."""
        extractor = MathContentExtractor()

        assert extractor.binary_column == "binary_content"
        assert extractor.url_column == "url"
        assert extractor.mime_type_column == "mime_type"
        assert extractor.lynx_timeout_sec == 20
        assert extractor._lynx is None
        assert extractor._magic is None

    def test_extractor_init_custom_values(self) -> None:
        """Test extractor initialization with custom values."""
        extractor = MathContentExtractor(
            binary_column="custom_binary", url_column="custom_url", mime_type_column="custom_mime", lynx_timeout_sec=30
        )

        assert extractor.binary_column == "custom_binary"
        assert extractor.url_column == "custom_url"
        assert extractor.mime_type_column == "custom_mime"
        assert extractor.lynx_timeout_sec == 30

    def test_input_columns(self) -> None:
        """Test input columns definition."""
        extractor = MathContentExtractor()
        expected = ["binary_content", "url", "mime_type"]
        assert extractor.input_columns() == expected

    def test_input_columns_custom(self) -> None:
        """Test input columns with custom column names."""
        extractor = MathContentExtractor(
            binary_column="custom_binary", url_column="custom_url", mime_type_column="custom_mime"
        )
        expected = ["custom_binary", "custom_url", "custom_mime"]
        assert extractor.input_columns() == expected

    def test_output_columns(self) -> None:
        """Test output columns definition."""
        extractor = MathContentExtractor()
        expected = ["text", "url", "type", "magic_mime_type"]
        assert extractor.output_columns() == expected

    def test_output_columns_custom_url(self) -> None:
        """Test output columns with custom URL column name."""
        extractor = MathContentExtractor(url_column="custom_url")
        expected = ["text", "custom_url", "type", "magic_mime_type"]
        assert extractor.output_columns() == expected

    def test_extract_edge_cases_none_binary_content(self, test_records: dict) -> None:
        """Test extraction with None binary content - integration style."""
        extractor = MathContentExtractor()
        record = test_records["none"]  # Has None binary_content

        result = extractor.extract(record)
        assert result is None

    def test_extract_edge_cases_empty_binary_content(self, test_records: dict) -> None:
        """Test extraction with empty binary content - integration style."""
        extractor = MathContentExtractor()
        # Create record with empty binary content that will decode to empty string
        record = test_records["text"].copy()
        record["binary_content"] = b""  # Empty bytes will decode to empty string

        with mock.patch("magic.Magic") as mock_magic:
            mock_magic.return_value.from_buffer.return_value = "text/plain"
            result = extractor.extract(record)

        assert result is None  # Empty content should return None

    @pytest.mark.parametrize(
        ("record_type", "expected_type"),
        [
            ("notebook", "notebook"),
            ("html", "html"),
            ("text", "text"),
        ],
    )
    def test_extract_content_types(  # noqa: PLR0913
        self,
        test_records: dict,
        record_type: str,
        expected_type: str,
        basic_notebook_json: str,
        html_with_content: str,
        plain_text: str,
        extracted_text_responses: dict,
    ) -> None:
        """Test extraction of different content types."""
        record = test_records[record_type].copy()

        # Use real binary content that can be decoded naturally
        if record_type == "notebook":
            record["binary_content"] = basic_notebook_json.encode("utf-8")
            with mock.patch("magic.Magic") as mock_magic:
                mock_magic.return_value.from_buffer.return_value = "application/json"
                extractor = MathContentExtractor()
                result = extractor.extract(record)
        elif record_type == "html":
            record["binary_content"] = html_with_content.encode("utf-8")
            mock_lynx = mock.Mock()
            mock_lynx.extract_text.return_value = extracted_text_responses["html"]
            with (
                mock.patch("nemo_curator.stages.math.download.extract.LynxExtractor", return_value=mock_lynx),
                mock.patch("magic.Magic") as mock_magic,
            ):
                mock_magic.return_value.from_buffer.return_value = "text/html"
                extractor = MathContentExtractor()
                result = extractor.extract(record)
        else:  # text
            record["binary_content"] = plain_text.encode("utf-8")
            with mock.patch("magic.Magic") as mock_magic:
                mock_magic.return_value.from_buffer.return_value = "text/plain"
                extractor = MathContentExtractor()
                result = extractor.extract(record)

        assert result is not None
        assert result["type"] == expected_type
        assert result["url"] == record["url"]
        assert "text" in result

    @mock.patch("magic.Magic")
    def test_extract_with_magic_mime_detection(
        self, mock_magic_class: mock.Mock, sample_text_content: str, sample_urls: dict
    ) -> None:
        """Test extraction with magic MIME type detection."""
        # Mock magic.Magic instance
        mock_magic_instance = mock.Mock()
        mock_magic_instance.from_buffer.return_value = "text/plain"
        mock_magic_class.return_value = mock_magic_instance

        extractor = MathContentExtractor()
        record = {
            "binary_content": sample_text_content.encode("utf-8"),
            "url": sample_urls["text"],
            "mime_type": "text/plain",
        }

        result = extractor.extract(record)

        assert result is not None
        assert result["magic_mime_type"] == "text/plain"
        assert result["text"] == sample_text_content
        assert result["type"] == "text"
        assert result["url"] == sample_urls["text"]
        mock_magic_class.assert_called_once_with(mime=True)
        mock_magic_instance.from_buffer.assert_called_once_with(record["binary_content"])

    @mock.patch("magic.Magic")
    def test_extract_magic_mime_exception(
        self, mock_magic_class: mock.Mock, sample_text_content: str, sample_urls: dict
    ) -> None:
        """Test extraction when magic MIME detection fails."""
        # Mock magic.Magic to raise exception
        mock_magic_instance = mock.Mock()
        mock_magic_instance.from_buffer.side_effect = Exception("Magic failed")
        mock_magic_class.return_value = mock_magic_instance

        extractor = MathContentExtractor()
        record = {
            "binary_content": sample_text_content.encode("utf-8"),
            "url": sample_urls["text"],
            "mime_type": "text/plain",
        }

        result = extractor.extract(record)

        assert result is not None
        assert result["magic_mime_type"] is None
        assert result["text"] == sample_text_content
        assert result["type"] == "text"  # Should still determine type correctly

    @pytest.mark.parametrize(
        ("record_setup", "expected_type"),
        [
            # Test notebook detection by URL
            ({"url": "http://example.com/notebook.ipynb", "content": "basic_notebook_json"}, "notebook"),
            # Test notebook detection by MIME type
            ({"mime_type": "application/json", "content": "basic_notebook_json"}, "notebook"),
            # Test HTML detection by MIME type
            ({"mime_type": "text/html", "content": "html_with_content"}, "html"),
            # Test text detection by MIME type
            ({"mime_type": "text/plain", "content": "plain_text"}, "text"),
            # Test HTML detection by structure (fallback)
            ({"content": "complex_html"}, "html"),
            # Test fallback to HTML for unknown content
            ({"content": "unknown_content"}, "html"),
        ],
    )
    def test_extract_type_detection(  # noqa: PLR0913
        self,
        record_setup: dict,
        expected_type: str,
        test_records: dict,
        basic_notebook_json: str,
        html_with_content: str,
        plain_text: str,
        complex_html: str,
        unknown_content: str,
        extracted_text_responses: dict,
    ) -> None:
        """Test type detection through public extract method - integration style."""
        # Get base record and override with test-specific values
        record = test_records["text"].copy()  # Start with text record as base

        # Apply setup overrides
        if "url" in record_setup:
            record["url"] = record_setup["url"]
        if "mime_type" in record_setup:
            record["mime_type"] = record_setup["mime_type"]

        # Set up real binary content based on content type
        content_map = {
            "basic_notebook_json": basic_notebook_json,
            "html_with_content": html_with_content,
            "plain_text": plain_text,
            "complex_html": complex_html,
            "unknown_content": unknown_content,
        }
        content = content_map[record_setup["content"]]
        record["binary_content"] = content.encode("utf-8")  # Use real binary content

        extractor = MathContentExtractor()

        # Mock external dependencies based on expected type
        if expected_type == "notebook":
            with mock.patch("magic.Magic") as mock_magic:
                mock_magic.return_value.from_buffer.return_value = "application/json"
                result = extractor.extract(record)
        elif expected_type == "html":
            mock_lynx = mock.Mock()
            mock_lynx.extract_text.return_value = extracted_text_responses["generic"]
            with (
                mock.patch("nemo_curator.stages.math.download.extract.LynxExtractor", return_value=mock_lynx),
                mock.patch("magic.Magic") as mock_magic,
            ):
                mock_magic.return_value.from_buffer.return_value = "text/html"
                result = extractor.extract(record)
        else:  # text
            with mock.patch("magic.Magic") as mock_magic:
                mock_magic.return_value.from_buffer.return_value = "text/plain"
                result = extractor.extract(record)

        assert result is not None
        assert result["type"] == expected_type

    def test_lazy_initialization_lynx(
        self, sample_html_content: str, sample_urls: dict, extracted_text_responses: dict
    ) -> None:
        """Test lazy initialization of lynx extractor."""
        extractor = MathContentExtractor()

        # Initially None
        assert extractor._lynx is None

        with (
            mock.patch("nemo_curator.stages.math.download.extract.LynxExtractor") as mock_lynx_class,
            mock.patch("magic.Magic") as mock_magic_class,
        ):
            mock_lynx_instance = mock.Mock()
            mock_lynx_instance.extract_text.return_value = extracted_text_responses["lynx"]
            mock_lynx_class.return_value = mock_lynx_instance

            mock_magic_instance = mock.Mock()
            mock_magic_instance.from_buffer.return_value = "text/html"
            mock_magic_class.return_value = mock_magic_instance

            record = {
                "binary_content": sample_html_content.encode("utf-8"),
                "url": sample_urls["html"],
                "mime_type": "text/html",
            }

            result = extractor.extract(record)

            # Verify lynx was initialized with correct timeout
            mock_lynx_class.assert_called_once_with(timeout_sec=20)
            assert extractor._lynx is mock_lynx_instance
            assert result["text"] == extracted_text_responses["lynx"]
            assert result["type"] == "html"

    def test_lazy_initialization_magic(self, sample_test_content: str, sample_urls: dict) -> None:
        """Test lazy initialization of magic MIME detector."""
        extractor = MathContentExtractor()

        # Initially None
        assert extractor._magic is None

        with mock.patch("magic.Magic") as mock_magic_class:
            mock_magic_instance = mock.Mock()
            mock_magic_instance.from_buffer.return_value = "text/plain"
            mock_magic_class.return_value = mock_magic_instance

            record = {
                "binary_content": sample_test_content.encode("utf-8"),
                "url": sample_urls["text"],
                "mime_type": "text/plain",
            }

            result = extractor.extract(record)

            # Verify magic was initialized correctly
            mock_magic_class.assert_called_once_with(mime=True)
            assert extractor._magic is mock_magic_instance
            assert result["magic_mime_type"] == "text/plain"
            assert result["text"] == sample_test_content
            assert result["type"] == "text"
