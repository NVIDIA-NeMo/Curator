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

import subprocess
from unittest import mock

from nemo_curator.stages.math.download.html_extractors.lynx import LynxExtractor


class TestLynxExtractor:
    """Test the LynxExtractor class."""

    def test_lynx_extractor_init_default_timeout(self) -> None:
        """Test LynxExtractor initialization with default timeout."""
        extractor = LynxExtractor()
        assert extractor.timeout_sec == 20

    def test_lynx_extractor_init_custom_timeout(self) -> None:
        """Test LynxExtractor initialization with custom timeout."""
        extractor = LynxExtractor(timeout_sec=30)
        assert extractor.timeout_sec == 30

    @mock.patch("subprocess.run")
    def test_lynx_extractor_extract_text_success(self, mock_run: mock.Mock, html_with_content: str) -> None:
        """Test successful lynx text extraction."""
        # Mock successful subprocess call
        mock_process = mock.Mock()
        mock_process.returncode = 0
        mock_process.stdout = b"Extracted text content"
        mock_run.return_value = mock_process

        extractor = LynxExtractor(timeout_sec=15)

        result = extractor.extract_text(html_with_content)

        assert result == "Extracted text content"
        mock_run.assert_called_once_with(
            [
                "lynx",
                "-dump",
                "-stdin",
                "-nolist",
                "-width=10000",
                "-assume_charset=utf-8",
                "-display_charset=utf-8",
                "-localhost",
                "-force_html",
            ],
            input=html_with_content.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=15,
        )

    @mock.patch("subprocess.run")
    def test_lynx_extractor_extract_text_timeout(self, mock_run: mock.Mock, simple_html: str) -> None:
        """Test LynxExtractor timeout handling."""
        mock_run.side_effect = subprocess.TimeoutExpired(["lynx"], timeout=20)

        extractor = LynxExtractor()

        result = extractor.extract_text(simple_html)

        assert result == ""
        mock_run.assert_called_once()

    @mock.patch("subprocess.run")
    def test_lynx_extractor_extract_text_failure(self, mock_run: mock.Mock, simple_html: str) -> None:
        """Test LynxExtractor when lynx returns non-zero exit code."""
        mock_process = mock.Mock()
        mock_process.returncode = 1
        mock_run.return_value = mock_process

        extractor = LynxExtractor()

        result = extractor.extract_text(simple_html)

        assert result == ""
        mock_run.assert_called_once()

    @mock.patch("subprocess.run")
    def test_lynx_extractor_extract_text_empty_input(self, mock_run: mock.Mock) -> None:
        """Test LynxExtractor with empty input."""
        extractor = LynxExtractor()

        result = extractor.extract_text("")

        assert result == ""
        mock_run.assert_not_called()  # Should return early without calling subprocess

    @mock.patch("subprocess.run")
    def test_lynx_extractor_extract_text_decode_error(self, mock_run: mock.Mock, simple_html: str) -> None:
        """Test LynxExtractor with decode error handling."""
        mock_process = mock.Mock()
        mock_process.returncode = 0
        # Invalid UTF-8 bytes that will cause decode error
        mock_process.stdout = b"\xff\xfe"
        mock_run.return_value = mock_process

        extractor = LynxExtractor()

        result = extractor.extract_text(simple_html)

        # Should handle decode error gracefully with error replacement
        assert isinstance(result, str)
        mock_run.assert_called_once()

    @mock.patch("subprocess.run")
    def test_lynx_extractor_extract_text_with_math_content(self, mock_run: mock.Mock, math_html: str) -> None:
        """Test LynxExtractor with mathematical content."""
        # Simulate lynx extracting LaTeX/math content
        mock_process = mock.Mock()
        mock_process.returncode = 0
        mock_process.stdout = "Quadratic Formula\n\nThe quadratic formula is:\n\nx = (-b ± √(b² - 4ac)) / 2a\n\nWhere a, b, and c are coefficients.".encode()
        mock_run.return_value = mock_process

        extractor = LynxExtractor()

        result = extractor.extract_text(math_html)

        assert "Quadratic Formula" in result
        assert "coefficients" in result
        mock_run.assert_called_once()
