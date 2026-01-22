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

import shutil
import subprocess
from unittest import mock

import pytest

from nemo_curator.stages.math.download.html_extractors.lynx import LynxExtractor


@pytest.mark.skipif(not shutil.which("lynx"), reason="lynx executable not found in PATH")
class TestLynxExtractor:
    """Test the LynxExtractor class."""

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

        # Should handle decode error gracefully with ftfy and error replacement
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

    @mock.patch("subprocess.run")
    def test_lynx_extractor_subprocess_error(self, mock_run: mock.Mock, simple_html: str) -> None:
        """Test LynxExtractor with subprocess error handling."""
        mock_run.side_effect = subprocess.SubprocessError("Subprocess failed")

        extractor = LynxExtractor()

        result = extractor.extract_text(simple_html)

        assert result == ""
        mock_run.assert_called_once()

    @mock.patch("subprocess.run")
    def test_lynx_extractor_os_error(self, mock_run: mock.Mock, simple_html: str) -> None:
        """Test LynxExtractor with OS error handling."""
        mock_run.side_effect = OSError("System error")

        extractor = LynxExtractor()

        result = extractor.extract_text(simple_html)

        assert result == ""
        mock_run.assert_called_once()

    def test_lynx_extractor_unicode_encode_error(self) -> None:
        """Test LynxExtractor with Unicode encode error handling."""
        extractor = LynxExtractor()

        # Create HTML with characters that might cause encoding issues
        problematic_html = "Test content with problematic chars: \udcff"

        # Mock subprocess.run to raise UnicodeEncodeError during input processing
        def mock_run_with_encode_error(*_args, **kwargs) -> mock.Mock:
            # Simulate the error happening when trying to encode the input
            if "input" in kwargs:
                encoding = "utf-8"
                error_msg = "invalid start byte"
                raise UnicodeEncodeError(encoding, "test", 0, 1, error_msg)
            return mock.Mock()

        with mock.patch("subprocess.run", side_effect=mock_run_with_encode_error):
            result = extractor.extract_text(problematic_html)

        assert result == ""
