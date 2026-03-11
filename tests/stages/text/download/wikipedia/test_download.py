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
from pathlib import Path
from unittest import mock

from nemo_curator.stages.text.download.wikipedia.download import WikipediaDownloader


class TestWikipediaDownloader:
    """Test suite for WikipediaDownloader."""

    def test_init_default_values(self, tmp_path: Path):
        """Test initialization with default values."""
        downloader = WikipediaDownloader(str(tmp_path))
        assert downloader._download_dir == str(tmp_path)
        assert downloader._verbose is False

    def test_init_custom_values(self, tmp_path: Path):
        """Test initialization with custom values."""
        downloader = WikipediaDownloader(str(tmp_path), verbose=True)
        assert downloader._download_dir == str(tmp_path)
        assert downloader._verbose is True

    def test_get_output_filename_simple(self, tmp_path: Path):
        """Test conversion of simple URL to output filename."""
        downloader = WikipediaDownloader(str(tmp_path))

        url = "https://dumps.wikimedia.org/enwiki/20230501/enwiki-20230501-pages-articles-multistream1.xml.bz2"
        expected_name = "enwiki-20230501-enwiki-20230501-pages-articles-multistream1.xml.bz2"

        result = downloader._get_output_filename(url)
        assert result == expected_name

    def test_get_output_filename_complex(self, tmp_path: Path):
        """Test conversion of complex URL to output filename."""
        downloader = WikipediaDownloader(str(tmp_path))

        url = "https://dumps.wikimedia.org/eswiki/20230401/eswiki-20230401-pages-articles-multistream15.xml.bz2"
        expected_name = "eswiki-20230401-eswiki-20230401-pages-articles-multistream15.xml.bz2"

        result = downloader._get_output_filename(url)
        assert result == expected_name

    def test_get_output_filename_different_language(self, tmp_path: Path):
        """Test conversion of URL with different language."""
        downloader = WikipediaDownloader(str(tmp_path))

        url = "https://dumps.wikimedia.org/frwiki/20230601/frwiki-20230601-pages-articles-multistream3.xml.bz2"
        expected_name = "frwiki-20230601-frwiki-20230601-pages-articles-multistream3.xml.bz2"

        result = downloader._get_output_filename(url)
        assert result == expected_name

    @mock.patch("subprocess.run", return_value=mock.Mock(returncode=0))
    def test_download_to_path_success(self, mock_run: mock.Mock, tmp_path: Path):
        """Test successful download using wget."""
        downloader = WikipediaDownloader(str(tmp_path), verbose=False)

        url = "https://dumps.wikimedia.org/enwiki/20230501/enwiki-20230501-pages-articles-multistream1.xml.bz2"
        temp_path = str(tmp_path / "temp_file.bz2")

        success, error_message = downloader._download_to_path(url, temp_path)

        assert success is True
        assert error_message is None
        mock_run.assert_called_once_with(
            ["wget", url, "-O", temp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    @mock.patch("subprocess.run", return_value=mock.Mock(returncode=0))
    def test_download_to_path_verbose_logging(self, mock_run: mock.Mock, tmp_path: Path):
        """Test download with verbose logging enabled."""
        downloader = WikipediaDownloader(str(tmp_path), verbose=True)

        url = "https://dumps.wikimedia.org/enwiki/20230501/enwiki-20230501-pages-articles-multistream1.xml.bz2"
        temp_path = str(tmp_path / "temp_file.bz2")

        success, error_message = downloader._download_to_path(url, temp_path)

        assert success is True
        assert error_message is None
        mock_run.assert_called_once_with(
            ["wget", url, "-O", temp_path],
            stdout=None,
            stderr=None,
        )

    @mock.patch("subprocess.run", return_value=mock.Mock(returncode=0))
    def test_download_to_path_quiet_mode(self, mock_run: mock.Mock, tmp_path: Path):
        """Test download with quiet mode (verbose=False)."""
        downloader = WikipediaDownloader(str(tmp_path), verbose=False)

        url = "https://dumps.wikimedia.org/enwiki/20230501/enwiki-20230501-pages-articles-multistream1.xml.bz2"
        temp_path = str(tmp_path / "temp_file.bz2")

        success, error_message = downloader._download_to_path(url, temp_path)

        assert success is True
        assert error_message is None
        mock_run.assert_called_once_with(
            ["wget", url, "-O", temp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    @mock.patch("subprocess.run", return_value=mock.Mock(returncode=1, stderr=b"File not found"))
    def test_download_to_path_failed(self, mock_run: mock.Mock, tmp_path: Path):
        """Test failed download with error message."""
        downloader = WikipediaDownloader(str(tmp_path), verbose=False)

        url = "https://dumps.wikimedia.org/enwiki/20230501/nonexistent-file.xml.bz2"
        temp_path = str(tmp_path / "temp_file.bz2")

        success, error_message = downloader._download_to_path(url, temp_path)

        assert success is False
        assert error_message == "File not found"
        mock_run.assert_called_once_with(
            ["wget", url, "-O", temp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    @mock.patch("subprocess.run", return_value=mock.Mock(returncode=2, stderr=b"Network error"))
    def test_download_to_path_network_error(self, mock_run: mock.Mock, tmp_path: Path):
        """Test download with network error."""
        downloader = WikipediaDownloader(str(tmp_path), verbose=False)

        url = "https://dumps.wikimedia.org/enwiki/20230501/enwiki-20230501-pages-articles-multistream1.xml.bz2"
        temp_path = str(tmp_path / "temp_file.bz2")

        success, error_message = downloader._download_to_path(url, temp_path)

        assert success is False
        assert error_message == "Network error"
        mock_run.assert_called_once_with(
            ["wget", url, "-O", temp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    @mock.patch("subprocess.run", return_value=mock.Mock(returncode=1, stderr=None))
    def test_download_to_path_failed_no_stderr(self, mock_run: mock.Mock, tmp_path: Path):
        """Test failed download without stderr."""
        downloader = WikipediaDownloader(str(tmp_path), verbose=False)

        url = "https://dumps.wikimedia.org/enwiki/20230501/enwiki-20230501-pages-articles-multistream1.xml.bz2"
        temp_path = str(tmp_path / "temp_file.bz2")

        success, error_message = downloader._download_to_path(url, temp_path)

        assert success is False
        assert error_message == "Unknown error"
        mock_run.assert_called_once_with(
            ["wget", url, "-O", temp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    @mock.patch("subprocess.run")
    def test_download_command_construction(self, mock_run: mock.Mock, tmp_path: Path):
        """Test that wget command is constructed correctly."""
        mock_run.return_value = mock.Mock(returncode=0)
        downloader = WikipediaDownloader(str(tmp_path), verbose=False)

        url = "https://dumps.wikimedia.org/enwiki/20230501/enwiki-20230501-pages-articles-multistream1.xml.bz2"
        temp_path = str(tmp_path / "temp_file.bz2")

        downloader._download_to_path(url, temp_path)

        expected_cmd = ["wget", url, "-O", temp_path]
        mock_run.assert_called_once_with(
            expected_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def test_inheritance_from_document_downloader(self, tmp_path: Path):
        """Test that WikipediaDownloader properly inherits from DocumentDownloader."""
        downloader = WikipediaDownloader(str(tmp_path))

        # Should have the parent's attributes
        assert hasattr(downloader, "_download_dir")
        assert hasattr(downloader, "_verbose")

        # Should have the parent's methods
        assert hasattr(downloader, "_get_output_filename")
        assert hasattr(downloader, "_download_to_path")


class TestWikipediaDownloaderIntegration:
    """Integration tests for WikipediaDownloader."""

    @mock.patch("subprocess.run")
    def test_realistic_download_scenario(self, mock_run: mock.Mock, tmp_path: Path):
        """Test a realistic download scenario."""
        mock_run.return_value = mock.Mock(returncode=0)

        downloader = WikipediaDownloader(str(tmp_path), verbose=True)

        # Test URLs that might be encountered in practice
        test_urls = [
            "https://dumps.wikimedia.org/enwiki/20230501/enwiki-20230501-pages-articles-multistream1.xml.bz2",
            "https://dumps.wikimedia.org/enwiki/20230501/enwiki-20230501-pages-articles-multistream2.xml.bz2",
            "https://dumps.wikimedia.org/eswiki/20230401/eswiki-20230401-pages-articles-multistream1.xml.bz2",
        ]

        for url in test_urls:
            temp_path = str(tmp_path / f"temp_{hash(url)}.bz2")
            success, error_message = downloader._download_to_path(url, temp_path)

            assert success is True
            assert error_message is None

        # Should have called wget for each URL
        assert mock_run.call_count == len(test_urls)

        # Check that all calls were made with correct parameters
        for i, call in enumerate(mock_run.call_args_list):
            args, kwargs = call
            assert args[0][0] == "wget"
            assert args[0][1] == test_urls[i]
            assert args[0][2] == "-O"
            assert kwargs["stdout"] is None  # verbose mode
            assert kwargs["stderr"] is None  # verbose mode

    @mock.patch("subprocess.run")
    def test_mixed_success_failure_scenario(self, mock_run: mock.Mock, tmp_path: Path):
        """Test scenario with both successful and failed downloads."""
        # First call succeeds, second fails
        mock_run.side_effect = [
            mock.Mock(returncode=0),  # Success
            mock.Mock(returncode=1, stderr=b"404 Not Found"),  # Failure
        ]

        downloader = WikipediaDownloader(str(tmp_path), verbose=False)

        # First download should succeed
        url1 = "https://dumps.wikimedia.org/enwiki/20230501/enwiki-20230501-pages-articles-multistream1.xml.bz2"
        temp_path1 = str(tmp_path / "temp1.bz2")
        success1, error1 = downloader._download_to_path(url1, temp_path1)

        assert success1 is True
        assert error1 is None

        # Second download should fail
        url2 = "https://dumps.wikimedia.org/enwiki/20230501/nonexistent-file.xml.bz2"
        temp_path2 = str(tmp_path / "temp2.bz2")
        success2, error2 = downloader._download_to_path(url2, temp_path2)

        assert success2 is False
        assert error2 == "404 Not Found"

        assert mock_run.call_count == 2
