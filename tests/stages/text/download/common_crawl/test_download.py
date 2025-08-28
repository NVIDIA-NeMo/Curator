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

import pytest

from nemo_curator.stages.text.download.common_crawl.download import CommonCrawlWARCDownloader


class TestCommonCrawlWARCDownloader:
    """Test suite for CommonCrawlWARCDownloader."""

    @mock.patch("subprocess.run", return_value=mock.Mock(returncode=0))
    def test_download_to_path_wget(self, mock_run: mock.Mock, tmp_path: Path) -> None:
        """Test _download_to_path with wget (use_aws_to_download=False)."""
        downloader = CommonCrawlWARCDownloader(str(tmp_path), use_aws_to_download=False, verbose=False)

        url = "http://dummy/commoncrawl.warc"
        temp_path = str(tmp_path / "temp_file.tmp")

        success, error_message = downloader._download_to_path(url, temp_path)

        assert success is True
        assert error_message is None
        mock_run.assert_called_once_with(
            ["wget", url, "-O", temp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    @mock.patch.object(CommonCrawlWARCDownloader, "_check_s5cmd_installed", return_value=True)
    @mock.patch("subprocess.run", return_value=mock.Mock(returncode=0))
    def test_download_to_path_s3(self, mock_run: mock.Mock, mock_s5cmd_check: mock.Mock, tmp_path: Path) -> None:  # noqa: ARG002
        """Test _download_to_path with s5cmd (use_aws_to_download=True)."""

        downloader = CommonCrawlWARCDownloader(str(tmp_path), use_aws_to_download=True, verbose=False)

        url = "http://dummy/crawl-data/CC-MAIN-2024-10/segments/1234567890/warc/CC-MAIN-123.warc.gz"
        temp_path = str(tmp_path / "temp_file.tmp")

        success, error_message = downloader._download_to_path(url, temp_path)

        assert success is True
        assert error_message is None
        expected_s3_url = "s3://commoncrawl/crawl-data/CC-MAIN-2024-10/segments/1234567890/warc/CC-MAIN-123.warc.gz"
        mock_run.assert_called_once_with(
            ["s5cmd", "cp", expected_s3_url, temp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    @mock.patch("subprocess.run", return_value=mock.Mock(returncode=0))
    def test_download_to_path_verbose_logging(self, mock_run: mock.Mock, tmp_path: Path) -> None:
        """Test _download_to_path with verbose logging enabled."""
        downloader = CommonCrawlWARCDownloader(str(tmp_path), use_aws_to_download=False, verbose=True)

        url = "http://dummy/commoncrawl.warc"
        temp_path = str(tmp_path / "temp_file.tmp")

        success, error_message = downloader._download_to_path(url, temp_path)

        assert success is True
        assert error_message is None
        mock_run.assert_called_once_with(
            ["wget", url, "-O", temp_path],
            stdout=None,
            stderr=None,
        )

    @mock.patch("subprocess.run", return_value=mock.Mock(returncode=0))
    def test_download_to_path_quiet(self, mock_run: mock.Mock, tmp_path: Path) -> None:
        """Test _download_to_path with quiet mode (verbose=False)."""
        downloader = CommonCrawlWARCDownloader(str(tmp_path), use_aws_to_download=False, verbose=False)

        url = "http://dummy/commoncrawl.warc"
        temp_path = str(tmp_path / "temp_file.tmp")

        success, error_message = downloader._download_to_path(url, temp_path)

        assert success is True
        assert error_message is None
        mock_run.assert_called_once_with(
            ["wget", url, "-O", temp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    @mock.patch("subprocess.run", return_value=mock.Mock(returncode=1, stderr=b"Download failed"))
    def test_download_to_path_failed(self, mock_run: mock.Mock, tmp_path: Path) -> None:
        """Test _download_to_path with failed download."""
        downloader = CommonCrawlWARCDownloader(str(tmp_path), use_aws_to_download=False, verbose=False)

        url = "http://dummy/commoncrawl.warc"
        temp_path = str(tmp_path / "temp_file.tmp")

        success, error_message = downloader._download_to_path(url, temp_path)

        assert success is False
        assert error_message == "Download failed"
        mock_run.assert_called_once_with(
            ["wget", url, "-O", temp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def test_check_s5cmd_installed_true(self, tmp_path: Path) -> None:
        """Test _check_s5cmd_installed when s5cmd is available."""
        downloader = CommonCrawlWARCDownloader(str(tmp_path), use_aws_to_download=False, verbose=False)

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = None
            result = downloader._check_s5cmd_installed()
            assert result is True

    def test_check_s5cmd_installed_false(self, tmp_path: Path) -> None:
        """Test _check_s5cmd_installed when s5cmd is not available."""
        downloader = CommonCrawlWARCDownloader(str(tmp_path), use_aws_to_download=False, verbose=False)

        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            result = downloader._check_s5cmd_installed()
            assert result is False

    @mock.patch.object(CommonCrawlWARCDownloader, "_check_s5cmd_installed", return_value=False)
    def test_init_aws_download_without_s5cmd(self, mock_s5cmd_check: mock.Mock, tmp_path: Path) -> None:  # noqa: ARG002
        """Test initialization with AWS download but s5cmd not installed."""
        with pytest.raises(RuntimeError, match="s5cmd is not installed"):
            CommonCrawlWARCDownloader(str(tmp_path), use_aws_to_download=True, verbose=False)

    def test_url_to_output_name_conversion(self, tmp_path: Path) -> None:
        """Test conversion of URL to output filename."""
        downloader = CommonCrawlWARCDownloader(str(tmp_path), use_aws_to_download=False, verbose=False)

        url = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2021-10/segments/12345/warc/file.warc.gz"
        expected_name = "crawl-data-CC-MAIN-2021-10-segments-12345-warc-file.warc.gz"

        result = downloader._get_output_filename(url)
        assert result == expected_name
