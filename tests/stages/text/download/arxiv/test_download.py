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

import os
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from nemo_curator.stages.text.download.arxiv.download import ArxivDownloader


class FakeCompletedProcess:
    def __init__(self) -> None:
        self.returncode = 0


def fake_run_success(cmd: list[str], stdout: str, stderr: str) -> subprocess.CompletedProcess:  # noqa: ARG001
    return FakeCompletedProcess()


class TestArxivDownloader:
    """Test suite for ArxivDownloader."""

    @mock.patch.object(ArxivDownloader, "_check_s5cmd_installed", return_value=True)
    @mock.patch("subprocess.run", return_value=mock.Mock(returncode=0))
    @pytest.mark.parametrize("verbose", [True, False])
    def test_download_to_path(self, mock_run: mock.Mock, tmp_path: Path, verbose: bool) -> None:
        """Test _download_to_path with s5cmd."""
        downloader = ArxivDownloader(str(tmp_path), verbose=verbose)

        tar_filename = "dummy.tar"
        temp_path = str(tmp_path / "temp_file.tmp")

        success, error_message = downloader._download_to_path(tar_filename, temp_path)

        if verbose:
            stdout, stderr = None, None
        else:
            stdout, stderr = subprocess.DEVNULL, subprocess.DEVNULL

        assert success is True
        assert error_message is None
        mock_run.assert_called_once_with(
            ["s5cmd", "--request-payer=requester", "cp", "s3://arxiv/src/" + tar_filename, temp_path],
            stdout=stdout,
            stderr=stderr,
        )

    @mock.patch.object(ArxivDownloader, "_check_s5cmd_installed", return_value=True)
    @mock.patch("subprocess.run", return_value=mock.Mock(returncode=1, stderr=b"Failed to download"))
    def test_download_to_path_failed(self, mock_run: mock.Mock, tmp_path: Path) -> None:
        """Test _download_to_path with failed download."""
        downloader = ArxivDownloader(str(tmp_path), verbose=False)

        tar_filename = "dummy.tar"
        temp_path = str(tmp_path / "temp_file.tmp")

        success, error_message = downloader._download_to_path(tar_filename, temp_path)

        assert success is False
        assert "Failed to download" in error_message
        mock_run.assert_called_once_with(
            ["s5cmd", "--request-payer=requester", "cp", "s3://arxiv/src/" + tar_filename, temp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    @mock.patch.object(ArxivDownloader, "_check_s5cmd_installed", return_value=True)
    def test_init_with_s5cmd(self, tmp_path: Path) -> None:
        """Test _check_s5cmd_installed when s5cmd is available."""
        downloader = ArxivDownloader(str(tmp_path), verbose=False)

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = None
            result = downloader._check_s5cmd_installed()
            assert result is True

    @mock.patch.object(ArxivDownloader, "_check_s5cmd_installed", return_value=False)
    def test_init_without_s5cmd(self, tmp_path: Path) -> None:
        """Test initialization but s5cmd not installed."""
        with pytest.raises(RuntimeError, match="s5cmd is not installed"):
            ArxivDownloader(str(tmp_path), verbose=False)

    @mock.patch.object(ArxivDownloader, "_check_s5cmd_installed", return_value=True)
    def test_get_output_filename(self, tmp_path: Path) -> None:
        """Test conversion of URL to output filename."""
        downloader = ArxivDownloader(str(tmp_path), verbose=False)

        url = "s3://arxiv/src/dummy.tar"

        result = downloader._get_output_filename(url)
        assert result == url

    def test_arxiv_downloader_existing_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        with mock.patch.object(ArxivDownloader, "_check_s5cmd_installed", return_value=True):
            # Create a temporary download directory and simulate an already-downloaded tar file.
            download_dir = tmp_path / "downloads"
            download_dir.mkdir()
            tar_filename = "dummy.tar"
            file_path = os.path.join(str(download_dir), tar_filename)
            # Write dummy content to simulate an existing download.
            with open(file_path, "w") as f:
                f.write("existing content")

            downloader = ArxivDownloader(str(download_dir), verbose=False)
            # Monkey-patch subprocess.run (should not be called since file exists).
            monkeypatch.setattr(subprocess, "run", fake_run_success)
            result = downloader.download(tar_filename)
            assert result == file_path
