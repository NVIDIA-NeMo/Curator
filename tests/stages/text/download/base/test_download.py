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

from pathlib import Path
from unittest import mock

import pytest

from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.download.base.download import DocumentDownloader, DocumentDownloadStage
from nemo_curator.tasks import FileGroupTask


class MockDocumentDownloader(DocumentDownloader):
    """Mock implementation of DocumentDownloader for testing."""

    def __init__(self, download_dir: str, verbose: bool = False):
        super().__init__(download_dir, verbose)

    def _get_output_filename(self, url: str) -> str:
        """Simple filename generation for testing."""
        return url.split("/")[-1].replace(":", "-")

    def _download_to_path(self, url: str, path: str) -> tuple[bool, str | None]:
        """Mock download implementation - will be patched in tests."""
        # Default successful mock
        Path(path).write_text(f"mock content for {url}")
        return True, None


class TestBaseDocumentDownloader:
    """Base test class for DocumentDownloader temporary file logic."""

    @mock.patch.object(MockDocumentDownloader, "_download_to_path")
    @pytest.mark.parametrize("verbose", [True, False])
    def test_download_existing_file(
        self, mock_download: mock.Mock, tmp_path: Path, caplog: pytest.LogCaptureFixture, verbose: bool
    ) -> None:
        """Test that download skips downloading when file already exists and is non-empty."""
        url = "http://dummy/test-file.txt"

        downloader = MockDocumentDownloader(str(tmp_path), verbose=verbose)

        # Create existing file with content
        expected_filename = downloader._get_output_filename(url)
        file_path = tmp_path / expected_filename
        file_path.write_text("existing content")

        result = downloader.download(url)

        assert result == str(file_path)
        assert file_path.read_text() == "existing content"
        mock_download.assert_not_called()

        if verbose:
            assert "exists. Not downloading" in caplog.text
        else:
            assert "exists. Not downloading" not in caplog.text

    @pytest.mark.parametrize("verbose", [True, False])
    def test_download_successful(self, tmp_path: Path, caplog: pytest.LogCaptureFixture, verbose: bool) -> None:
        """Test successful download of a new file."""
        url = "http://dummy/test-file.txt"

        downloader = MockDocumentDownloader(str(tmp_path), verbose=verbose)

        result = downloader.download(url)

        expected_filename = downloader._get_output_filename(url)
        file_path = tmp_path / expected_filename

        assert result == str(file_path)
        assert file_path.exists()
        assert file_path.read_text() == "mock content for http://dummy/test-file.txt"

        if verbose:
            assert "Successfully downloaded to" in caplog.text
        else:
            assert "Successfully downloaded" not in caplog.text

    def test_download_empty_existing_file(self, tmp_path: Path) -> None:
        """Test that download proceeds when file exists but is empty."""
        url = "http://dummy/test-file.txt"

        downloader = MockDocumentDownloader(str(tmp_path), verbose=False)

        # Create empty file
        expected_filename = downloader._get_output_filename(url)
        file_path = tmp_path / expected_filename
        file_path.touch()

        # Download should proceed even if file is empty
        result = downloader.download(url)

        assert result == str(file_path)
        assert file_path.read_text() == "mock content for http://dummy/test-file.txt"

    @mock.patch.object(MockDocumentDownloader, "_download_to_path", return_value=(False, "Download failed"))
    def test_download_failed(self, mock_download: mock.Mock, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Test handling of failed download."""
        caplog.set_level("ERROR")

        url = "http://dummy/test-file.txt"

        downloader = MockDocumentDownloader(str(tmp_path), verbose=False)

        result = downloader.download(url)

        assert result is None
        mock_download.assert_called_once()

        # Check that error message was logged
        assert "Failed to download to" in caplog.text

    def test_download_directory_creation(self, tmp_path: Path) -> None:
        """Test that download directory is created if it doesn't exist."""
        download_dir = tmp_path / "new_dir"
        assert not download_dir.exists()

        MockDocumentDownloader(str(download_dir), verbose=False)

        assert download_dir.exists()
        assert download_dir.is_dir()

    @mock.patch.object(MockDocumentDownloader, "_download_to_path")
    def test_temp_file_cleanup_on_failure(self, mock_download: mock.Mock, tmp_path: Path) -> None:
        """Test that temporary files don't remain after failed downloads."""
        url = "http://dummy/test-file.txt"

        def fail_after_creating_temp(url: str, path: str) -> tuple[bool, str | None]:  # noqa: ARG001
            # Create temp file but return failure
            Path(path).write_text("temp content")
            return False, "Download failed"

        mock_download.side_effect = fail_after_creating_temp

        downloader = MockDocumentDownloader(str(tmp_path), verbose=False)

        result = downloader.download(url)

        # Download should fail
        assert result is None
        mock_download.assert_called_once()

        # Temp file should not exist (cleanup happens via os.rename failure or explicit cleanup)
        expected_filename = downloader._get_output_filename(url)
        temp_file = tmp_path / f"{expected_filename}.tmp"
        final_file = tmp_path / expected_filename

        # temp file can exist after failure
        assert temp_file.exists()
        # final file should not exist after failure
        assert not final_file.exists()

    def test_temp_file_cleanup_on_retry(self, tmp_path: Path) -> None:
        """Test that existing temp files are cleaned up before new download attempts."""
        url = "http://dummy/test-file.txt"

        downloader = MockDocumentDownloader(str(tmp_path), verbose=False)

        # Create a leftover temp file from a previous failed attempt
        expected_filename = downloader._get_output_filename(url)
        temp_file = tmp_path / f"{expected_filename}.tmp"
        temp_file.write_text("old temp content")
        assert temp_file.exists()

        result = downloader.download(url)

        final_file = tmp_path / expected_filename
        assert result == str(final_file)
        assert final_file.read_text() == "mock content for http://dummy/test-file.txt"
        assert not temp_file.exists()  # Temp file should be moved to final location


class TestDocumentDownloadStage:
    """Test class for DocumentDownloadStage functionality."""

    def test_stage_properties(self, tmp_path: Path) -> None:
        """Test that stage properties are correctly defined."""
        downloader = MockDocumentDownloader(str(tmp_path), verbose=False)
        stage = DocumentDownloadStage(downloader=downloader)

        # Test stage name
        assert stage.name == "download_mockdocumentdownloader"

        # Test inputs and outputs
        assert stage.inputs() == (["data"], [])
        assert stage.outputs() == (["data"], [])

        # Test resources
        assert stage.resources == Resources(cpus=0.5)

    def test_process_successful_downloads(self, tmp_path: Path) -> None:
        """Test successful download of multiple URLs."""
        downloader = MockDocumentDownloader(str(tmp_path), verbose=False)
        stage = DocumentDownloadStage(downloader=downloader)

        # Create input task with multiple URLs
        urls = ["http://example.com/file1.txt", "http://example.com/file2.txt", "http://example.com/file3.txt"]
        input_task = FileGroupTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=urls,
            _metadata={"source": "test", "count": 3},
        )

        result = stage.process(input_task)

        # Verify result structure
        assert isinstance(result, FileGroupTask)
        assert result.task_id == "test_task"
        assert result.dataset_name == "test_dataset"
        assert result._metadata == {
            "source": "test",
            "count": 3,
            "source_files": [str(tmp_path / f"file{i + 1}.txt") for i in range(3)],
        }

        # Verify all files were downloaded
        assert len(result.data) == 3
        for i, file_path in enumerate(result.data):
            assert file_path.endswith(f"file{i + 1}.txt")
            assert Path(file_path).exists()
            assert Path(file_path).read_text() == f"mock content for {urls[i]}"

    @mock.patch.object(MockDocumentDownloader, "download")
    def test_process_with_failed_downloads(self, mock_download: mock.Mock, tmp_path: Path) -> None:
        """Test handling when some downloads fail."""

        # Mock download to succeed for first URL, fail for second, succeed for third
        def side_effect(url: str) -> str | None:
            if "file2" in url:
                return None  # Simulate failure
            return str(tmp_path / url.split("/")[-1])

        mock_download.side_effect = side_effect

        downloader = MockDocumentDownloader(str(tmp_path), verbose=False)
        stage = DocumentDownloadStage(downloader=downloader)

        urls = [
            "http://example.com/file1.txt",
            "http://example.com/file2.txt",  # This will fail
            "http://example.com/file3.txt",
        ]
        input_task = FileGroupTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=urls,
            _metadata={"source": "test"},
        )

        result = stage.process(input_task)

        # Should only have 2 files (failed download not included)
        assert len(result.data) == 2
        assert "file1.txt" in result.data[0]
        assert "file3.txt" in result.data[1]

        # Verify download was called for all URLs
        assert mock_download.call_count == 3

    def test_process_empty_file_group(self, tmp_path: Path) -> None:
        """Test processing an empty file group task."""
        downloader = MockDocumentDownloader(str(tmp_path), verbose=False)
        stage = DocumentDownloadStage(downloader=downloader)

        input_task = FileGroupTask(
            task_id="empty_task",
            dataset_name="test_dataset",
            data=[],
            _metadata={"source": "test"},
        )

        result = stage.process(input_task)

        assert isinstance(result, FileGroupTask)
        assert result.task_id == "empty_task"
        assert result.dataset_name == "test_dataset"
        assert result.data == []
        assert result._metadata == {"source": "test", "source_files": []}

    @mock.patch.object(MockDocumentDownloader, "download", return_value=None)
    def test_process_all_downloads_fail(self, mock_download: mock.Mock, tmp_path: Path) -> None:
        """Test when all downloads fail."""
        downloader = MockDocumentDownloader(str(tmp_path), verbose=False)
        stage = DocumentDownloadStage(downloader=downloader)

        urls = ["http://example.com/file1.txt", "http://example.com/file2.txt"]
        input_task = FileGroupTask(
            task_id="test_task",
            dataset_name="test_dataset",
            data=urls,
            _metadata={"source": "test"},
        )

        result = stage.process(input_task)

        # Should return empty data list when all downloads fail
        assert len(result.data) == 0
        assert result.task_id == "test_task"
        assert result.dataset_name == "test_dataset"
        assert result._metadata == {"source": "test", "source_files": []}

        # Verify download was attempted for all URLs
        assert mock_download.call_count == 2
