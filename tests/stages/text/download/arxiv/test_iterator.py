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

import io
import tarfile
from pathlib import Path

import pytest

from nemo_curator.stages.text.download.arxiv.iterator import ArxivIterator
from nemo_curator.utils.file_utils import tar_safe_extract


class TestArxivIterator:
    """Test suite for ArxivIterator."""

    def test_output_columns(self) -> None:
        """Test that output_columns returns the expected column names."""
        iterator = ArxivIterator()
        columns = iterator.output_columns()

        expected_columns = ["id", "source_id", "content"]
        assert columns == expected_columns

    def test_arxiv_iterator(self, tmp_path: Path) -> None:
        # Create an inner tar archive containing a .tex file.
        inner_tar_path = tmp_path / "2103.00001.tar"
        dummy_tex_filename = "2103.00001.tex"
        dummy_tex_content = "This is a dummy LaTeX content."
        with tarfile.open(inner_tar_path, "w") as inner_tar:
            # Create a temporary tex file to add into the inner tar archive.
            temp_tex_path = tmp_path / dummy_tex_filename
            with open(temp_tex_path, "w") as f:
                f.write(dummy_tex_content)
            inner_tar.add(temp_tex_path, arcname=dummy_tex_filename)

        # Create an outer tar archive that contains the inner tar archive.
        outer_tar_path = tmp_path / "dummy_main.tar"
        with tarfile.open(outer_tar_path, "w") as outer_tar:
            outer_tar.add(inner_tar_path, arcname="2103.00001.tar")

        iterator = ArxivIterator(log_frequency=1)
        results = list(iterator.iterate(str(outer_tar_path)))
        # Expect one paper extracted.
        assert len(results) == 1
        tex_files = results[0]
        # The ArxivIterator extracts the arxiv id from the inner archive's filename.
        assert tex_files["id"] == "2103.00001"
        # The source_id is set to the outer tar file's basename.
        assert tex_files["source_id"] == "dummy_main.tar"
        # Verify that the tex extraction returns the dummy content.
        assert isinstance(tex_files["content"], list)
        assert dummy_tex_content in tex_files["content"]


class TestSafeExtract:
    """Test suite for tar_safe_extract function."""

    def test_tar_safe_extract_path_traversal_prevention(self, tmp_path: Path) -> None:
        """Test that tar_safe_extract prevents path traversal attacks."""
        # Create a malicious tar file that tries to write outside the extraction directory
        malicious_tar_path = tmp_path / "malicious.tar"

        with tarfile.open(malicious_tar_path, "w") as tar:
            # Add a normal file first
            normal_data = io.BytesIO(b"normal content\n")
            normal_tarinfo = tarfile.TarInfo(name="normal.txt")
            normal_tarinfo.size = len(normal_data.getbuffer())
            tar.addfile(normal_tarinfo, fileobj=normal_data)

            # Add a malicious file that tries to escape the extraction directory
            malicious_data = io.BytesIO(b"malicious content\n")
            malicious_path = "../../../evil.txt"  # Path traversal attempt
            malicious_tarinfo = tarfile.TarInfo(name=malicious_path)
            malicious_tarinfo.size = len(malicious_data.getbuffer())
            tar.addfile(malicious_tarinfo, fileobj=malicious_data)

        # Create extraction directory
        extraction_dir = tmp_path / "extraction"
        extraction_dir.mkdir()

        # Test that tar_safe_extract raises ValueError for path traversal
        with (
            tarfile.open(malicious_tar_path, "r") as tar,
            pytest.raises(ValueError, match="Path traversal attempt detected"),
        ):
            tar_safe_extract(tar, str(extraction_dir))

        # Verify that the malicious file was not created outside the extraction directory
        evil_file_path = tmp_path / "evil.txt"
        assert not evil_file_path.exists(), "Malicious file should not have been created outside extraction directory"

        # Verify that the extraction directory is still safe
        extracted_files = list(extraction_dir.rglob("*"))
        for file_path in extracted_files:
            # All extracted files should be within the extraction directory
            assert str(file_path).startswith(str(extraction_dir)), (
                f"File {file_path} was extracted outside safe directory"
            )

    def test_tar_safe_extract_absolute_path_prevention(self, tmp_path: Path) -> None:
        """Test that tar_safe_extract prevents absolute path attacks."""
        # Create a malicious tar file with absolute path
        malicious_tar_path = tmp_path / "absolute_path.tar"

        with tarfile.open(malicious_tar_path, "w") as tar:
            # Add a file with absolute path
            malicious_data = io.BytesIO(b"absolute path content\n")
            absolute_path = str(tmp_path / "absolute_evil.txt")  # Absolute path within tmp_path
            malicious_tarinfo = tarfile.TarInfo(name=absolute_path)
            malicious_tarinfo.size = len(malicious_data.getbuffer())
            tar.addfile(malicious_tarinfo, fileobj=malicious_data)

        # Create extraction directory
        extraction_dir = tmp_path / "extraction"
        extraction_dir.mkdir()

        # Test that tar_safe_extract raises ValueError for absolute path
        with (
            tarfile.open(malicious_tar_path, "r") as tar,
            pytest.raises(ValueError, match="Absolute path not allowed"),
        ):
            tar_safe_extract(tar, str(extraction_dir))

    def test_tar_safe_extract_normal_files(self, tmp_path: Path) -> None:
        """Test that tar_safe_extract works correctly with normal files."""
        # Create a normal tar file
        normal_tar_path = tmp_path / "normal.tar"

        with tarfile.open(normal_tar_path, "w") as tar:
            # Add normal files
            for i in range(3):
                file_data = io.BytesIO(f"content of file {i}\n".encode())
                tarinfo = tarfile.TarInfo(name=f"file_{i}.txt")
                tarinfo.size = len(file_data.getbuffer())
                tar.addfile(tarinfo, fileobj=file_data)

            # Add a file in a subdirectory
            subdir_data = io.BytesIO(b"subdirectory content\n")
            subdir_tarinfo = tarfile.TarInfo(name="subdir/subfile.txt")
            subdir_tarinfo.size = len(subdir_data.getbuffer())
            tar.addfile(subdir_tarinfo, fileobj=subdir_data)

        # Create extraction directory
        extraction_dir = tmp_path / "extraction"
        extraction_dir.mkdir()

        # Test that tar_safe_extract works correctly with normal files
        with tarfile.open(normal_tar_path, "r") as tar:
            tar_safe_extract(tar, str(extraction_dir))

        # Verify all files were extracted correctly
        assert (extraction_dir / "file_0.txt").exists()
        assert (extraction_dir / "file_1.txt").exists()
        assert (extraction_dir / "file_2.txt").exists()
        assert (extraction_dir / "subdir" / "subfile.txt").exists()

        # Verify content
        with open(extraction_dir / "file_0.txt") as f:
            assert f.read() == "content of file 0\n"
        with open(extraction_dir / "subdir" / "subfile.txt") as f:
            assert f.read() == "subdirectory content\n"

    def test_tar_safe_extract_device_file_prevention(self, tmp_path: Path) -> None:
        """Test that tar_safe_extract prevents extraction of device files."""
        # Create a malicious tar file with a device file
        malicious_tar_path = tmp_path / "device_file.tar"

        with tarfile.open(malicious_tar_path, "w") as tar:
            # Add a device file (character device)
            device_tarinfo = tarfile.TarInfo(name="evil_device")
            device_tarinfo.type = tarfile.CHRTYPE  # Character device
            device_tarinfo.devmajor = 1
            device_tarinfo.devminor = 3
            tar.addfile(device_tarinfo)

        # Create extraction directory
        extraction_dir = tmp_path / "extraction"
        extraction_dir.mkdir()

        # Test that tar_safe_extract raises ValueError for device files
        with (
            tarfile.open(malicious_tar_path, "r") as tar,
            pytest.raises(ValueError, match="Device files not allowed"),
        ):
            tar_safe_extract(tar, str(extraction_dir))

    def test_tar_safe_extract_symlink_prevention(self, tmp_path: Path) -> None:
        """Test that tar_safe_extract prevents unsafe symlinks."""
        # Create a malicious tar file with unsafe symlinks
        malicious_tar_path = tmp_path / "symlink_attack.tar"

        with tarfile.open(malicious_tar_path, "w") as tar:
            # Add a normal file first
            normal_data = io.BytesIO(b"normal content\n")
            normal_tarinfo = tarfile.TarInfo(name="normal.txt")
            normal_tarinfo.size = len(normal_data.getbuffer())
            tar.addfile(normal_tarinfo, fileobj=normal_data)

            # Add a symlink that tries to escape the extraction directory
            symlink_tarinfo = tarfile.TarInfo(name="evil_symlink")
            symlink_tarinfo.type = tarfile.SYMTYPE
            symlink_tarinfo.linkname = "../../../etc/passwd"  # Path traversal via symlink
            tar.addfile(symlink_tarinfo)

        # Create extraction directory
        extraction_dir = tmp_path / "extraction"
        extraction_dir.mkdir()

        # Test that tar_safe_extract raises ValueError for unsafe symlinks
        with (
            tarfile.open(malicious_tar_path, "r") as tar,
            pytest.raises(ValueError, match="Symlink target outside extraction directory"),
        ):
            tar_safe_extract(tar, str(extraction_dir))

    def test_tar_safe_extract_absolute_symlink_prevention(self, tmp_path: Path) -> None:
        """Test that tar_safe_extract prevents symlinks with absolute targets."""
        # Create a malicious tar file with absolute symlink target
        malicious_tar_path = tmp_path / "absolute_symlink.tar"

        with tarfile.open(malicious_tar_path, "w") as tar:
            # Add a symlink with absolute target
            symlink_tarinfo = tarfile.TarInfo(name="absolute_symlink")
            symlink_tarinfo.type = tarfile.SYMTYPE
            symlink_tarinfo.linkname = "/etc/passwd"  # Absolute symlink target
            tar.addfile(symlink_tarinfo)

        # Create extraction directory
        extraction_dir = tmp_path / "extraction"
        extraction_dir.mkdir()

        # Test that tar_safe_extract raises ValueError for absolute symlink targets
        with (
            tarfile.open(malicious_tar_path, "r") as tar,
            pytest.raises(ValueError, match="Absolute symlink target not allowed"),
        ):
            tar_safe_extract(tar, str(extraction_dir))
