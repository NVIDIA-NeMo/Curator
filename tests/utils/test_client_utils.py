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

import tempfile
from unittest.mock import Mock

import fsspec
import pytest

from nemo_curator.utils.client_utils import FSPath


class TestFSPath:
    """Test suite for FSPath class."""

    def test_fspath_initialization(self) -> None:
        """Test FSPath initialization with filesystem and path."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        path = "/test/path"
        fs_path = FSPath(mock_fs, path)

        assert fs_path._fs == mock_fs
        assert fs_path._path == path

    def test_fspath_str_representation(self) -> None:
        """Test FSPath string representation."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        path = "/test/path"
        fs_path = FSPath(mock_fs, path)

        assert str(fs_path) == path

    def test_fspath_repr_representation(self) -> None:
        """Test FSPath repr representation."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        path = "/test/path"
        fs_path = FSPath(mock_fs, path)

        expected_repr = f"FSPath({path})"
        assert repr(fs_path) == expected_repr

    def test_fspath_open(self) -> None:
        """Test FSPath open method."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_file = Mock(spec=fsspec.spec.AbstractBufferedFile)
        path = "/test/path"
        fs_path = FSPath(mock_fs, path)

        mock_fs.open.return_value = mock_file

        result = fs_path.open("rb")

        mock_fs.open.assert_called_once_with(path, "rb")
        assert result == mock_file

    def test_fspath_open_with_kwargs(self) -> None:
        """Test FSPath open method with additional kwargs."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_file = Mock(spec=fsspec.spec.AbstractBufferedFile)
        path = "/test/path"
        fs_path = FSPath(mock_fs, path)

        mock_fs.open.return_value = mock_file

        result = fs_path.open("wb", compression="gzip")

        mock_fs.open.assert_called_once_with(path, "wb", compression="gzip")
        assert result == mock_file

    def test_as_posix_local_filesystem(self) -> None:
        """Test as_posix method for local filesystem."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_fs.protocol = "file"
        path = "/test/path"
        fs_path = FSPath(mock_fs, path)

        result = fs_path.as_posix()
        assert result == path

    def test_as_posix_s3_filesystem(self) -> None:
        """Test as_posix method for S3 filesystem."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_fs.protocol = "s3"
        path = "bucket/key"
        fs_path = FSPath(mock_fs, path)

        result = fs_path.as_posix()
        assert result == "s3://bucket/key"

    def test_as_posix_gcs_filesystem(self) -> None:
        """Test as_posix method for GCS filesystem."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_fs.protocol = "gcs"
        path = "bucket/key"
        fs_path = FSPath(mock_fs, path)

        result = fs_path.as_posix()
        assert result == "gcs://bucket/key"

    def test_as_posix_multiple_protocols(self) -> None:
        """Test as_posix method for filesystem with multiple protocols."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_fs.protocol = ["s3", "s3a"]
        path = "bucket/key"
        fs_path = FSPath(mock_fs, path)

        result = fs_path.as_posix()
        assert result == "s3://bucket/key"

    def test_as_posix_no_protocol(self) -> None:
        """Test as_posix method for filesystem without protocol."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        # Remove protocol attribute entirely
        delattr(mock_fs, "protocol")
        path = "/test/path"
        fs_path = FSPath(mock_fs, path)

        result = fs_path.as_posix()
        assert result == path

    def test_get_bytes_cat_ranges_empty_file(self) -> None:
        """Test get_bytes_cat_ranges with empty file."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_fs.size.return_value = 0
        path = "/test/path"
        fs_path = FSPath(mock_fs, path)

        result = fs_path.get_bytes_cat_ranges()
        assert result == b""

    def test_get_bytes_cat_ranges_small_file(self) -> None:
        """Test get_bytes_cat_ranges with file smaller than part_size."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_fs.size.return_value = 1024  # 1KB
        # Mock data that matches the file size
        mock_data = [b"test data" + b"\x00" * (1024 - len(b"test data"))]
        mock_fs.cat_ranges.return_value = mock_data
        path = "/test/path"
        fs_path = FSPath(mock_fs, path)

        result = fs_path.get_bytes_cat_ranges(part_size=32 * 1024**2)  # 32MB

        # Should call cat_ranges once for the entire file
        mock_fs.cat_ranges.assert_called_once()
        assert result == b"test data" + b"\x00" * (1024 - len(b"test data"))

    def test_get_bytes_cat_ranges_large_file(self) -> None:
        """Test get_bytes_cat_ranges with file larger than part_size."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        file_size = 100 * 1024**2  # 100MB
        part_size = 32 * 1024**2  # 32MB
        mock_fs.size.return_value = file_size

        # Mock data for 4 parts (100MB / 32MB = 4 parts)
        # The method will create ranges: [0:32MB], [32MB:64MB], [64MB:96MB], [96MB:100MB]
        # So we need 4 blocks with sizes: 32MB, 32MB, 32MB, 4MB
        mock_data = [
            b"part1" * (32 * 1024**2 // 6),  # 32MB
            b"part2" * (32 * 1024**2 // 6),  # 32MB
            b"part3" * (32 * 1024**2 // 6),  # 32MB
            b"part4" * (4 * 1024**2 // 6),  # 4MB
        ]
        mock_fs.cat_ranges.return_value = mock_data

        path = "/test/path"
        fs_path = FSPath(mock_fs, path)

        result = fs_path.get_bytes_cat_ranges(part_size=part_size)

        # Should call cat_ranges with 4 ranges
        assert mock_fs.cat_ranges.call_count == 1
        call_args = mock_fs.cat_ranges.call_args
        assert len(call_args[0][1]) == 4  # 4 start positions
        assert len(call_args[0][2]) == 4  # 4 end positions

        # Result should be the expected file size
        assert len(result) == file_size

    def test_get_bytes_cat_ranges_custom_part_size(self) -> None:
        """Test get_bytes_cat_ranges with custom part_size."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        file_size = 1000  # 1000 bytes
        part_size = 100  # 100 bytes
        mock_fs.size.return_value = file_size

        # Mock data for 10 parts (1000 / 100 = 10 parts)
        mock_data = [b"x" * 100] * 10
        mock_fs.cat_ranges.return_value = mock_data

        path = "/test/path"
        fs_path = FSPath(mock_fs, path)

        result = fs_path.get_bytes_cat_ranges(part_size=part_size)

        # Should call cat_ranges with 10 ranges
        assert mock_fs.cat_ranges.call_count == 1
        call_args = mock_fs.cat_ranges.call_args
        assert len(call_args[0][1]) == 10  # 10 start positions
        assert len(call_args[0][2]) == 10  # 10 end positions

        # Result should be 1000 bytes
        assert len(result) == 1000

    def test_get_bytes_cat_ranges_cat_ranges_failure(self) -> None:
        """Test get_bytes_cat_ranges when cat_ranges fails."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_fs.size.return_value = 1024
        mock_fs.cat_ranges.side_effect = Exception("cat_ranges failed")
        path = "/test/path"
        fs_path = FSPath(mock_fs, path)

        with pytest.raises(Exception, match="cat_ranges failed"):
            fs_path.get_bytes_cat_ranges()

    def test_get_bytes_cat_ranges_size_failure(self) -> None:
        """Test get_bytes_cat_ranges when size() fails."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_fs.size.side_effect = Exception("size failed")
        path = "/test/path"
        fs_path = FSPath(mock_fs, path)

        with pytest.raises(Exception, match="size failed"):
            fs_path.get_bytes_cat_ranges()

    def test_get_bytes_cat_ranges_with_real_fsspec(self) -> None:
        """Test get_bytes_cat_ranges with a real fsspec filesystem."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            test_data = b"Hello, World! This is a test file with some content."
            tmp_file.write(test_data)
            tmp_file.flush()

            fs = fsspec.filesystem("file")
            fs_path = FSPath(fs, tmp_file.name)

            result = fs_path.get_bytes_cat_ranges(part_size=10)

            assert result == test_data

            # Clean up
            fs.rm(tmp_file.name)

    def test_fspath_integration_with_fsspec(self) -> None:
        """Test FSPath integration with real fsspec operations."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = f"{tmp_dir}/test.txt"
            test_content = "Hello, World!"

            # Create file using fsspec
            fs = fsspec.filesystem("file")
            with fs.open(test_file, "w") as f:
                f.write(test_content)

            # Use FSPath to read the file
            fs_path = FSPath(fs, test_file)

            # Test open method
            with fs_path.open("r") as f:
                content = f.read()
                assert content == test_content

            # Test as_posix
            posix_path = fs_path.as_posix()
            # Local filesystem will have "file://" prefix
            assert posix_path == f"file://{test_file}"

            # Test get_bytes_cat_ranges
            bytes_content = fs_path.get_bytes_cat_ranges()
            assert bytes_content == test_content.encode()

    def test_fspath_with_s3_like_protocol(self) -> None:
        """Test FSPath with S3-like protocol simulation."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_fs.protocol = "s3"
        path = "my-bucket/path/to/file.txt"
        fs_path = FSPath(mock_fs, path)

        # Test as_posix with S3 protocol
        result = fs_path.as_posix()
        assert result == "s3://my-bucket/path/to/file.txt"

        # Test string representation
        assert str(fs_path) == path

        # Test repr
        assert repr(fs_path) == f"FSPath({path})"

    def test_fspath_edge_cases(self) -> None:
        """Test FSPath with edge cases."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)

        # Test with empty path
        fs_path = FSPath(mock_fs, "")
        assert str(fs_path) == ""
        assert repr(fs_path) == "FSPath()"

        # Test with None-like path (though this shouldn't happen in practice)
        fs_path = FSPath(mock_fs, "None")
        assert str(fs_path) == "None"

        # Test with very long path
        long_path = "/" + "a" * 1000
        fs_path = FSPath(mock_fs, long_path)
        assert str(fs_path) == long_path
        assert len(str(fs_path)) == 1001

    def test_is_remote_url_local_file(self) -> None:
        """Test is_remote_url with local file path."""
        from nemo_curator.utils.client_utils import is_remote_url

        # Test with absolute local path
        assert not is_remote_url("/home/user/file.txt")

        # Test with relative local path
        assert not is_remote_url("./file.txt")

        # Test with file:// protocol (local)
        assert not is_remote_url("file:///home/user/file.txt")

    def test_is_remote_url_with_mock_fsspec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test is_remote_url with mocked fsspec to test protocol handling."""
        from nemo_curator.utils.client_utils import is_remote_url

        # Mock the url_to_fs function
        def mock_url_to_fs(url: str) -> tuple:
            class MockFS:
                def __init__(self, protocol: str) -> None:
                    self.protocol = protocol

            # Return different protocols based on URL
            if "s3" in url:
                return MockFS("s3"), ""
            elif "gcs" in url:
                return MockFS("gcs"), ""
            elif "file" in url or "/" in url:
                return MockFS("file"), ""
            elif "http" in url:
                return MockFS("http"), ""
            else:
                return MockFS(None), ""

        monkeypatch.setattr("nemo_curator.utils.client_utils.url_to_fs", mock_url_to_fs)

        # Test with mocked protocols
        assert is_remote_url("s3://bucket/key")
        assert is_remote_url("gcs://bucket/key")
        assert not is_remote_url("file:///path")
        assert not is_remote_url("/local/path")
        assert not is_remote_url("unknown://path")

    def test_is_remote_url_multiple_protocols(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test is_remote_url with filesystem that has multiple protocols."""
        from nemo_curator.utils.client_utils import is_remote_url

        def mock_url_to_fs(_url: str) -> tuple:
            class MockFS:
                def __init__(self) -> None:
                    # Simulate filesystem with multiple protocols
                    self.protocol = ["s3", "s3a"]

            return MockFS(), ""

        monkeypatch.setattr("nemo_curator.utils.client_utils.url_to_fs", mock_url_to_fs)

        # Should return True since first protocol is "s3" (remote)
        assert is_remote_url("s3://bucket/key")

    def test_is_remote_url_no_protocol(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test is_remote_url with filesystem that has no protocol."""
        from nemo_curator.utils.client_utils import is_remote_url

        def mock_url_to_fs(_url: str) -> tuple:
            class MockFS:
                def __init__(self) -> None:
                    # Simulate filesystem with no protocol
                    self.protocol = None

            return MockFS(), ""

        monkeypatch.setattr("nemo_curator.utils.client_utils.url_to_fs", mock_url_to_fs)

        # Should return False since protocol is None
        assert not is_remote_url("unknown://path")
