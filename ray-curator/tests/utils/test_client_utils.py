from unittest.mock import Mock

import fsspec
import pytest

from ray_curator.utils.client_utils import FSPath


class TestFSPath:
    """Test cases for FSPath class."""

    def test_init(self):
        """Test FSPath initialization."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        path = "/test/path"
        fs_path = FSPath(mock_fs, path)

        assert fs_path._fs == mock_fs
        assert fs_path._path == path

    def test_open(self):
        """Test FSPath.open method."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_file = Mock(spec=fsspec.spec.AbstractBufferedFile)
        mock_fs.open.return_value = mock_file

        path = "/test/path"
        fs_path = FSPath(mock_fs, path)

        # Test default mode
        result = fs_path.open()
        mock_fs.open.assert_called_once_with(path, "rb")
        assert result == mock_file

        # Test custom mode
        result = fs_path.open("w")
        mock_fs.open.assert_called_with(path, "w")
        assert result == mock_file

        # Test with kwargs
        result = fs_path.open("r", encoding="utf-8")
        mock_fs.open.assert_called_with(path, "r", encoding="utf-8")
        assert result == mock_file

    def test_str(self):
        """Test FSPath string representation."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        path = "/test/path"
        fs_path = FSPath(mock_fs, path)

        assert str(fs_path) == path

    def test_repr(self):
        """Test FSPath repr representation."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        path = "/test/path"
        fs_path = FSPath(mock_fs, path)

        assert repr(fs_path) == f"FSPath({path})"

    def test_get_bytes_cat_ranges_empty_file(self):
        """Test get_bytes_cat_ranges with empty file."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_fs.size.return_value = 0

        path = "/test/path"
        fs_path = FSPath(mock_fs, path)

        result = fs_path.get_bytes_cat_ranges()
        assert result == b""
        mock_fs.size.assert_called_once_with(path)

    def test_get_bytes_cat_ranges_single_part(self):
        """Test get_bytes_cat_ranges with file smaller than part size."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_fs.size.return_value = 1024  # 1KB

        # Mock cat_ranges to return a single block
        mock_fs.cat_ranges.return_value = [b"test data"]

        path = "/test/path"
        fs_path = FSPath(mock_fs, path)

        result = fs_path.get_bytes_cat_ranges(part_size=32 * 1024**2)  # 32MB

        # Should only have one range
        mock_fs.cat_ranges.assert_called_once()
        call_args = mock_fs.cat_ranges.call_args
        assert call_args[0][0] == [path]  # paths
        assert call_args[0][1] == [0]     # starts
        assert call_args[0][2] == [1024]  # ends

        assert result == b"test data"

    def test_get_bytes_cat_ranges_multiple_parts(self):
        """Test get_bytes_cat_ranges with file larger than part size."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        file_size = 100 * 1024**2  # 100MB
        part_size = 32 * 1024**2   # 32MB
        mock_fs.size.return_value = file_size

        # Mock cat_ranges to return multiple blocks
        mock_fs.cat_ranges.return_value = [
            b"part1" * (16 * 1024**2),  # 16MB
            b"part2" * (16 * 1024**2),  # 16MB
            b"part3" * (16 * 1024**2),  # 16MB
            b"part4" * (16 * 1024**2),  # 16MB
        ]

        path = "/test/path"
        fs_path = FSPath(mock_fs, path)

        result = fs_path.get_bytes_cat_ranges(part_size=part_size)

        # Should have 4 ranges (100MB / 32MB = 4 parts)
        mock_fs.cat_ranges.assert_called_once()
        call_args = mock_fs.cat_ranges.call_args
        assert len(call_args[0][0]) == 4  # 4 paths
        assert len(call_args[0][1]) == 4  # 4 starts
        assert len(call_args[0][2]) == 4  # 4 ends

        # Check the ranges
        expected_starts = [0, 32 * 1024**2, 64 * 1024**2, 96 * 1024**2]
        expected_ends = [32 * 1024**2, 64 * 1024**2, 96 * 1024**2, 100 * 1024**2]

        assert call_args[0][1] == expected_starts
        assert call_args[0][2] == expected_ends

        # Result should be concatenated
        assert len(result) == file_size

    def test_get_bytes_cat_ranges_custom_part_size(self):
        """Test get_bytes_cat_ranges with custom part size."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        file_size = 1000
        part_size = 100
        mock_fs.size.return_value = file_size

        # Mock cat_ranges to return blocks
        mock_fs.cat_ranges.return_value = [
            b"a" * 100,  # 100 bytes
            b"b" * 100,  # 100 bytes
            b"c" * 100,  # 100 bytes
            b"d" * 100,  # 100 bytes
            b"e" * 100,  # 100 bytes
            b"f" * 100,  # 100 bytes
            b"g" * 100,  # 100 bytes
            b"h" * 100,  # 100 bytes
            b"i" * 100,  # 100 bytes
            b"j" * 100,  # 100 bytes
        ]

        path = "/test/path"
        fs_path = FSPath(mock_fs, path)

        result = fs_path.get_bytes_cat_ranges(part_size=part_size)

        # Should have 10 ranges (1000 / 100 = 10 parts)
        mock_fs.cat_ranges.assert_called_once()
        call_args = mock_fs.cat_ranges.call_args
        assert len(call_args[0][0]) == 10

        # Check the ranges
        expected_starts = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
        expected_ends = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

        assert call_args[0][1] == expected_starts
        assert call_args[0][2] == expected_ends

        # Result should be concatenated
        assert len(result) == file_size

    def test_get_bytes_cat_ranges_cat_ranges_error(self):
        """Test get_bytes_cat_ranges when cat_ranges raises an error."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_fs.size.return_value = 1024
        mock_fs.cat_ranges.side_effect = Exception("cat_ranges failed")

        path = "/test/path"
        fs_path = FSPath(mock_fs, path)

        with pytest.raises(Exception, match="cat_ranges failed"):
            fs_path.get_bytes_cat_ranges()

    def test_get_bytes_cat_ranges_with_raise_on_error(self):
        """Test get_bytes_cat_ranges with on_error='raise' parameter."""
        mock_fs = Mock(spec=fsspec.AbstractFileSystem)
        mock_fs.size.return_value = 1024
        mock_fs.cat_ranges.return_value = [b"test data"]

        path = "/test/path"
        fs_path = FSPath(mock_fs, path)

        result = fs_path.get_bytes_cat_ranges()

        # Verify on_error="raise" is passed to cat_ranges
        mock_fs.cat_ranges.assert_called_once()
        call_args = mock_fs.cat_ranges.call_args
        assert call_args[1]["on_error"] == "raise"

        assert result == b"test data"
