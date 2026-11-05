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

from __future__ import annotations

import io
import tarfile
from typing import TYPE_CHECKING

import fsspec
import pytest

from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.utils.file_utils import (
    get_all_file_paths_and_size_under,
    get_all_file_paths_under,
    infer_dataset_name_from_path,
    parse_bytes_string_to_int,
    read_json_file,
    tar_safe_extract,
    write_json_file,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestJsonFiles:
    def test_local_file(self, tmp_path: Path):
        path = tmp_path / "nested" / "record.json"
        fs = fsspec.filesystem("file")

        write_json_file(str(path), {"b": 2, "a": 1}, fs)

        assert path.read_text() == '{"a": 1, "b": 2}\n'
        assert read_json_file(str(path), fs) == {"a": 1, "b": 2}

    def test_fsspec_file(self, tmp_path: Path):
        path = f"/{tmp_path.name}/nested/record.json"
        fs = fsspec.filesystem("memory")

        write_json_file(path, {"value": 1}, fs)

        assert read_json_file(path, fs) == {"value": 1}


class TestInferDatasetNameFromPath:
    """Test cases for infer_dataset_name_from_path function."""

    def test_local_paths(self):
        """Test local path dataset name inference."""
        assert infer_dataset_name_from_path("/home/user/my_dataset/file.txt") == "my_dataset"
        assert infer_dataset_name_from_path("./file.txt") == "file"
        assert infer_dataset_name_from_path("file.txt") == "file"

    def test_cloud_paths(self):
        """Test cloud storage path dataset name inference."""
        assert infer_dataset_name_from_path("s3://bucket/datasets/my_dataset/") == "my_dataset"
        assert infer_dataset_name_from_path("s3://bucket/datasets/my_dataset/data.parquet") == "data.parquet"
        assert infer_dataset_name_from_path("s3://my-bucket") == "my-bucket"
        assert infer_dataset_name_from_path("abfs://container@account.dfs.core.windows.net/dataset") == "dataset"

    def test_directory_paths(self):
        """Test local and cloud directory dataset name inference."""
        assert infer_dataset_name_from_path("/home/user/MyDataSet/", path_kind="directory") == "mydataset"
        assert infer_dataset_name_from_path("s3://bucket/MyDataSet/", path_kind="directory") == "mydataset"

    def test_case_conversion(self):
        """Test that results are converted to lowercase."""
        assert infer_dataset_name_from_path("s3://bucket/MyDataSet") == "mydataset"
        assert infer_dataset_name_from_path("/home/user/MyDataSet/file.txt") == "mydataset"


def _write_test_file(path: Path, content: str = "test", size_bytes: int | None = None) -> None:
    """Helper to create test files with specific content or size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if size_bytes is not None:
        path.write_bytes(b"x" * size_bytes)
    else:
        path.write_text(content)


class TestGetAllFilePathsUnder:
    """Test cases for get_all_file_paths_under function."""

    def test_recursion_with_filtering(self, tmp_path: Path):
        """Test recursive directory traversal with extension filtering."""
        # Create directory structure with mixed file types
        root = tmp_path / "data"
        files_to_create = [
            (root / "file1.jsonl", "{}"),
            (root / "file2.json", "{}"),
            (root / "file3.parquet", "{}"),
            (root / "file4.txt", "text"),
            (root / "subdir" / "file5.jsonl", "{}"),
            (root / "subdir" / "nested" / "file6.json", "{}"),
            (root / "subdir" / "file7.txt", "text"),
        ]

        for file_path, content in files_to_create:
            _write_test_file(file_path, content)

        # Test with recursion and filtering
        result = get_all_file_paths_under(
            str(root),
            recurse_subdirectories=True,
            keep_extensions=[".jsonl", ".json"],
        )

        expected_files = [
            str(root / "file1.jsonl"),
            str(root / "file2.json"),
            str(root / "subdir" / "file5.jsonl"),
            str(root / "subdir" / "nested" / "file6.json"),
        ]
        assert sorted(result) == sorted(expected_files)

    def test_no_recursion(self, tmp_path: Path):
        """Test directory traversal without recursion."""
        root = tmp_path / "data"
        files_to_create = [
            (root / "file1.jsonl", "{}"),
            (root / "file2.txt", "text"),
            (root / "subdir" / "file3.jsonl", "{}"),
        ]

        for file_path, content in files_to_create:
            _write_test_file(file_path, content)

        result = get_all_file_paths_under(
            str(root),
            recurse_subdirectories=False,
            keep_extensions=[".jsonl"],
        )

        # Should only include top-level jsonl files
        assert result == [str(root / "file1.jsonl")]

    def test_single_file_input(self, tmp_path: Path):
        """Test with a single file as input."""
        test_file = tmp_path / "single.jsonl"
        _write_test_file(test_file, "{}")

        result = get_all_file_paths_under(
            str(test_file),
            recurse_subdirectories=False,
            keep_extensions=[".jsonl"],
        )

        assert result == [str(test_file)]

    def test_no_extension_filtering(self, tmp_path: Path):
        """Test without extension filtering."""
        root = tmp_path / "data"
        files_to_create = [
            (root / "file1.jsonl", "{}"),
            (root / "file2.txt", "text"),
            (root / "file_no_ext", "content"),
        ]

        for file_path, content in files_to_create:
            _write_test_file(file_path, content)

        result = get_all_file_paths_under(
            str(root),
            recurse_subdirectories=False,
            keep_extensions=None,
        )

        expected_files = [str(f[0]) for f in files_to_create]
        assert sorted(result) == sorted(expected_files)

    def test_empty_directory(self, tmp_path: Path):
        """Test with empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = get_all_file_paths_under(str(empty_dir))
        assert result == []


class TestGetAllFilePathsAndSizeUnder:
    """Test cases for get_all_file_paths_and_size_under function."""

    def test_files_with_sizes_sorted(self, tmp_path: Path):
        """Test that files are returned with sizes and sorted by size."""
        root = tmp_path / "sized_files"

        # Create files with different sizes
        files_with_sizes = [
            (root / "large.jsonl", 100),
            (root / "small.jsonl", 10),
            (root / "medium.jsonl", 50),
        ]

        for file_path, size in files_with_sizes:
            _write_test_file(file_path, size_bytes=size)

        result = get_all_file_paths_and_size_under(
            str(root),
            recurse_subdirectories=False,
            keep_extensions=[".jsonl"],
        )

        # Should be sorted by size (ascending)
        expected = [
            (str(root / "small.jsonl"), 10),
            (str(root / "medium.jsonl"), 50),
            (str(root / "large.jsonl"), 100),
        ]
        assert result == expected

    def test_files_with_paths_sorted(self, tmp_path: Path):
        """Test that files are returned with sizes and sorted by path."""
        root = tmp_path / "sized_files"

        # Create files with different sizes
        files_with_sizes = [
            (root / "large.jsonl", 100),
            (root / "small.jsonl", 10),
            (root / "medium.jsonl", 50),
        ]

        for file_path, size in files_with_sizes:
            _write_test_file(file_path, size_bytes=size)

        result = get_all_file_paths_and_size_under(
            str(root),
            recurse_subdirectories=False,
            keep_extensions=[".jsonl"],
            sort_by_size=False,
        )

        # Should be sorted by path (alphabetical)
        expected = [
            (str(root / "large.jsonl"), 100),
            (str(root / "medium.jsonl"), 50),
            (str(root / "small.jsonl"), 10),
        ]
        assert result == expected

    def test_with_recursion_and_filtering(self, tmp_path: Path):
        """Test recursive traversal with extension filtering and size info."""
        root = tmp_path / "nested_sized"

        files_to_create = [
            (root / "file1.jsonl", 20),
            (root / "file2.txt", 30),  # Should be filtered out
            (root / "subdir" / "file3.jsonl", 15),
            (root / "subdir" / "file4.parquet", 25),
        ]

        for file_path, size in files_to_create:
            _write_test_file(file_path, size_bytes=size)

        result = get_all_file_paths_and_size_under(
            str(root),
            recurse_subdirectories=True,
            keep_extensions=[".jsonl", ".parquet"],
        )

        expected = [
            (str(root / "subdir" / "file3.jsonl"), 15),
            (str(root / "file1.jsonl"), 20),
            (str(root / "subdir" / "file4.parquet"), 25),
        ]
        assert result == expected

    def test_single_file_with_size(self, tmp_path: Path):
        """Test single file input returns correct size."""
        test_file = tmp_path / "single.jsonl"
        _write_test_file(test_file, size_bytes=42)

        result = get_all_file_paths_and_size_under(str(test_file))
        assert result == [(str(test_file), 42)]


class TestFilePartitioningStageGetters:
    """Test cases for FilePartitioningStage private methods."""

    def test_get_file_list_with_sizes_directory_sort_by_size(self, tmp_path: Path):
        """Test _get_file_list_with_sizes with directory input."""
        root = tmp_path / "sized_stage_test"

        files_with_sizes = [
            (root / "large.jsonl", 80),
            (root / "small.json", 20),
            (root / "medium.parquet", 50),
            (root / "nested" / "tiny.jsonl", 5),
        ]

        for file_path, size in files_with_sizes:
            _write_test_file(file_path, size_bytes=size)

        stage = FilePartitioningStage(file_paths=str(root))
        # Default is to sort by size
        result = stage._get_file_list_with_sizes()

        # Should be sorted by size (ascending)
        expected = [
            (str(root / "nested" / "tiny.jsonl"), 5),
            (str(root / "small.json"), 20),
            (str(root / "medium.parquet"), 50),
            (str(root / "large.jsonl"), 80),
        ]
        assert result == expected

    def test_get_file_list_with_sizes_directory_sort_by_path(self, tmp_path: Path):
        """Test _get_file_list_with_sizes with directory input and sort_by_size=False."""
        root = tmp_path / "sized_stage_test"

        files_with_sizes = [
            (root / "large.jsonl", 80),
            (root / "small.json", 20),
            (root / "medium.parquet", 50),
            (root / "nested" / "tiny.jsonl", 5),
        ]

        for file_path, size in files_with_sizes:
            _write_test_file(file_path, size_bytes=size)

        stage = FilePartitioningStage(file_paths=str(root))
        result = stage._get_file_list_with_sizes(sort_by_size=False)

        # Should be sorted by path (alphabetical)
        expected = [
            (str(root / "large.jsonl"), 80),
            (str(root / "medium.parquet"), 50),
            (str(root / "nested" / "tiny.jsonl"), 5),
            (str(root / "small.json"), 20),
        ]
        assert result == expected

    def test_get_file_list_invalid_input(self):
        """Test that invalid input types raise TypeError."""
        stage = FilePartitioningStage(file_paths=123)  # Invalid type

        with pytest.raises(TypeError, match="Invalid file paths"):
            stage._get_file_list_with_sizes()

    def test_parse_bytes_string_to_int(self):
        """Test parse_bytes_string_to_int method."""
        assert parse_bytes_string_to_int("100B") == 100
        assert parse_bytes_string_to_int("1KB") == 1000
        assert parse_bytes_string_to_int("1KiB") == 1024
        assert parse_bytes_string_to_int("1MB") == 1000 * 1000
        assert parse_bytes_string_to_int("1MiB") == 1024 * 1024
        assert parse_bytes_string_to_int("1GB") == 1000 * 1000 * 1000
        assert parse_bytes_string_to_int("1GiB") == 1024 * 1024 * 1024
        assert parse_bytes_string_to_int("2TB") == 2 * 1000 * 1000 * 1000 * 1000
        assert parse_bytes_string_to_int("2TiB") == 2 * 1024 * 1024 * 1024 * 1024
        assert parse_bytes_string_to_int("100") == 100  # No unit defaults to bytes


def _make_tar_bytes(members: list[dict]) -> bytes:
    """Build an in-memory tar archive from a list of member specs.

    Each spec is a dict with keys:
        name: archive path
        kind: "file", "symlink", or "hardlink"
        target: link target (for symlink/hardlink)
        data: file content bytes (for file)
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for m in members:
            info = tarfile.TarInfo(name=m["name"])
            if m["kind"] == "file":
                data = m.get("data", b"")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
            elif m["kind"] == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = m["target"]
                tf.addfile(info)
            elif m["kind"] == "hardlink":
                info.type = tarfile.LNKTYPE
                info.linkname = m["target"]
                tf.addfile(info)
    return buf.getvalue()


class TestTarSafeExtract:
    """Regression tests for symlink/hardlink path-traversal checks."""

    def test_safe_symlink_in_subdirectory(self, tmp_path: Path):
        """A symlink whose target resolves inside the extraction root is allowed."""
        data = _make_tar_bytes(
            [
                {"name": "subdir/data.txt", "kind": "file", "data": b"hello"},
                {"name": "subdir/link.txt", "kind": "symlink", "target": "data.txt"},
            ]
        )
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            tar_safe_extract(tf, str(tmp_path))
        assert (tmp_path / "subdir" / "link.txt").is_symlink()
        assert (tmp_path / "subdir" / "link.txt").read_text() == "hello"

    def test_symlink_escape_via_parent_traversal(self, tmp_path: Path):
        """A symlink that escapes the extraction root via ../ must be rejected."""
        data = _make_tar_bytes(
            [
                {"name": "subdir/evil.txt", "kind": "symlink", "target": "../../etc/passwd"},
            ]
        )
        with tarfile.open(fileobj=io.BytesIO(data)) as tf, pytest.raises(ValueError, match="Symlink target outside"):
            tar_safe_extract(tf, str(tmp_path))

    def test_symlink_shallow_escape(self, tmp_path: Path):
        """A top-level symlink pointing to ../ must be rejected."""
        data = _make_tar_bytes(
            [
                {"name": "escape.txt", "kind": "symlink", "target": "../secret"},
            ]
        )
        with tarfile.open(fileobj=io.BytesIO(data)) as tf, pytest.raises(ValueError, match="Symlink target outside"):
            tar_safe_extract(tf, str(tmp_path))

    def test_deep_nested_safe_symlink(self, tmp_path: Path):
        """A symlink in a deep path that stays inside the root is allowed."""
        data = _make_tar_bytes(
            [
                {"name": "a/b/c/real.txt", "kind": "file", "data": b"deep"},
                {"name": "a/b/c/link.txt", "kind": "symlink", "target": "../../../a/b/c/real.txt"},
            ]
        )
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            tar_safe_extract(tf, str(tmp_path))
        assert (tmp_path / "a" / "b" / "c" / "link.txt").read_text() == "deep"

    def test_absolute_symlink_target_rejected(self, tmp_path: Path):
        """A symlink with an absolute target path must be rejected."""
        data = _make_tar_bytes(
            [
                {"name": "abs.txt", "kind": "symlink", "target": "/etc/passwd"},
            ]
        )
        with tarfile.open(fileobj=io.BytesIO(data)) as tf, pytest.raises(ValueError, match="Absolute symlink target"):
            tar_safe_extract(tf, str(tmp_path))

    def test_safe_hardlink(self, tmp_path: Path):
        """A hard link whose target is inside the archive root is allowed."""
        data = _make_tar_bytes(
            [
                {"name": "original.txt", "kind": "file", "data": b"content"},
                {"name": "link.txt", "kind": "hardlink", "target": "original.txt"},
            ]
        )
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            tar_safe_extract(tf, str(tmp_path))
        assert (tmp_path / "link.txt").read_text() == "content"

    def test_hardlink_escape_rejected(self, tmp_path: Path):
        """A hard link targeting outside the archive root must be rejected."""
        data = _make_tar_bytes(
            [
                {"name": "evil.txt", "kind": "hardlink", "target": "../outside.txt"},
            ]
        )
        with tarfile.open(fileobj=io.BytesIO(data)) as tf, pytest.raises(ValueError, match="Hard link target outside"):
            tar_safe_extract(tf, str(tmp_path))

    def test_chained_symlink_escape_rejected(self, tmp_path: Path):
        """A symlink inside an extracted symlinked directory must not escape the root."""
        data = _make_tar_bytes(
            [
                {"name": "a", "kind": "symlink", "target": "."},
                {"name": "a/link", "kind": "symlink", "target": "../secret"},
            ]
        )
        with tarfile.open(fileobj=io.BytesIO(data)) as tf, pytest.raises(ValueError, match="Symlink target outside"):
            tar_safe_extract(tf, str(tmp_path))

    def test_member_path_through_symlink_rejected(self, tmp_path: Path):
        """A member path that leaves the root through an extracted symlink must be rejected."""
        out = tmp_path / "out"
        data = _make_tar_bytes(
            [
                {"name": "a", "kind": "symlink", "target": "."},
                {"name": "a/../escaped.txt", "kind": "file", "data": b"x"},
            ]
        )
        with tarfile.open(fileobj=io.BytesIO(data)) as tf, pytest.raises(ValueError, match="Path traversal attempt"):
            tar_safe_extract(tf, str(out))
        assert not (tmp_path / "escaped.txt").exists()
