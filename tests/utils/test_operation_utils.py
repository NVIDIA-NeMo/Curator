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

import pathlib
import tempfile
from unittest.mock import Mock, patch

from nemo_curator.utils.operation_utils import (
    get_tmp_dir,
    make_named_temporary_file,
    make_pipeline_named_temporary_file,
    make_pipeline_temporary_dir,
    make_temporary_dir,
)


class TestGetTmpDir:
    """Test suite for get_tmp_dir function."""

    @patch("tempfile.gettempdir")
    def test_get_tmp_dir_success(self, mock_gettempdir: Mock) -> None:
        """Test successful temporary directory retrieval."""
        mock_gettempdir.return_value = "/tmp"  # noqa: S108

        result = get_tmp_dir()

        assert result == pathlib.Path("/tmp")  # noqa: S108
        mock_gettempdir.assert_called_once()

    @patch("tempfile.gettempdir")
    def test_get_tmp_dir_different_paths(self, mock_gettempdir: Mock) -> None:
        """Test temporary directory retrieval with different paths."""
        test_paths = ["/tmp", "/var/tmp", "C:\\temp"]  # noqa: S108

        for test_path in test_paths:
            mock_gettempdir.return_value = test_path
            result = get_tmp_dir()
            assert result == pathlib.Path(test_path)


class TestMakeTemporaryDir:
    """Test suite for make_temporary_dir context manager."""

    def test_make_temporary_dir_default_params(self) -> None:
        """Test temporary directory creation with default parameters."""
        with make_temporary_dir() as tmp_dir:
            assert isinstance(tmp_dir, pathlib.Path)
            assert tmp_dir.exists()
            assert tmp_dir.is_dir()

        # Directory should be deleted after context exit
        assert not tmp_dir.exists()

    def test_make_temporary_dir_with_prefix(self) -> None:
        """Test temporary directory creation with custom prefix."""
        prefix = "test_prefix_"

        with make_temporary_dir(prefix=prefix) as tmp_dir:
            assert isinstance(tmp_dir, pathlib.Path)
            assert tmp_dir.exists()
            assert tmp_dir.is_dir()
            assert prefix in tmp_dir.name

        assert not tmp_dir.exists()

    def test_make_temporary_dir_with_target_dir(self) -> None:
        """Test temporary directory creation with custom target directory."""
        with tempfile.TemporaryDirectory() as parent_dir:
            target_dir = pathlib.Path(parent_dir)

            with make_temporary_dir(target_dir=target_dir) as tmp_dir:
                assert isinstance(tmp_dir, pathlib.Path)
                assert tmp_dir.exists()
                assert tmp_dir.is_dir()
                assert tmp_dir.parent == target_dir

            assert not tmp_dir.exists()

    def test_make_temporary_dir_no_delete(self) -> None:
        """Test temporary directory creation without deletion."""
        with make_temporary_dir(delete=False) as tmp_dir:
            assert isinstance(tmp_dir, pathlib.Path)
            assert tmp_dir.exists()
            assert tmp_dir.is_dir()

        # Directory should still exist after context exit
        assert tmp_dir.exists()

        # Clean up manually
        tmp_dir.rmdir()

    def test_make_temporary_dir_with_all_params(self) -> None:
        """Test temporary directory creation with all parameters."""
        with tempfile.TemporaryDirectory() as parent_dir:
            target_dir = pathlib.Path(parent_dir)
            prefix = "full_test_"

            with make_temporary_dir(prefix=prefix, target_dir=target_dir, delete=True) as tmp_dir:
                assert isinstance(tmp_dir, pathlib.Path)
                assert tmp_dir.exists()
                assert tmp_dir.is_dir()
                assert tmp_dir.parent == target_dir
                assert prefix in tmp_dir.name

            assert not tmp_dir.exists()


class TestMakeNamedTemporaryFile:
    """Test suite for make_named_temporary_file context manager."""

    def test_make_named_temporary_file_default_params(self) -> None:
        """Test named temporary file creation with default parameters."""
        with make_named_temporary_file() as tmp_file:
            assert isinstance(tmp_file, pathlib.Path)
            assert tmp_file.exists()
            assert tmp_file.is_file()

        # File should be deleted after context exit
        assert not tmp_file.exists()

    def test_make_named_temporary_file_with_prefix(self) -> None:
        """Test named temporary file creation with custom prefix."""
        prefix = "test_prefix_"

        with make_named_temporary_file(prefix=prefix) as tmp_file:
            assert isinstance(tmp_file, pathlib.Path)
            assert tmp_file.exists()
            assert tmp_file.is_file()
            assert prefix in tmp_file.name

        assert not tmp_file.exists()

    def test_make_named_temporary_file_with_suffix(self) -> None:
        """Test named temporary file creation with custom suffix."""
        suffix = ".test"

        with make_named_temporary_file(suffix=suffix) as tmp_file:
            assert isinstance(tmp_file, pathlib.Path)
            assert tmp_file.exists()
            assert tmp_file.is_file()
            assert tmp_file.name.endswith(suffix)

        assert not tmp_file.exists()

    def test_make_named_temporary_file_with_target_dir(self) -> None:
        """Test named temporary file creation with custom target directory."""
        with tempfile.TemporaryDirectory() as parent_dir:
            target_dir = pathlib.Path(parent_dir)

            with make_named_temporary_file(target_dir=target_dir) as tmp_file:
                assert isinstance(tmp_file, pathlib.Path)
                assert tmp_file.exists()
                assert tmp_file.is_file()
                assert tmp_file.parent == target_dir

            assert not tmp_file.exists()

    def test_make_named_temporary_file_no_delete(self) -> None:
        """Test named temporary file creation without deletion."""
        with make_named_temporary_file(delete=False) as tmp_file:
            assert isinstance(tmp_file, pathlib.Path)
            assert tmp_file.exists()
            assert tmp_file.is_file()

        # File should still exist after context exit
        assert tmp_file.exists()

        # Clean up manually
        tmp_file.unlink()

    def test_make_named_temporary_file_with_all_params(self) -> None:
        """Test named temporary file creation with all parameters."""
        with tempfile.TemporaryDirectory() as parent_dir:
            target_dir = pathlib.Path(parent_dir)
            prefix = "full_test_"
            suffix = ".test"

            with make_named_temporary_file(
                prefix=prefix, suffix=suffix, delete=True, target_dir=target_dir
            ) as tmp_file:
                assert isinstance(tmp_file, pathlib.Path)
                assert tmp_file.exists()
                assert tmp_file.is_file()
                assert tmp_file.parent == target_dir
                assert prefix in tmp_file.name
                assert tmp_file.name.endswith(suffix)

            assert not tmp_file.exists()


class TestMakePipelineTemporaryDir:
    """Test suite for make_pipeline_temporary_dir context manager."""

    @patch("nemo_curator.utils.operation_utils.get_tmp_dir")
    def test_make_pipeline_temporary_dir_default_params(self, mock_get_tmp_dir: Mock) -> None:
        """Test pipeline temporary directory creation with default parameters."""
        mock_tmp_dir = pathlib.Path("/tmp")  # noqa: S108
        mock_get_tmp_dir.return_value = mock_tmp_dir

        with make_pipeline_temporary_dir() as tmp_dir:
            assert isinstance(tmp_dir, pathlib.Path)
            assert tmp_dir.exists()
            assert tmp_dir.is_dir()
            # Check that the path contains ray_pipeline
            assert "ray_pipeline" in str(tmp_dir)

        assert not tmp_dir.exists()

    @patch("nemo_curator.utils.operation_utils.get_tmp_dir")
    def test_make_pipeline_temporary_dir_with_sub_dir(self, mock_get_tmp_dir: Mock) -> None:
        """Test pipeline temporary directory creation with sub directory."""
        mock_tmp_dir = pathlib.Path("/tmp")  # noqa: S108
        mock_get_tmp_dir.return_value = mock_tmp_dir
        sub_dir = "test_sub_dir"

        with make_pipeline_temporary_dir(sub_dir=sub_dir) as tmp_dir:
            assert isinstance(tmp_dir, pathlib.Path)
            assert tmp_dir.exists()
            assert tmp_dir.is_dir()
            # Check that the path contains both ray_pipeline and sub_dir
            assert "ray_pipeline" in str(tmp_dir)
            assert sub_dir in str(tmp_dir)

        assert not tmp_dir.exists()

    @patch("nemo_curator.utils.operation_utils.get_tmp_dir")
    def test_make_pipeline_temporary_dir_creates_parent_dirs(self, mock_get_tmp_dir: Mock) -> None:
        """Test that pipeline temporary directory creates parent directories."""
        with tempfile.TemporaryDirectory() as base_dir:
            mock_tmp_dir = pathlib.Path(base_dir)
            mock_get_tmp_dir.return_value = mock_tmp_dir

            with make_pipeline_temporary_dir(sub_dir="nested/sub/dir") as tmp_dir:
                assert isinstance(tmp_dir, pathlib.Path)
                assert tmp_dir.exists()
                assert tmp_dir.is_dir()
                # Check that parent directories were created
                pipeline_dir = mock_tmp_dir / "ray_pipeline" / "nested" / "sub" / "dir"
                assert pipeline_dir.exists()

            assert not tmp_dir.exists()


class TestMakePipelineNamedTemporaryFile:
    """Test suite for make_pipeline_named_temporary_file context manager."""

    @patch("nemo_curator.utils.operation_utils.get_tmp_dir")
    def test_make_pipeline_named_temporary_file_default_params(self, mock_get_tmp_dir: Mock) -> None:
        """Test pipeline named temporary file creation with default parameters."""
        mock_tmp_dir = pathlib.Path("/tmp")  # noqa: S108
        mock_get_tmp_dir.return_value = mock_tmp_dir

        with make_pipeline_named_temporary_file() as tmp_file:
            assert isinstance(tmp_file, pathlib.Path)
            assert tmp_file.exists()
            assert tmp_file.is_file()
            # Check that the path contains ray_pipeline
            assert "ray_pipeline" in str(tmp_file)

        assert not tmp_file.exists()

    @patch("nemo_curator.utils.operation_utils.get_tmp_dir")
    def test_make_pipeline_named_temporary_file_with_sub_dir(self, mock_get_tmp_dir: Mock) -> None:
        """Test pipeline named temporary file creation with sub directory."""
        mock_tmp_dir = pathlib.Path("/tmp")  # noqa: S108
        mock_get_tmp_dir.return_value = mock_tmp_dir
        sub_dir = "test_sub_dir"

        with make_pipeline_named_temporary_file(sub_dir=sub_dir) as tmp_file:
            assert isinstance(tmp_file, pathlib.Path)
            assert tmp_file.exists()
            assert tmp_file.is_file()
            # Check that the path contains both ray_pipeline and sub_dir
            assert "ray_pipeline" in str(tmp_file)
            assert sub_dir in str(tmp_file)

        assert not tmp_file.exists()

    @patch("nemo_curator.utils.operation_utils.get_tmp_dir")
    def test_make_pipeline_named_temporary_file_with_suffix(self, mock_get_tmp_dir: Mock) -> None:
        """Test pipeline named temporary file creation with custom suffix."""
        mock_tmp_dir = pathlib.Path("/tmp")  # noqa: S108
        mock_get_tmp_dir.return_value = mock_tmp_dir
        suffix = ".test"

        with make_pipeline_named_temporary_file(suffix=suffix) as tmp_file:
            assert isinstance(tmp_file, pathlib.Path)
            assert tmp_file.exists()
            assert tmp_file.is_file()
            assert tmp_file.name.endswith(suffix)
            assert "ray_pipeline" in str(tmp_file)

        assert not tmp_file.exists()

    @patch("nemo_curator.utils.operation_utils.get_tmp_dir")
    def test_make_pipeline_named_temporary_file_creates_parent_dirs(self, mock_get_tmp_dir: Mock) -> None:
        """Test that pipeline named temporary file creates parent directories."""
        with tempfile.TemporaryDirectory() as base_dir:
            mock_tmp_dir = pathlib.Path(base_dir)
            mock_get_tmp_dir.return_value = mock_tmp_dir

            with make_pipeline_named_temporary_file(sub_dir="nested/sub/dir") as tmp_file:
                assert isinstance(tmp_file, pathlib.Path)
                assert tmp_file.exists()
                assert tmp_file.is_file()
                # Check that parent directories were created
                pipeline_dir = mock_tmp_dir / "ray_pipeline" / "nested" / "sub" / "dir"
                assert pipeline_dir.exists()

            assert not tmp_file.exists()

    @patch("nemo_curator.utils.operation_utils.get_tmp_dir")
    def test_make_pipeline_named_temporary_file_with_all_params(self, mock_get_tmp_dir: Mock) -> None:
        """Test pipeline named temporary file creation with all parameters."""
        mock_tmp_dir = pathlib.Path("/tmp")  # noqa: S108
        mock_get_tmp_dir.return_value = mock_tmp_dir
        sub_dir = "test_sub"
        suffix = ".test"

        with make_pipeline_named_temporary_file(sub_dir=sub_dir, suffix=suffix) as tmp_file:
            assert isinstance(tmp_file, pathlib.Path)
            assert tmp_file.exists()
            assert tmp_file.is_file()
            assert tmp_file.name.endswith(suffix)
            assert "ray_pipeline" in str(tmp_file)
            assert sub_dir in str(tmp_file)

        assert not tmp_file.exists()
