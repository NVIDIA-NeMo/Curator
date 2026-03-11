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

from __future__ import annotations

import contextlib
import pathlib
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator


def get_tmp_dir() -> pathlib.Path:
    """Retrieve the appropriate temporary directory based on the runtime environment.

    Returns:
        pathlib.Path: Path to the temporary directory.

    """
    # TODO: add cloud and slurm checks
    return pathlib.Path(tempfile.gettempdir())


@contextlib.contextmanager
def make_temporary_dir(
    *,
    prefix: str | None = None,
    target_dir: pathlib.Path | None = None,
    delete: bool = True,
) -> Generator[pathlib.Path, None, None]:
    """Context manager to create a temporary directory.

    Args:
        prefix (Optional[str], optional): Prefix for the directory name. Defaults to None.
        target_dir (Optional[pathlib.Path], optional): Parent directory for the temporary directory. Defaults to None.
        delete (bool, optional): If True, the directory will be deleted upon exit. Defaults to True.

    Yields:
        Generator[pathlib.Path, None, None]: Path of the created temporary directory.

    """
    if target_dir is None:
        target_dir = get_tmp_dir()
    if prefix is not None:
        prefix = str(prefix)

    # If not set to delete, make the directory and yield its path
    if not delete:
        yield pathlib.Path(tempfile.mkdtemp(prefix=prefix, dir=str(target_dir)))
    else:
        with tempfile.TemporaryDirectory(dir=target_dir, prefix=prefix) as tmp_dir:
            yield pathlib.Path(tmp_dir)


@contextlib.contextmanager
def make_named_temporary_file(
    *,
    prefix: str | None = None,
    suffix: str | None = None,
    delete: bool = True,
    target_dir: pathlib.Path | None = None,
) -> Generator[pathlib.Path, None, None]:
    """Context manager to create a named temporary file.

    Args:
        prefix (Optional[str], optional): Prefix for the file name. Defaults to None.
        suffix (Optional[str], optional): suffix for the file name. Defaults to None.
        delete (bool, optional): If True, the file will be deleted upon exit. Defaults to True.
        target_dir (Optional[pathlib.Path], optional): Directory where the file should be created. Defaults to None.

    Yields:
        Generator[pathlib.Path, None, None]: Path of the created temporary file.

    """
    if target_dir is None:
        target_dir = get_tmp_dir()

    with tempfile.NamedTemporaryFile(dir=target_dir, delete=delete, prefix=prefix, suffix=suffix) as file:
        yield pathlib.Path(file.name)


@contextlib.contextmanager
def make_pipeline_temporary_dir(
    sub_dir: str | None = None,
) -> Generator[pathlib.Path, None, None]:
    """Context manager to create a temporary directory for pipelines."""
    if sub_dir is not None:
        target_dir = get_tmp_dir() / pathlib.Path("ray_pipeline") / pathlib.Path(sub_dir)
    else:
        target_dir = get_tmp_dir() / pathlib.Path("ray_pipeline")
    target_dir.mkdir(parents=True, exist_ok=True)
    with make_temporary_dir(target_dir=target_dir, delete=True) as tdir:
        yield tdir


@contextlib.contextmanager
def make_pipeline_named_temporary_file(
    sub_dir: str | None = None,
    suffix: str | None = None,
) -> Generator[pathlib.Path, None, None]:
    """Context manager to create a named temporary file for pipelines."""
    if sub_dir is not None:
        target_dir = get_tmp_dir() / pathlib.Path("ray_pipeline") / pathlib.Path(sub_dir)
    else:
        target_dir = get_tmp_dir() / pathlib.Path("ray_pipeline")
    target_dir.mkdir(parents=True, exist_ok=True)
    with make_named_temporary_file(delete=True, target_dir=target_dir, suffix=suffix) as file:
        yield file
