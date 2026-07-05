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

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def fsync_directory(path: Path) -> None:
    """Flush directory metadata to disk."""
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY

    dir_fd = os.open(path, flags)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _unlink_best_effort(path: Path) -> None:
    """Remove a temporary file without masking the primary result."""
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def _write_json_temp_file(
    path: Path,
    payload: Any,  # noqa: ANN401
    *,
    indent: int | None = None,
    separators: tuple[str, str] | None = None,
    sort_keys: bool = True,
) -> Path:
    """Write JSON to a fsynced temporary file beside its destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            json.dump(payload, tmp, indent=indent, separators=separators, sort_keys=sort_keys)
            tmp.write("\n")
            tmp.flush()
            os.fsync(tmp.fileno())
    except Exception:
        if tmp_path is not None:
            _unlink_best_effort(tmp_path)
        raise
    else:
        return tmp_path


def _fsync_directory_best_effort(path: Path) -> None:
    """Flush directory metadata when supported by the filesystem."""
    with contextlib.suppress(OSError):
        fsync_directory(path)


def write_json_atomically(
    path: Path,
    payload: Any,  # noqa: ANN401
    *,
    indent: int | None = None,
    separators: tuple[str, str] | None = None,
    sort_keys: bool = True,
) -> None:
    """Write JSON through a fsynced temp file and atomic rename.

    Directory fsync is best-effort because the destination is already visible
    after ``os.replace`` and some shared filesystems do not support syncing a
    directory file descriptor.
    """
    tmp_path = _write_json_temp_file(
        path,
        payload,
        indent=indent,
        separators=separators,
        sort_keys=sort_keys,
    )
    try:
        os.replace(tmp_path, path)
        _fsync_directory_best_effort(path.parent)
    except Exception:
        _unlink_best_effort(tmp_path)
        raise


def write_json_atomically_if_absent(
    path: Path,
    payload: Any,  # noqa: ANN401
    *,
    indent: int | None = None,
    separators: tuple[str, str] | None = None,
    sort_keys: bool = True,
) -> bool:
    """Atomically create a JSON file without replacing an existing file."""
    tmp_path = _write_json_temp_file(
        path,
        payload,
        indent=indent,
        separators=separators,
        sort_keys=sort_keys,
    )
    try:
        os.link(tmp_path, path)
    except FileExistsError:
        _unlink_best_effort(tmp_path)
        return False
    except Exception:
        _unlink_best_effort(tmp_path)
        raise

    _unlink_best_effort(tmp_path)
    _fsync_directory_best_effort(path.parent)
    return True
