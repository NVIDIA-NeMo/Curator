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

"""Shared output-root resolution for resumable NeMo speech outputs (local or cloud)."""

from __future__ import annotations

import posixpath
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem

from nemo_curator.utils.client_utils import is_remote_url

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem


def resolve_output_dir(
    output_dir: str, storage_options: dict[str, Any] | None = None
) -> tuple[AbstractFileSystem, str]:
    """Return the filesystem and root of an absolute local path or a cloud URL (``s3://``, ``gs://``, ...)."""

    if not is_remote_url(output_dir) and (urlsplit(output_dir).scheme or not Path(output_dir).is_absolute()):
        msg = f"NeMo speech output_dir must be an absolute local path or a cloud URL, got {output_dir!r}"
        raise ValueError(msg)
    return url_to_fs(output_dir, **(storage_options or {}))


def contained_output_path(fs: AbstractFileSystem, root: str, relative: str | PurePosixPath) -> str:
    """Return a path below ``root`` while rejecting traversal and, locally, symlink-based escapes.

    The output tree is expected to be exclusively managed by the pipeline.
    Every existing local component below the configured root, including the
    leaf, must be a real directory/file rather than a symlink.
    """

    relative_path = PurePosixPath(str(relative).replace("\\", "/"))
    if not relative_path.parts or relative_path.is_absolute() or ".." in relative_path.parts:
        msg = f"NeMo speech output path must be a safe relative path, got {relative!r}"
        raise ValueError(msg)
    if not isinstance(fs, LocalFileSystem):
        # Object-store keys have no symlinks, so a validated relative key cannot leave the root.
        return posixpath.join(root, relative_path.as_posix())

    current = Path(root)
    for part in relative_path.parts:
        current /= part
        if current.is_symlink():
            msg = f"NeMo speech output path contains a symlink below {root}: {relative_path}"
            raise ValueError(msg)

    resolved_root = Path(root).resolve(strict=False)
    resolved_candidate = current.resolve(strict=False)
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as error:
        msg = f"NeMo speech output path escapes {resolved_root}: {relative_path}"
        raise ValueError(msg) from error
    return str(resolved_candidate)


__all__ = ["contained_output_path", "resolve_output_dir"]
