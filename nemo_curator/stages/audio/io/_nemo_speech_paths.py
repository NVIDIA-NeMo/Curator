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

"""Shared local-path validation for resumable NeMo speech outputs."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit


def validate_local_output_dir(output_dir: str) -> Path:
    """Require an absolute local path for atomic, shared-filesystem output."""

    parsed = urlsplit(output_dir)
    path = Path(output_dir)
    if parsed.scheme or not path.is_absolute():
        msg = f"NeMo speech output_dir must be an absolute local path, got {output_dir!r}"
        raise ValueError(msg)
    return path


def contained_output_path(root: Path, relative: str | PurePosixPath) -> Path:
    """Return a path below ``root`` while rejecting symlink-based escapes.

    The output tree is expected to be exclusively managed by the pipeline.
    Every existing component below the configured root, including the leaf,
    must be a real directory/file rather than a symlink.
    """

    relative_path = PurePosixPath(str(relative).replace("\\", "/"))
    if relative_path.is_absolute() or any(part in {"", ".", ".."} for part in relative_path.parts):
        msg = f"NeMo speech output path must be a safe relative path, got {relative!r}"
        raise ValueError(msg)

    current = root
    for part in relative_path.parts:
        current /= part
        if current.is_symlink():
            msg = f"NeMo speech output path contains a symlink below {root}: {relative_path}"
            raise ValueError(msg)

    resolved_root = root.resolve(strict=False)
    resolved_candidate = current.resolve(strict=False)
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as error:
        msg = f"NeMo speech output path escapes {resolved_root}: {relative_path}"
        raise ValueError(msg) from error
    return resolved_candidate


__all__ = ["contained_output_path", "validate_local_output_dir"]
