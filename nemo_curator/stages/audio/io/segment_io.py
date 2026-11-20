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

"""Encode-and-write helpers shared by the segment-writing stages.

``SegmentExtractionStage``, ``AudioSegmentWriterStage``, ``SnippetExtractionStage`` and
``SplitLongAudioStage`` all encode a slice of samples and publish it. Encoding happens fully
in memory first, so a failed encode never leaves a file behind, and the bytes are published
whole: a local file through a temporary file and a rename, an ``s3://`` / ``gs://`` object with
one upload.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import soundfile as sf
from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem

from nemo_curator.utils.atomic_io import fsync_directory

if TYPE_CHECKING:
    import numpy as np
    from fsspec import AbstractFileSystem


def encode_audio(audio: np.ndarray, sample_rate: int, *, file_format: str, subtype: str) -> bytes:
    """Encode samples (frames, or frames x channels) into an in-memory audio file."""
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format=file_format.upper(), subtype=subtype)
    return buffer.getvalue()


def write_audio(  # noqa: PLR0913
    path: str,
    audio: np.ndarray,
    sample_rate: int,
    *,
    file_format: str,
    subtype: str,
    storage_options: dict[str, Any] | None = None,
) -> None:
    """Encode ``audio`` and publish it at a local path or cloud URL (parent directories are created)."""
    payload = encode_audio(audio, sample_rate, file_format=file_format, subtype=subtype)
    fs, resolved = url_to_fs(path, **(storage_options or {}))
    write_bytes_atomically(fs, resolved, payload)


def write_bytes_atomically(fs: AbstractFileSystem, path_str: str, payload: bytes) -> None:
    """Publish ``payload`` at ``path_str`` so a reader never sees a partially written file."""
    if not isinstance(fs, LocalFileSystem):
        # An object becomes visible only once its single upload completes.
        fs.pipe_file(path_str, payload)
        return
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        # A short temp name: the final name may already be close to the 255-byte limit.
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=".", suffix=".tmp", delete=False
        ) as stream:
            temp_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        with contextlib.suppress(OSError):
            fsync_directory(path.parent)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
