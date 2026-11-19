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
in memory first, so a failed encode never leaves a file behind, and the bytes are written
through fsspec so a local path and an ``s3://`` / ``gs://`` URL behave the same.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import fsspec
import soundfile as sf

if TYPE_CHECKING:
    import numpy as np


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
    """Encode ``audio`` and write it to a local path or cloud URL (parent directories are created)."""
    payload = encode_audio(audio, sample_rate, file_format=file_format, subtype=subtype)
    with fsspec.open(path, "wb", **(storage_options or {})) as f:
        f.write(payload)
