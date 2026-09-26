# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TextGrid writers used by the on-the-fly RAM E2E pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from david_ai_common import (
    fastmss_textgrid_path,
    recording_textgrid_path,
    session_textgrid_path,
    words_from_json,
    write_textgrid,
)

if TYPE_CHECKING:
    from pathlib import Path

    from david_ai_mfa_align import SessionAlignResult


def write_recording_textgrids(align_result: SessionAlignResult, textgrid_dir: Path) -> None:
    """Write ordinary and FastMSS TextGrids for every speaker recording."""
    textgrid_dir.mkdir(parents=True, exist_ok=True)
    for rec_row in align_result.recordings:
        rec_id = rec_row["recording_id"]
        merged = words_from_json(rec_row["merged_words"])
        fallback = words_from_json(rec_row["fb_words"])
        duration = float(rec_row.get("audio_duration", 0.0))
        max_word_end = max((end for _, end, _ in merged + fallback), default=0.0)
        xmax = max(duration, max_word_end) + 0.01

        write_textgrid(
            merged,
            fastmss_textgrid_path(textgrid_dir, rec_id),
            xmin=0.0,
            xmax=xmax,
        )
        write_textgrid(
            sorted(merged + fallback, key=lambda word: word[0]),
            recording_textgrid_path(textgrid_dir, rec_id),
            xmin=0.0,
            xmax=xmax,
        )


def write_session_textgrids(align_result: SessionAlignResult, textgrid_dir: Path) -> None:
    """Write ordinary and FastMSS TextGrids for the mixed session timeline."""
    session_id = align_result.recordings[0]["session_id"] if align_result.recordings else ""
    if not session_id:
        return

    fastmss_words = [(start, end, word) for start, end, word, _ in align_result.merged_words]
    ordinary_words = sorted(
        fastmss_words + [(start, end, word) for start, end, word, _ in align_result.fb_words],
        key=lambda word: word[0],
    )
    max_word_end = max((end for _, end, _ in ordinary_words), default=0.0)
    xmax = max(float(align_result.audio_duration), max_word_end) + 0.01
    textgrid_dir.mkdir(parents=True, exist_ok=True)

    write_textgrid(
        fastmss_words,
        session_textgrid_path(textgrid_dir, session_id, variant="fastmss"),
        xmin=0.0,
        xmax=xmax,
    )
    write_textgrid(
        ordinary_words,
        session_textgrid_path(textgrid_dir, session_id, variant="ordinary"),
        xmin=0.0,
        xmax=xmax,
    )


def write_all_textgrids(align_result: SessionAlignResult, textgrid_dir: Path) -> None:
    """Persist session and per-recording TextGrids in both required variants."""
    write_session_textgrids(align_result, textgrid_dir)
    write_recording_textgrids(align_result, textgrid_dir)
