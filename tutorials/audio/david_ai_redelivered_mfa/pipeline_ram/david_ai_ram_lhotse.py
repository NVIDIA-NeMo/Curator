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

"""Lhotse cut building and manifest merge for the RAM-by-session pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from david_ai_common import (
    fastmss_textgrid_path,
    group_segments_by_recording,
    recording_id,
    recording_textgrid_path,
    words_from_json,
    write_textgrid,
)
from lhotse import CutSet, MonoCut, RecordingSet, SupervisionSet, fix_manifests
from stage4_build_final_outputs import build_cut_from_alignment
from stage4_build_lhotse import build_cut as build_cut_from_textgrid

if TYPE_CHECKING:
    from stage2_mfa_align_textgrids import SessionAlignResult

logger = logging.getLogger(__name__)


def session_lhotse_path(lhotse_dir: Path, session_id: str) -> Path:
    return lhotse_dir / "sessions" / f"{session_id}_cuts.jsonl.gz"


def session_has_recording_alignments(textgrid_dir: Path, speaker_ids: list[str], session_id: str) -> bool:
    return any(
        fastmss_textgrid_path(textgrid_dir, recording_id(speaker_id, session_id)).is_file()
        for speaker_id in speaker_ids
    )


def write_recording_textgrids(align_result: SessionAlignResult, textgrid_dir: Path) -> None:
    textgrid_dir.mkdir(parents=True, exist_ok=True)
    for rec_row in align_result.recordings:
        rec_id = rec_row["recording_id"]
        merged = words_from_json(rec_row["merged_words"])
        fb = words_from_json(rec_row["fb_words"])
        duration = float(rec_row.get("audio_duration", 0.0))
        max_word_end = max((end for _, end, _ in merged + fb), default=0.0)
        xmax = max(duration, max_word_end) + 0.01

        if merged:
            write_textgrid(
                merged,
                fastmss_textgrid_path(textgrid_dir, rec_id),
                xmin=0.0,
                xmax=xmax,
            )
        ordinary = sorted(merged + fb, key=lambda x: x[0])
        if ordinary:
            write_textgrid(
                ordinary,
                recording_textgrid_path(textgrid_dir, rec_id),
                xmin=0.0,
                xmax=xmax,
            )


def build_session_lhotse_cuts(
    session_id: str,
    norm_rows: list[dict],
    *,
    textgrid_dir: Path,
    lhotse_dir: Path,
    align_result: SessionAlignResult | None = None,
    force: bool = False,
) -> int:
    cut_path = session_lhotse_path(lhotse_dir, session_id)
    if cut_path.is_file() and not force:
        return len(CutSet.from_file(cut_path))

    grouped = group_segments_by_recording(norm_rows)
    cuts: list[MonoCut] = []

    if align_result is not None:
        for rec_row in align_result.recordings:
            rec_id = rec_row["recording_id"]
            segments = grouped.get(rec_id)
            if not segments:
                continue
            cut = build_cut_from_alignment(rec_id, rec_row, segments)
            if cut is not None:
                cuts.append(cut)
    else:
        for rec_id, segments in sorted(grouped.items()):
            tg_path = fastmss_textgrid_path(textgrid_dir, rec_id)
            cut = build_cut_from_textgrid(rec_id, segments, tg_path, textgrid_dir=textgrid_dir)
            if cut is not None:
                cuts.append(cut)

    if cuts:
        cut_path.parent.mkdir(parents=True, exist_ok=True)
        CutSet.from_cuts(cuts).to_file(cut_path)
        logger.info("%s: wrote %d Lhotse cuts with word alignment", session_id, len(cuts))
    else:
        logger.info("%s: no MFA-aligned Lhotse cuts (fallback-only session)", session_id)
    return len(cuts)


def merge_ram_lhotse_manifests(lhotse_dir: Path, *, prefix: str = "david_ai") -> int:
    session_dir = lhotse_dir / "sessions"
    if not session_dir.is_dir():
        return 0
    paths = sorted(session_dir.glob("*_cuts.jsonl.gz"))
    if not paths:
        return 0

    cuts: list[MonoCut] = []
    for path in paths:
        cuts.extend(CutSet.from_file(path))
    if not cuts:
        return 0

    cutset = CutSet.from_cuts(cuts)
    rec_set = RecordingSet.from_recordings([c.recording for c in cutset])
    sup_set = SupervisionSet.from_segments([s for c in cutset for s in c.supervisions])
    rec_set, sup_set = fix_manifests(rec_set, sup_set)

    lhotse_dir.mkdir(parents=True, exist_ok=True)
    rec_set.to_file(lhotse_dir / f"{prefix}_recordings.jsonl.gz")
    sup_set.to_file(lhotse_dir / f"{prefix}_supervisions.jsonl.gz")
    cutset.to_file(lhotse_dir / f"{prefix}_cuts.jsonl.gz")
    cutset.to_file(lhotse_dir / f"{prefix}_aligned_cuts.jsonl.gz")
    return len(cutset)
