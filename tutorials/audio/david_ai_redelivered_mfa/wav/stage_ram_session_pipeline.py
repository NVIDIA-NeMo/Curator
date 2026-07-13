#!/usr/bin/env python3
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

"""Strict on-the-fly E2E: raw sessions -> MFA/G2P -> audio, RTTM, and TextGrids."""

from __future__ import annotations

import argparse
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from david_ai_common import PipelineError, discover_sessions, resolve_mfa_dict, run_main
from david_ai_ram_session import SessionRamResult, is_session_done, process_session_ram

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RamSessionTask:
    session_dir: str
    work_dir: str
    audio_masked_dir: str
    audio_mixed_dir: str
    textgrid_dir: str
    mfa_dict: str
    mfa_acoustic: str
    mfa_g2p: str
    ram_dir: str
    num2words_lang: str
    mfa_num_jobs: int
    segment_padding: float
    rttm_merge_gap: float
    noise_level: float
    stitch_ms: float
    boundary_offset: float


def _run_session_task(task: RamSessionTask) -> SessionRamResult:
    return process_session_ram(
        Path(task.session_dir),
        work_dir=Path(task.work_dir),
        audio_masked_dir=Path(task.audio_masked_dir),
        audio_mixed_dir=Path(task.audio_mixed_dir),
        textgrid_dir=Path(task.textgrid_dir),
        mfa_dict=Path(task.mfa_dict),
        mfa_acoustic=task.mfa_acoustic,
        mfa_g2p=task.mfa_g2p,
        ram_dir=Path(task.ram_dir),
        num2words_lang=task.num2words_lang,
        mfa_num_jobs=task.mfa_num_jobs,
        segment_padding=task.segment_padding,
        rttm_merge_gap=task.rttm_merge_gap,
        noise_level=task.noise_level,
        stitch_ms=task.stitch_ms,
        boundary_offset=task.boundary_offset,
    )


def sessions_without_done_flags(sessions: list[Path], work_dir: Path) -> list[Path]:
    """Select only sessions that have not completed successfully."""
    return [session for session in sessions if not is_session_done(work_dir, session.name)]


def filter_sessions_from_file(sessions: list[Path], sessions_file: Path) -> list[Path]:
    """Restrict discovered sessions to IDs listed one per line."""
    requested = {
        line.strip()
        for line in sessions_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    available = {session.name for session in sessions}
    missing = sorted(requested - available)
    if missing:
        logger.warning("%d requested session IDs were not found under DATA_ROOT", len(missing))
    return [session for session in sessions if session.name in requested]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--audio-masked-dir", type=Path, default=None)
    ap.add_argument("--audio-mixed-dir", type=Path, default=None)
    ap.add_argument("--textgrid-dir", type=Path, default=None)
    ap.add_argument("--mfa-dict-name", default="english_us_arpa")
    ap.add_argument("--mfa-acoustic", default="english_us_arpa")
    ap.add_argument("--mfa-g2p", default="english_us_arpa")
    ap.add_argument("--ram-dir", type=Path, default=Path("/tmp/david_ai_ram_session"))
    ap.add_argument("--num2words-lang", default="en")
    ap.add_argument("--mfa-num-jobs", type=int, default=2)
    ap.add_argument("--segment-padding", type=float, default=0.5)
    ap.add_argument("--rttm-merge-gap", type=float, default=0.2)
    ap.add_argument("--noise-level", type=float, default=0.0002)
    ap.add_argument("--stitch-ms", type=float, default=5.0)
    ap.add_argument("--boundary-offset", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sessions-file", type=Path, default=None)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    args = ap.parse_args()

    if args.shard_count < 1:
        raise PipelineError(f"--shard-count must be >= 1, got {args.shard_count}")
    if not 0 <= args.shard_index < args.shard_count:
        raise PipelineError(
            f"--shard-index must be in [0, {args.shard_count}), got {args.shard_index}"
        )

    work_dir = args.work_dir.resolve()
    data_root = args.data_root.resolve()
    audio_masked_dir = (args.audio_masked_dir or work_dir / "audio_16k_masked").resolve()
    audio_mixed_dir = (args.audio_mixed_dir or work_dir / "audio_mixed").resolve()
    textgrid_dir = (args.textgrid_dir or work_dir / "textgrids").resolve()
    ram_dir = args.ram_dir.resolve()
    for path in (audio_masked_dir, audio_mixed_dir, textgrid_dir, ram_dir):
        path.mkdir(parents=True, exist_ok=True)

    mfa_dict_path = resolve_mfa_dict(args.mfa_dict_name)
    logger.info(
        "Using base MFA dictionary %s with runtime G2P model %s",
        mfa_dict_path,
        args.mfa_g2p,
    )

    sessions = discover_sessions(data_root)
    if args.sessions_file is not None:
        sessions_file = args.sessions_file.resolve()
        if not sessions_file.is_file():
            raise PipelineError(f"sessions file does not exist: {sessions_file}")
        sessions = filter_sessions_from_file(sessions, sessions_file)
        logger.info("Restricted run to %d sessions from %s", len(sessions), sessions_file)
    if not sessions:
        raise PipelineError(f"No sessions under {data_root}")
    if args.shard_count > 1:
        total = len(sessions)
        sessions = [
            session
            for index, session in enumerate(sessions)
            if index % args.shard_count == args.shard_index
        ]
        logger.info(
            "Shard %d/%d: processing %d of %d raw sessions",
            args.shard_index,
            args.shard_count,
            len(sessions),
            total,
        )

    pending_sessions = sessions_without_done_flags(sessions, work_dir)
    skipped_sessions = len(sessions) - len(pending_sessions)
    workers = max(1, args.workers)
    logger.info(
        "Resumable E2E START: sessions=%d pending=%d done=%d workers=%d mfa_jobs=%d ram_dir=%s",
        len(sessions),
        len(pending_sessions),
        skipped_sessions,
        workers,
        args.mfa_num_jobs,
        ram_dir,
    )
    if not pending_sessions:
        logger.info("All %d selected sessions already have validated done flags", len(sessions))
        return 0
    tasks = [
        RamSessionTask(
            session_dir=str(session.resolve()),
            work_dir=str(work_dir),
            audio_masked_dir=str(audio_masked_dir),
            audio_mixed_dir=str(audio_mixed_dir),
            textgrid_dir=str(textgrid_dir),
            mfa_dict=str(mfa_dict_path),
            mfa_acoustic=args.mfa_acoustic,
            mfa_g2p=args.mfa_g2p,
            ram_dir=str(ram_dir),
            num2words_lang=args.num2words_lang,
            mfa_num_jobs=args.mfa_num_jobs,
            segment_padding=args.segment_padding,
            rttm_merge_gap=args.rttm_merge_gap,
            noise_level=args.noise_level,
            stitch_ms=args.stitch_ms,
            boundary_offset=args.boundary_offset,
        )
        for session in pending_sessions
    ]

    ok = fail = completed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_session_task, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if result.ok:
                ok += 1
            else:
                fail += 1
                logger.warning("%s failed: %s", result.session_id, result.error)
            if completed % 50 == 0 or completed == len(futures):
                logger.info(
                    "E2E progress: %d/%d (ok=%d fail=%d)",
                    completed,
                    len(futures),
                    ok,
                    fail,
                )

    shutil.rmtree(ram_dir, ignore_errors=True)
    logger.info(
        "Resumable E2E DONE: ok=%d fail=%d previously_done=%d workers=%d",
        ok,
        fail,
        skipped_sessions,
        workers,
    )
    return 0 if fail == 0 and ok == len(pending_sessions) else 1


if __name__ == "__main__":
    run_main(main)
