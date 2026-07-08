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

"""RAM session pipeline: parallel per-session norm + 16k + MFA + Lhotse + mix (no norm JSON)."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from david_ai_common import (
    PipelineError,
    clear_stage_done,
    discover_sessions,
    finish_stage,
    resolve_mfa_dict,
    run_main,
)
from david_ai_ram_session import (
    SessionRamResult,
    process_session_ram,
    session_needs_ram_processing,
)
from david_ai_ram_lhotse import merge_ram_lhotse_manifests

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RamSessionTask:
    session_dir: str
    work_dir: str
    audio_16k_dir: str
    audio_mixed_dir: str
    lhotse_dir: str
    textgrid_dir: str
    mfa_dict: str
    mfa_acoustic: str
    mfa_g2p: str
    ram_dir: str
    num2words_lang: str
    mfa_num_jobs: int
    segment_padding: float
    rttm_merge_gap: float
    target_sr: int
    opus_bitrate: str
    noise_level: float
    preserve_speech: bool
    stitch_ms: float
    boundary_indent: float
    lexicon_dir: str
    force: bool


def _run_session_task(task: RamSessionTask) -> SessionRamResult:
    return process_session_ram(
        Path(task.session_dir),
        work_dir=Path(task.work_dir),
        audio_16k_dir=Path(task.audio_16k_dir),
        audio_mixed_dir=Path(task.audio_mixed_dir),
        lhotse_dir=Path(task.lhotse_dir),
        textgrid_dir=Path(task.textgrid_dir),
        mfa_dict=Path(task.mfa_dict),
        mfa_acoustic=task.mfa_acoustic,
        mfa_g2p=task.mfa_g2p or None,
        ram_dir=Path(task.ram_dir),
        num2words_lang=task.num2words_lang,
        mfa_num_jobs=task.mfa_num_jobs,
        segment_padding=task.segment_padding,
        rttm_merge_gap=task.rttm_merge_gap,
        target_sr=task.target_sr,
        opus_bitrate=task.opus_bitrate,
        noise_level=task.noise_level,
        preserve_speech=task.preserve_speech,
        stitch_ms=task.stitch_ms,
        boundary_indent=task.boundary_indent,
        lexicon_dir=Path(task.lexicon_dir) if task.lexicon_dir else None,
        force=task.force,
    )


def merge_session_lhotse(lhotse_dir: Path, *, prefix: str = "david_ai") -> int:
    return merge_ram_lhotse_manifests(lhotse_dir, prefix=prefix)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--audio-16k-dir", type=Path, default=None)
    ap.add_argument("--audio-mixed-dir", type=Path, default=None)
    ap.add_argument("--lhotse-dir", type=Path, default=None)
    ap.add_argument("--textgrid-dir", type=Path, default=None)
    ap.add_argument("--lexicon-dir", type=Path, default=None)
    ap.add_argument("--mfa-dict", type=Path, default=None, help="Merged dictionary path")
    ap.add_argument(
        "--mfa-dict-name",
        default="english_us_arpa",
        help="Base MFA dictionary when merged dict is missing (used with --mfa-g2p for OOV)",
    )
    ap.add_argument("--mfa-acoustic", default="english_us_arpa")
    ap.add_argument(
        "--mfa-g2p",
        default="english_us_arpa",
        help="MFA G2P model for OOV pronunciations at align time",
    )
    ap.add_argument(
        "--ram-dir",
        type=Path,
        default=None,
        help="tmpfs scratch for per-session MFA (default: /dev/shm/david_ai_ram_session)",
    )
    ap.add_argument("--num2words-lang", default="en")
    ap.add_argument("--mfa-num-jobs", type=int, default=4)
    ap.add_argument("--segment-padding", type=float, default=0.5)
    ap.add_argument("--rttm-merge-gap", type=float, default=0.2)
    ap.add_argument("--target-sr", type=int, default=16000)
    ap.add_argument("--opus-bitrate", default="32k")
    ap.add_argument("--noise-level", type=float, default=0.0002)
    ap.add_argument("--preserve-speech", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--stitch-ms", type=float, default=5.0)
    ap.add_argument("--boundary-indent", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--session", action="append", default=[])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-lexicon", action="store_true")
    ap.add_argument("--merge-lhotse", action="store_true", help="Merge per-session cuts into global Lhotse files")
    ap.add_argument("--lhotse-prefix", default="david_ai")
    ap.add_argument("--stage-done-name", default="ram_session_pipeline")
    ap.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Total number of shards (e.g. SLURM array size) for multi-node runs",
    )
    ap.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="This shard's index in [0, shard-count); processes sessions where i %% shard-count == shard-index",
    )
    args = ap.parse_args()

    if args.shard_count < 1:
        raise SystemExit(f"--shard-count must be >= 1, got {args.shard_count}")
    if not (0 <= args.shard_index < args.shard_count):
        raise SystemExit(
            f"--shard-index must be in [0, {args.shard_count}), got {args.shard_index}"
        )

    work_dir = args.work_dir.resolve()
    if args.force:
        clear_stage_done(work_dir, args.stage_done_name)

    data_root = args.data_root.resolve()
    audio_16k_dir = (args.audio_16k_dir or work_dir / "audio_16k").resolve()
    audio_mixed_dir = (args.audio_mixed_dir or work_dir / "audio_mixed").resolve()
    lhotse_dir = (args.lhotse_dir or work_dir / "lhotse").resolve()
    textgrid_dir = (args.textgrid_dir or work_dir / "textgrids").resolve()
    lexicon_dir = (args.lexicon_dir or work_dir / "lexicon").resolve()
    ram_dir = (args.ram_dir or Path("/dev/shm/david_ai_ram_session")).resolve()
    merged_dict = (
        args.mfa_dict.resolve()
        if args.mfa_dict
        else (lexicon_dir / "english_mfa_davidai_eng.dict").resolve()
    )

    for path in (audio_16k_dir, audio_mixed_dir, lhotse_dir, textgrid_dir, lexicon_dir, ram_dir):
        path.mkdir(parents=True, exist_ok=True)

    if not args.skip_lexicon and (args.force or not merged_dict.is_file()):
        import subprocess
        import sys

        lex_cmd = [
            sys.executable,
            str(Path(__file__).with_name("preprocess_build_lexicon.py")),
            "--data-root",
            str(data_root),
            "--lexicon-dir",
            str(lexicon_dir),
            "--num2words-lang",
            args.num2words_lang,
            "--workers",
            str(max(1, args.workers)),
        ]
        if args.session:
            lex_cmd.extend(["--session", *args.session])
        logger.info("Building lexicon from data root: %s", " ".join(lex_cmd))
        result = subprocess.run(lex_cmd, check=False)
        if result.returncode != 0:
            raise PipelineError("lexicon build failed")

    if merged_dict.is_file():
        mfa_dict_path = resolve_mfa_dict(str(merged_dict))
        logger.info("Using merged MFA dictionary: %s", mfa_dict_path)
    else:
        mfa_dict_path = resolve_mfa_dict(args.mfa_dict_name)
        logger.info(
            "Merged dictionary missing; using base dict %s with G2P (%s) for OOV",
            mfa_dict_path,
            args.mfa_g2p,
        )

    sessions = discover_sessions(data_root)
    if args.session:
        wanted = set(args.session)
        sessions = [s for s in sessions if s.name in wanted]
    if not sessions:
        raise SystemExit(f"No sessions under {data_root}")

    if args.shard_count > 1:
        total = len(sessions)
        sessions = [s for i, s in enumerate(sessions) if i % args.shard_count == args.shard_index]
        logger.info(
            "Shard %d/%d: processing %d of %d total sessions",
            args.shard_index,
            args.shard_count,
            len(sessions),
            total,
        )
        if not sessions:
            logger.info("Shard %d/%d has no sessions; nothing to do", args.shard_index, args.shard_count)
            return finish_stage(work_dir, args.stage_done_name, 0)

    todo_sessions = [
        session_dir
        for session_dir in sessions
        if session_needs_ram_processing(
            session_dir,
            work_dir=work_dir,
            audio_16k_dir=audio_16k_dir,
            audio_mixed_dir=audio_mixed_dir,
            lhotse_dir=lhotse_dir,
            textgrid_dir=textgrid_dir,
            force=args.force,
        )
    ]
    skip_sessions = len(sessions) - len(todo_sessions)
    workers = max(1, args.workers)
    logger.info(
        "RAM session pipeline START: %d sessions (%d todo, %d already done), workers=%d, ram_dir=%s, mfa_dict=%s",
        len(sessions),
        len(todo_sessions),
        skip_sessions,
        workers,
        ram_dir,
        mfa_dict_path,
    )

    tasks = [
        RamSessionTask(
            session_dir=str(session_dir.resolve()),
            work_dir=str(work_dir),
            audio_16k_dir=str(audio_16k_dir),
            audio_mixed_dir=str(audio_mixed_dir),
            lhotse_dir=str(lhotse_dir),
            textgrid_dir=str(textgrid_dir),
            mfa_dict=str(mfa_dict_path),
            mfa_acoustic=args.mfa_acoustic,
            mfa_g2p=args.mfa_g2p,
            ram_dir=str(ram_dir),
            num2words_lang=args.num2words_lang,
            mfa_num_jobs=args.mfa_num_jobs,
            segment_padding=args.segment_padding,
            rttm_merge_gap=args.rttm_merge_gap,
            target_sr=args.target_sr,
            opus_bitrate=args.opus_bitrate,
            noise_level=args.noise_level,
            preserve_speech=args.preserve_speech,
            stitch_ms=args.stitch_ms,
            boundary_indent=args.boundary_indent,
            lexicon_dir=str(lexicon_dir),
            force=args.force,
        )
        for session_dir in sessions
    ]

    ok = skip = fail = 0
    total_cuts = 0
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_session_task, task) for task in tasks]
        for fut in as_completed(futures):
            result = fut.result()
            completed += 1
            if result.skipped:
                skip += 1
            elif result.ok:
                ok += 1
                total_cuts += result.cuts
            else:
                fail += 1
                logger.warning("%s failed: %s", result.session_id, result.error)
            if completed % 200 == 0 or completed == len(futures):
                logger.info(
                    "RAM session progress: %d/%d (ok=%d skip=%d fail=%d)",
                    completed,
                    len(futures),
                    ok,
                    skip,
                    fail,
                )

    for child in ram_dir.glob("worker_*"):
        shutil.rmtree(child, ignore_errors=True)
    mfa_scratch = Path(os.environ.get("DAVIDAI_MFA_SCRATCH", "/tmp/david_ai_mfa_scratch"))
    if mfa_scratch.is_dir():
        for child in mfa_scratch.glob("worker_*"):
            shutil.rmtree(child, ignore_errors=True)

    merged_cuts = 0
    if args.merge_lhotse:
        merged_cuts = merge_session_lhotse(lhotse_dir, prefix=args.lhotse_prefix)
        logger.info("Merged %d Lhotse cuts into %s", merged_cuts, lhotse_dir)

    logger.info(
        "RAM session pipeline DONE: ok=%d skip=%d fail=%d cuts=%d merged=%d workers=%d",
        ok,
        skip,
        fail,
        total_cuts,
        merged_cuts,
        workers,
    )
    all_complete = fail == 0 and (ok + skip) == len(sessions)
    exit_code = 0 if all_complete else 1
    if not all_complete:
        logger.warning(
            "RAM session pipeline incomplete: %d/%d sessions finished (fail=%d); resubmit to continue undone sessions",
            ok + skip,
            len(sessions),
            fail,
        )
    return finish_stage(work_dir, args.stage_done_name, exit_code)


if __name__ == "__main__":
    run_main(main)
