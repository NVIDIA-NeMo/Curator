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

"""Preprocessing: gather all transcript text, normalize in parallel, build MFA lexicon.

Scans every session transcript (or existing normalized manifests), normalizes the
text in parallel worker processes, unions the vocabulary, then runs OOV G2P and
writes the merged MFA dictionary. Run once before the RAM session pipeline so the
per-session workers reuse the same dictionary.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from david_ai_common import (
    MFA_ROOT_DIR_DEFAULT,
    PipelineError,
    discover_sessions,
    load_jsonl,
    log_exception,
    run_main,
)
from stage0_build_lexicon import build_merged_dictionary
from david_ai_glued_words import count_words_in_text

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def _normalize_words(texts: list[str], *, num2words_lang: str) -> Counter[str]:
    from stage0_build_manifests import normalize_text

    word_freq: Counter[str] = Counter()
    for text in texts:
        text = (text or "").strip()
        if not text:
            continue
        try:
            norm = normalize_text(text, num2words_lang=num2words_lang)
        except Exception as exc:
            log_exception("normalization failed", exc)
            continue
        if norm:
            word_freq.update(count_words_in_text(norm))
    return word_freq


def _session_texts_from_transcript(session_dir: Path) -> list[str]:
    transcript_path = session_dir / "machine_generated_transcript.json"
    if not transcript_path.is_file():
        return []
    try:
        with transcript_path.open(encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log_exception(f"cannot read {transcript_path}", exc)
        return []
    segments = payload.get("transcript") or []
    if not isinstance(segments, list):
        return []
    return [seg.get("text") or "" for seg in segments if isinstance(seg, dict)]


def _worker_from_data_root(task: tuple[str, str]) -> Counter[str]:
    session_dir_str, num2words_lang = task
    texts = _session_texts_from_transcript(Path(session_dir_str))
    return _normalize_words(texts, num2words_lang=num2words_lang)


def _worker_from_manifest(task: tuple[str, str, bool]) -> Counter[str]:
    manifest_str, num2words_lang, renormalize = task
    path = Path(manifest_str)
    try:
        rows = load_jsonl(path)
    except Exception as exc:
        log_exception(f"cannot load manifest {path}", exc)
        return []
    if renormalize:
        texts = [(r.get("text_raw") or r.get("text") or "") for r in rows]
        return _normalize_words(texts, num2words_lang=num2words_lang)
    word_freq: Counter[str] = Counter()
    for row in rows:
        text = (row.get("text_norm") or row.get("text") or "").strip()
        if text:
            word_freq.update(count_words_in_text(text))
    return word_freq


def collect_words_parallel(
    *,
    data_root: Path | None,
    manifests_dir: Path | None,
    num2words_lang: str,
    workers: int,
    renormalize: bool,
    sessions: list[str] | None,
) -> Counter[str]:
    wanted = set(sessions) if sessions else None

    if data_root is not None:
        session_dirs = [
            s for s in discover_sessions(data_root) if wanted is None or s.name in wanted
        ]
        if not session_dirs:
            raise SystemExit(f"No sessions under {data_root}")
        tasks = [(str(s.resolve()), num2words_lang) for s in session_dirs]
        worker = _worker_from_data_root
        total = len(tasks)
        logger.info("Collecting vocabulary from %d transcripts (workers=%d)", total, workers)
    elif manifests_dir is not None:
        paths = [p for p in sorted(manifests_dir.glob("*_norm.jsonl")) if p.name != "all_norm.jsonl"]
        if wanted is not None:
            paths = [p for p in paths if p.name.removesuffix("_norm.jsonl") in wanted]
        if not paths:
            raise SystemExit(f"No *_norm.jsonl manifests under {manifests_dir}")
        tasks = [(str(p), num2words_lang, renormalize) for p in paths]
        worker = _worker_from_manifest
        total = len(tasks)
        logger.info("Collecting vocabulary from %d manifests (workers=%d)", total, workers)
    else:
        raise PipelineError("pass --data-root or --manifests-dir")

    all_words: Counter[str] = Counter()
    completed = 0
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(worker, task) for task in tasks]
        for fut in as_completed(futures):
            all_words.update(fut.result())
            completed += 1
            if completed % 1000 == 0 or completed == total:
                logger.info("Vocabulary progress: %d/%d (%d unique words)", completed, total, len(all_words))
    return all_words


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Scan session transcripts (machine_generated_transcript.json)",
    )
    ap.add_argument(
        "--manifests-dir",
        type=Path,
        default=None,
        help="Scan existing *_norm.jsonl manifests instead of raw transcripts",
    )
    ap.add_argument("--lexicon-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8, help="Parallel normalization workers")
    ap.add_argument("--num2words-lang", default="en")
    ap.add_argument("--session", action="append", default=[], help="Optional session_id filter")
    ap.add_argument(
        "--renormalize",
        action="store_true",
        help="With --manifests-dir, re-normalize text_raw instead of reusing text_norm",
    )
    ap.add_argument("--mfa-dict", default="english_us_arpa", help="Base English MFA dictionary")
    ap.add_argument("--mfa-g2p", default="english_us_arpa", help="MFA G2P model for OOV")
    ap.add_argument("--mfa-root-dir", default=MFA_ROOT_DIR_DEFAULT, help="MFA_ROOT_DIR")
    ap.add_argument("--output-name", default="english_mfa_davidai_eng.dict")
    ap.add_argument("--skip-g2p", action="store_true", help="Only write OOV list; skip mfa g2p")
    ap.add_argument(
        "--words-out",
        type=Path,
        default=None,
        help="Optional path to also dump the full sorted vocabulary",
    )
    ap.add_argument("--unglue-max-freq", type=int, default=5)
    args = ap.parse_args()

    os.environ["MFA_ROOT_DIR"] = str(Path(args.mfa_root_dir).expanduser().resolve())

    all_words = collect_words_parallel(
        data_root=args.data_root.resolve() if args.data_root else None,
        manifests_dir=args.manifests_dir.resolve() if args.manifests_dir else None,
        num2words_lang=args.num2words_lang,
        workers=args.workers,
        renormalize=args.renormalize,
        sessions=args.session or None,
    )

    if args.words_out is not None:
        words_out = args.words_out.resolve()
        words_out.parent.mkdir(parents=True, exist_ok=True)
        words_out.write_text("\n".join(sorted(all_words)) + ("\n" if all_words else ""), encoding="utf-8")
        logger.info("Wrote vocabulary (%d words): %s", len(all_words), words_out)

    return build_merged_dictionary(
        all_words,
        lexicon_dir=args.lexicon_dir,
        mfa_dict=args.mfa_dict,
        mfa_g2p=args.mfa_g2p,
        output_name=args.output_name,
        skip_g2p=args.skip_g2p,
        unglue_max_freq=args.unglue_max_freq,
    )


if __name__ == "__main__":
    run_main(main)
