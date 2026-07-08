#!/usr/bin/env python3
"""Stage 0b: English OOV list + G2P + merged MFA dictionary (*_davidai_eng)."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from pathlib import Path

from david_ai_common import (
    MFA_ROOT_DIR_DEFAULT,
    PipelineError,
    load_jsonl,
    log_exception,
    mfa_models_root,
    resolve_mfa_dict,
    resolve_mfa_g2p_model,
    run_main,
)

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dict_words(dict_path: Path) -> set[str]:
    words: set[str] = set()
    with dict_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                words.add(parts[0].lower())
    return words


def collect_words_from_manifests(manifests_dir: Path) -> set[str]:
    words: set[str] = set()
    for path in sorted(manifests_dir.glob("*_norm.jsonl")):
        if path.name == "all_norm.jsonl":
            continue
        try:
            rows = load_jsonl(path)
        except Exception as exc:
            log_exception(f"cannot load manifest {path}", exc)
            continue
        for row in rows:
            text = (row.get("text_norm") or row.get("text") or "").strip()
            if text:
                words.update(text.split())
    return words


def collect_words_from_data_root(
    data_root: Path,
    *,
    num2words_lang: str = "en",
    sessions: list[str] | None = None,
) -> set[str]:
    import json

    from david_ai_common import discover_sessions
    from stage0_build_manifests import normalize_text

    wanted = set(sessions) if sessions else None
    words: set[str] = set()
    for session_dir in discover_sessions(data_root):
        if wanted is not None and session_dir.name not in wanted:
            continue
        transcript_path = session_dir / "machine_generated_transcript.json"
        if not transcript_path.is_file():
            continue
        try:
            with transcript_path.open(encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            log_exception(f"cannot read {transcript_path}", exc)
            continue
        segments = payload.get("transcript") or []
        if not isinstance(segments, list):
            continue
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            try:
                norm = normalize_text(text, num2words_lang=num2words_lang)
            except Exception as exc:
                log_exception(f"normalization failed in {session_dir.name}", exc)
                continue
            if norm:
                words.update(norm.split())
    return words


def ensure_mfa_model(model_type: str, model_name: str) -> None:
    cmd = ["mfa", "model", "download", model_type, model_name]
    logger.info("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as exc:
        raise PipelineError(f"mfa model download failed to start: {exc}") from exc
    if result.returncode != 0:
        raise PipelineError(
            f"mfa model download {model_type} {model_name} failed: {result.stderr[-800:]}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifests-dir", type=Path, default=None)
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Scan transcripts directly (no *_norm.jsonl); use with RAM session pipeline",
    )
    ap.add_argument("--num2words-lang", default="en")
    ap.add_argument("--session", action="append", default=[], help="Optional session_id filter")
    ap.add_argument("--lexicon-dir", type=Path, required=True)
    ap.add_argument(
        "--mfa-dict",
        default="english_us_arpa",
        help="Base English MFA dictionary model name or path",
    )
    ap.add_argument(
        "--mfa-g2p",
        default="english_us_arpa",
        help="MFA G2P model for OOV pronunciations",
    )
    ap.add_argument(
        "--mfa-root-dir",
        default=MFA_ROOT_DIR_DEFAULT,
        help="MFA models root directory (MFA_ROOT_DIR)",
    )
    ap.add_argument(
        "--output-name",
        default="english_mfa_davidai_eng.dict",
        help="Merged dictionary filename",
    )
    ap.add_argument(
        "--skip-g2p",
        action="store_true",
        help="Only write OOV list; do not run mfa g2p",
    )
    args = ap.parse_args()

    os.environ["MFA_ROOT_DIR"] = str(Path(args.mfa_root_dir).expanduser().resolve())

    lexicon_dir = args.lexicon_dir.resolve()
    lexicon_dir.mkdir(parents=True, exist_ok=True)

    logger.info("MFA_ROOT_DIR=%s", mfa_models_root())
    base_dict_path = resolve_mfa_dict(args.mfa_dict)
    try:
        base_words = load_dict_words(base_dict_path)
    except OSError as exc:
        raise PipelineError(f"cannot read base dictionary {base_dict_path}: {exc}") from exc
    logger.info("Loaded base dictionary %s (%d entries)", base_dict_path, len(base_words))

    if args.data_root is not None:
        all_words = collect_words_from_data_root(
            args.data_root.resolve(),
            num2words_lang=args.num2words_lang,
            sessions=args.session or None,
        )
    elif args.manifests_dir is not None:
        all_words = collect_words_from_manifests(args.manifests_dir.resolve())
    else:
        raise PipelineError("pass --manifests-dir or --data-root")
    oov_words = sorted(w for w in all_words if w.lower() not in base_words)
    logger.info("Collected %d unique words, %d OOV", len(all_words), len(oov_words))

    oov_list_path = lexicon_dir / "oov_words.txt"
    oov_list_path.write_text("\n".join(oov_words) + ("\n" if oov_words else ""), encoding="utf-8")
    logger.info("Wrote OOV list: %s", oov_list_path)

    oov_pron_path = lexicon_dir / "oov_pronunciations.txt"
    if oov_words and not args.skip_g2p:
        try:
            g2p_model_path = resolve_mfa_g2p_model(args.mfa_g2p)
        except FileNotFoundError:
            ensure_mfa_model("g2p", args.mfa_g2p)
            g2p_model_path = resolve_mfa_g2p_model(args.mfa_g2p)
        g2p_cmd = [
            "mfa",
            "g2p",
            str(oov_list_path),
            str(g2p_model_path),
            str(oov_pron_path),
        ]
        logger.info("Running: %s", " ".join(g2p_cmd))
        try:
            result = subprocess.run(g2p_cmd, capture_output=True, text=True)
        except OSError as exc:
            raise PipelineError(f"mfa g2p failed to start: {exc}") from exc
        if result.returncode != 0:
            logger.error("mfa g2p failed: %s", result.stderr[-800:])
            return result.returncode
        logger.info("Wrote G2P pronunciations: %s", oov_pron_path)
    elif not oov_words:
        oov_pron_path.write_text("", encoding="utf-8")
        logger.info("No OOV words; skipping G2P")

    merged_path = lexicon_dir / args.output_name
    try:
        with merged_path.open("w", encoding="utf-8") as out:
            out.write(base_dict_path.read_text(encoding="utf-8"))
            if oov_pron_path.is_file() and oov_pron_path.stat().st_size > 0:
                out.write("\n")
                out.write(oov_pron_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PipelineError(f"cannot write merged dictionary {merged_path}: {exc}") from exc

    logger.info("Wrote merged dictionary: %s", merged_path)
    return 0


if __name__ == "__main__":
    run_main(main)
