#!/usr/bin/env python3
"""Stage 0b: English OOV list + G2P + merged MFA dictionary (*_davidai_eng)."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from collections import Counter
from pathlib import Path

from david_ai_common import (
    MFA_ROOT_DIR_DEFAULT,
    PipelineError,
    load_jsonl,
    log_exception,
    mfa_models_root,
    partition_list,
    resolve_mfa_dict,
    resolve_mfa_g2p_model,
    run_main,
)
from david_ai_glued_words import (
    build_unglue_repair_map,
    count_words_in_text,
    repaired_vocabulary,
    write_oov_frequency_report,
    write_unglue_repairs,
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


def collect_word_freq_from_manifests(
    manifests_dir: Path,
    *,
    num2words_lang: str = "en",
    renormalize_from_raw: bool = True,
) -> Counter[str]:
    from stage0_build_manifests import normalize_text

    word_freq: Counter[str] = Counter()
    for path in sorted(manifests_dir.glob("*_norm.jsonl")):
        if path.name == "all_norm.jsonl":
            continue
        try:
            rows = load_jsonl(path)
        except Exception as exc:
            log_exception(f"cannot load manifest {path}", exc)
            continue
        for row in rows:
            text_raw = (row.get("text_raw") or "").strip()
            text_norm = (row.get("text_norm") or row.get("text") or "").strip()
            if renormalize_from_raw and text_raw:
                try:
                    text_norm = normalize_text(text_raw, num2words_lang=num2words_lang)
                except Exception as exc:
                    log_exception(f"renormalize failed for {path.name}", exc)
            if text_norm:
                word_freq.update(count_words_in_text(text_norm))
    return word_freq


def collect_words_from_manifests(manifests_dir: Path) -> set[str]:
    return set(collect_word_freq_from_manifests(manifests_dir))


def collect_words_from_data_root(
    data_root: Path,
    *,
    num2words_lang: str = "en",
    sessions: list[str] | None = None,
) -> set[str]:
    return set(collect_word_freq_from_data_root(data_root, num2words_lang=num2words_lang, sessions=sessions))


def collect_word_freq_from_data_root(
    data_root: Path,
    *,
    num2words_lang: str = "en",
    sessions: list[str] | None = None,
) -> Counter[str]:
    import json

    from david_ai_common import discover_sessions
    from stage0_build_manifests import normalize_text

    wanted = set(sessions) if sessions else None
    word_freq: Counter[str] = Counter()
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
                word_freq.update(count_words_in_text(norm))
    return word_freq


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


def write_g2p_shards(oov_words: list[str], lexicon_dir: Path, shard_count: int) -> list[Path]:
    shard_dir = lexicon_dir / "g2p_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards = partition_list(oov_words, max(1, shard_count))
    paths: list[Path] = []
    for idx, shard in enumerate(shards):
        path = shard_dir / f"oov_shard_{idx:03d}.txt"
        path.write_text("\n".join(shard) + ("\n" if shard else ""), encoding="utf-8")
        paths.append(path)
    (shard_dir / "shard_count.txt").write_text(f"{len(shards)}\n", encoding="utf-8")
    logger.info("Wrote %d G2P shards under %s", len(paths), shard_dir)
    return paths


def run_g2p_shard(
    lexicon_dir: Path,
    shard_index: int,
    *,
    mfa_g2p: str = "english_us_arpa",
) -> int:
    lexicon_dir = lexicon_dir.resolve()
    shard_dir = lexicon_dir / "g2p_shards"
    oov_path = shard_dir / f"oov_shard_{shard_index:03d}.txt"
    out_path = shard_dir / f"pron_shard_{shard_index:03d}.txt"
    done_path = shard_dir / f"pron_shard_{shard_index:03d}.done"

    if done_path.is_file():
        logger.info("G2P shard %d already done (%s)", shard_index, done_path)
        return 0
    if not oov_path.is_file():
        raise PipelineError(f"Missing G2P shard input: {oov_path}")

    shard_words = [line.strip() for line in oov_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not shard_words:
        out_path.write_text("", encoding="utf-8")
        done_path.write_text("ok\n", encoding="utf-8")
        logger.info("G2P shard %d empty; skipping", shard_index)
        return 0

    try:
        g2p_model_path = resolve_mfa_g2p_model(mfa_g2p)
    except FileNotFoundError:
        ensure_mfa_model("g2p", mfa_g2p)
        g2p_model_path = resolve_mfa_g2p_model(mfa_g2p)

    g2p_cmd = ["mfa", "g2p", str(oov_path), str(g2p_model_path), str(out_path)]
    logger.info("G2P shard %d START (%d words): %s", shard_index, len(shard_words), " ".join(g2p_cmd))
    try:
        result = subprocess.run(g2p_cmd, capture_output=True, text=True)
    except OSError as exc:
        raise PipelineError(f"mfa g2p shard {shard_index} failed to start: {exc}") from exc
    if result.returncode != 0:
        logger.error("mfa g2p shard %d failed: %s", shard_index, result.stderr[-800:])
        return result.returncode

    done_path.write_text("ok\n", encoding="utf-8")
    logger.info("G2P shard %d DONE -> %s", shard_index, out_path)
    return 0


def merge_g2p_shards(
    lexicon_dir: Path,
    *,
    mfa_dict: str = "english_us_arpa",
    output_name: str = "english_mfa_davidai_eng.dict",
) -> int:
    lexicon_dir = lexicon_dir.resolve()
    shard_dir = lexicon_dir / "g2p_shards"
    if not shard_dir.is_dir():
        raise PipelineError(f"Missing G2P shard directory: {shard_dir}")

    base_dict_path = resolve_mfa_dict(mfa_dict)
    pron_paths = sorted(shard_dir.glob("pron_shard_*.txt"))
    if not pron_paths:
        raise PipelineError(f"No G2P shard outputs under {shard_dir}")

    for oov_path in sorted(shard_dir.glob("oov_shard_*.txt")):
        idx = oov_path.stem.removeprefix("oov_shard_")
        if not oov_path.read_text(encoding="utf-8").strip():
            continue
        if not (shard_dir / f"pron_shard_{idx}.done").is_file():
            raise PipelineError(f"G2P shard {idx} not finished")

    oov_pron_path = lexicon_dir / "oov_pronunciations.txt"
    with oov_pron_path.open("w", encoding="utf-8") as out:
        for path in pron_paths:
            text = path.read_text(encoding="utf-8")
            if not text:
                continue
            out.write(text)
            if not text.endswith("\n"):
                out.write("\n")
    logger.info("Merged %d G2P shard files -> %s", len(pron_paths), oov_pron_path)

    merged_path = lexicon_dir / output_name
    with merged_path.open("w", encoding="utf-8") as out:
        out.write(base_dict_path.read_text(encoding="utf-8"))
        if oov_pron_path.is_file() and oov_pron_path.stat().st_size > 0:
            out.write("\n")
            out.write(oov_pron_path.read_text(encoding="utf-8"))
    logger.info("Wrote merged dictionary: %s", merged_path)
    return 0


def build_merged_dictionary(
    all_words: set[str] | Counter[str],
    *,
    lexicon_dir: Path,
    mfa_dict: str = "english_us_arpa",
    mfa_g2p: str = "english_us_arpa",
    output_name: str = "english_mfa_davidai_eng.dict",
    skip_g2p: bool = False,
    unglue_max_freq: int = 5,
    g2p_shard_count: int = 0,
) -> int:
    """Build the merged MFA dictionary (base dict + G2P for OOV) from *all_words*.

    Returns 0 on success, non-zero if G2P failed.
    """
    lexicon_dir = lexicon_dir.resolve()
    lexicon_dir.mkdir(parents=True, exist_ok=True)

    logger.info("MFA_ROOT_DIR=%s", mfa_models_root())
    base_dict_path = resolve_mfa_dict(mfa_dict)
    try:
        base_words = load_dict_words(base_dict_path)
    except OSError as exc:
        raise PipelineError(f"cannot read base dictionary {base_dict_path}: {exc}") from exc
    logger.info("Loaded base dictionary %s (%d entries)", base_dict_path, len(base_words))

    word_freq = all_words if isinstance(all_words, Counter) else Counter({word: 1 for word in all_words})
    logger.info("Building frequency-based unglue map from %d unique tokens", len(word_freq))
    repair_map = build_unglue_repair_map(word_freq, base_words, max_freq=unglue_max_freq)
    write_unglue_repairs(lexicon_dir / "unglue_repairs.tsv", repair_map)
    write_oov_frequency_report(
        lexicon_dir / "oov_word_frequencies.tsv",
        word_freq=word_freq,
        dictionary=base_words,
        repair_map=repair_map,
    )
    if repair_map:
        logger.info("Unglued %d low-frequency OOV tokens (max_freq=%d)", len(repair_map), unglue_max_freq)

    repaired_words = repaired_vocabulary(word_freq, repair_map)
    oov_words = sorted(w for w in repaired_words if w.lower() not in base_words)
    logger.info(
        "Collected %d unique words, %d OOV after unglue (from %d raw unique)",
        len(repaired_words),
        len(oov_words),
        len(word_freq),
    )

    oov_list_path = lexicon_dir / "oov_words.txt"
    oov_list_path.write_text("\n".join(oov_words) + ("\n" if oov_words else ""), encoding="utf-8")
    logger.info("Wrote OOV list: %s", oov_list_path)

    if g2p_shard_count > 0:
        write_g2p_shards(oov_words, lexicon_dir, g2p_shard_count)
        if skip_g2p:
            logger.info("Prepared %d G2P shards; run shard jobs then --merge-g2p-only", g2p_shard_count)
            return 0

    oov_pron_path = lexicon_dir / "oov_pronunciations.txt"
    if oov_words and not skip_g2p:
        try:
            g2p_model_path = resolve_mfa_g2p_model(mfa_g2p)
        except FileNotFoundError:
            ensure_mfa_model("g2p", mfa_g2p)
            g2p_model_path = resolve_mfa_g2p_model(mfa_g2p)
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

    merged_path = lexicon_dir / output_name
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
    ap.add_argument(
        "--no-renormalize-from-raw",
        action="store_true",
        help="With --manifests-dir, count words from stored text_norm only (skip punctuation repair pass)",
    )
    ap.add_argument(
        "--unglue-max-freq",
        type=int,
        default=5,
        help="Repair glued OOV tokens with frequency <= this value when splittable",
    )
    ap.add_argument(
        "--g2p-shard-count",
        type=int,
        default=0,
        help="Split OOV list into N shards for parallel mfa g2p jobs",
    )
    ap.add_argument(
        "--g2p-shard-index",
        type=int,
        default=None,
        help="Run mfa g2p for a single shard (use with SLURM array)",
    )
    ap.add_argument(
        "--merge-g2p-only",
        action="store_true",
        help="Merge completed G2P shard outputs into final dictionary",
    )
    args = ap.parse_args()

    os.environ["MFA_ROOT_DIR"] = str(Path(args.mfa_root_dir).expanduser().resolve())

    if args.merge_g2p_only:
        return merge_g2p_shards(
            args.lexicon_dir,
            mfa_dict=args.mfa_dict,
            output_name=args.output_name,
        )

    if args.g2p_shard_index is not None:
        return run_g2p_shard(
            args.lexicon_dir,
            args.g2p_shard_index,
            mfa_g2p=args.mfa_g2p,
        )

    if args.data_root is not None:
        all_words = collect_word_freq_from_data_root(
            args.data_root.resolve(),
            num2words_lang=args.num2words_lang,
            sessions=args.session or None,
        )
    elif args.manifests_dir is not None:
        all_words = collect_word_freq_from_manifests(
            args.manifests_dir.resolve(),
            num2words_lang=args.num2words_lang,
            renormalize_from_raw=not args.no_renormalize_from_raw,
        )
    else:
        raise PipelineError("pass --manifests-dir or --data-root")

    return build_merged_dictionary(
        all_words,
        lexicon_dir=args.lexicon_dir,
        mfa_dict=args.mfa_dict,
        mfa_g2p=args.mfa_g2p,
        output_name=args.output_name,
        skip_g2p=args.skip_g2p,
        unglue_max_freq=args.unglue_max_freq,
        g2p_shard_count=args.g2p_shard_count,
    )


if __name__ == "__main__":
    run_main(main)
