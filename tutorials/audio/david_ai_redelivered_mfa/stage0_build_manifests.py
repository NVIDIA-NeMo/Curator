#!/usr/bin/env python3
"""Stage 0: build per-session manifests and normalized copies (*_norm)."""

from __future__ import annotations

import argparse
import json
import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from david_ai_glued_words import apply_repair_map_to_text, load_lexicon_unglue_repairs, separate_gluing_punctuation
from david_ai_common import (
    POSTPROCESSED_RE,
    PipelineError,
    append_jsonl,
    audio_16k_path,
    discover_sessions,
    log_exception,
    normalization_log_entry,
    recording_id,
    run_main,
    write_jsonl,
)

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

_DIGIT_DECADE_RE = re.compile(r"(?<![\w])'?(\d{1,4})s(?=[^\w]|[a-z]|$)", re.IGNORECASE)
_DIGIT_FEET_INCHES_RE = re.compile(r"\b(\d+)'(\d+)\"?")
_DIGIT_HYPHEN_PREFIX_RE = re.compile(r"\b(\d+)(-\w+)")
_DIGIT_GROUP_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")
_DIGIT_GENERAL_RE = re.compile(r"\d+")


def strip_digit_grouping_commas(text: str) -> str:
    """Remove thousand separators glued to digits (e.g. 2,000 -> 2000)."""
    return _DIGIT_GROUP_COMMA_RE.sub("", text)


def verbalize_digit_string(num_str: str, *, num2words_lang: str) -> str:
    from num2words import num2words

    from nemo_curator.stages.audio.preprocessing.transcript_num2words import (
        _split_verbalized_num2words,
    )

    spoken = num2words(int(num_str), lang=num2words_lang)
    return " ".join(_split_verbalized_num2words(spoken.casefold()))


def verbalize_decade(num_str: str, *, num2words_lang: str) -> str:
    n = int(num_str)
    if num2words_lang == "en":
        if 0 < n < 100 and n % 10 == 0:
            base = verbalize_digit_string(num_str, num2words_lang=num2words_lang)
            if base.endswith("y"):
                return base[:-1] + "ies"
            return f"{base}s"
        if 1000 <= n <= 2090 and n % 10 == 0:
            head = n // 100
            decade = n % 100
            if decade:
                head_word = verbalize_digit_string(str(head), num2words_lang=num2words_lang)
                decade_word = verbalize_decade(str(decade), num2words_lang=num2words_lang)
                return f"{head_word} {decade_word}"
    return f"{verbalize_digit_string(num_str, num2words_lang=num2words_lang)} s"


def preprocess_spoken_numbers(text: str, *, num2words_lang: str) -> str:
    """Expand common spoken-number patterns before strict token normalization."""
    lang = (num2words_lang or "").strip()
    if not lang:
        return text

    def _decade(match: re.Match[str]) -> str:
        return f" {verbalize_decade(match.group(1), num2words_lang=lang)} "

    def _feet_inches(match: re.Match[str]) -> str:
        feet = verbalize_digit_string(match.group(1), num2words_lang=lang)
        inches = verbalize_digit_string(match.group(2), num2words_lang=lang)
        return f"{feet} {inches}"

    def _hyphen_prefix(match: re.Match[str]) -> str:
        spoken = verbalize_digit_string(match.group(1), num2words_lang=lang)
        return f"{spoken}{match.group(2)}"

    def _general(match: re.Match[str]) -> str:
        return f" {verbalize_digit_string(match.group(0), num2words_lang=lang)} "

    text = _DIGIT_DECADE_RE.sub(_decade, text)
    text = _DIGIT_FEET_INCHES_RE.sub(_feet_inches, text)
    text = _DIGIT_HYPHEN_PREFIX_RE.sub(_hyphen_prefix, text)
    text = _DIGIT_GENERAL_RE.sub(_general, text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(
    text: str,
    *,
    num2words_lang: str = "en",
    repair_map: dict[str, str] | None = None,
) -> str:
    """Normalize transcript for MFA: digits -> words, keep only ' and -."""
    try:
        from nemo_curator.stages.audio.preprocessing.transcript_normalization import (
            normalize_audio_transcript,
            resolve_alphabet,
        )
    except ImportError as exc:
        raise PipelineError("nemo_curator is required for text normalization") from exc

    lang = (num2words_lang or "").strip()
    if lang:
        try:
            import num2words  # noqa: F401
        except ImportError as exc:
            raise PipelineError(
                "num2words is required for digit verbalization "
                "(pip install num2words or nemo-curator[audio_transcript])"
            ) from exc

    alphabet = resolve_alphabet("english", None, lowercase=True)
    try:
        prepared = separate_gluing_punctuation(strip_digit_grouping_commas(text))
        prepared = preprocess_spoken_numbers(prepared, num2words_lang=lang) if lang else prepared
        normalized = normalize_audio_transcript(
            prepared,
            alphabet=alphabet,
            permitted_symbols="'-",
            lowercase=True,
            remove_punctuation=True,
            map_symbols_to_space=True,
            unknown_word_replacement="spn",
            allow_digits=False,
            num2words_lang=None,
            num2words_lowercase_output=True,
        )
        if repair_map:
            normalized = apply_repair_map_to_text(normalized, repair_map)
        return normalized
    except Exception as exc:
        raise ValueError(f"normalization failed for text snippet: {text[:80]!r}") from exc


def session_norm_path(manifests_dir: Path, session_id: str) -> Path:
    return manifests_dir / f"{session_id}_norm.jsonl"


def session_manifests_done(manifests_dir: Path, session_id: str) -> bool:
    return session_norm_path(manifests_dir, session_id).is_file()


@dataclass
class SessionBuildResult:
    session_id: str
    norm_row_count: int
    normalization_logged: int
    log_entries: list[dict]
    failed: bool
    error: str = ""


def build_session_manifests(
    session_dir: Path,
    *,
    audio_16k_dir: Path,
    num2words_lang: str = "en",
    repair_map: dict[str, str] | None = None,
    lexicon_dir: Path | None = None,
) -> tuple[list[dict], list[dict], int, list[dict]]:
    session_id = session_dir.name
    if repair_map is None and lexicon_dir is not None:
        repair_map, repairs_path = load_lexicon_unglue_repairs(lexicon_dir)
        if repairs_path is not None:
            logger.debug("Using unglue repairs from %s (%d entries)", repairs_path, len(repair_map))
    transcript_path = session_dir / "machine_generated_transcript.json"
    if not transcript_path.is_file():
        raise FileNotFoundError(f"missing transcript: {transcript_path}")

    try:
        with transcript_path.open(encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {transcript_path}: {exc}") from exc
    except OSError as exc:
        raise PipelineError(f"cannot read {transcript_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {transcript_path}")

    segments = payload.get("transcript") or []
    if not isinstance(segments, list):
        raise ValueError(f"expected transcript list in {transcript_path}")
    postprocessed = sorted(session_dir.glob("*_postprocessed.wav"))
    speaker_ids = set()
    for wav in postprocessed:
        m = POSTPROCESSED_RE.match(wav.name)
        if m:
            speaker_ids.add(m.group(1))

    raw_rows: list[dict] = []
    norm_rows: list[dict] = []
    log_entries: list[dict] = []
    normalization_logged = 0

    for speaker_id in sorted(speaker_ids):
        audio_path = session_dir / f"{speaker_id}_postprocessed.wav"
        if not audio_path.is_file():
            logger.warning("%s: missing %s", session_id, audio_path.name)
            continue

        rec_id = recording_id(speaker_id, session_id)
        audio_16k = audio_16k_path(audio_16k_dir, speaker_id, session_id)
        spk_segments = [s for s in segments if s.get("speaker") == speaker_id]

        for idx, seg in enumerate(spk_segments):
            if not isinstance(seg, dict):
                logger.warning("%s/%s: skip non-object segment at index %d", session_id, speaker_id, idx)
                continue
            text = (seg.get("text") or "").strip()
            try:
                start = float(seg["start"])
                end = float(seg["end"])
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "%s/%s segment %d: invalid start/end: %s",
                    session_id,
                    speaker_id,
                    idx,
                    exc,
                )
                continue
            if end <= start:
                continue

            row = {
                "session_id": session_id,
                "speaker_id": speaker_id,
                "recording_id": rec_id,
                "segment_index": idx,
                "start": start,
                "end": end,
                "duration": round(end - start, 6),
                "text": text,
                "audio_filepath": str(audio_path.resolve()),
                "audio_filepath_16k": str(audio_16k.resolve()),
            }
            raw_rows.append(row)

            norm_row = dict(row)
            text_norm = ""
            norm_error = ""
            try:
                text_norm = (
                    normalize_text(
                        text,
                        num2words_lang=num2words_lang,
                        repair_map=repair_map,
                    )
                    if text
                    else ""
                )
            except Exception as exc:
                norm_error = str(exc)
                log_exception(f"{session_id}/{speaker_id} segment {idx} normalization", exc)
            norm_row["text_raw"] = text
            norm_row["text"] = text_norm
            norm_row["text_norm"] = text_norm
            norm_rows.append(norm_row)

            if text and (text != text_norm or norm_error):
                log_entries.append(
                    normalization_log_entry(
                        norm_row,
                        text_raw=text,
                        text_norm=text_norm,
                        num2words_lang=num2words_lang,
                        error=norm_error,
                    )
                )
                normalization_logged += 1

    return raw_rows, norm_rows, normalization_logged, log_entries


def process_one_session(task: tuple[str, str, str, str]) -> SessionBuildResult:
    session_dir_str, manifests_dir_str, audio_16k_dir_str, num2words_lang = task
    session_dir = Path(session_dir_str)
    manifests_dir = Path(manifests_dir_str)
    audio_16k_dir = Path(audio_16k_dir_str)
    session_id = session_dir.name
    try:
        _raw_rows, norm_rows, normalization_logged, log_entries = build_session_manifests(
            session_dir,
            audio_16k_dir=audio_16k_dir,
            num2words_lang=num2words_lang,
        )
        write_jsonl(session_norm_path(manifests_dir, session_id), norm_rows)
        return SessionBuildResult(
            session_id=session_id,
            norm_row_count=len(norm_rows),
            normalization_logged=normalization_logged,
            log_entries=log_entries,
            failed=False,
        )
    except Exception as exc:
        log_exception(f"session {session_id}", exc)
        return SessionBuildResult(
            session_id=session_id,
            norm_row_count=0,
            normalization_logged=0,
            log_entries=[],
            failed=True,
            error=str(exc),
        )


def write_combined_norm_manifest(manifests_dir: Path) -> int:
    combined_path = manifests_dir / "all_norm.jsonl"
    row_count = 0
    with combined_path.open("w", encoding="utf-8") as out:
        for norm_path in sorted(manifests_dir.glob("*_norm.jsonl")):
            if norm_path.name == "all_norm.jsonl":
                continue
            with norm_path.open(encoding="utf-8") as src:
                for line in src:
                    out.write(line)
                    row_count += 1
    return row_count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/ttimofeeva/FastMSS/DavidAI/d12/subser_251spk"),
        help="Root with session_id/machine_generated_transcript.json",
    )
    ap.add_argument(
        "--manifests-dir",
        type=Path,
        required=True,
        help="Output directory for raw and *_norm manifests",
    )
    ap.add_argument(
        "--audio-16k-dir",
        type=Path,
        required=True,
        help="Target directory for 16 kHz Opus audio paths in manifests",
    )
    ap.add_argument(
        "--session",
        action="append",
        default=[],
        help="Optional session_id filter (repeatable)",
    )
    ap.add_argument(
        "--num2words-lang",
        default="en",
        help="Verbalize all-digit tokens via num2words (default: en). Set empty to disable.",
    )
    ap.add_argument(
        "--normalization-log",
        type=Path,
        default=None,
        help="JSONL log of segments where normalization changed text or failed",
    )
    ap.add_argument("--workers", type=int, default=1, help="Parallel session workers")
    ap.add_argument("--force", action="store_true", help="Clear normalization log before writing")
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    manifests_dir = args.manifests_dir.resolve()
    audio_16k_dir = args.audio_16k_dir.resolve()
    manifests_dir.mkdir(parents=True, exist_ok=True)
    audio_16k_dir.mkdir(parents=True, exist_ok=True)

    sessions = discover_sessions(data_root)
    if args.session:
        wanted = set(args.session)
        sessions = [s for s in sessions if s.name in wanted]

    if not sessions:
        raise SystemExit(f"No sessions found under {data_root}")

    normalization_log = (
        args.normalization_log.resolve()
        if args.normalization_log
        else (manifests_dir.parent / "logs" / "normalization.jsonl")
    )
    normalization_log.parent.mkdir(parents=True, exist_ok=True)
    if args.force and normalization_log.is_file() and not args.session:
        normalization_log.unlink()

    if args.force:
        to_process = sessions
        skipped = 0
    else:
        to_process = []
        skipped = 0
        for session_dir in sessions:
            if session_manifests_done(manifests_dir, session_dir.name):
                skipped += 1
            else:
                to_process.append(session_dir)

    workers = max(1, args.workers)
    failed_sessions = 0
    total_normalization_logged = 0
    total_norm_rows = 0

    if to_process:
        logger.info(
            "Stage 0 START: %d sessions to process (%d skipped, workers=%d)",
            len(to_process),
            skipped,
            workers,
        )
        tasks = [
            (
                str(session_dir.resolve()),
                str(manifests_dir),
                str(audio_16k_dir),
                args.num2words_lang,
            )
            for session_dir in to_process
        ]
        completed = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(process_one_session, task) for task in tasks]
            for fut in as_completed(futures):
                result = fut.result()
                completed += 1
                if result.failed:
                    failed_sessions += 1
                else:
                    total_norm_rows += result.norm_row_count
                    total_normalization_logged += result.normalization_logged
                    for entry in result.log_entries:
                        append_jsonl(normalization_log, entry)
                if completed % 500 == 0 or completed == len(futures):
                    logger.info("Stage 0 progress: %d/%d sessions", completed, len(futures))
    else:
        logger.info("Stage 0: all %d sessions already have manifests (%d skipped)", len(sessions), skipped)

    combined_rows = write_combined_norm_manifest(manifests_dir)
    if combined_rows == 0:
        raise PipelineError("no manifests were produced")

    combined_path = manifests_dir / "all_norm.jsonl"
    logger.info(
        "Wrote combined normalized manifest: %s (%d rows, failed_sessions=%d, normalization_log=%s, changed=%d, workers=%d)",
        combined_path,
        combined_rows,
        failed_sessions,
        normalization_log,
        total_normalization_logged,
        workers,
    )
    return 1 if failed_sessions else 0


if __name__ == "__main__":
    run_main(main)
