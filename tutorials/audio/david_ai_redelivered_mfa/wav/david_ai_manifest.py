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

"""Build normalized MFA rows directly from one raw session, without persisted manifests."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from david_ai_common import (
    PipelineError,
    log_exception,
    recording_id,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_DIGIT_DECADE_RE = re.compile(r"(?<![\w])'?(\d{1,4})s(?=[^\w]|[a-z]|$)", re.IGNORECASE)
_DIGIT_FEET_INCHES_RE = re.compile(r"\b(\d+)'(\d+)\"?")
_DIGIT_HYPHEN_PREFIX_RE = re.compile(r"\b(\d+)(-\w+)")
_DIGIT_GROUP_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")
_DIGIT_GENERAL_RE = re.compile(r"\d+")


def separate_gluing_punctuation(text: str) -> str:
    """Insert spaces around punctuation, keeping apostrophes and hyphens word-internal."""
    text = re.sub(r"([^\w\s'-])", r" \1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_digit_grouping_commas(text: str) -> str:
    return _DIGIT_GROUP_COMMA_RE.sub("", text)


def verbalize_digit_string(num_str: str, *, num2words_lang: str) -> str:
    from nemo_curator.stages.audio.preprocessing.transcript_num2words import (
        _split_verbalized_num2words,
    )
    from num2words import num2words

    spoken = num2words(int(num_str), lang=num2words_lang)
    return " ".join(_split_verbalized_num2words(spoken.casefold()))


def verbalize_decade(num_str: str, *, num2words_lang: str) -> str:
    n = int(num_str)
    if num2words_lang == "en":
        if 0 < n < 100 and n % 10 == 0:
            base = verbalize_digit_string(num_str, num2words_lang=num2words_lang)
            return f"{base[:-1]}ies" if base.endswith("y") else f"{base}s"
        if 1000 <= n <= 2090 and n % 10 == 0:
            head = n // 100
            decade = n % 100
            if decade:
                return (
                    f"{verbalize_digit_string(str(head), num2words_lang=num2words_lang)} "
                    f"{verbalize_decade(str(decade), num2words_lang=num2words_lang)}"
                )
    return f"{verbalize_digit_string(num_str, num2words_lang=num2words_lang)} s"


def preprocess_spoken_numbers(text: str, *, num2words_lang: str) -> str:
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
        return f"{verbalize_digit_string(match.group(1), num2words_lang=lang)}{match.group(2)}"

    def _general(match: re.Match[str]) -> str:
        return f" {verbalize_digit_string(match.group(0), num2words_lang=lang)} "

    text = _DIGIT_DECADE_RE.sub(_decade, text)
    text = _DIGIT_FEET_INCHES_RE.sub(_feet_inches, text)
    text = _DIGIT_HYPHEN_PREFIX_RE.sub(_hyphen_prefix, text)
    text = _DIGIT_GENERAL_RE.sub(_general, text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str, *, num2words_lang: str = "en") -> str:
    """Normalize one transcript string for MFA without reading repair/cache files."""
    try:
        from nemo_curator.stages.audio.preprocessing.transcript_normalization import (
            normalize_audio_transcript,
            resolve_alphabet,
        )
    except ImportError as exc:
        msg = "nemo_curator is required for text normalization"
        raise PipelineError(msg) from exc

    lang = (num2words_lang or "").strip()
    alphabet = resolve_alphabet("english", None, lowercase=True)
    try:
        prepared = separate_gluing_punctuation(strip_digit_grouping_commas(text))
        prepared = preprocess_spoken_numbers(prepared, num2words_lang=lang) if lang else prepared
        return normalize_audio_transcript(
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
    except Exception as exc:
        msg = f"normalization failed for text snippet: {text[:80]!r}"
        raise ValueError(msg) from exc


def resolve_speaker_audio_path(session_dir: Path, speaker_id: str) -> Path:
    """Resolve one speaker WAV using the supported filename priority."""
    candidates = (
        session_dir / f"{speaker_id}_postprocess.wav",
        session_dir / f"{speaker_id}_postprocessed.wav",
        session_dir / f"{speaker_id}.wav",
        session_dir / f"{speaker_id}_preprocessed.wav",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    expected = ", ".join(path.name for path in candidates)
    msg = f"no speaker audio for {speaker_id}; tried: {expected}"
    raise FileNotFoundError(msg)


def build_session_rows(
    session_dir: Path,
    *,
    num2words_lang: str = "en",
) -> list[dict]:
    """Read one raw session and create normalized rows entirely in memory."""
    session_id = session_dir.name
    transcript_path = session_dir / "machine_generated_transcript.json"
    if not transcript_path.is_file():
        msg = f"missing transcript: {transcript_path}"
        raise FileNotFoundError(msg)
    try:
        with transcript_path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except json.JSONDecodeError as exc:
        msg = f"invalid JSON in {transcript_path}: {exc}"
        raise ValueError(msg) from exc
    except OSError as exc:
        msg = f"cannot read {transcript_path}: {exc}"
        raise PipelineError(msg) from exc

    segments = payload.get("transcript") if isinstance(payload, dict) else None
    if not isinstance(segments, list):
        msg = f"expected transcript list in {transcript_path}"
        raise TypeError(msg)

    speaker_ids = {
        str(segment["speaker"])
        for segment in segments
        if isinstance(segment, dict) and segment.get("speaker")
    }
    norm_rows: list[dict] = []
    for speaker_id in sorted(speaker_ids):
        audio_path = resolve_speaker_audio_path(session_dir, speaker_id)
        rec_id = recording_id(speaker_id, session_id)
        speaker_segments = [
            segment
            for segment in segments
            if isinstance(segment, dict) and segment.get("speaker") == speaker_id
        ]

        for index, segment in enumerate(speaker_segments):
            text_raw = (segment.get("text") or "").strip()
            try:
                start = float(segment["start"])
                end = float(segment["end"])
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("%s/%s segment %d: invalid boundaries: %s", session_id, speaker_id, index, exc)
                continue
            if end <= start:
                continue

            text_norm = ""
            try:
                text_norm = normalize_text(text_raw, num2words_lang=num2words_lang) if text_raw else ""
            except Exception as exc:
                log_exception(f"{session_id}/{speaker_id} segment {index} normalization", exc)

            row = {
                "session_id": session_id,
                "speaker_id": speaker_id,
                "recording_id": rec_id,
                "segment_index": index,
                "start": start,
                "end": end,
                "duration": round(end - start, 6),
                "text": text_norm,
                "text_raw": text_raw,
                "text_norm": text_norm,
                "audio_filepath": str(audio_path.resolve()),
                "audio_filepath_16k": str(audio_path.resolve()),
            }
            norm_rows.append(row)
    return norm_rows
