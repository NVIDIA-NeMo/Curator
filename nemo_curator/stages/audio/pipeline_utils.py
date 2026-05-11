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

"""Shared utilities for the audio ASR pipeline.

Provides:
- ``set_note`` — per-stage decision tracking via an ``additional_notes`` dict
- ``LANG_CODE_TO_NAME`` — ISO 639-1 code to full English name mapping
- ``INDIC_CONFORMER_LANGUAGE_CODES`` — languages supported by AI4Bharat Indic Conformer (ISO codes)
- ``WHISPER_ROUTED_LANGUAGE_CODES`` — languages routed to Faster-Whisper in recovery ASR
- ``resolve_indic_language_code`` — normalize ``source_lang`` to an Indic Conformer code when applicable
- ``resolve_whisper_language_code`` — normalize ``source_lang`` to a faster-whisper ``language`` code when applicable
"""

from typing import Any

NOTES_KEY = "additional_notes"

LANG_CODE_TO_NAME: dict[str, str] = {
    "en": "English", "de": "German", "es": "Spanish", "fr": "French",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
    "pl": "Polish", "cs": "Czech", "ro": "Romanian", "hu": "Hungarian",
    "el": "Greek", "fi": "Finnish", "da": "Danish", "sv": "Swedish",
    "lt": "Lithuanian", "lv": "Latvian", "hr": "Croatian", "et": "Estonian",
    "bg": "Bulgarian", "sk": "Slovak", "sl": "Slovenian", "mt": "Maltese",
    "uk": "Ukrainian", "sr": "Serbian", "mk": "Macedonian", "no": "Norwegian",
    "as": "Assamese", "bn": "Bengali", "brx": "Bodo", "doi": "Dogri", "gu": "Gujarati",
    "hi": "Hindi", "kn": "Kannada", "kok": "Konkani", "ks": "Kashmiri", "mai": "Maithili",
    "ml": "Malayalam", "mni": "Manipuri", "mr": "Marathi", "ne": "Nepali", "or": "Odia",
    "pa": "Punjabi", "sa": "Sanskrit", "sat": "Santali", "sd": "Sindhi", "ta": "Tamil",
    "te": "Telugu", "ur": "Urdu",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
    "he": "Hebrew", "id": "Indonesian", "vi": "Vietnamese", "th": "Thai",
    "tr": "Turkish", "fil": "Filipino", "tl": "Tagalog", "fa": "Persian",
}

INDIC_CONFORMER_LANGUAGE_CODES: frozenset[str] = frozenset({
    "as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai", "ml", "mni",
    "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur",
})

# Manifest/source_lang codes covered by language-routed Faster-Whisper recovery (see MODEL_LANG_CODE_TO_WHISPER).
WHISPER_ROUTED_LANGUAGE_CODES: frozenset[str] = frozenset({
    "ro", "hu", "el", "fi", "da", "sv", "th", "fil", "tl", "fa",
})

# Map manifest ISO codes to faster-whisper ``transcribe(language=...)`` codes when they differ.
MODEL_LANG_CODE_TO_WHISPER: dict[str, str] = {
    "fil": "tl",
    "tl": "tl",
}


def resolve_indic_language_code(
    raw: str | None,
    *,
    indic_codes: frozenset[str] = INDIC_CONFORMER_LANGUAGE_CODES,
) -> str | None:
    """If ``raw`` denotes an Indic Conformer language, return its ISO code; else ``None``.

    Accepts ISO codes (e.g. ``hi``) or full English names from ``LANG_CODE_TO_NAME``
    (e.g. ``Hindi``, ``hindi``).
    """
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in indic_codes:
        return s
    for code, name in LANG_CODE_TO_NAME.items():
        if code in indic_codes and name.lower() == s:
            return code
    return None


def resolve_whisper_language_code(
    raw: str | None,
    *,
    whisper_codes: frozenset[str] = WHISPER_ROUTED_LANGUAGE_CODES,
) -> str | None:
    """If ``raw`` is a Whisper-routed language, return the faster-whisper ``language`` code; else ``None``.

    Accepts ISO codes (e.g. ``ro``, ``fil``) or English names from ``LANG_CODE_TO_NAME``.
    Filipino manifest codes ``fil`` / ``tl`` map to Whisper ``tl``.
    """
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in whisper_codes:
        return MODEL_LANG_CODE_TO_WHISPER.get(s, s)
    for code, name in LANG_CODE_TO_NAME.items():
        if code in whisper_codes and name.lower() == s:
            return MODEL_LANG_CODE_TO_WHISPER.get(code, code)
    return None


def set_note(task_data: dict[str, Any], stage_name: str, value: str, notes_key: str = NOTES_KEY) -> None:
    """Write a stage decision note into the ``additional_notes`` dict.

    Each stage writes its own key so downstream analysis can query
    per-stage decisions without string parsing.
    """
    notes = task_data.get(notes_key)
    if not isinstance(notes, dict):
        notes = {}
        task_data[notes_key] = notes
    notes[stage_name] = value
