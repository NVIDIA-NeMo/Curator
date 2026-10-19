# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Parse transcript language predictions and preserve multilingual records."""

LANG_CODE_TO_NAME: dict[str, str] = {
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "pl": "Polish",
    "cs": "Czech",
    "ro": "Romanian",
    "hu": "Hungarian",
    "el": "Greek",
    "fi": "Finnish",
    "da": "Danish",
    "sv": "Swedish",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "hr": "Croatian",
    "et": "Estonian",
    "bg": "Bulgarian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "mt": "Maltese",
    "uk": "Ukrainian",
    "sr": "Serbian",
    "mk": "Macedonian",
    "no": "Norwegian",
    "as": "Assamese",
    "bn": "Bengali",
    "brx": "Bodo",
    "doi": "Dogri",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "kok": "Konkani",
    "ks": "Kashmiri",
    "mai": "Maithili",
    "ml": "Malayalam",
    "mni": "Manipuri",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "sat": "Santali",
    "sd": "Sindhi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "he": "Hebrew",
    "id": "Indonesian",
    "vi": "Vietnamese",
    "th": "Thai",
    "tr": "Turkish",
    "fil": "Filipino",
    "tl": "Tagalog",
    "fa": "Persian",
}

_LANG_NAME_TO_CODE = {name: code for code, name in LANG_CODE_TO_NAME.items()}
_LANG_PREFIXES = ("the text is in ", "the language is ", "it is ", "this is ", "written in ")
_VERIFICATION_NAME = "LLMLanguageVerification"


def normalize_language_to_code(value: str | None) -> str | None:  # noqa: C901, PLR0911
    """Resolve a code, English name, or ``Hindi-hi`` manifest label."""
    if value is None:
        return None
    text = str(value).strip().strip(".,;:!?\"'")
    if not text:
        return None
    lowered = text.lower()
    if lowered in LANG_CODE_TO_NAME:
        return lowered
    if "-" in text:
        suffix = text.rsplit("-", 1)[-1].lower()
        if suffix in LANG_CODE_TO_NAME:
            return suffix
    title = text.title()
    if title in _LANG_NAME_TO_CODE:
        return _LANG_NAME_TO_CODE[title]
    for name, code in _LANG_NAME_TO_CODE.items():
        if name.lower() == lowered:
            return code
    for prefix in _LANG_PREFIXES:
        if lowered.startswith(prefix):
            return normalize_language_to_code(text[len(prefix) :])
    return None


def parse_language_prediction(raw: str) -> tuple[str | None, list[str]]:
    """Parse ``Primary:`` / ``Languages:`` lines or a single language label.

    Detected codes retain their order, with the primary first when parseable.
    Unknown labels are ignored and repeated languages are counted once.
    """
    primary_raw: str | None = None
    langs_raw: list[str] = []
    for line in raw.splitlines():
        lowered = line.strip().lower()
        if lowered.startswith("primary:"):
            primary_raw = line.split(":", 1)[1].strip()
        elif lowered.startswith("languages:"):
            langs_raw = [part.strip() for part in line.split(":", 1)[1].split(",") if part.strip()]
    if primary_raw is None:
        primary_raw = langs_raw[0] if langs_raw else raw.strip()
    primary_code = normalize_language_to_code(primary_raw)
    detected_codes: list[str] = []
    for value in [primary_raw, *langs_raw]:
        code = normalize_language_to_code(value)
        if code and code not in detected_codes:
            detected_codes.append(code)
    return primary_code, detected_codes


def _format_lang_label(code: str) -> str:
    return f"{LANG_CODE_TO_NAME.get(code, code)}-{code}"


def verify_language(
    data: dict,
    prediction_key: str,
    source_lang_key: str,
    skip_me_key: str,
    notes_key: str,
) -> None:
    """Flag single-language mismatches in place and retain code-switched text.

    A parseable multilingual result is kept even if its primary is malformed
    or its language list omits the source language. Existing skip flags survive.
    """
    raw_prediction = data.get(prediction_key, "")
    raw_expected = data.get(source_lang_key, "")
    if data.get(skip_me_key):
        note = "skipped (already flagged)"
    elif not isinstance(raw_prediction, str) or not raw_prediction.strip():
        note = "skipped (empty prediction)"
    else:
        expected_code = normalize_language_to_code(str(raw_expected))
        primary_code, detected_codes = parse_language_prediction(raw_prediction)
        if expected_code is None:
            note = f"skipped (unknown source_lang={raw_expected!r})"
        else:
            expected_label = _format_lang_label(expected_code)
            # Check multilingual results before rejecting an unparseable primary.
            if len(detected_codes) > 1:
                primary_label = _format_lang_label(primary_code) if primary_code is not None else "unparseable"
                langs_label = ", ".join(_format_lang_label(code) for code in detected_codes)
                source_status = "source present" if expected_code in detected_codes else "source absent"
                note = (
                    f"kept code-switch (primary={primary_label}, langs=[{langs_label}], "
                    f"expected={expected_label}, {source_status})"
                )
            elif primary_code is None:
                data[skip_me_key] = f"Unparseable language:{_VERIFICATION_NAME}"
                note = f"unparseable prediction ({raw_prediction.strip()!r}, expected={expected_label})"
            elif primary_code != expected_code:
                data[skip_me_key] = f"Wrong language:{_VERIFICATION_NAME}"
                note = f"wrong language (predicted={_format_lang_label(primary_code)}, expected={expected_label})"
            else:
                note = f"passed ({_format_lang_label(primary_code)})"
    notes = data.get(notes_key)
    if not isinstance(notes, dict):
        notes = {}
        data[notes_key] = notes
    notes[_VERIFICATION_NAME] = note
