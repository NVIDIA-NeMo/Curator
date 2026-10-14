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

"""Language and recovery-model routing for Granary-v2 audio pipelines."""

from dataclasses import dataclass
from typing import Literal

ModelRouteName = Literal[
    "qwen_omni",
    "qwen_asr",
    "parakeet_v3",
    "whisper",
    "parakeet_riva",
    "indic_monolingual",
    "none",
]

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
    "he": "Hebrew",
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
    "id": "Indonesian",
    "vi": "Vietnamese",
    "th": "Thai",
    "tr": "Turkish",
    "fil": "Filipino",
    "tl": "Tagalog",
    "fa": "Persian",
}

INDIC_CONFORMER_LANGUAGE_CODES = frozenset(
    {
        "as",
        "bn",
        "brx",
        "doi",
        "gu",
        "hi",
        "kn",
        "kok",
        "ks",
        "mai",
        "ml",
        "mni",
        "mr",
        "ne",
        "or",
        "pa",
        "sa",
        "sat",
        "sd",
        "ta",
        "te",
        "ur",
    }
)
PARAKEET_RIVA_PRIMARY_LANGS = frozenset({"hi", "ta", "bn"})
INDIC_MONOLINGUAL_PRIMARY_LANGS = INDIC_CONFORMER_LANGUAGE_CODES - PARAKEET_RIVA_PRIMARY_LANGS
INDIC_MONOLINGUAL_QWEN_RECOVERY_LANGS = frozenset({"ur"})
QWEN_OMNI_PRIMARY_LANGS = frozenset(
    {"en", "de", "es", "fr", "it", "pt", "ru", "nl", "zh", "ja", "ko", "ar", "id", "vi", "tr"}
)
QWEN_ASR_PRIMARY_LANGS = frozenset({"th", "fil", "fa"})
PARAKEET_V3_PRIMARY_LANGS = frozenset({"pl", "cs", "ro", "hu", "el", "fi", "da", "sv"})
WHISPER_PRIMARY_LANGS = frozenset({"lt", "lv", "hr", "et", "bg", "sk", "sl", "mt", "uk", "he"})
WHISPER_NO_RECOVERY_LANGS = frozenset({"he"})

MODEL_LANG_CODE_TO_WHISPER = {
    "fil": "tl",
    "tl": "tl",
    "jv": "jw",
    "iw": "he",
    "in": "id",
    "ji": "yi",
    "nb": "no",
}

_PRIMARY_TO_RECOVERY: dict[ModelRouteName, ModelRouteName] = {
    "qwen_omni": "qwen_asr",
    "qwen_asr": "whisper",
    "parakeet_v3": "whisper",
    "whisper": "parakeet_v3",
    "parakeet_riva": "indic_monolingual",
    "indic_monolingual": "none",
    "none": "none",
}


@dataclass(frozen=True)
class ModelRoute:
    """Resolved primary/recovery pair for one language."""

    language: str
    primary: ModelRouteName
    recovery: ModelRouteName

    @property
    def has_recovery(self) -> bool:
        return self.recovery != "none"


def normalize_language_code(raw: str) -> str:
    """Normalize a supported ISO code or full English language name."""
    value = str(raw).strip().lower()
    if value in LANG_CODE_TO_NAME:
        return value
    for code, name in LANG_CODE_TO_NAME.items():
        if value == name.lower():
            return code
    msg = f"Unsupported Granary-v2 language: {raw!r}"
    raise ValueError(msg)


def resolve_indic_language_code(raw: str | None) -> str | None:
    if raw is None:
        return None
    try:
        code = normalize_language_code(raw)
    except ValueError:
        return None
    return code if code in INDIC_CONFORMER_LANGUAGE_CODES else None


def resolve_whisper_language_code(raw: str | None) -> str | None:
    if raw is None:
        return None
    try:
        code = normalize_language_code(raw)
    except ValueError:
        return None
    return MODEL_LANG_CODE_TO_WHISPER.get(code, code)


def _default_primary(language: str) -> ModelRouteName:
    if language in PARAKEET_RIVA_PRIMARY_LANGS:
        return "parakeet_riva"
    if language in INDIC_MONOLINGUAL_PRIMARY_LANGS:
        return "indic_monolingual"
    if language in QWEN_OMNI_PRIMARY_LANGS:
        return "qwen_omni"
    if language in QWEN_ASR_PRIMARY_LANGS:
        return "qwen_asr"
    if language in PARAKEET_V3_PRIMARY_LANGS:
        return "parakeet_v3"
    if language in WHISPER_PRIMARY_LANGS:
        return "whisper"
    msg = f"No Granary-v2 primary model route for language {language!r}"
    raise ValueError(msg)


def resolve_model_route(
    language: str,
    *,
    primary: ModelRouteName | None = None,
    recovery: ModelRouteName | None = None,
) -> ModelRoute:
    """Resolve a model pair while preserving explicit caller overrides."""
    code = normalize_language_code(language)
    resolved_primary = primary or _default_primary(code)

    if recovery is not None:
        resolved_recovery = recovery
    elif code in PARAKEET_RIVA_PRIMARY_LANGS:
        resolved_recovery = "indic_monolingual"
    elif code in INDIC_MONOLINGUAL_QWEN_RECOVERY_LANGS:
        resolved_recovery = "qwen_omni"
    elif code in INDIC_MONOLINGUAL_PRIMARY_LANGS:
        resolved_recovery = "none"
    elif code in QWEN_ASR_PRIMARY_LANGS:
        resolved_recovery = "whisper"
    elif code in WHISPER_NO_RECOVERY_LANGS:
        resolved_recovery = "none"
    else:
        resolved_recovery = _PRIMARY_TO_RECOVERY[resolved_primary]

    return ModelRoute(language=code, primary=resolved_primary, recovery=resolved_recovery)
