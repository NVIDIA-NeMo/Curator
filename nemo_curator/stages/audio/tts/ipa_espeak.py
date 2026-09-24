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

"""Local espeak/espeak-ng IPA conversion, ported from TTS Granary."""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field

from loguru import logger

ESPEAK_VOICE_BY_LANG: dict[str, str] = {
    "ar": "ar",
    "ar-ae": "ar",
    "ar-msa": "ar",
    "ar-sa": "ar",
    "ar-sy": "ar",
    "bg": "bg",
    "cs": "cs",
    "da": "da",
    "de": "de",
    "el": "el",
    "en": "en",
    "en-us": "en-us",
    "es": "es",
    "et": "et",
    "fi": "fi",
    "fr": "fr",
    "he": "he",
    "hi": "hi",
    "hr": "hr",
    "hu": "hu",
    "it": "it",
    "ja": "ja",
    "ko": "ko",
    "ko-kr": "ko",
    "lt": "lt",
    "lv": "lv",
    "nl": "nl",
    "pl": "pl",
    "pt": "pt",
    "ro": "ro",
    "ru": "ru",
    "sk": "sk",
    "sl": "sl",
    "sv": "sv",
    "uk": "uk",
    "vi": "vi",
    "zh": "zh",
}

_DISPLAY_NAME_TO_LANG: dict[str, str] = {
    "arabic": "ar",
    "bulgarian": "bg",
    "chinese": "zh",
    "croatian": "hr",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "estonian": "et",
    "finnish": "fi",
    "french": "fr",
    "german": "de",
    "greek": "el",
    "hebrew": "he",
    "hindi": "hi",
    "hungarian": "hu",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "latvian": "lv",
    "lithuanian": "lt",
    "polish": "pl",
    "portuguese": "pt",
    "romanian": "ro",
    "russian": "ru",
    "slovak": "sk",
    "slovenian": "sl",
    "spanish": "es",
    "swedish": "sv",
    "ukrainian": "uk",
}

IPA_FLAG = "--ipa"
COMMON_FLAGS = ["-q"]
MAX_ESPEAK_RETRIES = 2
ESPEAK_RETRY_SLEEP_SECONDS = 0.1
_WS_RE = re.compile(r"\s+")


def find_espeak_binaries() -> list[str]:
    """Prefer espeak-ng, then optionally fall back to espeak."""
    binaries = [exe for exe in ("espeak-ng", "espeak") if shutil.which(exe)]
    if binaries:
        return binaries
    msg = "Neither 'espeak-ng' nor 'espeak' was found on PATH. Install espeak-ng (recommended) or espeak."
    raise RuntimeError(msg)


def language_to_espeak_voice(language: str | None) -> str:
    """Map a manifest language (ISO code or display name) to an espeak voice."""
    if not language:
        return "en"
    text = str(language).strip()
    if not text:
        return "en"
    lower = text.lower()
    if lower in ESPEAK_VOICE_BY_LANG:
        return ESPEAK_VOICE_BY_LANG[lower]
    iso = _DISPLAY_NAME_TO_LANG.get(lower)
    if iso:
        return ESPEAK_VOICE_BY_LANG.get(iso, iso)
    return lower


@dataclass(frozen=True)
class EspeakRunner:
    exe: str
    voice: str
    fallback_exe: str | None = None

    def text_to_ipa(self, text: str) -> str:
        """Convert text to normalized IPA using espeak/espeak-ng."""
        errors: list[str] = []
        executables = [self.exe]
        if self.fallback_exe and self.fallback_exe != self.exe:
            executables.append(self.fallback_exe)

        for exe in executables:
            cmd = [exe, "-v", self.voice, IPA_FLAG, *COMMON_FLAGS]
            for attempt in range(1, MAX_ESPEAK_RETRIES + 2):
                try:
                    proc = subprocess.run(  # noqa: S603
                        cmd,
                        input=text.encode("utf-8"),
                        capture_output=True,
                        check=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"cmd={' '.join(cmd)} attempt={attempt} error={exc}")
                    break

                if proc.returncode == 0:
                    out = proc.stdout.decode("utf-8", errors="replace").strip()
                    return _WS_RE.sub(" ", out).strip()

                stderr = proc.stderr.decode("utf-8", errors="replace")
                errors.append(f"cmd={' '.join(cmd)} attempt={attempt} rc={proc.returncode} stderr={stderr}")
                if proc.returncode < 0 and attempt <= MAX_ESPEAK_RETRIES:
                    logger.warning(
                        "espeak process crash (rc={}) for voice={}; retrying attempt {}",
                        proc.returncode,
                        self.voice,
                        attempt + 1,
                    )
                    time.sleep(ESPEAK_RETRY_SLEEP_SECONDS)
                    continue
                break

        msg = "espeak command failed after retries/fallback:\n" + "\n".join(errors)
        raise RuntimeError(msg)


@dataclass
class IPACache:
    """Process-local cache for repeated (voice, text) pairs."""

    _cache: dict[tuple[str, str], str] = field(default_factory=dict)

    def get(self, voice: str, text: str) -> str | None:
        return self._cache.get((voice, text))

    def set(self, voice: str, text: str, ipa: str) -> None:
        self._cache[(voice, text)] = ipa
