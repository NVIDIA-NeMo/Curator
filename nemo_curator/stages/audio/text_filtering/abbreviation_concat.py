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

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

_LANG_CHAR_CLASS: dict[str, str] = {
    "en": r"[A-Za-z]",
    "nl": r"[A-Za-z]",
    "de": r"[A-Za-zÄÖÜäöüß]",
    "fr": r"[A-Za-zÀ-ÖØ-öø-ÿ]",
    "es": r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]",
    "it": r"[A-Za-zÀÈÉÌÒÓÙàèéìòóù]",
    "pt": r"[A-Za-zÀ-ÖØ-öø-ÿ]",
    "pl": r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]",
    "cs": r"[A-Za-zÁČĎÉĚÍŇÓŘŠŤÚŮÝŽáčďéěíňóřšťúůýž]",
    "sk": r"[A-Za-zÁÄČĎÉÍĽĻŇÓÔŔŠŤÚÝŽáäčďéíľļňóôŕšťúýž]",
    "sv": r"[A-Za-zÅÄÖåäö]",
    "no": r"[A-Za-zÆØÅæøå]",
    "da": r"[A-Za-zÆØÅæøå]",
    "fi": r"[A-Za-zÄÖÅäöå]",
    "hu": r"[A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű]",
    "ro": r"[A-Za-zĂÂÎȘȚăâîșț]",
    "hr": r"[A-Za-zČĆĐŠŽčćđšž]",
    "sl": r"[A-Za-zČŠŽčšž]",
    "ru": r"[А-ЯЁа-яё]",  # noqa: RUF001
    "bg": r"[А-Яа-я]",  # noqa: RUF001
    "uk": r"[А-ЯҐЄІЇа-яґєії]",  # noqa: RUF001
    "sr": r"[А-ЯЂЈЉЊЋЏа-яђјљњћџ]",  # noqa: RUF001
    "mk": r"[А-Яа-яѓѕѝ]",  # noqa: RUF001
    "el": r"[Α-Ωα-ω]",  # noqa: RUF001
}
_LANG_PARTICLES: dict[str, frozenset[str]] = {
    "en": frozenset({"a"}),
    "it": frozenset({"a", "e"}),
    "pt": frozenset({"a", "e"}),
    "es": frozenset({"a"}),
}
_CONTRACTION_SUFFIXES = ("m", "ll", "ve", "d", "re", "ma")
_APOSTROPHES = "'’‘ʼ"  # noqa: RUF001
_VOWELS = frozenset("AEIOUaeiou")

_TAIL_SLICE = 2
_MIN_PARTS = 2
_PLURAL_SUFFIX_LEN = 2
_MULTI_CHAR_LEN = 3


def _set_note(data: dict[str, Any], stage: str, value: str, notes_key: str) -> None:
    notes = data.get(notes_key)
    if not isinstance(notes, dict):
        notes = {}
        data[notes_key] = notes
    notes[stage] = value


@functools.lru_cache(maxsize=32)
def _pattern(language: str) -> re.Pattern[str]:
    char_class = _LANG_CHAR_CLASS.get(language, _LANG_CHAR_CLASS["en"])
    # ASCII and left-curly apostrophes are token boundaries except after the pronoun I.
    return re.compile(
        rf"(?<![\w’ʼ])(?<![Ii]['‘])({char_class}(?: {char_class}){{1,}}s?)(?!\w)"  # noqa: RUF001
    )


def _strip_particles(raw: str, particles: frozenset[str]) -> str:
    if not particles:
        return raw
    parts = raw.split(" ")
    if parts[0] in particles:
        parts = parts[1:]
    if parts and parts[-1] in particles:
        keep_trailing = (["D", "N"], ["R", "N"])
        preceding = [part.upper() for part in parts[:-1]]
        if preceding[-_TAIL_SLICE:] not in keep_trailing:
            parts = parts[:-1]
    if len(parts) < _MIN_PARTS:
        return raw
    return " ".join(parts)


def _is_mixed_case_pair(letters: str) -> bool:
    return len(letters) == _MIN_PARTS and letters[0].islower() != letters[1].islower()


def _join_match(match: re.Match[str], particles: frozenset[str]) -> str:  # noqa: C901, PLR0911
    raw = match.group(0)
    original = raw
    if raw == "I I":
        return original

    parts = raw.split(" ")
    if any(len(part) >= _MULTI_CHAR_LEN for part in parts):
        return original

    # Keep trailing words such as "Is", "As", and "Os" separate. The
    # sequence "A P Is" is normalized to the plural abbreviation "APIs".
    tail = ""
    is_api_plural = parts == ["A", "P", "Is"]
    if (
        len(parts) >= _MIN_PARTS
        and len(parts[-1]) == _PLURAL_SUFFIX_LEN
        and parts[-1][1] == "s"
        and not is_api_plural
        and (parts[-1][0] in _VOWELS or not parts[-1][0].isupper())
    ):
        tail = " " + parts.pop()
        if len(parts) < _MIN_PARTS:
            return original
        raw = " ".join(parts)

    if sum(1 for part in parts if len(part) == 1) < _MIN_PARTS:
        has_plural_suffix = any(
            len(part) == _PLURAL_SUFFIX_LEN and part[1] == "s" and part[0].isupper() and part[0].lower() not in "aeiou"
            for part in parts
        )
        if not has_plural_suffix:
            return original

    if len(parts) == _MIN_PARTS and particles and any(part in particles for part in parts):
        return original

    letters = raw.replace(" ", "")
    if len(set(letters.upper())) <= 1:
        return original
    if _is_mixed_case_pair(letters):
        return original

    stripped = _strip_particles(raw, particles)
    if stripped == raw:
        return letters + tail
    if len(stripped.replace(" ", "")) < _MIN_PARTS:
        return original

    stripped_start = raw.index(stripped)
    stripped_end = stripped_start + len(stripped)
    prefix = raw[:stripped_start]
    suffix = raw[stripped_end:]
    return prefix + stripped.replace(" ", "") + suffix + tail


def concat_abbreviations(text: str, language: str = "en") -> tuple[str, list[str]]:
    """Join ASR-spelled letter sequences and return the changed abbreviations."""
    found: list[str] = []
    particles = _LANG_PARTICLES.get(language, frozenset())

    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        end = match.end()
        contraction_tail = (
            raw[-2:-1] == " "
            and raw[-1:].upper() == "I"
            and end < len(text)
            and text[end] in _APOSTROPHES
            and text[end + 1 : end + 4].lower().startswith(_CONTRACTION_SUFFIXES)
        )
        if contraction_tail:
            core = raw[:-2]
            core_match = _pattern(language).fullmatch(core)
            joined_core = _join_match(core_match, particles) if core_match else core
            joined = joined_core + " " + raw[-1] if joined_core != core else raw
        else:
            joined = _join_match(match, particles)
        if joined != raw:
            abbreviation = joined.strip()
            raw_parts = raw.split(" ")
            last_part = raw_parts[-1]
            if len(last_part) == _PLURAL_SUFFIX_LEN and last_part[0].islower() and last_part[1] == "s":
                abbreviation = abbreviation.removesuffix(last_part).rstrip()
            elif not (raw == "A P Is" and abbreviation == "APIs"):
                abbreviation = abbreviation.rstrip("’s").rstrip("’s")  # noqa: RUF001
            if abbreviation:
                found.append(abbreviation)
        return joined

    return _pattern(language).sub(replace, text), found


@dataclass
class AbbreviationConcatStage(ProcessingStage[AudioTask, AudioTask]):
    """Concatenate spaced single-letter abbreviations in a transcript."""

    text_key: str = "text"
    output_text_key: str = "text"
    skip_me_key: str = "_skipme"
    notes_key: str = "additional_notes"
    source_lang_key: str = "source_lang"
    default_language: str = "en"
    name: str = "AbbreviationConcat"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_text_key, self.notes_key]

    def process(self, task: AudioTask) -> AudioTask:
        if task.data.get(self.skip_me_key, ""):
            task.data.setdefault(self.output_text_key, "")
            return task
        text = task.data.get(self.text_key, "")
        if not isinstance(text, str) or not text.strip():
            task.data.setdefault(self.output_text_key, text if isinstance(text, str) else "")
            return task

        language_value = task.data.get(self.source_lang_key) or self.default_language
        language = str(language_value).strip().lower()
        result, found = concat_abbreviations(text, language=language)
        task.data[self.output_text_key] = result
        if found:
            logger.trace("AbbreviationConcat: {!r} -> {!r} abbrevs={}", text, result, found)
            changes = ", ".join(f"{' '.join(abbreviation)} -> {abbreviation}" for abbreviation in found)
            _set_note(task.data, self.name, f"applied ({changes})", self.notes_key)
        return task
