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

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

_LANG_CHAR_CLASS = {
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
_LANG_PARTICLES = {
    "en": frozenset({"a"}),
    "it": frozenset({"a", "e"}),
    "pt": frozenset({"a", "e"}),
    "es": frozenset({"a"}),
}
_CONTRACTION_SUFFIXES = ("m", "ll", "ve", "d", "re", "ma")
_MIN_ABBREVIATION_CHARS = 2
_MAX_COMPONENT_CHARS = 2
_MIN_CONTRACTION_PREFIX_CHARS = 3


def _set_note(data: dict[str, Any], stage: str, value: str, notes_key: str) -> None:
    notes = data.get(notes_key)
    if not isinstance(notes, dict):
        notes = {}
        data[notes_key] = notes
    notes[stage] = value


@functools.lru_cache(maxsize=32)
def _pattern(language: str) -> re.Pattern[str]:
    char_class = _LANG_CHAR_CLASS.get(language, _LANG_CHAR_CLASS["en"])
    return re.compile(
        rf"(?<![\w’’’ʼ])({char_class}(?: {char_class}){{1,}}(?:(?<=[A-Z])s)?)(?!\w)"  # noqa: RUF001
    )


def _strip_particles(raw: str, particles: frozenset[str]) -> str:
    parts = raw.split()
    if parts and parts[0] in particles:
        parts = parts[1:]
    if parts and parts[-1] in particles:
        preceding = [part.upper() for part in parts[:-1]]
        if preceding[-2:] not in (["D", "N"], ["R", "N"]):
            parts = parts[:-1]
    return " ".join(parts) if len(parts) >= _MIN_ABBREVIATION_CHARS else raw


def _join_match(match: re.Match[str], particles: frozenset[str]) -> str:
    raw = match.group(0)
    if raw == "I I":
        return raw
    parts = raw.split()
    if any(len(part) > _MAX_COMPONENT_CHARS for part in parts):
        return raw
    if len(parts) == _MIN_ABBREVIATION_CHARS and particles and any(part in particles for part in parts):
        return raw

    candidate = _strip_particles(raw, particles)
    letters = candidate.replace(" ", "")
    if len(letters) < _MIN_ABBREVIATION_CHARS or len(set(letters.upper())) <= 1:
        return raw
    if len(letters) == _MIN_ABBREVIATION_CHARS and letters[0].islower() != letters[1].islower():
        return raw

    prefix = raw[: raw.index(candidate[0])]
    suffix = raw[raw.rindex(candidate[-1]) + 1 :]
    return prefix + letters + suffix


def concat_abbreviations(text: str, language: str = "en") -> tuple[str, list[str]]:
    """Join ASR-spelled letter sequences and return the changed abbreviations."""
    found: list[str] = []
    particles = _LANG_PARTICLES.get(language, frozenset())

    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        joined = _join_match(match, particles)
        end = match.end()
        if (
            joined != raw
            and joined[-1:].upper() == "I"
            and len(joined) >= _MIN_CONTRACTION_PREFIX_CHARS
            and end < len(text)
            and text[end] in "’’’ʼ"  # noqa: RUF001
            and text[end + 1 : end + 4].lower().startswith(_CONTRACTION_SUFFIXES)
        ):
            joined = joined[:-1]
        if joined != raw:
            found.append(joined.strip())
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
        return [], [self.output_text_key]

    def _process_single(self, task: AudioTask) -> None:
        if task.data.get(self.skip_me_key, ""):
            task.data.setdefault(self.output_text_key, "")
            return
        text = task.data.get(self.text_key, "")
        if not isinstance(text, str) or not text.strip():
            task.data.setdefault(self.output_text_key, text if isinstance(text, str) else "")
            return

        language = str(task.data.get(self.source_lang_key, self.default_language)).lower()
        result, found = concat_abbreviations(text, language=language)
        task.data[self.output_text_key] = result
        if found:
            _set_note(task.data, self.name, f"joined: {', '.join(found)}", self.notes_key)

    def process(self, task: AudioTask) -> AudioTask:
        self._process_single(task)
        return task

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        for task in tasks:
            self._process_single(task)
        return tasks
