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

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

# Per-language single-letter character classes.
# Suffix 's' (plural) uses ASCII [a-z] even for Cyrillic/Greek — the suffix group simply
# won't fire for pure non-Latin matches, which is acceptable.
_LANG_CHAR_CLASS: dict[str, str] = {
    # Pure ASCII Latin
    "en": r"[A-Za-z]",
    "nl": r"[A-Za-z]",
    # Latin + national characters
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
    # Cyrillic
    "ru": r"[А-ЯЁа-яё]",  # noqa: RUF001
    "bg": r"[А-Яа-я]",  # noqa: RUF001
    "uk": r"[А-ЯҐЄІЇа-яґєії]",  # noqa: RUF001
    "sr": r"[А-ЯЂЈЉЊЋЏа-яђјљњћџ]",  # noqa: RUF001
    "mk": r"[А-Яа-яѓѕѝ]",  # noqa: RUF001
    # Greek
    "el": r"[Α-Ωα-ω]",  # noqa: RUF001
}

# Per-language single-letter particles to strip at match edges.
# Empty frozenset means no stripping for that language.
_LANG_PARTICLES: dict[str, frozenset[str]] = {
    "en": frozenset({"a"}),
    "it": frozenset({"a", "e"}),  # "e" = "and" in Italian
    "pt": frozenset({"a", "e"}),
    "es": frozenset({"a"}),
}

_CONTRACTION_SUFFIXES = frozenset({"m", "ll", "ve", "d", "re", "ma"})
_VOWELS = frozenset("AEIOUaeiou")

_TAIL_SLICE = 2       # last N elements checked for DNA/RNA guard
_MIN_PARTS = 2        # abbreviation needs at least this many parts
_PLURAL_SUFFIX_LEN = 2  # "Xs" has 2 chars
_MULTI_CHAR_LEN = 3   # parts with >= 3 chars are not single letters


@functools.lru_cache(maxsize=32)
def _get_pattern(lang: str) -> re.Pattern:
    cc = _LANG_CHAR_CLASS.get(lang, _LANG_CHAR_CLASS["en"])
    return re.compile(
        r"(?<![\w’’’ʼ])"  # noqa: RUF001  # not preceded by word char or apostrophe
        r"(" + cc + r"(?:[ ]" + cc + r"){1,}"  # 2+ spaced single letters
        r"(?:(?<=[A-Z])s)?)"              # optional ASCII plural 's' after uppercase
        r"(?!\w)"                         # not followed by a word character
    )


def _strip_particles(raw: str, particles: frozenset[str]) -> str:
    """Generalization of the old _strip_article_a for any set of single-letter particles."""
    if not particles:
        return raw
    parts = raw.split(" ")
    if parts[0] in particles:
        parts = parts[1:]
    if parts and parts[-1] in particles:
        keep_trailing = (["D", "N"], ["R", "N"])  # DNA, RNA
        preceding = [p.upper() for p in parts[:-1]]
        if preceding[-_TAIL_SLICE:] not in keep_trailing:
            parts = parts[:-1]
    if len(parts) < _MIN_PARTS:
        return raw
    return " ".join(parts)


def _is_mixed_case_pair(letters: str) -> bool:
    """True for a 2-letter mix of lower+upper like 'xI' or 'Ia' — not a real abbreviation."""
    return len(letters) == _MIN_PARTS and letters[0].islower() != letters[1].islower()


def _is_double_i(raw: str) -> bool:
    """True when the match is exactly two uppercase 'I' letters ("I I")."""
    return raw == "I I"


def _join_letters(match: re.Match, particles: frozenset[str]) -> str:  # noqa: C901, PLR0911
    """Remove spaces between matched single letters, skipping false positives."""
    raw = match.group(0)
    if _is_double_i(raw):
        return raw

    parts = raw.split(" ")
    if any(len(p) >= _MULTI_CHAR_LEN for p in parts):
        return raw

    # "Is", "As", "Os" etc. are words, not plural abbreviation suffixes — pop and reattach.
    tail = ""
    if (len(parts) >= _MIN_PARTS
            and len(parts[-1]) == _PLURAL_SUFFIX_LEN
            and parts[-1][1] == "s"
            and parts[-1][0] in _VOWELS):
        tail = " " + parts.pop()
        if len(parts) < _MIN_PARTS:
            return raw
        raw = " ".join(parts)

    if sum(1 for p in parts if len(p) == 1) < _MIN_PARTS:
        has_plural_suffix = any(
            len(p) == _PLURAL_SUFFIX_LEN and p[1] == "s" and p[0].lower() not in "aeiou"
            for p in parts
        )
        if not has_plural_suffix:
            return raw

    if len(parts) == _MIN_PARTS and particles and any(p in particles for p in parts):
        return raw

    letters = raw.replace(" ", "")
    if len(set(letters.upper())) <= 1:
        return raw
    if _is_mixed_case_pair(letters):
        return raw

    stripped = _strip_particles(raw, particles)
    if stripped == raw:
        return letters + tail
    if len(stripped.replace(" ", "")) < _MIN_PARTS:
        return raw

    prefix = raw[: raw.index(stripped[0])]
    suffix = raw[raw.rindex(stripped[-1]) + 1 :]
    return prefix + stripped.replace(" ", "") + suffix + tail


def concat_abbreviations(text: str, language: str = "en") -> tuple[str, list[str]]:
    """Join sequences of spaced-out single letters into abbreviations.

    Returns ``(result_text, abbreviations)`` where *abbreviations* is a
    list of the joined abbreviation strings that were produced.

    >>> concat_abbreviations("the A P I uses A D X format")
    ('the API uses ADX format', ['API', 'ADX'])
    >>> concat_abbreviations("at the U K's major conference")
    ("at the UK's major conference", ['UK'])
    """
    pattern = _get_pattern(language)
    particles = _LANG_PARTICLES.get(language, frozenset())
    found: list[str] = []

    def _collect_and_join(match: re.Match) -> str:
        raw = match.group(0)
        replaced = _join_letters(match, particles)
        if replaced != raw:
            # Trailing "I" absorbed from "I'm", "I'll" etc.: trim it.
            end = match.end()
            if (replaced
                    and replaced[-1].upper() == "I"
                    and len(replaced) >= _MULTI_CHAR_LEN
                    and end < len(text)
                    and text[end] in "’’’ʼ"):  # noqa: RUF001
                after = text[end + 1 : end + 4].lower()
                if any(after.startswith(s) for s in _CONTRACTION_SUFFIXES):
                    replaced = replaced[:-1]
            abbrev = replaced.strip().rstrip("’s").rstrip("’s")  # noqa: RUF001
            if abbrev:
                found.append(abbrev)
        return replaced

    result = pattern.sub(_collect_and_join, text)
    return result, found


@dataclass
class AbbreviationConcatStage(ProcessingStage[AudioTask, AudioTask]):
    """Concatenate spaced-out single-letter sequences into abbreviations.

    ASR models sometimes transcribe abbreviations as individual letters
    separated by spaces (e.g. "A P I" instead of "API").  This stage
    detects runs of two or more standalone single letters and joins them.

    Handles trailing possessives/contractions naturally:
    "U K's" becomes "UK's" because the apostrophe is not a word character.

    Reads from ``text_key``, writes to ``output_text_key``.  When both
    keys are the same the field is updated in-place.

    Language is resolved per-sample from ``source_lang_key`` in the task data,
    enabling language-specific character classes and particle stripping.
    """

    text_key: str = "text"
    output_text_key: str = "abbreviated_text"
    abbreviations_key: str = "abbreviations"
    skip_me_key: str = "_skip_me"
    source_lang_key: str = "source_lang"
    name: str = "AbbreviationConcat"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_text_key, self.abbreviations_key]

    def _process_single(self, task: AudioTask) -> AudioTask:
        skip = task.data.get(self.skip_me_key, "")
        if skip:
            task.data.setdefault(self.output_text_key, "")
            task.data.setdefault(self.abbreviations_key, [])
            return task

        text = task.data.get(self.text_key, "")
        if not isinstance(text, str) or not text.strip():
            task.data.setdefault(self.output_text_key, text if isinstance(text, str) else "")
            task.data.setdefault(self.abbreviations_key, [])
            return task

        result, found = concat_abbreviations(text, language=task.data[self.source_lang_key])
        if result != text:
            logger.trace("AbbreviationConcat: {!r} → {!r}  abbrevs={}", text, result, found)
        task.data[self.output_text_key] = result
        task.data[self.abbreviations_key] = found
        return task

    def process(self, task: AudioTask) -> AudioTask:
        return self._process_single(task)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        return [self._process_single(task) for task in tasks]
