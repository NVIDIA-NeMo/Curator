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

import re
from dataclasses import dataclass, field

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

# Matches 2+ standalone single letters separated by spaces.
# Examples: "A P I" → "API", "U K" → "UK", "A D X" → "ADX"
# Word-boundary anchors prevent matching letters inside longer words.
_ABBREV_PATTERN = re.compile(
    r"(?<![\w'\u2018\u2019\u02BC])" # not preceded by a word char or apostrophe
    r"([A-Za-z]"                  # first standalone letter
    r"(?:[ ][A-Za-z]){1,}"        # one or more: space + standalone letter
    r"(?:(?<=[A-Z])[a-z]{1,2})?)" # optional lowercase suffix after uppercase (e.g. Rs, Ds)
    r"(?!\w)"                     # not followed by a word character
)


def _strip_article_a(raw: str) -> str:
    """Strip a leading or trailing article 'a' (lowercase only) from the match.

    Returns the trimmed match string (still with inner spaces) so that
    only the real abbreviation letters get joined.  If stripping leaves
    fewer than 2 single letters the original string is returned unchanged.
    """
    parts = raw.split(" ")

    if parts[0] == "a":
        parts = parts[1:]
    if parts and parts[-1] == "a":
        preceding = [p.upper() for p in parts[:-1]]
        _KEEP_TRAILING_A = (["D", "N"], ["R", "N"])
        if preceding[-2:] not in _KEEP_TRAILING_A:
            parts = parts[:-1]

    if len(parts) < 2:
        return raw
    return " ".join(parts)


def _is_mixed_case_pair(letters: str) -> bool:
    """True for a 2-letter mix of lower+upper like 'xI' or 'Ia' — not a real abbreviation."""
    return len(letters) == 2 and letters[0].islower() != letters[1].islower()


def _is_double_I(raw: str) -> bool:
    """True when the match is exactly two uppercase 'I' letters ("I I")."""
    return raw == "I I"


def _join_letters(match: re.Match) -> str:
    """Remove spaces between matched single letters, skipping false positives."""
    raw = match.group(0)
    if _is_double_I(raw):
        return raw

    parts = raw.split(" ")
    if any(len(p) >= 3 for p in parts):
        return raw
    if sum(1 for p in parts if len(p) == 1) < 2:
        has_plural_suffix = any(
            len(p) == 2 and p[1] == "s" and p[0].lower() not in "aeiou"
            for p in parts
        )
        if not has_plural_suffix:
            return raw

    letters = raw.replace(" ", "")
    if len(set(letters.upper())) <= 1:
        return raw
    if _is_mixed_case_pair(letters):
        return raw

    stripped = _strip_article_a(raw)
    if stripped == raw:
        return letters
    if len(stripped.replace(" ", "")) < 2:
        return raw

    prefix = raw[: raw.index(stripped[0])]
    suffix = raw[raw.rindex(stripped[-1]) + 1 :]
    return prefix + stripped.replace(" ", "") + suffix


def concat_abbreviations(text: str) -> tuple[str, list[str]]:
    """Join sequences of spaced-out single letters into abbreviations.

    Returns ``(result_text, abbreviations)`` where *abbreviations* is a
    list of the joined abbreviation strings that were produced.

    >>> concat_abbreviations("the A P I uses A D X format")
    ('the API uses ADX format', ['API', 'ADX'])
    >>> concat_abbreviations("at the U K's major conference")
    ("at the UK's major conference", ['UK'])
    """
    found: list[str] = []

    def _collect_and_join(match: re.Match) -> str:
        raw = match.group(0)
        replaced = _join_letters(match)
        if replaced != raw:
            abbrev = replaced.strip().rstrip("'s").rstrip("\u2019s")
            if abbrev:
                found.append(abbrev)
        return replaced

    result = _ABBREV_PATTERN.sub(_collect_and_join, text)
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
    """

    text_key: str = "cleaned_text"
    output_text_key: str = "cleaned_text"
    abbreviations_key: str = "abbreviations"
    skip_me_key: str = "skip_me"
    name: str = "AbbreviationConcat"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_text_key, self.abbreviations_key]

    def process(self, task: AudioTask) -> AudioTask:
        skip = task.data.get(self.skip_me_key, "")
        if skip:
            task.data.setdefault(self.abbreviations_key, [])
            return task

        text = task.data.get(self.text_key, "")
        if not isinstance(text, str) or not text.strip():
            task.data.setdefault(self.abbreviations_key, [])
            return task

        result, found = concat_abbreviations(text)
        if result != text:
            logger.trace("AbbreviationConcat: {!r} → {!r}  abbrevs={}", text, result, found)
        task.data[self.output_text_key] = result
        task.data[self.abbreviations_key] = found
        return task
