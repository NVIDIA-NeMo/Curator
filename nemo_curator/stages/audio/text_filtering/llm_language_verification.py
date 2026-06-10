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

"""Verify LLM language-ID predictions against per-sample ``source_lang``."""

from __future__ import annotations

from dataclasses import dataclass, field

from nemo_curator.stages.audio.pipeline_utils import LANG_CODE_TO_NAME, set_note
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

# Supported languages for verification (Name-code labels written to ``skip_me_key``).
_LANG_NAME_TO_CODE: dict[str, str] = {name: code for code, name in LANG_CODE_TO_NAME.items()}

_LANG_PREFIXES = (
    "the text is in ",
    "the language is ",
    "it is ",
    "this is ",
    "written in ",
)


def _format_lang_label(code: str) -> str:
    """Return ``English-en`` style label for a normalized ISO 639-1 code."""
    name = LANG_CODE_TO_NAME.get(code, code)
    return f"{name}-{code}"


def normalize_language_to_code(value: str) -> str | None:
    """Map a manifest value or LLM prediction to an ISO 639-1 code."""
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
    """Parse an LLM language-ID output into ``(primary_code, detected_codes)``.

    Recognizes the ``Primary: <name>`` / ``Languages: <a, b>`` format; falls back
    to treating the whole string as a single language name.  ``detected_codes``
    is de-duplicated with the primary first; >1 entry means code-switching.
    """
    primary_raw: str | None = None
    langs_raw: list[str] = []
    for line in raw.splitlines():
        lowered = line.strip().lower()
        if lowered.startswith("primary:"):
            primary_raw = line.split(":", 1)[1].strip()
        elif lowered.startswith("languages:"):
            langs_raw = [p.strip() for p in line.split(":", 1)[1].split(",") if p.strip()]

    if primary_raw is None:
        primary_raw = langs_raw[0] if langs_raw else raw.strip()

    primary_code = normalize_language_to_code(primary_raw)

    detected_codes: list[str] = []
    for value in [primary_raw, *langs_raw]:
        code = normalize_language_to_code(value)
        if code and code not in detected_codes:
            detected_codes.append(code)

    return primary_code, detected_codes


@dataclass
class LLMLanguageVerificationStage(ProcessingStage[AudioTask, AudioTask]):
    """Compare ``llm_language_prediction`` to ``source_lang`` and flag mismatches.

    The prediction reports a primary language plus all languages present.
    Code-switched transcripts that still contain ``source_lang`` are kept (only
    noted); mismatches and code-switches missing ``source_lang`` are flagged
    ``"Wrong language:<stage>"`` (``"Unparseable language:<stage>"`` when the
    prediction can't be parsed), following ``FastTextLIDStage``'s
    ``"<reason>:<stage>"`` convention.  An already non-empty ``skip_me_key`` is
    never overwritten.

    Args:
        prediction_key: Manifest key holding the LLM language name/code output.
        source_lang_key: Manifest key holding the expected language code or name.
        skip_me_key: Key used to flag entries to skip downstream.
        notes_key: Key holding the ``additional_notes`` dict.
    """

    prediction_key: str = "llm_language_prediction"
    source_lang_key: str = "source_lang"
    skip_me_key: str = "_skipme"
    notes_key: str = "additional_notes"
    name: str = "LLMLanguageVerification"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.prediction_key, self.source_lang_key, self.skip_me_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.skip_me_key]

    def _process_single(self, task: AudioTask) -> AudioTask:
        if task.data.get(self.skip_me_key, ""):
            set_note(task.data, self.name, "skipped (already flagged)", self.notes_key)
            return task

        raw_prediction = task.data.get(self.prediction_key, "")
        raw_expected = task.data.get(self.source_lang_key, "")

        if not isinstance(raw_prediction, str) or not raw_prediction.strip():
            set_note(task.data, self.name, "skipped (empty prediction)", self.notes_key)
            return task

        expected_code = normalize_language_to_code(str(raw_expected))
        primary_code, detected_codes = parse_language_prediction(raw_prediction)

        if expected_code is None:
            set_note(
                task.data,
                self.name,
                f"skipped (unknown source_lang={raw_expected!r})",
                self.notes_key,
            )
            return task

        if primary_code is None:
            if not task.data.get(self.skip_me_key):
                task.data[self.skip_me_key] = f"Unparseable language:{self.name}"
            set_note(
                task.data,
                self.name,
                f"unparseable prediction ({raw_prediction.strip()!r}, expected={_format_lang_label(expected_code)})",
                self.notes_key,
            )
            return task

        langs_label = ", ".join(_format_lang_label(c) for c in detected_codes)
        expected_label = _format_lang_label(expected_code)

        # Keep code-switched samples that still contain source_lang; flag only when it's absent.
        if len(detected_codes) > 1:
            if expected_code in detected_codes:
                set_note(
                    task.data,
                    self.name,
                    f"passed code-switch (primary={_format_lang_label(primary_code)}, "
                    f"langs=[{langs_label}], expected={expected_label})",
                    self.notes_key,
                )
            else:
                if not task.data.get(self.skip_me_key):
                    task.data[self.skip_me_key] = f"Wrong language:{self.name}"
                set_note(
                    task.data,
                    self.name,
                    f"wrong language code-switch (langs=[{langs_label}], expected={expected_label})",
                    self.notes_key,
                )
            return task

        if primary_code != expected_code:
            if not task.data.get(self.skip_me_key):
                task.data[self.skip_me_key] = f"Wrong language:{self.name}"
            set_note(
                task.data,
                self.name,
                f"wrong language (predicted={_format_lang_label(primary_code)}, expected={expected_label})",
                self.notes_key,
            )
        else:
            set_note(
                task.data,
                self.name,
                f"passed ({_format_lang_label(primary_code)})",
                self.notes_key,
            )
        return task

    def process(self, task: AudioTask) -> AudioTask:
        return self._process_single(task)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        return [self._process_single(task) for task in tasks]
