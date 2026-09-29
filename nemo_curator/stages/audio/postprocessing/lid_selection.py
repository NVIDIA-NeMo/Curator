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

"""Select one language from the audio-LID ensemble's component results."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

_DEFAULT_INDIC_LANGUAGES: frozenset[str] = frozenset(
    {
        "hi",
        "ta",
        "bn",
        "ur",
        "gu",
        "mr",
        "ml",
        "kn",
        "te",
        "or",
        "as",
        "pa",
        "ne",
        "sa",
        "sd",
        "si",
        "kok",
        "mai",
        "doi",
        "ks",
        "mni",
        "sat",
        "brx",
        "bo",
    }
)

_MISSING_PREDICTIONS = "skipped due to missing langID predictions."
_MISSING_WHISPER = "skipped due to missing or empty whisper langID prediction."
_DISAGREEMENT = "skipped due to disagreement between langID models."
_SELECTOR_SKIP_REASONS = frozenset({_MISSING_PREDICTIONS, _MISSING_WHISPER, _DISAGREEMENT})


@dataclass(frozen=True)
class _Prediction:
    language: str
    confidence: float
    tag: str


def _set_note(data: dict[str, Any], key: str, value: Any, notes_key: str) -> None:  # noqa: ANN401
    """Write a note while repairing a missing or non-mapping notes field."""
    notes = data.get(notes_key)
    if not isinstance(notes, dict):
        notes = {}
        data[notes_key] = notes
    notes[key] = value


@dataclass
class SelectAudioLanguageStage(ProcessingStage[AudioTask, AudioTask]):
    """Apply the reference ensemble precedence policy to JSON-safe LID results.

    The stable component IDs are configurable, while their defaults preserve
    the reference pipeline's exact names. Results are removed after selection
    because their model-specific confidence values have served their routing
    purpose; the chosen confidence remains under ``notes_key``.

    Successful selection clears a pre-existing ``_skipme`` only when its value
    is one of this selector's own three rejection reasons. An unrelated skip
    reason belongs to another pipeline stage and is intentionally preserved.

    Precedence is: missing results; unconditional Whisper English; missing or
    empty Whisper; three-way primary/Canary/Whisper agreement; primary/Whisper
    agreement for non-Indic languages; disagreement.
    """

    results_key: str = "lid"
    output_key: str = "source_lang"
    confidence_key: str = "source_lid_confidence"
    skip_me_key: str = "_skipme"
    notes_key: str = "additional_notes"

    speechbrain_model_id: str = "SpeechBrainLangID"
    ambernet_model_id: str = "AmberNetLangID"
    indic_canary_model_id: str = "IndicCanaryLangID"
    whisper_model_id: str = "WhisperLangID"
    indic_languages: frozenset[str] = field(default_factory=lambda: _DEFAULT_INDIC_LANGUAGES)

    name: str = "SelectBestLIDPrediction"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self) -> None:
        key_fields = (
            "results_key",
            "output_key",
            "confidence_key",
            "skip_me_key",
            "notes_key",
            "speechbrain_model_id",
            "ambernet_model_id",
            "indic_canary_model_id",
            "whisper_model_id",
            "name",
        )
        for field_name in key_fields:
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                msg = f"SelectAudioLanguageStage.{field_name} must be a non-empty string"
                raise ValueError(msg)

        model_ids = (
            self.speechbrain_model_id,
            self.ambernet_model_id,
            self.indic_canary_model_id,
            self.whisper_model_id,
        )
        if len(set(model_ids)) != len(model_ids):
            msg = "SelectAudioLanguageStage model IDs must be distinct"
            raise ValueError(msg)
        if not all(isinstance(language, str) and language for language in self.indic_languages):
            msg = "SelectAudioLanguageStage.indic_languages must contain only non-empty strings"
            raise ValueError(msg)
        self.indic_languages = frozenset(self.indic_languages)

    def inputs(self) -> tuple[list[str], list[str]]:
        # ``results_key`` is optional by design: upstream terminal skips and
        # model preparation failures can legitimately reach the selector with
        # no predictions, which is one of its explicit routing branches.
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_key, self.notes_key, self.skip_me_key]

    def process(self, task: AudioTask) -> AudioTask:  # noqa: C901
        raw_results = task.data.pop(self.results_key, {})
        if not raw_results:
            return self._reject(
                task,
                output="",
                reason=_MISSING_PREDICTIONS,
                note="skipped (missing predictions)",
            )
        if not isinstance(raw_results, dict):
            msg = f"task.data[{self.results_key!r}] must be a mapping, got {type(raw_results).__name__}"
            raise TypeError(msg)

        predictions = self._parse_predictions(raw_results)
        self._add_component_notes(task, predictions)

        primary: _Prediction | None = None
        canary: _Prediction | None = None
        whisper: _Prediction | None = None
        primary_ids = {self.speechbrain_model_id, self.ambernet_model_id}
        for model_id, prediction in predictions.items():
            if model_id in primary_ids:
                primary = prediction
            elif model_id == self.indic_canary_model_id:
                canary = prediction
            elif model_id == self.whisper_model_id:
                whisper = prediction

        if whisper is not None and whisper.language == "en":
            return self._accept(task, whisper, f"used {whisper.tag}, English language.")

        if whisper is None or not whisper.language:
            return self._reject(
                task,
                output="",
                reason=_MISSING_WHISPER,
                note="skipped (missing or empty whisper langID prediction)",
            )

        if (
            primary is not None
            and canary is not None
            and canary.language == primary.language
            and whisper.language == primary.language
        ):
            return self._accept(task, canary, f"used {canary.tag}, agreement between all 3 langID models.")

        if (
            whisper.language not in self.indic_languages
            and primary is not None
            and primary.language == whisper.language
        ):
            note = f"used {whisper.tag}, agreement between {primary.tag} and {whisper.tag} langID models."
            return self._accept(task, whisper, note)

        return self._reject(
            task,
            output="skipped",
            reason=_DISAGREEMENT,
            note=_DISAGREEMENT,
        )

    def _parse_predictions(self, raw_results: dict[str, Any]) -> dict[str, _Prediction]:
        allowed_ids = {
            self.speechbrain_model_id,
            self.ambernet_model_id,
            self.indic_canary_model_id,
            self.whisper_model_id,
        }
        predictions: dict[str, _Prediction] = {}
        for model_id, raw in raw_results.items():
            if model_id not in allowed_ids:
                msg = f"Invalid model name: {model_id}"
                raise ValueError(msg)
            if not isinstance(raw, dict):
                msg = f"Invalid LID result: {raw}"
                raise ValueError(msg)  # noqa: TRY004 - preserve the reference selector's public error type
            try:
                language = raw["language"]
                confidence = raw["confidence"]
                tag = raw["tag"]
            except KeyError as exc:
                msg = f"Invalid LID result for {model_id}: missing {exc.args[0]!r}"
                raise ValueError(msg) from exc
            if not isinstance(language, str) or not isinstance(tag, str):
                msg = f"Invalid LID result: {raw}"
                raise ValueError(msg)  # noqa: TRY004 - preserve the reference selector's public error type
            try:
                numeric_confidence = float(confidence)
            except (TypeError, ValueError) as exc:
                msg = f"Invalid LID result: {raw}"
                raise ValueError(msg) from exc
            if not math.isfinite(numeric_confidence) or not 0.0 <= numeric_confidence <= 1.0:
                msg = f"Invalid LID result: {raw}"
                raise ValueError(msg)
            predictions[model_id] = _Prediction(
                language=language,
                confidence=numeric_confidence,
                tag=tag,
            )
        return predictions

    def _add_component_notes(self, task: AudioTask, predictions: dict[str, _Prediction]) -> None:
        labels = {
            self.speechbrain_model_id: "speechbrain",
            self.ambernet_model_id: "ambernet",
            self.indic_canary_model_id: "indic_canary",
            self.whisper_model_id: "whisper",
        }
        for model_id, prediction in predictions.items():
            _set_note(task.data, f"{prediction.tag}_lid_model", labels[model_id], self.notes_key)
            _set_note(task.data, f"{prediction.tag}_lid_prediction", prediction.language, self.notes_key)
            _set_note(
                task.data,
                f"{prediction.tag}_lid_confidence",
                f"{prediction.confidence:.3f}",
                self.notes_key,
            )

    def _accept(self, task: AudioTask, prediction: _Prediction, note: str) -> AudioTask:
        task.data[self.output_key] = prediction.language
        _set_note(task.data, self.confidence_key, prediction.confidence, self.notes_key)
        _set_note(task.data, self.name, note, self.notes_key)
        if task.data.get(self.skip_me_key) in _SELECTOR_SKIP_REASONS:
            task.data.pop(self.skip_me_key, None)
        return task

    def _reject(self, task: AudioTask, *, output: str, reason: str, note: str) -> AudioTask:
        task.data[self.output_key] = output
        _set_note(task.data, self.confidence_key, 0.0, self.notes_key)
        existing_reason = task.data.get(self.skip_me_key)
        if not existing_reason or existing_reason in _SELECTOR_SKIP_REASONS:
            task.data[self.skip_me_key] = reason
        _set_note(task.data, self.name, note, self.notes_key)
        return task
