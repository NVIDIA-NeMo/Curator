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

import re
from dataclasses import dataclass, field
from typing import Any

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)


def _set_note(data: dict[str, Any], stage: str, value: str, notes_key: str) -> None:
    notes = data.get(notes_key)
    if not isinstance(notes, dict):
        notes = {}
        data[notes_key] = notes
    notes[stage] = value


def _normalized_words(text: str) -> list[str]:
    return _PUNCTUATION.sub("", text).lower().split()


def _word_error_rate_percent(reference: str, hypothesis: str) -> float:
    """Return word-level Levenshtein distance as a percentage."""
    ref = _normalized_words(reference)
    hyp = _normalized_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 100.0

    previous = list(range(len(hyp) + 1))
    for ref_index, ref_word in enumerate(ref, start=1):
        current = [ref_index]
        for hyp_index, hyp_word in enumerate(hyp, start=1):
            current.append(
                min(
                    previous[hyp_index] + 1,
                    current[hyp_index - 1] + 1,
                    previous[hyp_index - 1] + (ref_word != hyp_word),
                )
            )
        previous = current
    return round(previous[-1] / len(ref) * 100.0, 2)


@dataclass
class SelectBestPredictionStage(ProcessingStage[AudioTask, AudioTask]):
    """Choose one stable transcript from primary, recovery, or reference text.

    Selection is ordered so explicit reference policies run before model
    recovery. Short Qwen-Omni clips can use reference text to avoid known
    sub-second hallucinations, and ``force_reference`` can make reference text
    authoritative for datasets whose model output is not trusted. Otherwise,
    the stage handles unsupported languages, recovered fallback predictions,
    reference recovery, cross-model agreement, and the normal primary result.
    ``asr_text_key`` is a compatibility alias that overrides
    ``fallback_text_key`` when supplied.
    """

    primary_text_key: str = "primary_model_prediction"
    fallback_text_key: str = "fallback_model_prediction"
    asr_text_key: str | None = None
    output_key: str = "best_prediction"
    source_key: str = "best_prediction_source"
    notes_key: str = "additional_notes"
    skip_me_key: str = "_skipme"
    duration_key: str = "duration"
    min_agreement_pct: float = 80.0
    agreement_wer_key: str = "primary_fallback_agreement_wer"
    primary_source_label: str = "primary"
    fallback_source_label: str = "fallback"
    reference_text_key: str | None = None
    use_reference_on_hallucination: bool = False
    force_reference: bool = False
    use_ground_truth_for_short_audio: bool = True
    short_audio_threshold: float = 1.0
    primary_model_type: str | None = None
    reference_source_label: str = "reference"
    ground_truth_source_label: str = "ground_truth"
    name: str = "SelectBestPrediction"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        keys = [self.primary_text_key]
        if self.reference_text_key:
            keys.append(self.reference_text_key)
        return [], keys

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_key, self.source_key, self.skip_me_key, self.agreement_wer_key, self.notes_key]

    def process(self, task: AudioTask) -> AudioTask:  # noqa: C901, PLR0911, PLR0912, PLR0915
        task.data.pop(self.agreement_wer_key, None)

        if (
            self.use_ground_truth_for_short_audio
            and self.reference_text_key
            and self.primary_model_type == "qwen_omni"
        ):
            duration_raw = task.data.get(self.duration_key)
            try:
                duration = float(duration_raw)
            except (TypeError, ValueError):
                duration = None
            if duration is not None and 0.0 < duration < self.short_audio_threshold:
                reference = str(task.data.get(self.reference_text_key, "") or "").strip()
                if reference:
                    task.data[self.output_key] = reference
                    task.data[self.source_key] = self.ground_truth_source_label
                    task.data[self.skip_me_key] = ""
                    _set_note(
                        task.data,
                        self.name,
                        f"Ground Truth (short audio {duration:.2f}s < {self.short_audio_threshold}s)",
                        self.notes_key,
                    )
                    return task

        resolved_fallback_text_key = self.asr_text_key or self.fallback_text_key
        primary = str(task.data.get(self.primary_text_key, "") or "")
        fallback = str(task.data.get(resolved_fallback_text_key, "") or "")
        notes = task.data.get(self.notes_key)
        notes = notes if isinstance(notes, dict) else {}
        skip_reason = str(task.data.get(self.skip_me_key, "") or "")

        if self.force_reference and self.reference_text_key:
            reference = str(task.data.get(self.reference_text_key, "") or "").strip()
            task.data[self.output_key] = reference
            task.data[self.source_key] = self.ground_truth_source_label
            task.data[self.skip_me_key] = ""
            _set_note(task.data, self.name, "forced:ground_truth", self.notes_key)
            return task

        primary_unsupported = "lang_not_supported" in str(notes.get(self.primary_text_key, ""))
        fallback_unsupported = "lang_not_supported" in str(notes.get(resolved_fallback_text_key, ""))

        if primary_unsupported and not fallback and self.reference_text_key:
            reference = str(task.data.get(self.reference_text_key, "") or "").strip()
            if reference:
                task.data[self.output_key] = reference
                task.data[self.source_key] = self.ground_truth_source_label
                task.data[self.skip_me_key] = ""
                _set_note(task.data, self.name, "Ground Truth", self.notes_key)
                return task

        if primary_unsupported and fallback_unsupported:
            task.data[self.output_key] = ""
            task.data[self.source_key] = "none"
            task.data[self.skip_me_key] = "not_supported"
            _set_note(task.data, self.name, "skipped:both_models_lang_not_supported", self.notes_key)
            return task

        if primary_unsupported and fallback:
            task.data[self.output_key] = fallback
            task.data[self.source_key] = self.fallback_source_label
            task.data[self.skip_me_key] = ""
            _set_note(
                task.data,
                self.name,
                f"used {self.fallback_source_label} (primary lang unsupported)",
                self.notes_key,
            )
            return task

        recovered = any("recovered" in str(value).lower() for value in notes.values())
        if recovered and fallback:
            task.data[self.output_key] = fallback
            task.data[self.source_key] = self.fallback_source_label
            _set_note(task.data, self.name, f"used {self.fallback_source_label}", self.notes_key)
            return task

        if self.use_reference_on_hallucination and self.reference_text_key and skip_reason.startswith("Hallucination"):
            reference = str(task.data.get(self.reference_text_key, "") or "").strip()
            if reference:
                task.data[self.output_key] = reference
                task.data[self.source_key] = self.reference_source_label
                task.data[self.skip_me_key] = ""
                _set_note(
                    task.data,
                    self.name,
                    f"recovered:reference_text (hallucination_detected, fallback={self.reference_text_key})",
                    self.notes_key,
                )
                return task

        if skip_reason.startswith("Hallucination") and primary and fallback:
            wer = _word_error_rate_percent(primary, fallback)
            task.data[self.agreement_wer_key] = wer
            if wer <= 100.0 - self.min_agreement_pct:
                task.data[self.output_key] = primary
                task.data[self.source_key] = self.primary_source_label
                task.data[self.skip_me_key] = ""
                _set_note(
                    task.data,
                    self.name,
                    f"recovered:cross_model_agreement (wer={wer:.1f}%)",
                    self.notes_key,
                )
                return task

        task.data[self.output_key] = primary
        task.data[self.source_key] = self.primary_source_label
        _set_note(task.data, self.name, f"used {self.primary_source_label}", self.notes_key)
        return task
