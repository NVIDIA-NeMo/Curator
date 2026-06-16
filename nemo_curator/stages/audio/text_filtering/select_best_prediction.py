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

from loguru import logger

from nemo_curator.stages.audio.metrics.get_wer import get_wer
from nemo_curator.stages.audio.pipeline_utils import set_note
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize_for_wer(text: str) -> str:
    """Lowercase and strip punctuation so WER focuses on word content."""
    return _PUNCT_RE.sub("", text).lower()


@dataclass
class SelectBestPredictionStage(ProcessingStage[AudioTask, AudioTask]):
    """Select the best available prediction and write it to ``best_prediction``.

    Selection priority:

    1. **ASR recovery** -- if ``notes_key`` contains "Recovered" and
       ``asr_text_key`` is non-empty, the ASR prediction is used.
    2. **Cross-model agreement** -- if *both* omni and ASR were flagged as
       hallucinated yet their texts agree (WER between them ≤
       ``100 - min_agreement_pct``), the omni prediction is kept and the
       sample is marked recovered because two independent models producing
       near-identical output is strong evidence the text is correct.
    3. **Fallback** -- the primary (omni) prediction is used as-is.

    When ``use_reference_on_hallucination`` is enabled and the primary
  output is flagged as a hallucination, the text at
  ``reference_text_key`` (e.g. the dataset's original transcript) is
  used instead, if non-empty.

    This allows downstream stages (FastTextLID, RegexSubstitution) to
    always read from ``best_prediction`` regardless of which model
    produced the final text.

    The model that produced the final text is recorded in
    ``source_key`` (default ``best_prediction_source``): ``"primary"``
    when the primary model's prediction is kept, or ``"fallback"``
    when the recovery model's prediction is chosen.
    """

    primary_text_key: str = "primary_model_prediction"
    asr_text_key: str = "fallback_model_prediction"
    output_key: str = "best_prediction"
    source_key: str = "best_prediction_source"
    notes_key: str = "additional_notes"
    skip_me_key: str = "_skipme"
    min_agreement_pct: float = 80.0
    agreement_wer_key: str = "omni_asr_agreement_wer"
    primary_source_label: str = "primary"
    fallback_source_label: str = "fallback"
    reference_text_key: str | None = None
    use_reference_on_hallucination: bool = False
    reference_source_label: str = "reference"
    name: str = "SelectBestPrediction"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        keys = [self.primary_text_key]
        if self.reference_text_key:
            keys.append(self.reference_text_key)
        return [], keys

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_key, self.skip_me_key, self.agreement_wer_key, self.source_key]

    def process(self, task: AudioTask) -> AudioTask:
        primary_pred = task.data.get(self.primary_text_key, "")
        asr_pred = task.data.get(self.asr_text_key, "")
        notes = task.data.get(self.notes_key, {})
        skip_me = str(task.data.get(self.skip_me_key, ""))

        notes_dict = notes if isinstance(notes, dict) else {}
        primary_lang_skipped = "lang_not_supported" in str(notes_dict.get(self.primary_text_key, ""))
        fallback_lang_skipped = "lang_not_supported" in str(notes_dict.get(self.asr_text_key, ""))

        # Case 3: both models skipped (language not supported by either)
        if primary_lang_skipped and fallback_lang_skipped:
            task.data[self.output_key] = ""
            task.data[self.source_key] = "none"
            task.data[self.skip_me_key] = "not_supported"
            set_note(task.data, self.name, "skipped:both_models_lang_not_supported", self.notes_key)
            return task

        # Case 1: primary skipped (lang unsupported), fallback has a transcription
        if primary_lang_skipped and asr_pred:
            task.data[self.output_key] = asr_pred
            task.data[self.source_key] = self.fallback_source_label
            set_note(task.data, self.name, f"used {self.fallback_source_label} (primary lang unsupported)", self.notes_key)
            return task

        # Hallucination recovery: primary was hallucinated, fallback available
        has_recovery = any("recovered" in str(v).lower() for v in notes_dict.values())
        if has_recovery and asr_pred:
            task.data[self.output_key] = asr_pred
            task.data[self.source_key] = self.fallback_source_label
            set_note(task.data, self.name, f"used {self.fallback_source_label}", self.notes_key)
            return task

        # Reference fallback: primary hallucinated, use existing dataset text
        if (
            self.use_reference_on_hallucination
            and self.reference_text_key
            and skip_me.startswith("Hallucination")
        ):
            ref_text = str(task.data.get(self.reference_text_key, "") or "").strip()
            if ref_text:
                task.data[self.output_key] = ref_text
                task.data[self.source_key] = self.reference_source_label
                task.data[self.skip_me_key] = ""
                set_note(
                    task.data,
                    self.name,
                    f"used {self.reference_source_label} (primary hallucination)",
                    self.notes_key,
                )
                return task

        # Cross-model agreement: both hallucinated but texts match
        both_hallucinated = skip_me.startswith("Hallucination") and asr_pred
        if both_hallucinated and primary_pred:
            wer = get_wer(_normalize_for_wer(primary_pred), _normalize_for_wer(asr_pred))
            task.data[self.agreement_wer_key] = wer
            if wer <= (100.0 - self.min_agreement_pct):
                logger.debug(
                    f"[{self.name}] cross-model agreement recovery: WER={wer:.1f}% "
                    f"(threshold {100.0 - self.min_agreement_pct:.1f}%), keeping omni prediction"
                )
                task.data[self.output_key] = primary_pred
                task.data[self.source_key] = self.primary_source_label
                task.data[self.skip_me_key] = ""
                set_note(task.data, self.name, f"recovered:cross_model_agreement (wer={wer:.1f}%)", self.notes_key)
                return task

        # Case 2 (default): primary OK (fallback may have been skipped or is irrelevant)
        task.data[self.output_key] = primary_pred
        task.data[self.source_key] = self.primary_source_label
        set_note(task.data, self.name, f"used {self.primary_source_label}", self.notes_key)
        return task
