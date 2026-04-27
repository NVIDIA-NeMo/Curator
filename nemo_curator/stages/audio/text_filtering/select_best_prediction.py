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

    This allows downstream stages (FastTextLID, RegexSubstitution) to
    always read from ``best_prediction`` regardless of which model
    produced the final text.
    """

    primary_text_key: str = "qwen3_prediction_s1"
    asr_text_key: str = "qwen3_asr_prediction"
    output_key: str = "best_prediction"
    notes_key: str = "additional_notes"
    skip_me_key: str = "_skip_me"
    min_agreement_pct: float = 80.0
    agreement_wer_key: str = "omni_asr_agreement_wer"
    name: str = "SelectBestPrediction"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.primary_text_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_key, self.skip_me_key, self.agreement_wer_key]

    def process(self, task: AudioTask) -> AudioTask:
        primary_pred = task.data.get(self.primary_text_key, "")
        asr_pred = task.data.get(self.asr_text_key, "")
        notes = str(task.data.get(self.notes_key, ""))
        skip_me = str(task.data.get(self.skip_me_key, ""))

        if "Recovered" in notes and asr_pred:
            task.data[self.output_key] = asr_pred
            return task

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
                task.data[self.notes_key] = "Recovered:CrossModelAgreement"
                task.data[self.skip_me_key] = ""
                return task

        task.data[self.output_key] = primary_pred
        return task
