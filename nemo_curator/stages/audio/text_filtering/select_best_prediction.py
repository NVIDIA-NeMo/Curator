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

from dataclasses import dataclass, field

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class SelectBestPredictionStage(ProcessingStage[AudioTask, AudioTask]):
    """Select the best available prediction and write it to ``best_prediction``.

    If QwenASR recovered a hallucinated sample (``notes_key`` contains
    "Recovered" and ``asr_text_key`` has a non-empty value), the ASR
    prediction is used.  Otherwise the primary prediction
    (``primary_text_key``) is used.

    This allows downstream stages (FastTextLID, RegexSubstitution) to
    always read from ``best_prediction`` regardless of which model
    produced the final text.
    """

    primary_text_key: str = "qwen3_prediction_s1"
    asr_text_key: str = "qwen3_asr_prediction"
    output_key: str = "best_prediction"
    notes_key: str = "additional_notes"
    name: str = "SelectBestPrediction"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.primary_text_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_key]

    def process(self, task: AudioTask) -> AudioTask:
        asr_pred = task.data.get(self.asr_text_key, "")
        notes = str(task.data.get(self.notes_key, ""))

        if "Recovered" in notes and asr_pred:
            task.data[self.output_key] = asr_pred
        else:
            task.data[self.output_key] = task.data.get(self.primary_text_key, "")

        return task
