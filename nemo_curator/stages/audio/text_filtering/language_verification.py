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

"""Flag rows whose LLM language prediction contradicts ``source_lang``."""

from dataclasses import dataclass, field

from nemo_curator.stages.audio.text_filtering.language_utils import verify_language
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class LanguageVerificationStage(ProcessingStage[AudioTask, AudioTask]):
    """Check ``prediction_key`` against ``language_key`` and set ``skip_me_key`` on a mismatch.

    This is the model-free half of language ID. Keeping it out of the LLM call lets the
    prediction run as a plain operation inside one fused ``AudioTextLLMStage``.
    """

    prediction_key: str = "llm_language_prediction"
    language_key: str = "source_lang"
    skip_me_key: str = "_skipme"
    notes_key: str = "additional_notes"
    name: str = "LanguageVerification"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.prediction_key, self.language_key, self.skip_me_key, self.notes_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.skip_me_key, self.notes_key]

    def process(self, task: AudioTask) -> AudioTask:
        verify_language(task.data, self.prediction_key, self.language_key, self.skip_me_key, self.notes_key)
        return task
