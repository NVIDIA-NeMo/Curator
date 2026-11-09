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

"""Guard a disfluency-removal pass against excessive transcript changes."""

from dataclasses import dataclass, field

from nemo_curator.stages.audio.text_filtering.scriptio_continua import is_scriptio_continua
from nemo_curator.stages.audio.text_filtering.text_metrics import (
    character_error_rate_percent,
    word_error_rate_percent,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class DisfluencyWerGuardStage(ProcessingStage[AudioTask, AudioTask]):
    """Restore the original prediction when disfluency removal diverges.

    The stage compares ``ref_text_key`` with the disfluency-cleaned text at
    ``hyp_text_key``. It uses character error rate (CER) for languages written
    without word-separating spaces and word error rate (WER) otherwise. If the
    selected error rate exceeds ``max_wer_pct``, the cleaned prediction is
    replaced with the original prediction.

    The selected rate is stored in the compatibility field ``wer_key`` even
    when CER is used, matching the reference pipeline contract.
    """

    ref_text_key: str = "qwen3_prediction_s1"
    hyp_text_key: str = "qwen3_prediction_s2"
    wer_key: str = "disfluency_wer"
    max_wer_pct: float = 50.0
    skip_me_key: str = "_skipme"
    language_key: str = "source_lang"
    name: str = "DisfluencyWerGuard"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.ref_text_key, self.hyp_text_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.hyp_text_key, self.wer_key]

    def process(self, task: AudioTask) -> AudioTask:
        if task.data.get(self.skip_me_key, ""):
            task.data.setdefault(self.wer_key, -1.0)
            return task

        reference = task.data.get(self.ref_text_key, "")
        hypothesis = task.data.get(self.hyp_text_key, "")
        if not reference or not hypothesis:
            task.data[self.wer_key] = -1.0
            return task

        metric_fn = (
            character_error_rate_percent
            if is_scriptio_continua(task.data.get(self.language_key))
            else word_error_rate_percent
        )
        error_rate = metric_fn(str(reference), str(hypothesis))
        task.data[self.wer_key] = error_rate
        if error_rate > self.max_wer_pct:
            task.data[self.hyp_text_key] = reference
        return task
