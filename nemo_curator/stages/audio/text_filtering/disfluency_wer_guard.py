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

from dataclasses import dataclass, field

from nemo_curator.stages.audio.metrics.get_wer import get_wer
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class DisfluencyWerGuardStage(ProcessingStage[AudioTask, AudioTask]):
    """Fall back to the base prediction when disfluency removal diverges too much.

    Computes word-error-rate between ``ref_text_key`` (turn-1 prediction)
    and ``hyp_text_key`` (turn-2 / disfluency-cleaned prediction).  If
    WER exceeds ``max_wer_pct``, the turn-2 field is overwritten with the
    turn-1 value because the disfluency pass changed the content too
    aggressively.

    The computed WER is always stored in ``wer_key`` for downstream
    inspection regardless of whether a fallback occurred.
    """

    name: str = "DisfluencyWerGuard"
    ref_text_key: str = "qwen3_prediction_s1"
    hyp_text_key: str = "qwen3_prediction_s2"
    wer_key: str = "disfluency_wer"
    max_wer_pct: float = 50.0
    skip_me_key: str = "_skip_me"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.ref_text_key, self.hyp_text_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.hyp_text_key, self.wer_key]

    def _process_single(self, task: AudioTask) -> AudioTask:
        if task.data.get(self.skip_me_key, ""):
            task.data.setdefault(self.wer_key, -1.0)
            return task

        ref = task.data.get(self.ref_text_key, "")
        hyp = task.data.get(self.hyp_text_key, "")

        if not ref or not hyp:
            task.data[self.wer_key] = -1.0
            return task

        wer = get_wer(ref, hyp)
        task.data[self.wer_key] = wer

        if wer > self.max_wer_pct:
            task.data[self.hyp_text_key] = ref

        return task

    def process(self, task: AudioTask) -> AudioTask:
        return self._process_single(task)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        return [self._process_single(task) for task in tasks]
