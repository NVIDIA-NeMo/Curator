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

"""Evaluation stage: compare the value a model recognized with the value that was spoken."""

import time
from dataclasses import dataclass

from nemo_curator.stages.audio.numpipe.numeric_norm import numeric_eq, parse_numeric
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


@dataclass
class NumericEqualityStage(ProcessingStage[AudioTask, AudioTask]):
    """Score a transcript by value against the canonical ``number``.

    The hypothesis may be words, digits or a hybrid; units belong to the value, so ``"50"`` is not
    accepted for ``"50%"``. Writes:

    - ``pred_canonical``: the parsed hypothesis in canonical written form (empty if unparseable),
    - ``numeric_match``: exact value match,
    - ``numeric_match_corrected``: value match after mapping digit-word homophones
      (``"for two oh"`` -> ``"four two zero"``), the headline metric for number-only audio.
    """

    pred_text_key: str = "pred_text"
    number_key: str = "number"
    name: str = "numeric_equality"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.pred_text_key, self.number_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], ["pred_canonical", "numeric_match", "numeric_match_corrected"]

    def process(self, task: AudioTask) -> AudioTask:
        t0 = time.perf_counter()
        pred = task.data.get(self.pred_text_key) or ""
        number = str(task.data[self.number_key])
        parsed = parse_numeric(pred)
        task.data["pred_canonical"] = parsed.canonical() if parsed else ""
        task.data["numeric_match"] = numeric_eq(pred, number)
        task.data["numeric_match_corrected"] = numeric_eq(pred, number, homophones=True)
        self._log_metrics(
            {"process_time": time.perf_counter() - t0, "match_corrected": float(task.data["numeric_match_corrected"])}
        )
        return task
