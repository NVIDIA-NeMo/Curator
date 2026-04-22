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

import string
from dataclasses import dataclass, field

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _words_match(a: str, b: str) -> bool:
    """True when *a* and *b* differ only in punctuation and capitalisation."""
    normalise = lambda t: t.strip().lower().translate(_PUNCT_TABLE).split()
    return normalise(a) == normalise(b)


@dataclass
class PnCContentGuardStage(ProcessingStage[AudioTask, AudioTask]):
    """Revert PnC restoration when the model changed actual word content.

    Compares ``text_key`` (pre-PnC) with ``pnc_text_key`` (post-PnC) after
    stripping punctuation and lower-casing.  If the normalised word sequences
    differ — meaning the LLM added, removed, or substituted words — the
    entry is considered corrupted by the model:

    * ``pnc_text_key`` is reverted to the original ``text_key`` value.
    * The bad PnC output is saved to ``rejected_text_key`` for debugging.
    """

    text_key: str = "text"
    pnc_text_key: str = "text"
    rejected_text_key: str = "rejected_pnc_text"
    name: str = "PnCContentGuard"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key, self.pnc_text_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.pnc_text_key, self.rejected_text_key]

    def process(self, task: AudioTask) -> AudioTask:
        original = task.data.get(self.text_key, "")
        pnc = task.data.get(self.pnc_text_key, "")

        if original and pnc and not _words_match(original, pnc):
            task.data[self.rejected_text_key] = pnc
            task.data[self.pnc_text_key] = original
        else:
            task.data[self.rejected_text_key] = ""

        return task
