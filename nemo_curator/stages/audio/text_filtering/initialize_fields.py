# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

_DEFAULT_DROP_KEYS: list[str] = [
    "answer",
    "source_lang",
    "target_lang",
    "decodercontext",
    "emotion",
    "diarize",
    "pnc",
    "itn",
    "timestamp",
]


@dataclass
class InitializeFieldsStage(ProcessingStage[AudioTask, AudioTask]):
    """Prepare fields for the text-filtering pipeline.

    Unconditionally:

    - Sets ``skip_me`` to ``""`` (empty string = not skipped).
    - Renames ``original_text_key`` → ``granary_v1_key`` (if the key
      exists in the task data).
    - Drops all keys listed in ``drop_keys``.

    Downstream stages store a human-readable reason string in
    ``skip_me`` when they flag an entry (e.g. ``"Hallucination"``).
    """

    name: str = "InitializeFields"
    skip_me_key: str = "skip_me"
    original_text_key: str = "text"
    granary_v1_key: str = "granary_v1_prediction"
    drop_keys: list[str] = field(default_factory=lambda: list(_DEFAULT_DROP_KEYS))
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        out = [self.skip_me_key]
        if self.granary_v1_key and self.original_text_key:
            out.append(self.granary_v1_key)
        return [], out

    def process(self, task: AudioTask) -> AudioTask:
        task.data[self.skip_me_key] = ""
        if self.original_text_key and self.original_text_key in task.data:
            task.data[self.granary_v1_key] = task.data.pop(self.original_text_key)
        for key in self.drop_keys:
            task.data.pop(key, None)
        return task
