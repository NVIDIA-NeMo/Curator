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
    "selected_transcript",
    "taskname",
    "orig_text",
    "orig_answer",
]


@dataclass
class InitializeFieldsStage(ProcessingStage[AudioTask, AudioTask]):
    """Prepare fields for the text-filtering pipeline.

    Unconditionally:

    - Preserves any existing ``_skipme`` value from Granary-v1 into
      ``additional_notes["v1_skipme"]`` before resetting.
    - Sets ``_skipme`` to ``""`` (empty string = not skipped).
    - Renames ``original_text_key`` → ``granary_v1_key`` (if the key
      exists in the task data).
    - Drops all keys listed in ``drop_keys``.

    Downstream stages store a human-readable reason string in
    ``_skipme`` when they flag an entry (e.g. ``"Hallucination"``).
    """

    skip_me_key: str = "_skipme"
    notes_key: str = "additional_notes"
    original_text_key: str = "text"
    granary_v1_key: str = "granary_v1_prediction"
    source_lang_key: str = "source_lang"
    default_source_lang: str = "en"
    drop_keys: list[str] = field(default_factory=lambda: list(_DEFAULT_DROP_KEYS))
    name: str = "InitializeFields"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        out = [self.skip_me_key]
        if self.granary_v1_key and self.original_text_key:
            out.append(self.granary_v1_key)
        return [], out

    def _init_task(self, task: AudioTask) -> None:
        v1_skipme = task.data.get(self.skip_me_key, "")
        notes = task.data.get(self.notes_key, {})
        if not isinstance(notes, dict):
            notes = {}
        if v1_skipme:
            notes["v1_skipme"] = v1_skipme
        task.data[self.notes_key] = notes
        task.data[self.skip_me_key] = ""
        if self.source_lang_key and self.source_lang_key not in task.data:
            task.data[self.source_lang_key] = self.default_source_lang
        if self.original_text_key and self.original_text_key in task.data:
            task.data[self.granary_v1_key] = task.data.pop(self.original_text_key)
        for key in self.drop_keys:
            task.data.pop(key, None)

    def process(self, task: AudioTask) -> AudioTask:
        self._init_task(task)
        return task

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        for task in tasks:
            self._init_task(task)
        return tasks
