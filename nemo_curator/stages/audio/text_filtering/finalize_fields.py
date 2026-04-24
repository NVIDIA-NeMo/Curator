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


@dataclass
class FinalizeFieldsStage(ProcessingStage[AudioTask, AudioTask]):
    """Drop unwanted fields and validate the final manifest schema.

    Drops all keys listed in ``drop_keys`` (silently ignores missing keys).
    The ``cleaned_text`` field produced by earlier stages is kept as-is.
    """

    name: str = "FinalizeFields"
    cleaned_text_key: str = "cleaned_text"
    drop_keys: list[str] = field(default_factory=lambda: ["pnc", "itn", "timestamp"])
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.cleaned_text_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.cleaned_text_key]

    def process(self, task: AudioTask) -> AudioTask:
        for key in self.drop_keys:
            task.data.pop(key, None)
        return task
