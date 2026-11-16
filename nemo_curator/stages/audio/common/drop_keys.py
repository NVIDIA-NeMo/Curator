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
from nemo_curator.tasks import AudioTask


@dataclass
class DropKeysStage(ProcessingStage[AudioTask, AudioTask]):
    """Remove the given keys from every task's data; missing keys are ignored.

    Use it to release an in-memory waveform once the last stage that reads it has run, so a JSON
    sink such as ``ManifestWriterStage`` never sees a tensor. It does no audio or file I/O.

    Args:
        keys: Task-data keys to remove.
    """

    keys: list[str] = field(default_factory=list)
    name: str = "DropKeys"

    def process(self, task: AudioTask) -> AudioTask:
        for key in self.keys:
            task.data.pop(key, None)
        return task
