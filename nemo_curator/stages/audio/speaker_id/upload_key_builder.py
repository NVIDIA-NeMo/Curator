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

import posixpath
from dataclasses import dataclass

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


@dataclass
class BuildUploadKeyStage(ProcessingStage[AudioTask, AudioTask]):
    """Build an upload object key from constants and task fields."""

    source_field: str = "output_key"
    output_field: str = "speaker_embedding_upload_key"
    key_prefix: str = ""
    key_suffix: str = ".npz"
    name: str = "build_upload_key"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.source_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_field]

    def process(self, task: AudioTask) -> AudioTask:
        source_value = str(task.data[self.source_field]).strip("/")
        object_key = source_value
        if self.key_prefix:
            object_key = posixpath.join(self.key_prefix.strip("/"), object_key)
        if self.key_suffix and not object_key.endswith(self.key_suffix):
            object_key = f"{object_key}{self.key_suffix}"
        task.data[self.output_field] = object_key
        return task
