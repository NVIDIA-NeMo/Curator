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

import os
from dataclasses import dataclass, field

from loguru import logger

from .tasks import Task


@dataclass
class AudioEntry(Task[dict]):
    """A single audio manifest entry.

    Represents one line from a JSONL manifest file (e.g. one audio file
    with its metadata).  ``data`` is always a single ``dict``, never a list.

    Args:
        data: Manifest entry dict (e.g. ``{"audio_filepath": "...", "text": "..."}``).
        filepath_key: Optional key whose value is validated as an existing path.
    """

    task_id: str = ""
    dataset_name: str = ""
    data: dict = field(default_factory=dict)
    filepath_key: str | None = None

    @property
    def num_items(self) -> int:
        return 1

    def validate(self) -> bool:
        """Validate the task data."""
        if self.filepath_key and self.filepath_key in self.data:
            path = self.data[self.filepath_key]
            if not os.path.exists(path):
                logger.warning(f"File {path} does not exist")
                return False
        return True


# Backward-compatible alias — will be removed in a future release.
AudioBatch = AudioEntry
