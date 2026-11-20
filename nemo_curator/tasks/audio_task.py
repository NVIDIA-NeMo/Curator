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

import os
from dataclasses import dataclass, field

from loguru import logger

from nemo_curator.utils.file_utils import get_fs

from .tasks import Task


class _AttrDict(dict):
    """Dict subclass exposing keys as attributes so ``hasattr`` works."""

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key: str, value: object) -> None:
        self[key] = value

    def __delattr__(self, key: str):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key) from None


@dataclass
class AudioTask(Task[dict]):
    """A single audio manifest entry.

    Represents one line from a JSONL manifest file (e.g. one audio file
    with its metadata).  ``data`` is always a single ``dict``, never a list.

    Matches the ``VideoTask`` naming convention used by the video modality.

    Args:
        data: Manifest entry dict (e.g. ``{"audio_filepath": "...", "text": "..."}``).
        filepath_key: Optional key whose value must be an existing path (local or
            any fsspec URL such as ``s3://``). :meth:`validate` runs on creation and
            logs a warning when the path is missing.
    """

    task_id: str = ""
    dataset_name: str = ""
    data: dict = field(default_factory=_AttrDict)
    filepath_key: str | None = None

    def __post_init__(self):
        if not isinstance(self.data, _AttrDict):
            self.data = _AttrDict(self.data)
        self.validate()

    @property
    def num_items(self) -> int:
        return 1

    def validate(self) -> bool:
        """Validate the task data."""
        if self.filepath_key and self.filepath_key in self.data:
            path = self.data[self.filepath_key]
            # fsspec resolves "" to the working directory, so reject empty values first.
            if (
                not path
                or not isinstance(path, (str, os.PathLike))
                or not get_fs(os.fspath(path)).exists(os.fspath(path))
            ):
                logger.warning(f"File {path} does not exist")
                return False
        return True
