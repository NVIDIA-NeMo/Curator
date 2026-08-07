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

import json
import os
from dataclasses import dataclass, field
from numbers import Integral

from loguru import logger

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
        filepath_key: Optional key whose value is validated as an existing path.
    """

    task_id: str = ""
    dataset_name: str = ""
    data: dict = field(default_factory=_AttrDict)
    filepath_key: str | None = None

    def __post_init__(self):
        if not isinstance(self.data, _AttrDict):
            self.data = _AttrDict(self.data)

    @property
    def num_items(self) -> int:
        return 1

    def input_data_size_bytes(self) -> int:
        """Return compact JSON-envelope bytes plus in-memory array storage.

        Audio stages may carry waveform tensors or NumPy arrays between stages.
        Representing those transient values as JSON ``null`` keeps telemetry
        non-blocking and avoids materializing huge, misleading array strings;
        their actual element-storage bytes are counted separately.
        """
        array_bytes = 0

        def _replace_non_json(value: object) -> None:
            nonlocal array_bytes
            nbytes = getattr(value, "nbytes", None)
            if isinstance(nbytes, Integral):
                array_bytes += max(int(nbytes), 0)
            else:
                numel = getattr(value, "numel", None)
                element_size = getattr(value, "element_size", None)
                if callable(numel) and callable(element_size):
                    array_bytes += max(int(numel()) * int(element_size()), 0)

        json_bytes = len(
            json.dumps(
                self.data,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                default=_replace_non_json,
            ).encode("utf-8")
        )
        return json_bytes + array_bytes

    def validate(self) -> bool:
        """Validate the task data."""
        if self.filepath_key and self.filepath_key in self.data:
            path = self.data[self.filepath_key]
            if not os.path.exists(path):
                logger.warning(f"File {path} does not exist")
                return False
        return True
