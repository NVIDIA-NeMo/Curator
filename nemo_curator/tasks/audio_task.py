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


def _array_storage_size_bytes(value: object) -> int | None:
    """Return array storage bytes without copying an in-memory payload."""
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, Integral):
        return max(int(nbytes), 0)

    numel = getattr(value, "numel", None)
    element_size = getattr(value, "element_size", None)
    if callable(numel) and callable(element_size):
        return max(int(numel()) * int(element_size()), 0)
    return None


def _json_envelope_and_array_bytes(value: object) -> tuple[object, int]:
    """Replace arrays with JSON nulls and total their storage bytes."""
    array_bytes = _array_storage_size_bytes(value)
    if array_bytes is not None:
        return None, array_bytes
    if isinstance(value, dict):
        envelope = {}
        total = 0
        for key, item in value.items():
            envelope[key], item_bytes = _json_envelope_and_array_bytes(item)
            total += item_bytes
        return envelope, total
    if isinstance(value, (list, tuple)):
        envelope = []
        total = 0
        for item in value:
            safe_item, item_bytes = _json_envelope_and_array_bytes(item)
            envelope.append(safe_item)
            total += item_bytes
        return envelope, total
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return None, 0
    return value, 0


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
        envelope, array_bytes = _json_envelope_and_array_bytes(self.data)
        json_bytes = len(
            json.dumps(envelope, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
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
