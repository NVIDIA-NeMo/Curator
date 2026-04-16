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

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from .tasks import Task

AUDIO_SAMPLE_KEY_FIELD = "sample_key"


class _AttrDict(dict):
    """Dict subclass exposing keys as attributes so ``hasattr`` works."""

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None


def _normalize_sample_key_value(value: Any) -> Any:  # noqa: ANN401
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip()
        return normalized if normalized else None
    return str(value)


def build_audio_sample_key(
    data: Mapping[str, Any],
    *,
    dataset_name: str = "",
    audio_filepath_key: str = "audio_filepath",
    tar_path_key: str = "_tar_path",
    tar_member_key: str = "_tar_member",
    shard_id_key: str = "_shard_id",
    source_type_key: str = "_audio_source_type",
    offset_key: str = "offset",
    duration_key: str = "duration",
    sample_key_field: str = AUDIO_SAMPLE_KEY_FIELD,
) -> str:
    """Build a stable sample key for an audio entry.

    If the input already contains an explicit ``sample_key`` value, preserve it.
    Otherwise derive a deterministic hash from the sample identity fields.
    """

    existing = _normalize_sample_key_value(data.get(sample_key_field))
    if existing is not None:
        return str(existing)

    identity = {
        "dataset_name": _normalize_sample_key_value(dataset_name),
        "source_type": _normalize_sample_key_value(data.get(source_type_key)),
        "audio_filepath": _normalize_sample_key_value(data.get(audio_filepath_key)),
        "tar_path": _normalize_sample_key_value(data.get(tar_path_key)),
        "tar_member": _normalize_sample_key_value(data.get(tar_member_key)),
        "shard_id": _normalize_sample_key_value(data.get(shard_id_key)),
        "offset": _normalize_sample_key_value(data.get(offset_key)),
        "duration": _normalize_sample_key_value(data.get(duration_key)),
    }
    identity_json = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(identity_json.encode("utf-8")).hexdigest()

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
    sample_key: str = ""
    filepath_key: str | None = None

    def __post_init__(self):
        if not isinstance(self.data, _AttrDict):
            self.data = _AttrDict(self.data)
        if not self.sample_key:
            existing = self.data.get(AUDIO_SAMPLE_KEY_FIELD)
            if isinstance(existing, str) and existing.strip():
                self.sample_key = existing.strip()

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
