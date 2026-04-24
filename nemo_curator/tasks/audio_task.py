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
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from .tasks import Task

AUDIO_SAMPLE_KEY_FIELD = "sample_key"
CHECKPOINT_SHARD_ID_KEY = "checkpoint_shard_id"
MAX_CHECKPOINT_SHARD_ID_LEN = 80


class _AttrDict(dict):
    """Dict subclass exposing keys as attributes so ``hasattr`` works."""

    def __getattr__(self: "_AttrDict", key: str) -> object:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self: "_AttrDict", key: str, value: object) -> None:
        self[key] = value

    def __delattr__(self: "_AttrDict", key: str) -> None:
        try:
            del self[key]
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


def _strip_known_extensions(path: str) -> str:
    name = os.path.basename(path)
    for suffix in (".jsonl", ".json", ".parquet", ".tar", ".gz", ".bz2", ".xz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _sanitize_checkpoint_shard_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return sanitized or "partition_unknown"


def build_checkpoint_shard_id(
    *,
    source_files: list[str] | None = None,
    explicit_shard_id: str | int | None = None,
    partition_index: int | None = None,
) -> str:
    """Build a stable checkpoint shard identifier."""
    if explicit_shard_id is not None:
        normalized_shard = _normalize_sample_key_value(explicit_shard_id)
        return f"shard_{normalized_shard}"

    normalized_files = [_sanitize_checkpoint_shard_component(_strip_known_extensions(path)) for path in source_files or []]
    if len(normalized_files) == 1:
        return normalized_files[0]
    if normalized_files:
        joined = "__".join(normalized_files)
        if len(joined) <= MAX_CHECKPOINT_SHARD_ID_LEN:
            return joined
        digest = hashlib.sha256(";".join(source_files or []).encode("utf-8")).hexdigest()[:12]
        return f"{normalized_files[0]}__{normalized_files[-1]}__{digest}"
    if partition_index is not None:
        return f"partition_{partition_index}"
    return "partition_unknown"


def build_audio_sample_key(
    data: Mapping[str, Any],
    *,
    dataset_name: str = "",
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
        "source_type": _normalize_sample_key_value(data.get("_audio_source_type")),
        "audio_filepath": _normalize_sample_key_value(data.get("audio_filepath")),
        "tar_path": _normalize_sample_key_value(data.get("_tar_path")),
        "tar_member": _normalize_sample_key_value(data.get("_tar_member")),
        "shard_id": _normalize_sample_key_value(data.get("_shard_id")),
        "offset": _normalize_sample_key_value(data.get("offset")),
        "duration": _normalize_sample_key_value(data.get("duration")),
    }
    identity_json = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(identity_json.encode("utf-8")).hexdigest()


def ensure_sample_key(task: "AudioTask") -> str:
    """Return the task sample key, deriving a root key when missing."""
    if task.sample_key:
        return task.sample_key
    task.sample_key = build_audio_sample_key(task.data, dataset_name=task.dataset_name)
    return task.sample_key


def ensure_checkpoint_shard_id(task: "AudioTask") -> str:
    """Return the checkpoint shard id, deriving it from task metadata when missing."""
    existing = task._metadata.get(CHECKPOINT_SHARD_ID_KEY)
    if isinstance(existing, str) and existing.strip():
        return existing

    shard_id = build_checkpoint_shard_id(
        source_files=task._metadata.get("source_files"),
        explicit_shard_id=task.data.get("_shard_id"),
        partition_index=task._metadata.get("partition_index"),
    )
    task._metadata[CHECKPOINT_SHARD_ID_KEY] = shard_id
    return shard_id


def carry_sample_key(parent_task: "AudioTask", *, data: Mapping[str, Any] | None = None) -> str:
    """Carry forward the same sample identity for a 1:1 transform."""
    if parent_task.sample_key:
        return parent_task.sample_key
    if data is not None:
        return build_audio_sample_key(data, dataset_name=parent_task.dataset_name)
    return ensure_sample_key(parent_task)


def derive_child_sample_key(
    parent_task: "AudioTask",
    *,
    child_kind: str,
    child_identity: Mapping[str, Any],
) -> str:
    """Derive a deterministic child sample key for fan-out outputs."""
    payload = {
        "parent_sample_key": ensure_sample_key(parent_task),
        "child_kind": child_kind,
        "child_identity": {
            key: _normalize_sample_key_value(value)
            for key, value in sorted(child_identity.items())
        },
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


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

    def __post_init__(self) -> None:
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
