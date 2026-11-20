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

"""Durable shard registration shared by the NeMo speech reader and writer."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from fsspec.implementations.local import LocalFileSystem

from nemo_curator.stages.audio.io._nemo_speech_paths import contained_output_path
from nemo_curator.stages.audio.io.shard_key import validate_shard_key
from nemo_curator.utils.atomic_io import write_json_atomically_if_absent

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem

SHARD_REGISTRATION_VERSION = 1


def write_json_if_absent(fs: AbstractFileSystem, path: str, payload: Any) -> bool:  # noqa: ANN401
    """Create ``path`` holding ``payload`` unless it exists; return whether it was created.

    Locally this keeps the atomic link-based create. Object stores publish an
    object only when its upload completes, and ``xb`` makes the create exclusive
    where the backend supports it.
    """

    if isinstance(fs, LocalFileSystem):
        return write_json_atomically_if_absent(Path(path), payload, separators=(",", ":"))
    # Not every backend makes a small ``xb`` upload conditional (s3fs 2024.12 does a plain PUT).
    if fs.exists(path):
        return False
    try:
        with fs.open(path, "xb") as stream:
            stream.write((json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8"))
    except FileExistsError:
        return False
    return True


def shard_registration_root(fs: AbstractFileSystem, output_dir: str) -> str:
    return contained_output_path(fs, output_dir, ".nemo_curator/nemo_speech_shards")


def shard_registration_path(fs: AbstractFileSystem, output_dir: str, shard_key: str) -> str:
    relative = PurePosixPath(".nemo_curator/nemo_speech_shards") / f"{validate_shard_key(shard_key)}.json"
    return contained_output_path(fs, output_dir, relative)


def register_nemo_speech_shard(fs: AbstractFileSystem, output_dir: str, shard_key: str, expected_inputs: int) -> str:
    """Atomically register a source shard before any downstream row can vanish."""

    normalized_key = validate_shard_key(shard_key)
    if expected_inputs <= 0:
        msg = f"NeMo speech shard {normalized_key!r} must contain at least one input"
        raise ValueError(msg)
    payload = {
        "version": SHARD_REGISTRATION_VERSION,
        "shard_key": normalized_key,
        "expected_inputs": expected_inputs,
    }
    path = shard_registration_path(fs, output_dir, normalized_key)
    if write_json_if_absent(fs, path, payload):
        return path
    try:
        existing: Any = json.loads(fs.read_text(path, encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"Invalid NeMo speech shard registration {path}: {error}"
        raise ValueError(msg) from error
    if existing != payload:
        msg = f"Conflicting NeMo speech shard registration for {normalized_key!r}"
        raise ValueError(msg)
    return path


__all__ = [
    "SHARD_REGISTRATION_VERSION",
    "register_nemo_speech_shard",
    "shard_registration_path",
    "shard_registration_root",
    "write_json_if_absent",
]
