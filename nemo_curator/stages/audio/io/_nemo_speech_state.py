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
from typing import Any

from nemo_curator.stages.audio.io._nemo_speech_paths import contained_output_path
from nemo_curator.stages.audio.io.shard_key import validate_shard_key
from nemo_curator.utils.atomic_io import write_json_atomically_if_absent

SHARD_REGISTRATION_VERSION = 1


def shard_registration_root(output_dir: Path) -> Path:
    return contained_output_path(output_dir, ".nemo_curator/nemo_speech_shards")


def shard_registration_path(output_dir: Path, shard_key: str) -> Path:
    relative = PurePosixPath(".nemo_curator/nemo_speech_shards") / f"{validate_shard_key(shard_key)}.json"
    return contained_output_path(output_dir, relative)


def register_nemo_speech_shard(output_dir: Path, shard_key: str, expected_inputs: int) -> Path:
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
    path = shard_registration_path(output_dir, normalized_key)
    created = write_json_atomically_if_absent(path, payload, separators=(",", ":"))
    if created:
        return path
    try:
        existing: Any = json.loads(path.read_text(encoding="utf-8"))
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
]
