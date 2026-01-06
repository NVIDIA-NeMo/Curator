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

import json
import uuid
from typing import Any

import fsspec
import ray
from loguru import logger
from ray.actor import ActorHandle

from nemo_curator.backends.utils import register_loguru_serializer

CURATOR_DEDUP_ID_STR = "_curator_dedup_id"
CURATOR_ID_GENERATOR_ACTOR_NAME = "curator_deduplication_id_generator"


class IdGeneratorBase:
    """Base IdGenerator class without Ray decorator for testing and direct use."""

    def __init__(self, start_id: int = 0, batch_registry: dict[str, tuple[int, int]] | None = None):
        self.next_id = start_id
        self.batch_registry = batch_registry or {}  # {batch_hash: (min_id, max_id)}

    def register_batch(self, files: str | list[str], count: int) -> int:
        batch_hash = self.hash_files(files)
        if _ids := self.batch_registry.get(batch_hash):
            return _ids[0]

        current_id = self.next_id
        self.next_id += count
        self.batch_registry[batch_hash] = (current_id, self.next_id - 1)
        return current_id

    def hash_files(self, filepath: str | list[str]) -> str:
        filepath = filepath if isinstance(filepath, list) else [filepath]
        return str(uuid.uuid5(uuid.NAMESPACE_URL, ";".join(filepath)))

    def get_batch_range(self, files: str | list[str] | None, key: str | None) -> tuple[int, int]:
        if (files is None and key is None) or (files is not None and key is not None):
            msg = "Either files or key must be provided"
            raise ValueError(msg)

        if files is not None:
            key = self.hash_files(files)

        return self.batch_registry[key]

    def to_disk(self, filepath: str, storage_options: dict[str, Any] | None = None) -> None:
        storage_options = storage_options or {}
        with fsspec.open(filepath, mode="w", **storage_options) as f:
            json.dump(
                {
                    "next_id": self.next_id,
                    "batch_registry": self.batch_registry,
                },
                f,
            )

    @classmethod
    def from_disk(cls, filepath: str, storage_options: dict[str, Any] | None = None) -> "IdGeneratorBase":
        storage_options = storage_options or {}
        with fsspec.open(filepath, mode="r", **storage_options) as f:
            data = json.load(f)
        return cls(start_id=data["next_id"], batch_registry=data["batch_registry"])


@ray.remote
class IdGenerator(IdGeneratorBase):
    """Ray actor version of IdGenerator."""


def get_id_generator_actor() -> ActorHandle[IdGenerator]:
    return ray.get_actor(name=CURATOR_ID_GENERATOR_ACTOR_NAME, namespace=CURATOR_ID_GENERATOR_ACTOR_NAME)


def kill_id_generator_actor() -> None:
    ray.kill(get_id_generator_actor())


def create_id_generator_actor(filepath: str | None = None, storage_options: dict[str, Any] | None = None) -> None:
    """Create an id generator actor.

    Args:
        filepath (str): Path from where we want to load the id generator state json file.
            If None, a new actor is created.
        storage_options (dict[str, Any] | None): Storage options to pass to fsspec.open.
    """
    register_loguru_serializer()  # TODO: instead of calling before each ray.init we can call it a packages __init__
    ray.init(ignore_reinit_error=True)

    try:
        if filepath is None:
            _ = IdGenerator.options(
                name=CURATOR_ID_GENERATOR_ACTOR_NAME, namespace=CURATOR_ID_GENERATOR_ACTOR_NAME, lifetime="detached"
            ).remote()
        else:
            # Create actor from saved state on disk
            # First load the data from disk
            storage_options = storage_options or {}
            with fsspec.open(filepath, mode="r", **storage_options) as f:
                data = json.load(f)
            # Create actor with loaded data
            _ = IdGenerator.options(
                name=CURATOR_ID_GENERATOR_ACTOR_NAME, namespace=CURATOR_ID_GENERATOR_ACTOR_NAME, lifetime="detached"
            ).remote(start_id=data["next_id"], batch_registry=data["batch_registry"])
    except Exception as e:
        logger.error(f"Error creating id generator actor: {e}")
        raise

    finally:
        # Shutdown Ray to allow future pipelines to call ray.init with their own configuration
        ray.shutdown()


def write_id_generator_to_disk(filepath: str, storage_options: dict[str, Any] | None = None) -> None:
    storage_options = storage_options or {}
    ray.get(get_id_generator_actor().to_disk.remote(filepath, storage_options))
