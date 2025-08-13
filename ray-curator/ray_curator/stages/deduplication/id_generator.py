# TODO: Add headers incl reference to source

import uuid

import ray

CURATOR_DEDUP_ID_STR = "_curator_dedup_id"
CURATOR_ID_GENERATOR_ACTOR_NAME = "curator_deduplication_id_generator"


@ray.remote
class IdGenerator:
    def __init__(self, start_id: int = 0):
        self.next_id = start_id
        self.batch_registry = {}  # {batch_hash: (min_id, max_id)}

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
