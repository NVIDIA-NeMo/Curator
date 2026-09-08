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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from nemo_curator.utils.atomic_io import write_json_atomically

METADATA_DIRNAME = ".nemo_curator_metadata"


@dataclass(frozen=True)
class CompletionManifestRecord:
    """One durable completion manifest read from disk."""

    path: Path
    payload: dict[str, object]


def _safe_token(value: object) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value))


def _mapping_digest(mapping: Mapping[str, object]) -> str:
    encoded = json.dumps(mapping, default=str, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def read_completion_manifests(
    checkpoint_path: str | Path,
    *,
    namespace: str,
    completion_dirname: str | None = None,
) -> list[CompletionManifestRecord]:
    """Read completed identities for one manifest namespace."""
    resolved_dirname = completion_dirname or f".{_safe_token(namespace)}_completion"
    manifest_dir = Path(checkpoint_path, METADATA_DIRNAME, resolved_dirname).absolute()
    if not manifest_dir.exists():
        return []

    records = []
    pattern = f"completed_{_safe_token(namespace)}_*.json"
    for manifest_file in sorted(manifest_dir.glob(pattern)):
        if not manifest_file.is_file():
            continue

        try:
            payload = json.loads(manifest_file.read_text())
        except (OSError, json.JSONDecodeError) as e:
            msg = f"Failed to read completion manifest {manifest_file}: {e}"
            raise ValueError(msg) from e

        if not isinstance(payload, dict):
            msg = f"Completion manifest must contain a JSON object: {manifest_file}"
            raise TypeError(msg)
        status = payload.get("status")
        if not isinstance(status, str):
            msg = f"Completion manifest must contain a string status: {manifest_file}"
            raise TypeError(msg)
        if status != "completed":
            msg = f"Completion manifest must have status 'completed': {manifest_file}"
            raise ValueError(msg)

        records.append(CompletionManifestRecord(path=manifest_file, payload=payload))

    return records


class CompletionManifest:
    """Durable proof that work identified by stable fields completed successfully."""

    def __init__(  # noqa: PLR0913
        self,
        checkpoint_path: str | Path,
        namespace: str,
        identity: Mapping[str, object],
        *,
        metadata: Mapping[str, object] | None = None,
        completion_dirname: str | None = None,
        enabled: bool = True,
        flatten_identity: bool = True,
        flatten_metadata: bool = False,
    ) -> None:
        """Create a completion manifest manager under ``checkpoint_path``."""
        self.checkpoint_path = Path(checkpoint_path)
        self.namespace = namespace
        self.identity = dict(identity)
        self.metadata = dict(metadata or {})
        self.completion_dirname = completion_dirname or f".{_safe_token(namespace)}_completion"
        self.enabled = enabled
        self.flatten_identity = flatten_identity
        self.flatten_metadata = flatten_metadata
        self.manifest_file: Path | None = None

    @property
    def manifest_dir(self) -> Path:
        return Path(self.checkpoint_path, METADATA_DIRNAME, self.completion_dirname).absolute()

    @property
    def filename_prefix(self) -> str:
        return f"completed_{_safe_token(self.namespace)}_{_mapping_digest(self.identity)}"

    def _payload(self, extra: Mapping[str, object] | None = None) -> dict[str, object]:
        payload: dict[str, object] = {}
        if self.flatten_identity:
            payload.update(self.identity)
        else:
            payload["identity"] = self.identity
        if self.metadata:
            if self.flatten_metadata:
                payload.update(self.metadata)
            else:
                payload["metadata"] = self.metadata
        if extra is not None:
            payload.update(extra)
        payload["status"] = "completed"
        return payload

    def mark_completed(self, extra: Mapping[str, object] | None = None) -> Path | None:
        """Atomically record successful completion."""
        if not self.enabled:
            return None
        if self.manifest_file is None:
            self.manifest_file = self.manifest_dir / f"{self.filename_prefix}.json"

        write_json_atomically(
            self.manifest_file,
            self._payload(extra),
            separators=(",", ":"),
            sort_keys=True,
        )
        return self.manifest_file

    def __enter__(self) -> "CompletionManifest":
        return self

    def __exit__(self, _exc_type: type[BaseException] | None, exc: BaseException | None, _: object) -> bool:
        if exc is None:
            self.mark_completed()
        return False
