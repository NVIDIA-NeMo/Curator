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

from nemo_curator.utils.atomic_io import fsync_directory, write_json_atomically

METADATA_DIRNAME = ".nemo_curator_metadata"


@dataclass(frozen=True)
class RetryManifestRecord:
    """One outstanding retry manifest read from disk."""

    path: Path
    status: str
    payload: dict[str, object]


def _safe_token(value: object) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value))


def _mapping_digest(mapping: Mapping[str, object]) -> str:
    encoded = json.dumps(mapping, default=str, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def read_retry_manifests(
    checkpoint_path: str | Path,
    *,
    namespace: str,
    retry_dirname: str | None = None,
) -> list[RetryManifestRecord]:
    """Read outstanding manifests for one retry namespace."""
    resolved_retry_dirname = retry_dirname or f".{_safe_token(namespace)}_retry"
    manifest_dir = Path(checkpoint_path, METADATA_DIRNAME, resolved_retry_dirname).absolute()
    if not manifest_dir.exists():
        return []

    records = []
    pattern = f"manifest_{_safe_token(namespace)}_*.json"
    for manifest_file in sorted(manifest_dir.glob(pattern)):
        if not manifest_file.is_file():
            continue

        try:
            payload = json.loads(manifest_file.read_text())
        except (OSError, json.JSONDecodeError) as e:
            msg = f"Failed to read retry manifest {manifest_file}: {e}"
            raise ValueError(msg) from e

        if not isinstance(payload, dict):
            msg = f"Retry manifest must contain a JSON object: {manifest_file}"
            raise ValueError(msg)

        status = payload.get("status")
        if not isinstance(status, str):
            msg = f"Retry manifest must contain a string status: {manifest_file}"
            raise ValueError(msg)

        records.append(RetryManifestRecord(path=manifest_file, status=status, payload=payload))

    return records


# TODO: Reverse
class RetryManifest:
    """Compact marker for retryable work, keyed by stable identity fields."""

    def __init__(  # noqa: PLR0913
        self,
        checkpoint_path: str | Path,
        namespace: str,
        identity: Mapping[str, object],
        *,
        metadata: Mapping[str, object] | None = None,
        retry_dirname: str | None = None,
        enabled: bool = True,
        flatten_identity: bool = True,
        flatten_metadata: bool = False,
    ) -> None:
        """Create a manifest manager under ``checkpoint_path``."""
        self.checkpoint_path = Path(checkpoint_path)
        self.namespace = namespace
        self.identity = dict(identity)
        self.metadata = dict(metadata or {})
        self.retry_dirname = retry_dirname or f".{_safe_token(namespace)}_retry"
        self.enabled = enabled
        self.flatten_identity = flatten_identity
        self.flatten_metadata = flatten_metadata
        self.manifest_file: Path | None = None

    @property
    def manifest_dir(self) -> Path:
        return Path(self.checkpoint_path, METADATA_DIRNAME, self.retry_dirname).absolute()

    @property
    def filename_prefix(self) -> str:
        return f"manifest_{_safe_token(self.namespace)}_{_mapping_digest(self.identity)}"

    def _payload(
        self,
        status: str,
        *,
        error: BaseException | None = None,
        extra: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": status,
        }

        if self.flatten_identity:
            payload.update(self.identity)
        else:
            payload["identity"] = self.identity
        if self.flatten_metadata:
            payload.update(self.metadata)
        if error is not None:
            payload["error_type"] = type(error).__name__
        if extra is not None:
            payload.update(extra)

        return payload

    def write(
        self,
        status: str,
        *,
        error: BaseException | None = None,
        extra: Mapping[str, object] | None = None,
    ) -> Path | None:
        """Write or update this manifest."""
        if not self.enabled:
            return None

        if self.manifest_file is None:
            self.manifest_file = self.manifest_dir / f"{self.filename_prefix}.json"

        write_json_atomically(
            self.manifest_file,
            self._payload(status, error=error, extra=extra),
            separators=(",", ":"),
            sort_keys=True,
        )
        return self.manifest_file

    def mark_pending(self) -> Path | None:
        return self.write("pending")

    def mark_failed(self, error: BaseException) -> Path | None:
        return self.write("failed", error=error)

    def mark_retryable(self, status: str, extra: Mapping[str, object] | None = None) -> Path | None:
        return self.write(status, extra=extra)

    def mark_success(self) -> int:
        """Remove manifests for this identity and return the count."""
        if not self.enabled:
            return 0

        removed = 0
        if not self.manifest_dir.exists():
            return removed

        for manifest_file in self.manifest_dir.glob(f"{self.filename_prefix}*.json"):
            try:
                manifest_file.unlink()
                removed += 1
            except FileNotFoundError:
                continue
        if removed:
            fsync_directory(self.manifest_dir)
        return removed

    def __enter__(self) -> "RetryManifest":
        self.mark_pending()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, _: object) -> bool:
        if exc is not None:
            self.mark_failed(exc)
        else:
            self.mark_success()
        return False
