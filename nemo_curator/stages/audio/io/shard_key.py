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

"""Stable, path-safe shard keys for NeMo speech manifests.

Shard keys are shared by discovery, output manifests, Opus paths, progress
records, and ``.done`` markers.  They therefore need to be both stable across
restarts and safe to join below a caller-provided output directory.
"""

from __future__ import annotations

from pathlib import PurePosixPath


def _strip_manifest_suffix(path: str) -> str:
    for suffix in (".jsonl.gz", ".jsonl", ".json"):
        if path.endswith(suffix):
            return path[: -len(suffix)]
    return path


def validate_shard_key(shard_key: str) -> str:
    """Return a normalized relative shard key or raise ``ValueError``.

    ``shard_key`` becomes part of a local output path, so absolute paths,
    empty keys, and traversal components are rejected.
    """

    raw = shard_key.replace("\\", "/")
    normalized = raw.strip("/")
    path = PurePosixPath(normalized)
    traversal = any(part in {".", ".."} for part in raw.split("/"))
    reserved = bool(path.parts and path.parts[0] == ".nemo_curator")
    if not normalized or PurePosixPath(raw).is_absolute() or traversal or reserved:
        msg = f"Unsafe NeMo shard key: {shard_key!r}"
        raise ValueError(msg)
    return path.as_posix()


def derive_manifest_shard_key(
    manifest_path: str,
    corpus: str,
    *,
    shard_key_prefix: str | None = None,
) -> str:
    """Derive the stable output key for a physical NeMo manifest shard.

    With ``shard_key_prefix``, the last prefix component is used as an
    anchor and the distinguishing path below that anchor is retained.  If
    the anchor is absent, the manifest basename is appended to the prefix.
    Without a prefix, ``corpus`` must occur exactly once as a path component.
    Manifest suffixes are removed in both cases.
    """

    parts = [part for part in manifest_path.replace("\\", "/").split("/") if part]
    if not parts:
        msg = "manifest_path must contain a filename"
        raise ValueError(msg)
    parts[-1] = _strip_manifest_suffix(parts[-1])
    lowered = [part.lower() for part in parts]

    if shard_key_prefix is not None:
        prefix = validate_shard_key(shard_key_prefix)
        anchor = prefix.rsplit("/", 1)[-1].lower()
        anchor_index = next((index for index in range(len(parts) - 1, -1, -1) if lowered[index] == anchor), None)
        tail = parts[anchor_index + 1 :] if anchor_index is not None else [parts[-1]]
        return validate_shard_key("/".join([prefix, *tail]) if tail else prefix)

    corpus_component = corpus.strip().lower()
    if not corpus_component:
        msg = "corpus must be non-empty when shard_key_prefix is not set"
        raise ValueError(msg)
    matches = [index for index, part in enumerate(lowered) if part == corpus_component]
    if len(matches) != 1:
        detail = "not found" if not matches else f"found {len(matches)} times"
        msg = (
            f"Corpus {corpus!r} was {detail} in manifest path {manifest_path!r}; "
            "it must occur exactly once, or the input_cfg entry must set shard_key_prefix."
        )
        raise ValueError(msg)
    return validate_shard_key("/".join(parts[matches[0] :]))


__all__ = ["derive_manifest_shard_key", "validate_shard_key"]
