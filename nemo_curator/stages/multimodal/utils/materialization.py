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

from __future__ import annotations

import tarfile
from typing import Any

import fsspec

from nemo_curator.tasks import MultiBatchTask, Task


def require_source_id_field(source_id_field: str) -> str:
    if source_id_field:
        return source_id_field
    msg = "source_id_field must be provided explicitly (e.g., 'pdf_name')"
    raise ValueError(msg)


def resolve_storage_options(
    task: Task[Any] | None = None,
    io_kwargs: dict[str, object] | None = None,
) -> dict[str, object]:
    source_storage_options = task._metadata.get("source_storage_options") if task is not None else None
    if isinstance(source_storage_options, dict) and source_storage_options:
        return source_storage_options
    storage_options = (io_kwargs or {}).get("storage_options")
    return storage_options if isinstance(storage_options, dict) else {}


def load_bytes_from_content_reference(
    content_path: str | None,
    content_key: str | None,
    storage_options: dict[str, object],
    byte_cache: dict[tuple[str, str], bytes | None],
) -> bytes | None:
    if not content_path:
        return None

    cache_key = (str(content_path), str(content_key or ""))
    if cache_key in byte_cache:
        return byte_cache[cache_key]

    try:
        with fsspec.open(str(content_path), mode="rb", **storage_options) as fobj:
            if content_key:
                with tarfile.open(fileobj=fobj, mode="r:*") as tf:
                    try:
                        extracted = tf.extractfile(content_key)
                    except KeyError:
                        extracted = None
                    payload = extracted.read() if extracted is not None else None
                    byte_cache[cache_key] = payload
                    return payload
            payload = fobj.read()
            byte_cache[cache_key] = payload
            return payload
    except Exception:  # noqa: BLE001
        byte_cache[cache_key] = None
        return None


def load_bytes_from_metadata_source(
    source_value: str | None,
    storage_options: dict[str, object],
    byte_cache: dict[tuple[str, str], bytes | None],
) -> bytes | None:
    source = MultiBatchTask.parse_metadata_source(source_value)
    return load_bytes_from_content_reference(
        content_path=source.get("content_path"),
        content_key=source.get("content_key"),
        storage_options=storage_options,
        byte_cache=byte_cache,
    )
