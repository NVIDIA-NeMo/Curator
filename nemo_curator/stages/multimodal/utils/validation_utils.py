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

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_curator.tasks import Task


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


def validate_and_project_source_fields(
    sample: dict[str, Any],
    fields: tuple[str, ...] | None,
    excluded_fields: set[str],
) -> dict[str, Any]:
    """Validate requested source `fields` and normalize selected values for tabular output."""
    selected = [key for key in sample if key not in excluded_fields] if fields is None else list(fields)
    if fields is not None:
        reserved = sorted(field for field in selected if field in excluded_fields)
        if reserved:
            msg = f"fields contains reserved keys: {reserved}"
            raise ValueError(msg)
        missing = sorted(field for field in selected if field not in sample)
        if missing:
            msg = f"fields not found in source sample: {missing}"
            raise ValueError(msg)
    return {
        field: (
            json.dumps(sample[field], ensure_ascii=True)
            if isinstance(sample[field], (dict, list))
            else sample[field]
        )
        for field in selected
    }
