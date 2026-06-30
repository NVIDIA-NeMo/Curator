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
# ruff: noqa: BLE001, C901, PLR0912

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_curator.stages.base import ProcessingStage
    from nemo_curator.tasks import Task


TERMINAL_GROUP_ID_KEY = "_curator_terminal_group_id"
TERMINAL_INDEX_KEY = "_curator_terminal_idx"
TERMINAL_COUNT_KEY = "_curator_terminal_count"
TERMINAL_SOURCE_INDEX_KEY = "_curator_terminal_source_index"
TERMINAL_DROPPED_KEY = "_curator_terminal_dropped"
TERMINAL_DROPPED_BY_STAGE_KEY = "_curator_terminal_dropped_by_stage"
TERMINAL_DROP_REASON_KEY = "_curator_terminal_drop_reason"


def preserve_dropped_terminal_tasks(
    stage: ProcessingStage,
    input_tasks: list[Task],
    output_tasks: list[Task | None],
) -> list[Task | None]:
    """Preserve terminal records required by downstream aggregators.

    Normal filters should still drop rows. A planner may, however, split a
    logical row into ordered terminal records that a later assembler must see
    exactly once. If an intermediate stage filters such a terminal record, turn
    it into a lightweight tombstone so the assembler can finish the parent
    instead of buffering forever.

    Terminal records are identified by the generic ``_curator_terminal_*``
    fields. Audio global planning also carries ``_curator_segment_*`` debug
    aliases, but the terminal fields are the backend-preserved contract.
    """
    if getattr(stage, "_curator_consumes_segment_rows", False):
        return output_tasks
    if not any(_terminal_row_key(task) is not None for task in input_tasks):
        return output_tasks

    if len(output_tasks) == len(input_tasks):
        changed = False
        preserved: list[Task | None] = []
        for input_task, output_task in zip(input_tasks, output_tasks, strict=True):
            if output_task is None and _terminal_row_key(input_task) is not None:
                preserved.append(_terminal_row_tombstone(stage, input_task))
                changed = True
            else:
                preserved.append(output_task)
        if changed:
            return preserved

    output_by_key: dict[tuple[str, int, int], Task] = {}
    duplicate_output_key = False
    non_segment_outputs: list[Task] = []
    for output_task in output_tasks:
        if output_task is None:
            continue
        key = _terminal_row_key(output_task)
        if key is None:
            non_segment_outputs.append(output_task)
            continue
        if key in output_by_key:
            duplicate_output_key = True
        output_by_key[key] = output_task

    input_keys = [_terminal_row_key(task) for task in input_tasks]
    if not any(key is not None and key not in output_by_key for key in input_keys):
        return output_tasks

    if not duplicate_output_key and not non_segment_outputs and all(key is not None for key in input_keys):
        return [
            output_by_key.get(key) or _terminal_row_tombstone(stage, input_task)
            for input_task, key in zip(input_tasks, input_keys, strict=True)
        ]

    preserved = [task for task in output_tasks if task is not None]
    for input_task, key in zip(input_tasks, input_keys, strict=True):
        if key is not None and key not in output_by_key:
            preserved.append(_terminal_row_tombstone(stage, input_task))
    return preserved


def _terminal_row_key(task: object) -> tuple[str, int, int] | None:
    data = getattr(task, "data", None)
    if not isinstance(data, dict):
        return None
    if TERMINAL_GROUP_ID_KEY in data:
        try:
            return (
                str(data[TERMINAL_GROUP_ID_KEY]),
                int(data.get(TERMINAL_INDEX_KEY, 0)),
                int(data.get(TERMINAL_COUNT_KEY, 1)),
            )
        except (TypeError, ValueError):
            return None
    if "_curator_segment_parent_id" not in data:
        return None
    try:
        return (
            str(data["_curator_segment_parent_id"]),
            int(data.get("_curator_segment_idx", 0)),
            int(data.get("_curator_segment_count", 1)),
        )
    except (TypeError, ValueError):
        return None


def _terminal_row_tombstone(stage: ProcessingStage, task: Task) -> Task:
    data = dict(getattr(task, "data", {}) or {})
    data = _strip_payload_refs(data)
    for key in _terminal_tombstone_drop_data_keys(stage):
        data.pop(key, None)
    skip_key = str(getattr(stage, "skip_me_key", "_skip_me") or "_skip_me")
    data.setdefault(skip_key, "dropped_segment_row")
    stage_name = str(getattr(stage, "_curator_stage_id", None) or getattr(stage, "name", None) or type(stage).__name__)
    data[TERMINAL_DROPPED_KEY] = True
    data[TERMINAL_DROPPED_BY_STAGE_KEY] = stage_name
    data.setdefault(TERMINAL_DROP_REASON_KEY, "dropped_before_terminal_assembly")
    if "_curator_segment_parent_id" in data:
        data["_curator_segment_dropped"] = True
        data["_curator_segment_dropped_by_stage"] = stage_name
        data.setdefault("_curator_segment_drop_reason", "dropped_before_segment_assembly")
    tombstone = copy.copy(task)
    tombstone_data = copy.copy(task.data)
    tombstone_data.clear()
    tombstone_data.update(data)
    tombstone.data = tombstone_data
    tombstone._stage_perf = list(task._stage_perf)
    tombstone._metadata = dict(task._metadata)
    tombstone.task_id = task.task_id
    return tombstone


def _strip_payload_refs(data: dict) -> dict:
    try:
        from nemo_curator.pipeline.payload_refs import strip_payload_refs
    except Exception:
        return data
    stripped = strip_payload_refs(data)
    return stripped if isinstance(stripped, dict) else data


def _terminal_tombstone_drop_data_keys(stage: ProcessingStage) -> tuple[str, ...]:
    drop_keys = getattr(stage, "terminal_tombstone_drop_data_keys", None)
    if not callable(drop_keys):
        return ()
    return tuple(str(key) for key in drop_keys() if str(key))
