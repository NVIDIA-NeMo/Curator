# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Explicit keyed fan-in task support for executor-backed pipeline phases."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .tasks import Task


@dataclass
class TaskGroup(Task[list[Task]]):
    """A deterministic, driver-created group of tasks sharing one data key.

    Grouping is deliberately explicit: an executor may split an ordinary
    ``process_batch`` call arbitrarily, so a stage that needs a complete key
    group must consume ``TaskGroup`` instances rather than infer a group from
    its input batch.  A group is non-resumable at the consuming fan-in stage
    unless that stage can define source attribution for every member.
    """

    group_key: str = ""
    data: list[Task] = field(default_factory=list)

    @property
    def num_items(self) -> int:
        return sum(task.num_items for task in self.data)

    def validate(self) -> bool:
        return bool(self.group_key) and bool(self.data)

    def get_deterministic_id(self) -> str:
        identity = {
            "group_key": self.group_key,
            "member_task_ids": sorted(task.task_id for task in self.data),
        }
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def group_tasks_by_data_key(tasks: list[Task], key: str) -> list[TaskGroup]:
    """Create deterministic task groups from a required string data key.

    This is the boundary between two executor-backed phases: callers collect
    a completed phase, group its task outputs by an application key, then pass
    the resulting ``TaskGroup`` objects as ``Pipeline.run(initial_tasks=...)``
    to a fan-in phase.  Metadata and performance history are retained from all
    members rather than selecting an arbitrary representative task.
    """
    groups: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        data = getattr(task, "data", None)
        if not isinstance(data, dict) or not isinstance(data.get(key), str) or not data[key]:
            msg = f"Task {task.task_id!r} has no non-empty string data[{key!r}] for keyed grouping"
            raise ValueError(msg)
        groups[data[key]].append(task)

    grouped_tasks: list[TaskGroup] = []
    for group_key in sorted(groups):
        members = sorted(
            groups[group_key],
            key=lambda task: (getattr(task, "data", {}).get("turn_index", 0), task.task_id),
        )
        metadata: dict[str, Any] = {}
        source_files = [source_file for task in members for source_file in task._metadata.get("source_files", [])]
        if source_files:
            metadata["source_files"] = list(dict.fromkeys(source_files))
        dataset_name = ",".join(dict.fromkeys(task.dataset_name for task in members))
        grouped_tasks.append(
            TaskGroup(
                dataset_name=dataset_name,
                data=members,
                group_key=group_key,
                _metadata=metadata,
                _stage_perf=[perf for task in members for perf in task._stage_perf],
            )
        )
    return grouped_tasks
