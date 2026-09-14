# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import pytest

from nemo_curator.tasks import AudioTask, group_tasks_by_data_key


def _task(conversation_id: str, turn_index: int, task_id: str) -> AudioTask:
    task = AudioTask(
        dataset_name="audio",
        data={"conversation_id": conversation_id, "turn_index": turn_index},
        task_id=task_id,
    )
    task._metadata = {"source_files": [f"{task_id}.jsonl"]}
    return task


def test_group_tasks_by_data_key_is_deterministic_and_preserves_metadata() -> None:
    groups = group_tasks_by_data_key(
        [_task("b", 1, "b1"), _task("a", 1, "a1"), _task("b", 0, "b0")],
        "conversation_id",
    )

    assert [group.group_key for group in groups] == ["a", "b"]
    assert [task.task_id for task in groups[1].data] == ["b0", "b1"]
    assert groups[1]._metadata["source_files"] == ["b0.jsonl", "b1.jsonl"]
    assert groups[1].get_deterministic_id() == groups[1].get_deterministic_id()


def test_group_tasks_by_data_key_rejects_missing_or_non_string_keys() -> None:
    invalid = AudioTask(dataset_name="audio", data={"conversation_id": 2})
    with pytest.raises(ValueError, match="non-empty string"):
        group_tasks_by_data_key([invalid], "conversation_id")
