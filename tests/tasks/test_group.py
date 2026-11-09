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


def test_grouping_retains_all_member_provenance_without_aliasing() -> None:
    members = [_task("conversation", i, str(i)) for i in range(2)]
    for i, member in enumerate(members):
        member._source_id = f"source-{i}"
        member._metadata["nested"] = {"value": i}
    group = group_tasks_by_data_key(members, "conversation_id")[0]
    assert group._metadata["source_ids"] == ["source-0", "source-1"]
    assert group._metadata["member_task_ids"] == ["0", "1"]
    group._metadata["member_metadata"][0]["nested"]["value"] = "changed"
    assert members[0]._metadata["nested"]["value"] == 0
    assert not group._source_id
