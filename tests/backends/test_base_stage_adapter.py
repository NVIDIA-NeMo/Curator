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

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from nemo_curator.backends.base import (
    FAILED_TASKS_DIR_ENV_VAR,
    SLURM_ARRAY_ENABLED_ENV_VAR,
    SLURM_ARRAY_MINIMUM_SHARD_INDEX_ENV_VAR,
    SLURM_ARRAY_SHARD_INDEX_ENV_VAR,
    SLURM_ARRAY_TOTAL_SHARDS_ENV_VAR,
    BaseStageAdapter,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import EmptyTask, FileGroupTask, Task
from nemo_curator.tasks.sentinels import FailedTask


@dataclass
class _FailedStage(ProcessingStage[Task, Task]):
    name: str = "failed"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: Task) -> Task:
        return FailedTask()


@dataclass
class _SourceFanoutStage(ProcessingStage[Task, FileGroupTask]):
    name: str = "source"
    is_source_stage: bool = True
    partitions: list[list[str]] = field(default_factory=list)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: Task) -> list[FileGroupTask]:
        return [FileGroupTask(dataset_name="d", data=list(partition)) for partition in self.partitions]


@dataclass
class _SimpleTask(Task[list[int]]):
    @property
    def num_items(self) -> int:
        return 0

    def validate(self) -> bool:
        return True


def _task(task_id: str = "") -> _SimpleTask:
    task = _SimpleTask(dataset_name="d", data=[])
    task.task_id = task_id
    return task


def _enable_slurm_array(
    monkeypatch: MonkeyPatch,
    shard_index: int | str | None,
    total_shards: int | str | None,
    minimum_shard_index: int | str | None = 0,
) -> None:
    monkeypatch.setenv(SLURM_ARRAY_ENABLED_ENV_VAR, "1")
    if shard_index is None:
        monkeypatch.delenv(SLURM_ARRAY_SHARD_INDEX_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(SLURM_ARRAY_SHARD_INDEX_ENV_VAR, str(shard_index))

    if total_shards is None:
        monkeypatch.delenv(SLURM_ARRAY_TOTAL_SHARDS_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(SLURM_ARRAY_TOTAL_SHARDS_ENV_VAR, str(total_shards))

    if minimum_shard_index is None:
        monkeypatch.delenv(SLURM_ARRAY_MINIMUM_SHARD_INDEX_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(SLURM_ARRAY_MINIMUM_SHARD_INDEX_ENV_VAR, str(minimum_shard_index))


class TestBaseStageAdapter:
    def test_process_batch_writes_failed_task_marker_when_enabled(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        marker_dir = tmp_path / "failed-tasks"
        monkeypatch.setenv(FAILED_TASKS_DIR_ENV_VAR, str(marker_dir))

        output = BaseStageAdapter(_FailedStage()).process_batch([_task("0_7")])

        assert output == []
        marker_files = list(marker_dir.glob("failed_task_*.json"))
        assert len(marker_files) == 1

        payload = json.loads(marker_files[0].read_text())
        assert payload["stage_name"] == "failed"
        assert payload["task_id"] == "0_7_0"
        assert payload["dataset_name"] == "failed"
        assert payload["task_type"] == "FailedTask"
        assert isinstance(payload["hostname"], str)
        assert isinstance(payload["pid"], int)
        assert isinstance(payload["created_at"], str)

    def test_process_batch_does_not_write_failed_task_marker_by_default(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        marker_dir = tmp_path / "failed-tasks"
        monkeypatch.delenv(FAILED_TASKS_DIR_ENV_VAR, raising=False)

        output = BaseStageAdapter(_FailedStage()).process_batch([_task("0_7")])

        assert output == []
        assert not marker_dir.exists()

    def test_source_filtering_is_disabled_by_default(self) -> None:
        partitions = [[f"file{i}.parquet"] for i in range(3)]
        stage = _SourceFanoutStage(partitions=partitions)

        output = BaseStageAdapter(stage).process_batch([EmptyTask()])

        assert [task.data for task in output] == partitions

    def test_assigns_each_source_task_to_one_shard(self, monkeypatch: MonkeyPatch) -> None:
        partitions = [[f"file{i}.parquet"] for i in range(8)]
        expected_partitions = {tuple(partition) for partition in partitions}
        assigned_partitions = []

        for shard_index in range(3):
            _enable_slurm_array(monkeypatch, shard_index=shard_index, total_shards=3)
            stage = _SourceFanoutStage(partitions=partitions)

            output = BaseStageAdapter(stage).process_batch([EmptyTask()])
            assigned_partitions.extend(tuple(task.data) for task in output)

        assert set(assigned_partitions) == expected_partitions
        assert len(assigned_partitions) == len(expected_partitions)

    def test_supports_minimum_shard_index(self, monkeypatch: MonkeyPatch) -> None:
        partitions = [[f"file{i}.parquet"] for i in range(8)]
        zero_indexed_stage = _SourceFanoutStage(partitions=partitions)
        _enable_slurm_array(monkeypatch, shard_index=0, total_shards=3)

        zero_indexed_result = [task.data for task in BaseStageAdapter(zero_indexed_stage).process_batch([EmptyTask()])]

        one_indexed_stage = _SourceFanoutStage(partitions=partitions)
        _enable_slurm_array(monkeypatch, shard_index=1, total_shards=3, minimum_shard_index=1)

        one_indexed_result = [task.data for task in BaseStageAdapter(one_indexed_stage).process_batch([EmptyTask()])]

        assert one_indexed_result == zero_indexed_result

    def test_reads_slurm_env_vars(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv(SLURM_ARRAY_ENABLED_ENV_VAR, "1")
        monkeypatch.delenv(SLURM_ARRAY_SHARD_INDEX_ENV_VAR, raising=False)
        monkeypatch.delenv(SLURM_ARRAY_TOTAL_SHARDS_ENV_VAR, raising=False)
        monkeypatch.delenv(SLURM_ARRAY_MINIMUM_SHARD_INDEX_ENV_VAR, raising=False)
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
        monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "11")

        adapter = BaseStageAdapter(_SourceFanoutStage(partitions=[["file.parquet"]]))

        adapter.process_batch([EmptyTask()])

        assert adapter._resolved_slurm_array.shard_index == 7
        assert adapter._resolved_slurm_array.total_shards == 11
        assert adapter._resolved_slurm_array.minimum_shard_index == 0

    def test_supports_curator_env_vars(self, monkeypatch: MonkeyPatch) -> None:
        _enable_slurm_array(monkeypatch, shard_index=3, total_shards=8, minimum_shard_index=1)
        adapter = BaseStageAdapter(_SourceFanoutStage(partitions=[["file.parquet"]]))

        adapter.process_batch([EmptyTask()])

        assert adapter._resolved_slurm_array.shard_index == 3
        assert adapter._resolved_slurm_array.total_shards == 8
        assert adapter._resolved_slurm_array.minimum_shard_index == 1

    def test_requires_slurm_env_vars_by_default(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv(SLURM_ARRAY_ENABLED_ENV_VAR, "1")
        monkeypatch.delenv(SLURM_ARRAY_SHARD_INDEX_ENV_VAR, raising=False)
        monkeypatch.delenv(SLURM_ARRAY_TOTAL_SHARDS_ENV_VAR, raising=False)
        monkeypatch.delenv(SLURM_ARRAY_MINIMUM_SHARD_INDEX_ENV_VAR, raising=False)
        monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
        monkeypatch.delenv("SLURM_ARRAY_TASK_COUNT", raising=False)
        stage = _SourceFanoutStage(partitions=[["file.parquet"]])

        with pytest.raises(ValueError, match="SLURM_ARRAY_TASK_ID"):
            BaseStageAdapter(stage).process_batch([EmptyTask()])

    def test_rejects_non_integer_env_var(self, monkeypatch: MonkeyPatch) -> None:
        _enable_slurm_array(monkeypatch, shard_index="not-an-int", total_shards=4)
        stage = _SourceFanoutStage(partitions=[["file.parquet"]])

        with pytest.raises(ValueError, match=rf"{SLURM_ARRAY_SHARD_INDEX_ENV_VAR}.*not-an-int"):
            BaseStageAdapter(stage).process_batch([EmptyTask()])

    def test_requires_positive_total_shards(self, monkeypatch: MonkeyPatch) -> None:
        _enable_slurm_array(monkeypatch, shard_index=0, total_shards=0)
        stage = _SourceFanoutStage(partitions=[["file.parquet"]])

        with pytest.raises(ValueError, match="total_shards must be greater than 0"):
            BaseStageAdapter(stage).process_batch([EmptyTask()])

    def test_warns_for_out_of_range_shard(
        self, monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _enable_slurm_array(monkeypatch, shard_index=0, total_shards=10, minimum_shard_index=1)
        stage = _SourceFanoutStage(partitions=[["file.parquet"]])

        with caplog.at_level("WARNING"):
            output = BaseStageAdapter(stage).process_batch([EmptyTask()])

        assert output == []
        assert "outside the assignable shard range [1, 10]" in caplog.text

    def test_non_source_stage_ignores_slurm_array(self, monkeypatch: MonkeyPatch) -> None:
        _enable_slurm_array(monkeypatch, shard_index=0, total_shards=100)
        partitions = [[f"file{i}.parquet"] for i in range(3)]
        stage = _SourceFanoutStage(
            is_source_stage=False,
            partitions=partitions,
        )

        output = BaseStageAdapter(stage).process_batch([EmptyTask()])

        assert [task.data for task in output] == partitions

    def test_rejects_nondeterministic_source_task_ids(self, monkeypatch: MonkeyPatch) -> None:
        _enable_slurm_array(monkeypatch, shard_index=0, total_shards=3)
        stage = _SourceFanoutStage(partitions=[["a.parquet"], ["b.parquet"], ["c.parquet"]])

        with pytest.raises(ValueError, match="requires deterministic task IDs"):
            BaseStageAdapter(stage).process_batch([_task("0_0"), _task("0_1")])
