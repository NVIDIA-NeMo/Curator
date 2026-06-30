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
from dataclasses import dataclass
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from nemo_curator.backends.slurm_array import (
    SLURM_ARRAY_ENABLED_ENV_VAR,
    SLURM_ARRAY_MINIMUM_SHARD_INDEX_ENV_VAR,
    SLURM_ARRAY_SHARD_INDEX_ENV_VAR,
    SLURM_ARRAY_TOTAL_SHARDS_ENV_VAR,
    SlurmArrayConfig,
    SlurmArrayRetryPlan,
    build_slurm_array_retry_manifest,
    configure_slurm_array_source_filtering,
    find_slurm_array_retries,
    filter_slurm_array_source_tasks,
    format_slurm_array_indices,
    is_slurm_array_driver_process,
    resolve_slurm_array_config,
)
from nemo_curator.tasks import Task
from nemo_curator.utils.retry_manifest import METADATA_DIRNAME


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


class TestSlurmArray:
    def test_config_inactive_without_array_environment(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.delenv(SLURM_ARRAY_ENABLED_ENV_VAR, raising=False)
        monkeypatch.delenv(SLURM_ARRAY_SHARD_INDEX_ENV_VAR, raising=False)
        monkeypatch.delenv(SLURM_ARRAY_TOTAL_SHARDS_ENV_VAR, raising=False)
        monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
        monkeypatch.delenv("SLURM_ARRAY_TASK_COUNT", raising=False)

        assert SlurmArrayConfig.from_env() is None

    def test_config_enabled_by_default_with_slurm_env_vars(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.delenv(SLURM_ARRAY_ENABLED_ENV_VAR, raising=False)
        monkeypatch.delenv(SLURM_ARRAY_SHARD_INDEX_ENV_VAR, raising=False)
        monkeypatch.delenv(SLURM_ARRAY_TOTAL_SHARDS_ENV_VAR, raising=False)
        monkeypatch.delenv(SLURM_ARRAY_MINIMUM_SHARD_INDEX_ENV_VAR, raising=False)
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
        monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "11")

        slurm_array = SlurmArrayConfig.from_env()

        assert slurm_array == SlurmArrayConfig(shard_index=7, total_shards=11, minimum_shard_index=0)

    @pytest.mark.parametrize("disabled_value", ["0", "false", "no", "off"])
    def test_config_can_be_explicitly_disabled(
        self, monkeypatch: MonkeyPatch, disabled_value: str
    ) -> None:
        monkeypatch.setenv(SLURM_ARRAY_ENABLED_ENV_VAR, disabled_value)
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
        monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "11")

        assert SlurmArrayConfig.from_env() is None

    def test_config_supports_curator_env_vars(self, monkeypatch: MonkeyPatch) -> None:
        _enable_slurm_array(monkeypatch, shard_index=3, total_shards=8, minimum_shard_index=1)

        slurm_array = SlurmArrayConfig.from_env()

        assert slurm_array == SlurmArrayConfig(shard_index=3, total_shards=8, minimum_shard_index=1)

    def test_configure_source_filtering_sets_curator_env_vars(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.delenv(SLURM_ARRAY_ENABLED_ENV_VAR, raising=False)
        monkeypatch.delenv(SLURM_ARRAY_SHARD_INDEX_ENV_VAR, raising=False)
        monkeypatch.delenv(SLURM_ARRAY_TOTAL_SHARDS_ENV_VAR, raising=False)
        monkeypatch.delenv(SLURM_ARRAY_MINIMUM_SHARD_INDEX_ENV_VAR, raising=False)

        configure_slurm_array_source_filtering(
            shard_index=3,
            total_shards=8,
            minimum_shard_index=1,
        )

        assert SlurmArrayConfig.from_env() == SlurmArrayConfig(
            shard_index=3,
            total_shards=8,
            minimum_shard_index=1,
        )

    def test_explicitly_enabled_config_requires_slurm_env_vars(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv(SLURM_ARRAY_ENABLED_ENV_VAR, "1")
        monkeypatch.delenv(SLURM_ARRAY_SHARD_INDEX_ENV_VAR, raising=False)
        monkeypatch.delenv(SLURM_ARRAY_TOTAL_SHARDS_ENV_VAR, raising=False)
        monkeypatch.delenv(SLURM_ARRAY_MINIMUM_SHARD_INDEX_ENV_VAR, raising=False)
        monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
        monkeypatch.delenv("SLURM_ARRAY_TASK_COUNT", raising=False)

        with pytest.raises(ValueError, match="SLURM_ARRAY_TASK_ID"):
            SlurmArrayConfig.from_env()

    def test_config_rejects_non_integer_env_var(self, monkeypatch: MonkeyPatch) -> None:
        _enable_slurm_array(monkeypatch, shard_index="not-an-int", total_shards=4)

        with pytest.raises(ValueError, match=rf"{SLURM_ARRAY_SHARD_INDEX_ENV_VAR}.*not-an-int"):
            SlurmArrayConfig.from_env()

    def test_resolution_non_source_stage_ignores_slurm_array(self, monkeypatch: MonkeyPatch) -> None:
        _enable_slurm_array(monkeypatch, shard_index=0, total_shards=100)

        assert resolve_slurm_array_config(is_source_stage=False) is None

    def test_resolution_requires_positive_total_shards(self, monkeypatch: MonkeyPatch) -> None:
        _enable_slurm_array(monkeypatch, shard_index=0, total_shards=0)

        with pytest.raises(ValueError, match="total_shards must be greater than 0"):
            resolve_slurm_array_config(is_source_stage=True)

    def test_resolution_warns_for_out_of_range_shard(
        self, monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _enable_slurm_array(monkeypatch, shard_index=0, total_shards=10, minimum_shard_index=1)

        with caplog.at_level("WARNING"):
            slurm_array = resolve_slurm_array_config(is_source_stage=True)

        assert slurm_array == SlurmArrayConfig(shard_index=0, total_shards=10, minimum_shard_index=1)
        assert "outside the assignable shard range [1, 10]" in caplog.text

    def test_filtering_is_disabled_without_config(self) -> None:
        tasks = [_task(f"0_{i}") for i in range(3)]

        assert filter_slurm_array_source_tasks(tasks, None, "source") == tasks

    def test_assigns_each_source_task_to_one_shard(self) -> None:
        tasks = [_task(f"0_{i}") for i in range(8)]
        assigned_task_ids = []

        for shard_index in range(3):
            slurm_array = SlurmArrayConfig(shard_index=shard_index, total_shards=3)
            assigned_task_ids.extend(
                task.task_id for task in filter_slurm_array_source_tasks(tasks, slurm_array, "source")
            )

        assert set(assigned_task_ids) == {task.task_id for task in tasks}
        assert len(assigned_task_ids) == len(tasks)

    def test_supports_minimum_shard_index(self) -> None:
        tasks = [_task(f"0_{i}") for i in range(8)]

        zero_indexed_result = [
            task.task_id
            for task in filter_slurm_array_source_tasks(
                tasks,
                SlurmArrayConfig(shard_index=0, total_shards=3),
                "source",
            )
        ]
        one_indexed_result = [
            task.task_id
            for task in filter_slurm_array_source_tasks(
                tasks,
                SlurmArrayConfig(shard_index=1, total_shards=3, minimum_shard_index=1),
                "source",
            )
        ]

        assert one_indexed_result == zero_indexed_result

    def test_rejects_nondeterministic_source_task_ids(self) -> None:
        tasks = [_task("r123"), _task("0_1")]
        slurm_array = SlurmArrayConfig(shard_index=0, total_shards=3)

        with pytest.raises(ValueError, match="requires deterministic task IDs"):
            filter_slurm_array_source_tasks(tasks, slurm_array, "source")

    def test_is_driver_process_for_local_and_slurm_head(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.delenv("SLURM_NODEID", raising=False)

        assert is_slurm_array_driver_process(use_slurm=False) is True
        assert is_slurm_array_driver_process(use_slurm=True) is True

        monkeypatch.setenv("SLURM_NODEID", "1")

        assert is_slurm_array_driver_process(use_slurm=True) is False

    def test_build_retry_manifest_writes_slurm_shard_identity(self, tmp_path: Path) -> None:
        manifest = build_slurm_array_retry_manifest(
            checkpoint_path=str(tmp_path),
            shard_index=7,
            total_shards=11,
            minimum_shard_index=1,
        )

        assert manifest is not None
        manifest_file = manifest.mark_pending()
        assert manifest_file is not None
        assert manifest_file.parent == tmp_path / METADATA_DIRNAME / ".slurm_array_retry"

        payload = json.loads(manifest_file.read_text())
        assert payload == {
            "minimum_shard_index": 1,
            "shard_index": 7,
            "status": "pending",
            "total_shards": 11,
        }

    def test_build_retry_manifest_disabled_without_checkpoint(self) -> None:
        assert (
            build_slurm_array_retry_manifest(
                checkpoint_path=None,
                shard_index=7,
                total_shards=11,
                minimum_shard_index=1,
            )
            is None
        )

    def test_find_retries_returns_original_config_and_all_outstanding_statuses(self, tmp_path: Path) -> None:
        pending = build_slurm_array_retry_manifest(str(tmp_path), 7, 11, 1)
        failed = build_slurm_array_retry_manifest(str(tmp_path), 3, 11, 1)
        failed_tasks = build_slurm_array_retry_manifest(str(tmp_path), 4, 11, 1)
        assert pending is not None
        assert failed is not None
        assert failed_tasks is not None
        pending.mark_pending()
        failed.mark_failed(RuntimeError("boom"))
        failed_tasks.mark_retryable("failed_tasks", {"failed_task_marker_count": 2})

        retry_plan = find_slurm_array_retries(tmp_path)

        assert retry_plan == SlurmArrayRetryPlan(
            shard_indices=(3, 4, 7),
            total_shards=11,
            minimum_shard_index=1,
        )

    def test_find_retries_returns_none_without_manifests(self, tmp_path: Path) -> None:
        assert find_slurm_array_retries(tmp_path) is None

    def test_find_retries_rejects_mixed_logical_runs(self, tmp_path: Path) -> None:
        first = build_slurm_array_retry_manifest(str(tmp_path), 1, 10, 0)
        second = build_slurm_array_retry_manifest(str(tmp_path), 2, 20, 0)
        assert first is not None
        assert second is not None
        first.mark_pending()
        second.mark_pending()

        with pytest.raises(ValueError, match="multiple shard configurations"):
            find_slurm_array_retries(tmp_path)

    @pytest.mark.parametrize(
        ("indices", "expected"),
        [
            ([], ""),
            ([7], "7"),
            ([1, 2, 5, 6, 7, 99], "1-2,5-7,99"),
            ([7, 5, 6, 5], "5-7"),
        ],
    )
    def test_format_slurm_array_indices(self, indices: list[int], expected: str) -> None:
        assert format_slurm_array_indices(indices) == expected
