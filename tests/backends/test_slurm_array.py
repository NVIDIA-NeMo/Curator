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

import nemo_curator.backends.slurm_array as slurm_array_module
from nemo_curator.backends.slurm_array import (
    SLURM_ARRAY_COMPLETION_MANIFEST_NAMESPACE,
    SLURM_ARRAY_ENABLED_ENV_VAR,
    SLURM_ARRAY_MINIMUM_SHARD_INDEX_ENV_VAR,
    SLURM_ARRAY_SHARD_INDEX_ENV_VAR,
    SLURM_ARRAY_TOTAL_SHARDS_ENV_VAR,
    SlurmArrayConfig,
    SlurmArrayRetryPlan,
    SlurmArrayRetrySubmission,
    build_slurm_array_completion_manifest,
    build_slurm_array_retry_submissions,
    configure_slurm_array_source_filtering,
    filter_slurm_array_source_tasks,
    find_slurm_array_retries,
    format_slurm_array_indices,
    is_slurm_array_driver_process,
    resolve_slurm_array_config,
)
from nemo_curator.tasks import Task
from nemo_curator.utils.atomic_io import write_json_atomically
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

    @pytest.mark.parametrize(
        ("shard_index", "minimum_shard_index", "expected_error"),
        [
            (-1, 0, "shard_index must be non-negative"),
            (0, -1, "minimum_shard_index must be non-negative"),
        ],
    )
    def test_resolution_rejects_negative_shard_indices(
        self,
        monkeypatch: MonkeyPatch,
        shard_index: int,
        minimum_shard_index: int,
        expected_error: str,
    ) -> None:
        _enable_slurm_array(
            monkeypatch,
            shard_index=shard_index,
            total_shards=10,
            minimum_shard_index=minimum_shard_index,
        )

        with pytest.raises(ValueError, match=expected_error):
            resolve_slurm_array_config(is_source_stage=True)

    def test_resolution_warns_for_out_of_range_shard(self, monkeypatch: MonkeyPatch) -> None:
        _enable_slurm_array(monkeypatch, shard_index=0, total_shards=10, minimum_shard_index=1)
        warnings: list[str] = []

        def capture_warning(message: str, *args: object) -> None:
            warnings.append(message.format(*args))

        monkeypatch.setattr(slurm_array_module.logger, "warning", capture_warning)

        slurm_array = resolve_slurm_array_config(is_source_stage=True)

        assert slurm_array == SlurmArrayConfig(shard_index=0, total_shards=10, minimum_shard_index=1)
        assert warnings == [
            "shard_index=0 is outside the assignable shard range [1, 10]. "
            "This task will not receive any source tasks."
        ]

    def test_filtering_is_disabled_without_config(self) -> None:
        tasks = [_task(f"0_{i}") for i in range(3)]

        assert filter_slurm_array_source_tasks(tasks, None, "source") == tasks

    def test_filtering_returns_empty_for_empty_input(self) -> None:
        slurm_array = SlurmArrayConfig(shard_index=0, total_shards=3)

        assert filter_slurm_array_source_tasks([], slurm_array, "source") == []

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
        assert is_slurm_array_driver_process() is True

        monkeypatch.setenv("SLURM_NODEID", "0")
        assert is_slurm_array_driver_process() is True

        monkeypatch.setenv("SLURM_NODEID", "1")
        assert is_slurm_array_driver_process() is False

    def test_build_completion_manifest_writes_run_config_and_shard_identity(self, tmp_path: Path) -> None:
        manifest = build_slurm_array_completion_manifest(
            checkpoint_path=str(tmp_path),
            shard_index=7,
            total_shards=11,
            minimum_shard_index=1,
        )

        assert manifest is not None
        manifest_file = manifest.mark_completed()
        assert manifest_file is not None
        assert manifest_file.parent == tmp_path / METADATA_DIRNAME / ".slurm_array_completion"

        payload = json.loads(manifest_file.read_text())
        assert payload == {
            "minimum_shard_index": 1,
            "shard_index": 7,
            "status": "completed",
            "total_shards": 11,
        }
        run_config = json.loads((manifest_file.parent / "run.json").read_text())
        assert run_config == {
            "minimum_shard_index": 1,
            "total_shards": 11,
        }

    def test_build_completion_manifest_disabled_without_checkpoint(self) -> None:
        assert (
            build_slurm_array_completion_manifest(
                checkpoint_path=None,
                shard_index=7,
                total_shards=11,
                minimum_shard_index=1,
            )
            is None
        )

    def test_find_retries_returns_shards_without_completion_manifests(self, tmp_path: Path) -> None:
        first = build_slurm_array_completion_manifest(str(tmp_path), 1, 4, 1)
        third = build_slurm_array_completion_manifest(str(tmp_path), 3, 4, 1)
        assert first is not None
        assert third is not None
        first.mark_completed()
        third.mark_completed()

        retry_plan = find_slurm_array_retries(tmp_path)

        assert retry_plan == SlurmArrayRetryPlan(
            shard_indices=(2, 4),
            total_shards=4,
            minimum_shard_index=1,
        )

    def test_find_retries_returns_all_shards_when_none_completed(self, tmp_path: Path) -> None:
        manifest = build_slurm_array_completion_manifest(str(tmp_path), 1, 3, 1)
        assert manifest is not None

        assert find_slurm_array_retries(tmp_path) == SlurmArrayRetryPlan(
            shard_indices=(1, 2, 3),
            total_shards=3,
            minimum_shard_index=1,
        )

    def test_find_retries_returns_empty_plan_when_all_shards_completed(self, tmp_path: Path) -> None:
        for shard_index in range(3):
            manifest = build_slurm_array_completion_manifest(str(tmp_path), shard_index, 3, 0)
            assert manifest is not None
            manifest.mark_completed()

        assert find_slurm_array_retries(tmp_path) == SlurmArrayRetryPlan(
            shard_indices=(),
            total_shards=3,
            minimum_shard_index=0,
        )

    def test_find_retries_returns_none_without_run_config(self, tmp_path: Path) -> None:
        assert find_slurm_array_retries(tmp_path) is None

    def test_find_retries_rejects_zero_total_shards_in_run_config(self, tmp_path: Path) -> None:
        manifest_dir = tmp_path / METADATA_DIRNAME / ".slurm_array_completion"
        manifest_dir.mkdir(parents=True)
        (manifest_dir / "run.json").write_text('{"minimum_shard_index":0,"total_shards":0}\n')

        with pytest.raises(ValueError, match="total_shards greater than 0"):
            find_slurm_array_retries(tmp_path)

    def test_find_retries_rejects_out_of_range_completed_shard(self, tmp_path: Path) -> None:
        manifest_dir = tmp_path / METADATA_DIRNAME / ".slurm_array_completion"
        manifest_dir.mkdir(parents=True)
        (manifest_dir / "run.json").write_text('{"minimum_shard_index":1,"total_shards":3}\n')
        # shard_index=5 is outside the valid range [1, 3]
        filename = f"completed_{SLURM_ARRAY_COMPLETION_MANIFEST_NAMESPACE}_bad.json"
        (manifest_dir / filename).write_text(
            '{"minimum_shard_index":1,"shard_index":5,"status":"completed","total_shards":3}'
        )

        with pytest.raises(ValueError, match="outside the original shard range"):
            find_slurm_array_retries(tmp_path)

    def test_find_retries_rejects_negative_minimum_in_run_config(self, tmp_path: Path) -> None:
        manifest_dir = tmp_path / METADATA_DIRNAME / ".slurm_array_completion"
        manifest_dir.mkdir(parents=True)
        (manifest_dir / "run.json").write_text('{"minimum_shard_index":-1,"total_shards":3}\n')

        with pytest.raises(ValueError, match="non-negative minimum_shard_index"):
            find_slurm_array_retries(tmp_path)

    def test_build_retry_submissions_uses_logical_indices_without_limit(self) -> None:
        retry_plan = SlurmArrayRetryPlan(shard_indices=(1, 1001), total_shards=2000, minimum_shard_index=0)

        assert build_slurm_array_retry_submissions(retry_plan) == (
            SlurmArrayRetrySubmission(array_indices=(1, 1001), shard_index_offset=0),
        )

    def test_build_retry_submissions_derives_offset_windows(self) -> None:
        retry_plan = SlurmArrayRetryPlan(
            shard_indices=(1, 2, 1005, 1006, 2999),
            total_shards=3000,
            minimum_shard_index=0,
        )

        assert build_slurm_array_retry_submissions(retry_plan, max_array_size=1000) == (
            SlurmArrayRetrySubmission(array_indices=(1, 2), shard_index_offset=0),
            SlurmArrayRetrySubmission(array_indices=(5, 6), shard_index_offset=1000),
            SlurmArrayRetrySubmission(array_indices=(999,), shard_index_offset=2000),
        )

    def test_build_retry_submissions_returns_no_submissions_when_complete(self) -> None:
        retry_plan = SlurmArrayRetryPlan(shard_indices=(), total_shards=3, minimum_shard_index=0)

        assert build_slurm_array_retry_submissions(retry_plan, max_array_size=1000) == ()

    def test_build_retry_submissions_rejects_negative_indices_without_limit(self) -> None:
        retry_plan = SlurmArrayRetryPlan(shard_indices=(-1, 0), total_shards=2, minimum_shard_index=-1)

        with pytest.raises(ValueError, match="must be non-negative"):
            build_slurm_array_retry_submissions(retry_plan)

    def test_build_completion_manifest_rejects_mixed_logical_runs(self, tmp_path: Path) -> None:
        first = build_slurm_array_completion_manifest(str(tmp_path), 1, 10, 0)
        assert first is not None

        with pytest.raises(ValueError, match="use a separate checkpoint path"):
            build_slurm_array_completion_manifest(str(tmp_path), 2, 20, 0)

    def test_build_completion_manifest_validates_config_created_by_concurrent_writer(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        def competing_writer(path: Path, _payload: object, **_kwargs: object) -> bool:
            write_json_atomically(
                path,
                {"minimum_shard_index": 0, "total_shards": 20},
                separators=(",", ":"),
            )
            return False

        monkeypatch.setattr(slurm_array_module, "write_json_atomically_if_absent", competing_writer)

        with pytest.raises(ValueError, match="use a separate checkpoint path"):
            build_slurm_array_completion_manifest(str(tmp_path), 1, 10, 0)

    @pytest.mark.parametrize(("shard_index", "minimum_shard_index"), [(-1, -1), (0, -1)])
    def test_build_completion_manifest_rejects_negative_indices(
        self,
        tmp_path: Path,
        shard_index: int,
        minimum_shard_index: int,
    ) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            build_slurm_array_completion_manifest(
                str(tmp_path),
                shard_index=shard_index,
                total_shards=10,
                minimum_shard_index=minimum_shard_index,
            )

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
