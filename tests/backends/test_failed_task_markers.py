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

import os
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from nemo_curator.backends.failed_task_markers import (
    FAILED_TASK_MANIFEST_FILENAME,
    FAILED_TASKS_DIR_ENV_VAR,
    configure_failed_task_manifest_dir,
    configure_slurm_array_failed_task_manifest_dir,
    failed_task_manifest_exists,
    record_failed_tasks,
)


class TestFailedTaskManifest:
    def test_configure_manifest_dir_uses_local_attempt_identity(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.delenv(FAILED_TASKS_DIR_ENV_VAR, raising=False)

        manifest_dir = configure_failed_task_manifest_dir(tmp_path)

        assert manifest_dir.parent == tmp_path / ".nemo_curator_metadata" / ".failed_tasks"
        assert manifest_dir.name.startswith("local_attempt_")
        assert os.environ[FAILED_TASKS_DIR_ENV_VAR] == str(manifest_dir)

    def test_configure_slurm_array_manifest_dir_uses_attempt_identity(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.delenv(FAILED_TASKS_DIR_ENV_VAR, raising=False)
        monkeypatch.setenv("SLURM_JOB_ID", "123")
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
        monkeypatch.setenv("SLURM_RESTART_COUNT", "2")

        manifest_dir = configure_slurm_array_failed_task_manifest_dir(tmp_path, shard_index=9)

        assert manifest_dir == (
            tmp_path
            / ".nemo_curator_metadata"
            / ".failed_tasks"
            / "slurm_job_123"
            / "array_task_7"
            / "restart_2"
            / "shard_9"
        )
        assert os.environ[FAILED_TASKS_DIR_ENV_VAR] == str(manifest_dir)

    def test_configure_slurm_array_manifest_dir_preserves_explicit_override(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        override = tmp_path / "custom-failed-tasks"
        monkeypatch.setenv(FAILED_TASKS_DIR_ENV_VAR, str(override))

        assert configure_slurm_array_failed_task_manifest_dir(tmp_path / "checkpoint", shard_index=9) == override

    def test_record_failed_tasks_writes_single_manifest(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        manifest_dir = tmp_path / "failed-tasks"
        monkeypatch.setenv(FAILED_TASKS_DIR_ENV_VAR, str(manifest_dir))

        record_failed_tasks()

        manifest_files = list(manifest_dir.glob("*.json"))
        assert manifest_files == [manifest_dir / FAILED_TASK_MANIFEST_FILENAME]
        assert failed_task_manifest_exists()

    def test_additional_failed_tasks_leave_existing_manifest_unchanged(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        manifest_dir = tmp_path / "failed-tasks"
        monkeypatch.setenv(FAILED_TASKS_DIR_ENV_VAR, str(manifest_dir))
        record_failed_tasks()
        manifest_file = manifest_dir / FAILED_TASK_MANIFEST_FILENAME

        record_failed_tasks()

        assert list(manifest_dir.glob("*.json")) == [manifest_file]

    def test_record_failed_tasks_without_configured_attempt_is_noop(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.delenv(FAILED_TASKS_DIR_ENV_VAR, raising=False)
        monkeypatch.chdir(tmp_path)

        record_failed_tasks()

        assert not (tmp_path / ".nemo_curator_metadata").exists()

    def test_record_failed_tasks_propagates_manifest_write_failure(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        manifest_dir = tmp_path / "failed-tasks"
        monkeypatch.setenv(FAILED_TASKS_DIR_ENV_VAR, str(manifest_dir))

        def fail_touch(_self: Path, *_args: object, **_kwargs: object) -> None:
            msg = "storage unavailable"
            raise OSError(msg)

        monkeypatch.setattr(Path, "touch", fail_touch)

        with pytest.raises(OSError, match="storage unavailable"):
            record_failed_tasks()

    def test_failed_task_manifest_exists_accepts_explicit_directory(self, tmp_path: Path) -> None:
        manifest_dir = tmp_path / "failed-tasks"
        manifest_dir.mkdir()
        (manifest_dir / FAILED_TASK_MANIFEST_FILENAME).touch()

        assert failed_task_manifest_exists(manifest_dir)
        assert not failed_task_manifest_exists(tmp_path / "missing")
