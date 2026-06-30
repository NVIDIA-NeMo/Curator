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

from pathlib import Path

from pytest import MonkeyPatch

from nemo_curator.backends.failed_task_markers import (
    FAILED_TASK_MANIFEST_FILENAME,
    FAILED_TASKS_DIR_ENV_VAR,
    failed_task_manifest_exists,
    record_failed_tasks,
)
from nemo_curator.tasks.sentinels import FailedTask


def _failed_task(task_id: str = "0_7_0") -> FailedTask:
    task = FailedTask()
    task.task_id = task_id
    return task


class TestFailedTaskManifest:
    def test_record_failed_tasks_writes_single_manifest(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        manifest_dir = tmp_path / "failed-tasks"
        monkeypatch.setenv(FAILED_TASKS_DIR_ENV_VAR, str(manifest_dir))

        record_failed_tasks("failed", [_failed_task("0_7_0"), _failed_task("0_8_0")])

        manifest_files = list(manifest_dir.glob("*.json"))
        assert manifest_files == [manifest_dir / FAILED_TASK_MANIFEST_FILENAME]
        assert manifest_files[0].read_text() == '{"status":"failed_tasks"}\n'
        assert failed_task_manifest_exists()

    def test_additional_failed_tasks_leave_existing_manifest_unchanged(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        manifest_dir = tmp_path / "failed-tasks"
        monkeypatch.setenv(FAILED_TASKS_DIR_ENV_VAR, str(manifest_dir))
        record_failed_tasks("stage-a", [_failed_task("0_7_0")])
        manifest_file = manifest_dir / FAILED_TASK_MANIFEST_FILENAME
        original_manifest = manifest_file.read_text()

        record_failed_tasks("stage-b", [_failed_task("0_8_0")])

        assert list(manifest_dir.glob("*.json")) == [manifest_file]
        assert manifest_file.read_text() == original_manifest

    def test_record_failed_tasks_does_not_write_manifest_by_default(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        manifest_dir = tmp_path / "failed-tasks"
        monkeypatch.delenv(FAILED_TASKS_DIR_ENV_VAR, raising=False)

        record_failed_tasks("failed", [_failed_task()])

        assert not manifest_dir.exists()
        assert not failed_task_manifest_exists()

    def test_record_failed_tasks_does_not_write_manifest_for_empty_list(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        manifest_dir = tmp_path / "failed-tasks"
        monkeypatch.setenv(FAILED_TASKS_DIR_ENV_VAR, str(manifest_dir))

        record_failed_tasks("failed", [])

        assert not manifest_dir.exists()
        assert not failed_task_manifest_exists()

    def test_failed_task_manifest_exists_accepts_explicit_directory(self, tmp_path: Path) -> None:
        manifest_dir = tmp_path / "failed-tasks"
        manifest_dir.mkdir()
        (manifest_dir / FAILED_TASK_MANIFEST_FILENAME).write_text('{"status":"failed_tasks"}\n')

        assert failed_task_manifest_exists(manifest_dir)
        assert not failed_task_manifest_exists(tmp_path / "missing")
