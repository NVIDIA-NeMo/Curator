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
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from nemo_curator.backends.failed_task_markers import (
    FAILED_TASKS_DIR_ENV_VAR,
    read_failed_task_markers,
    record_failed_tasks,
    summarize_failed_task_markers,
)
from nemo_curator.tasks.sentinels import FailedTask


def _failed_task(task_id: str = "0_7_0") -> FailedTask:
    task = FailedTask()
    task.task_id = task_id
    return task


class TestFailedTaskMarkers:
    def test_read_failed_task_markers_returns_identities(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        marker_dir = tmp_path / "failed-tasks"
        monkeypatch.setenv(FAILED_TASKS_DIR_ENV_VAR, str(marker_dir))
        record_failed_tasks("stage-b", [_failed_task("0_8_0")])
        record_failed_tasks("stage-a", [_failed_task("0_7_0")])

        markers = read_failed_task_markers()

        assert {(marker.stage_name, marker.task_id) for marker in markers} == {
            ("stage-a", "0_7_0"),
            ("stage-b", "0_8_0"),
        }
        assert all(marker.path.parent == marker_dir for marker in markers)

    def test_read_failed_task_markers_handles_missing_dir(self, tmp_path: Path) -> None:
        assert read_failed_task_markers(tmp_path / "missing") == []

    def test_read_failed_task_markers_rejects_malformed_marker(self, tmp_path: Path) -> None:
        marker_dir = tmp_path / "failed-tasks"
        marker_dir.mkdir()
        (marker_dir / "failed_task_bad.json").write_text('{"task_id":"0_7_0"}')

        with pytest.raises(ValueError, match="must contain string stage_name and task_id"):
            read_failed_task_markers(marker_dir)

    def test_record_failed_tasks_writes_marker_when_enabled(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        marker_dir = tmp_path / "failed-tasks"
        monkeypatch.setenv(FAILED_TASKS_DIR_ENV_VAR, str(marker_dir))

        record_failed_tasks("failed", [_failed_task()])

        marker_files = list(marker_dir.glob("failed_task_*.json"))
        assert len(marker_files) == 1

        marker_text = marker_files[0].read_text()
        assert marker_text == '{"stage_name":"failed","task_id":"0_7_0"}\n'

        payload = json.loads(marker_text)
        assert payload == {
            "stage_name": "failed",
            "task_id": "0_7_0",
        }

    def test_record_failed_tasks_does_not_write_marker_by_default(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        marker_dir = tmp_path / "failed-tasks"
        monkeypatch.delenv(FAILED_TASKS_DIR_ENV_VAR, raising=False)

        record_failed_tasks("failed", [_failed_task()])

        assert not marker_dir.exists()

    def test_record_failed_tasks_does_not_write_marker_for_empty_list(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        marker_dir = tmp_path / "failed-tasks"
        monkeypatch.setenv(FAILED_TASKS_DIR_ENV_VAR, str(marker_dir))

        record_failed_tasks("failed", [])

        assert not marker_dir.exists()

    def test_record_failed_tasks_reuses_marker_for_same_task_identity(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        marker_dir = tmp_path / "failed-tasks"
        monkeypatch.setenv(FAILED_TASKS_DIR_ENV_VAR, str(marker_dir))

        record_failed_tasks("failed", [_failed_task("0_7_0")])
        record_failed_tasks("failed", [_failed_task("0_7_0")])

        marker_files = list(marker_dir.glob("failed_task_*.json"))
        assert len(marker_files) == 1
        assert marker_files[0].read_text() == '{"stage_name":"failed","task_id":"0_7_0"}\n'

    def test_summarize_failed_task_markers_reads_env_dir(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        marker_dir = tmp_path / "failed-tasks"
        monkeypatch.setenv(FAILED_TASKS_DIR_ENV_VAR, str(marker_dir))
        record_failed_tasks("failed", [_failed_task("0_7_0"), _failed_task("0_8_0")])

        summary = summarize_failed_task_markers()

        assert summary == {"failed_task_marker_count": 2}

    def test_summarize_failed_task_markers_handles_missing_dir(self, tmp_path: Path) -> None:
        marker_dir = tmp_path / "missing"

        summary = summarize_failed_task_markers(marker_dir)

        assert summary == {"failed_task_marker_count": 0}
