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

from nemo_curator.utils.retry_manifest import METADATA_DIRNAME, RetryManifest


class TestRetryManifest:
    def test_mark_pending_writes_compact_manifest_with_flattened_identity(self, tmp_path: Path) -> None:
        manifest = RetryManifest(
            checkpoint_path=tmp_path,
            namespace="slurm_array",
            retry_dirname=".slurm_array_retry",
            identity={
                "minimum_shard_index": 0,
                "shard_index": 7,
                "total_shards": 11,
            },
        )

        manifest_file = manifest.mark_pending()

        assert manifest_file is not None
        assert manifest_file.parent == tmp_path / METADATA_DIRNAME / ".slurm_array_retry"
        marker_text = manifest_file.read_text()
        payload = json.loads(marker_text)
        assert marker_text == ('{"minimum_shard_index":0,"shard_index":7,"status":"pending","total_shards":11}\n')
        assert payload["status"] == "pending"
        assert payload["minimum_shard_index"] == 0
        assert payload["shard_index"] == 7
        assert payload["total_shards"] == 11

    def test_mark_pending_can_write_nested_identity_and_metadata(self, tmp_path: Path) -> None:
        manifest = RetryManifest(
            checkpoint_path=tmp_path,
            namespace="example",
            identity={"partition_id": 3},
            metadata={"attempt": 2},
            flatten_identity=False,
            flatten_metadata=True,
        )

        manifest_file = manifest.mark_pending()

        assert manifest_file is not None
        payload = json.loads(manifest_file.read_text())
        assert payload == {
            "attempt": 2,
            "identity": {"partition_id": 3},
            "status": "pending",
        }

    def test_mark_failed_updates_same_manifest_with_error(self, tmp_path: Path) -> None:
        manifest = RetryManifest(
            checkpoint_path=tmp_path,
            namespace="example",
            identity={"partition_id": 3},
        )

        pending_file = manifest.mark_pending()
        failed_file = manifest.mark_failed(RuntimeError("boom"))

        assert failed_file == pending_file
        assert failed_file is not None
        payload = json.loads(failed_file.read_text())
        assert payload["status"] == "failed"
        assert payload["error_type"] == "RuntimeError"

    def test_mark_retryable_updates_same_manifest_with_extra_metadata(self, tmp_path: Path) -> None:
        manifest = RetryManifest(
            checkpoint_path=tmp_path,
            namespace="example",
            identity={"partition_id": 3},
        )

        pending_file = manifest.mark_pending()
        retryable_file = manifest.mark_retryable(
            "failed_tasks",
            {
                "failed_task_marker_count": 2,
            },
        )

        assert retryable_file == pending_file
        assert retryable_file is not None
        payload = json.loads(retryable_file.read_text())
        assert payload["status"] == "failed_tasks"
        assert payload["failed_task_marker_count"] == 2

    def test_same_identity_reuses_manifest_file_across_instances(self, tmp_path: Path) -> None:
        manifest = RetryManifest(
            checkpoint_path=tmp_path,
            namespace="example",
            identity={"partition_id": 3},
        )
        same_identity_manifest = RetryManifest(
            checkpoint_path=tmp_path,
            namespace="example",
            identity={"partition_id": 3},
        )

        failed_file = manifest.mark_failed(RuntimeError("boom"))
        pending_file = same_identity_manifest.mark_pending()

        assert pending_file == failed_file
        assert pending_file is not None
        assert len(list(pending_file.parent.glob("manifest_*.json"))) == 1
        payload = json.loads(pending_file.read_text())
        assert payload == {
            "partition_id": 3,
            "status": "pending",
        }

    def test_mark_success_removes_matching_manifests(self, tmp_path: Path) -> None:
        manifest = RetryManifest(
            checkpoint_path=tmp_path,
            namespace="example",
            identity={"partition_id": 3},
        )
        other_manifest = RetryManifest(
            checkpoint_path=tmp_path,
            namespace="example",
            identity={"partition_id": 4},
        )
        manifest_file = manifest.mark_pending()
        other_manifest_file = other_manifest.mark_pending()

        removed_count = manifest.mark_success()

        assert removed_count == 1
        assert manifest_file is not None
        assert other_manifest_file is not None
        assert not manifest_file.exists()
        assert other_manifest_file.exists()

    def test_mark_success_does_not_remove_prefix_collision_identity(self, tmp_path: Path) -> None:
        manifest = RetryManifest(
            checkpoint_path=tmp_path,
            namespace="example",
            identity={"partition_id": "3"},
        )
        other_manifest = RetryManifest(
            checkpoint_path=tmp_path,
            namespace="example",
            identity={"partition_id": "3_extra"},
        )
        manifest_file = manifest.mark_pending()
        other_manifest_file = other_manifest.mark_pending()

        removed_count = manifest.mark_success()

        assert removed_count == 1
        assert manifest_file is not None
        assert other_manifest_file is not None
        assert not manifest_file.exists()
        assert other_manifest_file.exists()

    def test_context_manager_removes_manifest_on_success(self, tmp_path: Path) -> None:
        manifest = RetryManifest(
            checkpoint_path=tmp_path,
            namespace="example",
            identity={"partition_id": 3},
        )

        with manifest as active_manifest:
            manifest_file = active_manifest.manifest_file
            assert manifest_file is not None
            assert manifest_file.exists()

        assert manifest_file is not None
        assert not manifest_file.exists()

    def test_context_manager_marks_manifest_failed_on_exception(self, tmp_path: Path) -> None:
        manifest = RetryManifest(
            checkpoint_path=tmp_path,
            namespace="example",
            identity={"partition_id": 3},
        )
        manifest_file: Path | None = None

        def run_with_error() -> None:
            nonlocal manifest_file
            with manifest:
                manifest_file = manifest.manifest_file
                msg = "boom"
                raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="boom"):
            run_with_error()

        assert manifest_file is not None
        assert manifest_file.exists()
        payload = json.loads(manifest_file.read_text())
        assert payload["status"] == "failed"
        assert payload["error_type"] == "RuntimeError"

    def test_disabled_manifest_is_noop(self, tmp_path: Path) -> None:
        manifest = RetryManifest(
            checkpoint_path=tmp_path,
            namespace="example",
            identity={"partition_id": 3},
            enabled=False,
        )

        assert manifest.mark_pending() is None
        assert manifest.mark_failed(RuntimeError("boom")) is None
        assert manifest.mark_retryable("failed_tasks") is None
        assert manifest.mark_success() == 0
        assert not (tmp_path / METADATA_DIRNAME).exists()
