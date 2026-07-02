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

from nemo_curator.utils.retry_manifest import METADATA_DIRNAME, CompletionManifest, read_completion_manifests


class TestCompletionManifest:
    def test_mark_completed_writes_compact_manifest_with_flattened_identity(self, tmp_path: Path) -> None:
        manifest = CompletionManifest(
            checkpoint_path=tmp_path,
            namespace="slurm_array",
            completion_dirname=".slurm_array_completion",
            identity={
                "minimum_shard_index": 0,
                "shard_index": 7,
                "total_shards": 11,
            },
        )

        manifest_file = manifest.mark_completed()

        assert manifest_file is not None
        assert manifest_file.parent == tmp_path / METADATA_DIRNAME / ".slurm_array_completion"
        manifest_text = manifest_file.read_text()
        assert manifest_text == ('{"minimum_shard_index":0,"shard_index":7,"status":"completed","total_shards":11}\n')

    def test_mark_completed_supports_nested_identity_and_metadata(self, tmp_path: Path) -> None:
        manifest = CompletionManifest(
            checkpoint_path=tmp_path,
            namespace="example",
            identity={"partition_id": 3},
            metadata={"attempt": 2},
            flatten_identity=False,
            flatten_metadata=True,
        )

        manifest_file = manifest.mark_completed({"worker": "node-1"})

        assert manifest_file is not None
        assert json.loads(manifest_file.read_text()) == {
            "attempt": 2,
            "identity": {"partition_id": 3},
            "status": "completed",
            "worker": "node-1",
        }

    def test_mark_completed_nests_metadata_by_default_and_preserves_status(self, tmp_path: Path) -> None:
        manifest = CompletionManifest(
            checkpoint_path=tmp_path,
            namespace="example",
            identity={"partition_id": 3, "status": "identity-value"},
            metadata={"attempt": 2, "status": "metadata-value"},
        )

        manifest_file = manifest.mark_completed({"status": "extra-value"})

        assert manifest_file is not None
        assert json.loads(manifest_file.read_text()) == {
            "metadata": {"attempt": 2, "status": "metadata-value"},
            "partition_id": 3,
            "status": "completed",
        }

    def test_read_completion_manifests_returns_namespace_records(self, tmp_path: Path) -> None:
        manifest = CompletionManifest(tmp_path, "example", {"partition_id": 3})
        other_namespace = CompletionManifest(tmp_path, "other", {"partition_id": 4})
        manifest_file = manifest.mark_completed()
        other_namespace.mark_completed()

        records = read_completion_manifests(tmp_path, namespace="example")

        assert len(records) == 1
        assert records[0].path == manifest_file
        assert records[0].payload == {
            "partition_id": 3,
            "status": "completed",
        }

    def test_read_completion_manifests_handles_missing_directory(self, tmp_path: Path) -> None:
        assert read_completion_manifests(tmp_path, namespace="missing") == []

    def test_read_completion_manifests_rejects_malformed_manifest(self, tmp_path: Path) -> None:
        manifest_dir = tmp_path / METADATA_DIRNAME / ".example_completion"
        manifest_dir.mkdir(parents=True)
        (manifest_dir / "completed_example_bad.json").write_text("not json")

        with pytest.raises(ValueError, match="Failed to read completion manifest"):
            read_completion_manifests(tmp_path, namespace="example")

    def test_read_completion_manifests_requires_completed_status(self, tmp_path: Path) -> None:
        manifest_dir = tmp_path / METADATA_DIRNAME / ".example_completion"
        manifest_dir.mkdir(parents=True)
        (manifest_dir / "completed_example_bad.json").write_text('{"status":"failed"}')

        with pytest.raises(ValueError, match="must have status 'completed'"):
            read_completion_manifests(tmp_path, namespace="example")

    def test_read_completion_manifests_requires_string_status(self, tmp_path: Path) -> None:
        manifest_dir = tmp_path / METADATA_DIRNAME / ".example_completion"
        manifest_dir.mkdir(parents=True)
        (manifest_dir / "completed_example_bad.json").write_text('{"partition_id":3}')

        with pytest.raises(TypeError, match="must contain a string status"):
            read_completion_manifests(tmp_path, namespace="example")

    def test_same_identity_reuses_completion_file(self, tmp_path: Path) -> None:
        manifest = CompletionManifest(tmp_path, "example", {"partition_id": 3})
        same_identity = CompletionManifest(tmp_path, "example", {"partition_id": 3})

        first_file = manifest.mark_completed()
        second_file = same_identity.mark_completed()

        assert second_file == first_file
        assert first_file is not None
        assert len(list(first_file.parent.glob("completed_*.json"))) == 1

    def test_context_manager_marks_completion_on_success(self, tmp_path: Path) -> None:
        manifest = CompletionManifest(tmp_path, "example", {"partition_id": 3})

        with manifest:
            pass

        assert manifest.manifest_file is not None
        assert manifest.manifest_file.exists()

    def test_context_manager_writes_nothing_on_exception(self, tmp_path: Path) -> None:
        manifest = CompletionManifest(tmp_path, "example", {"partition_id": 3})

        with pytest.raises(RuntimeError, match="boom"), manifest:
            raise RuntimeError("boom")  # noqa: EM101

        assert manifest.manifest_file is None
        assert not manifest.manifest_dir.exists()

    def test_disabled_manifest_is_noop(self, tmp_path: Path) -> None:
        manifest = CompletionManifest(
            checkpoint_path=tmp_path,
            namespace="example",
            identity={"partition_id": 3},
            enabled=False,
        )

        assert manifest.mark_completed() is None
        assert not (tmp_path / METADATA_DIRNAME).exists()
