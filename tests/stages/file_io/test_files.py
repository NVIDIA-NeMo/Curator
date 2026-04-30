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

import fsspec

from nemo_curator.stages.file_io import (
    DeleteFilesStage,
    MaterializeFilesStage,
    UploadFilesStage,
    UploadManifestStage,
)
from nemo_curator.tasks import AudioTask, FileGroupTask


def _write_remote_bytes(path: str, payload: bytes) -> None:
    with fsspec.open(path, "wb") as fout:
        fout.write(payload)


class TestFileStages:
    def test_materialize_files_stage_writes_nested_output_field(self, tmp_path: Path) -> None:
        remote_path = f"memory://audio/{tmp_path.name}/sample.wav"
        _write_remote_bytes(remote_path, b"sample-bytes")

        task = AudioTask(
            task_id="t1",
            dataset_name="ds",
            data={"artifacts": {"remote": {"audio_path": remote_path}}},
        )

        stage = MaterializeFilesStage(
            source_field_path="artifacts.remote.audio_path",
            output_field_path="artifacts.local.materialized_path",
            temp_dir=str(tmp_path / "tmp"),
        )
        [materialized] = stage.process_batch([task])

        local_path = Path(materialized.data["artifacts"]["local"]["materialized_path"])
        assert local_path.exists()
        assert local_path.read_bytes() == b"sample-bytes"
        assert materialized.data["artifacts"]["remote"]["audio_path"] == remote_path

    def test_upload_files_stage_uses_nested_paths(self, tmp_path: Path) -> None:
        local_path = tmp_path / "sample.wav"
        local_path.write_bytes(b"upload-me")

        task = AudioTask(
            task_id="t1",
            dataset_name="ds",
            data={
                "artifacts": {"local": {"path": str(local_path)}},
                "names": {"object_key": "nested/sample.wav"},
            },
        )

        stage = UploadFilesStage(
            source_field_path="artifacts.local.path",
            output_field_path="artifacts.remote.uri",
            protocol="memory",
            bucket="uploaded-files",
            key_field_path="names.object_key",
        )
        uploaded = stage.process(task)

        uploaded_uri = uploaded.data["artifacts"]["remote"]["uri"]
        assert uploaded_uri == "memory://uploaded-files/nested/sample.wav"
        with fsspec.open(uploaded_uri, "rb") as fin:
            assert fin.read() == b"upload-me"

    def test_delete_files_stage_removes_nested_field_and_object(self, tmp_path: Path) -> None:
        remote_path = f"memory://delete/{tmp_path.name}/sample.wav"
        _write_remote_bytes(remote_path, b"delete-me")

        task = AudioTask(
            task_id="t1",
            dataset_name="ds",
            data={"artifacts": {"remote": {"uri": remote_path}}},
        )

        stage = DeleteFilesStage(source_field_path="artifacts.remote.uri")
        deleted = stage.process(task)

        fs, resolved = fsspec.core.url_to_fs(remote_path)
        assert not fs.exists(resolved)
        assert "uri" not in deleted.data["artifacts"]["remote"]

    def test_upload_manifest_stage_uploads_file_group_outputs(self, tmp_path: Path) -> None:
        manifest = tmp_path / "output.jsonl"
        manifest.write_text(json.dumps({"audio_filepath": "a.wav"}) + "\n")

        stage = UploadManifestStage(protocol="memory", bucket="manifest-bucket", key_prefix="jsonl")
        result = stage.process(FileGroupTask(task_id="fg", dataset_name="ds", data=[str(manifest)]))

        assert result.data == ["memory://manifest-bucket/jsonl/output.jsonl"]
        with fsspec.open(result.data[0], "rt", encoding="utf-8") as fin:
            assert json.loads(fin.read().strip())["audio_filepath"] == "a.wav"
        assert result._metadata["local_source_files"] == [str(manifest)]
