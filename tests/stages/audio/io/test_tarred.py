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

import io
import json
import tarfile
import wave
from dataclasses import dataclass
from pathlib import Path

import pytest

from nemo_curator.stages.audio.io.materialize import CleanupTemporaryAudioStage
from nemo_curator.stages.audio.io.tarred import (
    MaterializeTarredAudioStage,
    TarredAudioManifestPartitionStage,
    TarredAudioManifestReader,
    TarredAudioManifestReaderStage,
    _PipeStream,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, FileGroupTask, _EmptyTask


def _write_tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as tar:
        for name, content in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))


def _make_wav_bytes(*, sample_rate: int = 16000, duration_sec: float = 1.0) -> bytes:
    frames = int(sample_rate * duration_sec)
    data = (b"\x00\x00" * frames)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(data)
    return buffer.getvalue()


def _make_file_group_task(paths: list[str]) -> FileGroupTask:
    return FileGroupTask(task_id="group", dataset_name="dataset", data=paths)


@dataclass
class _PathConsumerStage(ProcessingStage[AudioTask, AudioTask]):
    audio_filepath_key: str = "audio_filepath"
    seen_exists_key: str = "_path_exists"
    name: str = "path_consumer"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.seen_exists_key]

    def process(self, task: AudioTask) -> AudioTask:
        task.data[self.seen_exists_key] = Path(task.data[self.audio_filepath_key]).exists()
        return task


class TestTarredAudioManifestReader:
    def test_partition_stage_expands_op_cl_pattern(self, tmp_path: Path) -> None:
        for shard_id in range(2):
            (tmp_path / f"manifest_{shard_id}.json").write_text(
                json.dumps({"audio_filepath": f"{shard_id}.wav", "text": f"text-{shard_id}"}) + "\n"
            )

        stage = TarredAudioManifestPartitionStage(
            manifest_paths=str(tmp_path / "manifest__OP_0..1_CL_.json"),
            files_per_partition=1,
        )
        result = stage.process(_EmptyTask)

        assert len(result) == 2
        assert result[0].data == [str(tmp_path / "manifest_0.json")]
        assert result[1].data == [str(tmp_path / "manifest_1.json")]

    def test_reader_maps_manifest_entries_to_tar_members(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest_0.json"
        tar_path = tmp_path / "audio_0.tar"
        manifest.write_text(
            "\n".join(
                [
                    json.dumps({"audio_filepath": "a.wav", "text": "alpha"}),
                    json.dumps({"audio_filepath": "b.wav-sub1", "text": "beta", "offset": 0.0, "duration": 0.5}),
                ]
            )
        )
        _write_tar(tar_path, {"a.wav": b"a-bytes", "b.wav": _make_wav_bytes(duration_sec=1.0)})

        stage = TarredAudioManifestReaderStage(tar_paths=str(tar_path))
        result = stage.process(_make_file_group_task([str(manifest)]))

        assert len(result) == 2
        assert result[0].data["_tar_path"] == str(tar_path)
        assert result[0].data["_tar_member"] == "a.wav"
        assert result[0].sample_key
        assert result[1].data["_tar_member"] == "b.wav"
        assert result[1].data["audio_filepath"] == "b.wav-sub1"
        assert result[1].data["_audio_source_type"] == "tarred"
        assert result[1].sample_key
        assert result[0].sample_key != result[1].sample_key

    def test_reader_raises_when_manifest_entry_missing_in_tar(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest_0.json"
        tar_path = tmp_path / "audio_0.tar"
        manifest.write_text(
            "\n".join(
                [
                    json.dumps({"audio_filepath": "a.wav"}),
                    json.dumps({"audio_filepath": "missing.wav"}),
                ]
            )
        )
        _write_tar(tar_path, {"a.wav": b"a-bytes"})

        stage = TarredAudioManifestReaderStage(tar_paths=str(tar_path), skip_missing_entries=False)

        with pytest.raises(RuntimeError, match=r"Cannot locate tar member 'missing\.wav'"):
            stage.process(_make_file_group_task([str(manifest)]))

    def test_reader_skips_missing_entries_when_enabled(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest_0.json"
        tar_path = tmp_path / "audio_0.tar"
        manifest.write_text(
            "\n".join(
                [
                    json.dumps({"audio_filepath": "a.wav"}),
                    json.dumps({"audio_filepath": "missing.wav"}),
                ]
            )
        )
        _write_tar(tar_path, {"a.wav": b"a-bytes"})

        stage = TarredAudioManifestReaderStage(tar_paths=str(tar_path), skip_missing_entries=True)
        result = stage.process(_make_file_group_task([str(manifest)]))

        assert len(result) == 1
        assert result[0].data["audio_filepath"] == "a.wav"

    def test_composite_decomposes_and_reads_sharded_specs(self, tmp_path: Path) -> None:
        manifest_0 = tmp_path / "manifest_0.json"
        manifest_1 = tmp_path / "manifest_1.json"
        tar_0 = tmp_path / "audio_0.tar"
        tar_1 = tmp_path / "audio_1.tar"

        manifest_0.write_text(json.dumps({"audio_filepath": "a.wav", "text": "alpha"}) + "\n")
        manifest_1.write_text(json.dumps({"audio_filepath": "b.wav", "text": "beta"}) + "\n")
        _write_tar(tar_0, {"a.wav": b"a-bytes"})
        _write_tar(tar_1, {"b.wav": b"b-bytes"})

        reader = TarredAudioManifestReader(
            manifest_paths=str(tmp_path / "manifest__OP_0..1_CL_.json"),
            tar_paths=str(tmp_path / "audio__OP_0..1_CL_.tar"),
            files_per_partition=1,
        )
        partition_stage, reader_stage = reader.decompose()

        file_tasks = partition_stage.process(_EmptyTask)
        results = reader_stage.process(file_tasks[0])

        assert len(file_tasks) == 2
        assert len(results) == 1
        assert results[0].data["_tar_path"] == str(tar_0)

    def test_composite_propagates_limit_to_partition_stage(self, tmp_path: Path) -> None:
        for shard_id in range(2):
            (tmp_path / f"manifest_{shard_id}.json").write_text(
                json.dumps({"audio_filepath": f"{shard_id}.wav", "text": f"text-{shard_id}"}) + "\n"
            )
            _write_tar(tmp_path / f"audio_{shard_id}.tar", {f"{shard_id}.wav": b"bytes"})

        reader = TarredAudioManifestReader(
            manifest_paths=str(tmp_path / "manifest__OP_0..1_CL_.json"),
            tar_paths=str(tmp_path / "audio__OP_0..1_CL_.tar"),
            files_per_partition=1,
            limit=1,
        )
        partition_stage, _reader_stage = reader.decompose()
        file_tasks = partition_stage.process(_EmptyTask)

        assert len(file_tasks) == 1


class TestTarredAudioMaterialization:
    def test_pipe_stream_allows_sigpipe_when_opted_in(self) -> None:
        class _FakeProcess:
            def __init__(self, return_code: int):
                self.stdout = io.BytesIO(b"")
                self.stderr = io.BytesIO(b"")
                self._return_code = return_code

            def wait(self) -> int:
                return self._return_code

        pipe_stream = _PipeStream("dummy", allow_sigpipe=True)
        pipe_stream.process = _FakeProcess(return_code=141)  # type: ignore[assignment]

        assert pipe_stream.__exit__(None, None, None) is False

    def test_pipe_stream_raises_for_sigpipe_by_default(self) -> None:
        class _FakeProcess:
            def __init__(self, return_code: int):
                self.stdout = io.BytesIO(b"")
                self.stderr = io.BytesIO(b"")
                self._return_code = return_code

            def wait(self) -> int:
                return self._return_code

        pipe_stream = _PipeStream("dummy")
        pipe_stream.process = _FakeProcess(return_code=141)  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="Pipe command failed with exit code 141"):
            pipe_stream.__exit__(None, None, None)

    def test_materialize_and_cleanup_roundtrip(self, tmp_path: Path) -> None:
        tar_path = tmp_path / "audio_0.tar"
        raw_audio = b"test-bytes"
        _write_tar(tar_path, {"sample.wav": raw_audio})

        task = AudioTask(
            task_id="t1",
            dataset_name="ds",
            data={
                "audio_filepath": "sample.wav",
                "_tar_path": str(tar_path),
                "_tar_member": "sample.wav",
            },
        )

        materialize = MaterializeTarredAudioStage(temp_dir=str(tmp_path / "tmp"))
        [materialized] = materialize.process_batch([task])

        temp_path = Path(materialized.data["_temporary_audio_path"])
        assert temp_path.exists()
        assert temp_path.read_bytes() == raw_audio
        assert materialized.data["audio_filepath"] == temp_path.as_posix()
        assert materialized.data["_manifest_audio_filepath"] == "sample.wav"
        assert materialized.data["_materialization_mode"] == "member"

        cleanup = CleanupTemporaryAudioStage()
        cleaned = cleanup.process(materialized)

        assert not temp_path.exists()
        assert cleaned.data["audio_filepath"] == "sample.wav"
        assert "_temporary_audio_path" not in cleaned.data
        assert "_materialization_mode" not in cleaned.data
        assert "_manifest_audio_filepath" not in cleaned.data

    def test_materialize_segment_for_offset_entries(self, tmp_path: Path) -> None:
        tar_path = tmp_path / "audio_0.tar"
        _write_tar(tar_path, {"sample.wav": _make_wav_bytes(duration_sec=1.0)})

        task = AudioTask(
            task_id="t1",
            dataset_name="ds",
            data={
                "audio_filepath": "sample.wav-sub1",
                "offset": 0.25,
                "duration": 0.5,
                "_tar_path": str(tar_path),
                "_tar_member": "sample.wav",
            },
        )

        materialize = MaterializeTarredAudioStage(temp_dir=str(tmp_path / "tmp"))
        [materialized] = materialize.process_batch([task])

        temp_path = Path(materialized.data["_temporary_audio_path"])
        assert temp_path.exists()
        assert materialized.data["_materialization_mode"] == "segment"

        with wave.open(temp_path.as_posix(), "rb") as wav_file:
            assert wav_file.getframerate() == 16000
            assert wav_file.getnframes() == 8000  # 0.5 sec at 16kHz

    def test_materialize_segment_for_duration_only_entries(self, tmp_path: Path) -> None:
        tar_path = tmp_path / "audio_0.tar"
        _write_tar(tar_path, {"sample.wav": _make_wav_bytes(duration_sec=1.0)})

        task = AudioTask(
            task_id="t1",
            dataset_name="ds",
            data={
                "audio_filepath": "sample.wav",
                "offset": 0.0,
                "duration": 0.5,
                "_tar_path": str(tar_path),
                "_tar_member": "sample.wav",
            },
        )

        materialize = MaterializeTarredAudioStage(temp_dir=str(tmp_path / "tmp"))
        [materialized] = materialize.process_batch([task])

        temp_path = Path(materialized.data["_temporary_audio_path"])
        assert temp_path.exists()
        assert materialized.data["_materialization_mode"] == "segment"

        with wave.open(temp_path.as_posix(), "rb") as wav_file:
            assert wav_file.getnframes() == 8000  # 0.5 sec at 16kHz

    def test_materialize_segment_raises_for_offset_past_audio_end(self, tmp_path: Path) -> None:
        tar_path = tmp_path / "audio_0.tar"
        _write_tar(tar_path, {"sample.wav": _make_wav_bytes(duration_sec=0.25)})

        task = AudioTask(
            task_id="t1",
            dataset_name="ds",
            data={
                "audio_filepath": "sample.wav-sub1",
                "offset": 1.0,
                "duration": 0.25,
                "_tar_path": str(tar_path),
                "_tar_member": "sample.wav",
            },
        )

        materialize = MaterializeTarredAudioStage(temp_dir=str(tmp_path / "tmp"))

        with pytest.raises(RuntimeError, match=r"Offset 1\.0s exceeds audio length"):
            materialize.process_batch([task])

    def test_materialize_to_durable_directory_keeps_file_after_cleanup(self, tmp_path: Path) -> None:
        tar_path = tmp_path / "audio_0.tar"
        raw_audio = b"durable-bytes"
        _write_tar(tar_path, {"sample.wav": raw_audio})

        task = AudioTask(
            task_id="t1",
            dataset_name="ds",
            sample_key="sample-key-1",
            data={
                "audio_filepath": "sample.wav",
                "_tar_path": str(tar_path),
                "_tar_member": "sample.wav",
            },
        )

        materialization_dir = tmp_path / "materialized"
        materialize = MaterializeTarredAudioStage(materialization_dir=str(materialization_dir))
        [materialized] = materialize.process_batch([task])

        durable_path = Path(materialized.data["audio_filepath"])
        assert durable_path.exists()
        assert durable_path.read_bytes() == raw_audio
        assert durable_path.is_relative_to(materialization_dir)
        assert "_temporary_audio_path" not in materialized.data

        cleanup = CleanupTemporaryAudioStage()
        cleaned = cleanup.process(materialized)

        assert durable_path.exists()
        assert cleaned.data["audio_filepath"] == "sample.wav"

    def test_pipe_transport_reads_manifest_and_tar(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest_0.json"
        tar_path = tmp_path / "audio_0.tar"
        manifest.write_text(json.dumps({"audio_filepath": "sample.wav", "text": "alpha"}) + "\n")
        _write_tar(tar_path, {"sample.wav": b"pipe-bytes"})

        manifest_cmd = (
            f'pipe:python3 -c "from pathlib import Path; import sys; '
            f"sys.stdout.buffer.write(Path(r'{manifest}').read_bytes())\""
        )
        tar_cmd = (
            f'pipe:python3 -c "from pathlib import Path; import sys; '
            f"sys.stdout.buffer.write(Path(r'{tar_path}').read_bytes())\""
        )

        reader_stage = TarredAudioManifestReaderStage(tar_paths=tar_cmd, transport="auto")
        [audio_task] = reader_stage.process(_make_file_group_task([manifest_cmd]))

        materialize = MaterializeTarredAudioStage(temp_dir=str(tmp_path / "tmp"), transport="auto")
        [materialized] = materialize.process_batch([audio_task])

        temp_path = Path(materialized.data["_temporary_audio_path"])
        assert temp_path.exists()
        assert temp_path.read_bytes() == b"pipe-bytes"

    def test_materialize_decodes_shared_member_once_for_segment_tasks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import nemo_curator.stages.audio.io.materialize as materialize_module

        tar_path = tmp_path / "audio_0.tar"
        _write_tar(tar_path, {"sample.wav": _make_wav_bytes(duration_sec=1.0)})

        tasks = [
            AudioTask(
                task_id="t1",
                dataset_name="ds",
                data={
                    "audio_filepath": "sample.wav-sub1",
                    "offset": 0.0,
                    "duration": 0.25,
                    "_tar_path": str(tar_path),
                    "_tar_member": "sample.wav",
                },
            ),
            AudioTask(
                task_id="t2",
                dataset_name="ds",
                data={
                    "audio_filepath": "sample.wav-sub2",
                    "offset": 0.25,
                    "duration": 0.25,
                    "_tar_path": str(tar_path),
                    "_tar_member": "sample.wav",
                },
            ),
        ]

        read_calls = 0
        original_read = materialize_module.soundfile.read

        def counting_read(*args: object, **kwargs: object) -> object:
            nonlocal read_calls
            read_calls += 1
            return original_read(*args, **kwargs)

        monkeypatch.setattr(materialize_module.soundfile, "read", counting_read)

        materialize = MaterializeTarredAudioStage(temp_dir=str(tmp_path / "tmp"))
        materialize.process_batch(tasks)

        assert read_calls == 1

    def test_manual_end_to_end_reader_materialize_consume_cleanup(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest_0.json"
        tar_path = tmp_path / "audio_0.tar"
        manifest.write_text(json.dumps({"audio_filepath": "sample.wav", "text": "alpha"}) + "\n")
        _write_tar(tar_path, {"sample.wav": b"consumer-bytes"})

        reader = TarredAudioManifestReader(
            manifest_paths=str(manifest),
            tar_paths=str(tar_path),
        )
        partition_stage, reader_stage = reader.decompose()
        [file_group] = partition_stage.process(_EmptyTask)
        audio_tasks = reader_stage.process(file_group)

        materialize = MaterializeTarredAudioStage(temp_dir=str(tmp_path / "tmp"))
        materialized = materialize.process_batch(audio_tasks)

        consumer = _PathConsumerStage()
        consumed = consumer.process_batch(materialized)
        assert consumed[0].data["_path_exists"] is True

        cleanup = CleanupTemporaryAudioStage()
        cleaned = cleanup.process_batch(consumed)

        assert cleaned[0].data["audio_filepath"] == "sample.wav"
        assert "_temporary_audio_path" not in cleaned[0].data
