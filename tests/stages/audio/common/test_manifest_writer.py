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
import pickle
from pathlib import Path

import numpy as np
import pytest

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.audio.common import (
    GetAudioDurationStage,
    ManifestCheckpointStage,
    ManifestWriterStage,
    PreserveByValueStage,
)
from nemo_curator.tasks import AudioTask


@pytest.mark.parametrize("prepared", [False, True])
def test_writer_actor_restart_retains_committed_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prepared: bool
) -> None:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata
    from nemo_curator.backends.ray_data import adapter

    monkeypatch.setattr(adapter, "get_worker_metadata_and_node_id", lambda: (NodeInfo(), WorkerMetadata()))
    path = tmp_path / "output.jsonl"
    stage = ManifestWriterStage(str(path))
    if prepared:
        stage.prepare_on_driver()
    first = adapter.create_actor_from_stage(pickle.loads(pickle.dumps(stage)))()  # noqa: S301
    first.stage.process(AudioTask(data={"row": 1}))
    restarted = adapter.create_actor_from_stage(pickle.loads(pickle.dumps(stage)))()  # noqa: S301
    restarted.stage.process(AudioTask(data={"row": 2}))
    expected = [{"row": 1}, {"row": 2}] if prepared else [{"row": 2}]
    assert [json.loads(line) for line in path.read_text().splitlines()] == expected
    stage.reset_for_retry()
    if prepared:
        stage.prepare_on_driver()
    fresh_attempt = adapter.create_actor_from_stage(pickle.loads(pickle.dumps(stage)))()  # noqa: S301
    fresh_attempt.stage.process(AudioTask(data={"row": 3}))
    assert [json.loads(line) for line in path.read_text().splitlines()] == [{"row": 3}]


def test_writer_direct_setup_replaces_existing_output(tmp_path: Path) -> None:
    path = tmp_path / "output.jsonl"
    path.write_text('{"old": true}\n')
    writer = ManifestWriterStage(str(path))
    writer.setup()
    writer.process(AudioTask(data={"new": True}))
    writer.setup()
    assert path.read_text() == ""


def test_writer_driver_preparation_occurs_once_and_worker_hooks_never_truncate(tmp_path: Path) -> None:
    path = tmp_path / "output.jsonl"
    path.write_text('{"old": true}\n')
    driver = ManifestWriterStage(str(path))
    driver.prepare_on_driver()
    assert path.read_text() == ""
    first = pickle.loads(pickle.dumps(driver))  # noqa: S301
    second = pickle.loads(pickle.dumps(driver))  # noqa: S301
    first.setup_on_node()
    first.setup()
    first.process(AudioTask(data={"row": 1}))
    with pytest.raises(FileExistsError):
        driver.prepare_on_driver()
    second.setup_on_node()
    second.setup()
    second.process(AudioTask(data={"row": 2}))
    driver.finalize()
    assert [json.loads(line) for line in path.read_text().splitlines()] == [{"row": 1}, {"row": 2}]
    assert not Path(f"{path}._RUN").exists()


@pytest.mark.parametrize("stage_cls", [ManifestWriterStage, ManifestCheckpointStage])
def test_prepared_worker_rejects_a_separate_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage_cls: type
) -> None:
    import fsspec

    from nemo_curator.stages.audio import common

    namespace = {"name": "driver"}
    filesystem = fsspec.filesystem("file")
    monkeypatch.setattr(
        common, "url_to_fs", lambda _path: (filesystem, str(tmp_path / namespace["name"] / "output.jsonl"))
    )
    driver = stage_cls("/logical/output.jsonl")
    driver.prepare_on_driver()
    worker = pickle.loads(pickle.dumps(driver))  # noqa: S301
    namespace["name"] = "worker"
    with pytest.raises(RuntimeError, match=r"cannot (verify|see)"):
        worker.setup_on_node()
    with pytest.raises(RuntimeError, match=r"cannot (verify|see)"):
        worker.setup(WorkerMetadata())
    assert not (tmp_path / "worker" / "output.jsonl").exists()
    assert not (tmp_path / "worker" / "output.jsonl._COMPLETE").exists()


@pytest.mark.parametrize("stage_cls", [ManifestWriterStage, ManifestCheckpointStage])
@pytest.mark.parametrize("tamper", ["missing_output", "foreign_owner"])
def test_prepared_workers_refuse_missing_or_foreign_state(tmp_path: Path, stage_cls: type, tamper: str) -> None:
    path = tmp_path / "output.jsonl"
    driver = stage_cls(str(path))
    driver.prepare_on_driver()
    worker = pickle.loads(pickle.dumps(driver))  # noqa: S301
    suffix = "._RUN" if stage_cls is ManifestWriterStage else "._RETRY_OWNER"
    owner = Path(f"{path}{suffix}")
    if tamper == "missing_output":
        path.unlink()
    else:
        value = "foreign-token" if stage_cls is ManifestWriterStage else '{"token": "foreign-token"}'
        owner.write_text(value)
    before = path.read_bytes() if path.exists() else None
    owner_before = owner.read_bytes()
    with pytest.raises(RuntimeError, match=r"cannot (verify|see)"):
        worker.setup(WorkerMetadata())
    assert (path.read_bytes() if path.exists() else None) == before
    assert owner.read_bytes() == owner_before


class TestManifestWriterStage:
    """Unit tests for ManifestWriterStage."""

    def test_writes_entry_to_jsonl(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonl"
        writer = ManifestWriterStage(output_path=str(out))
        writer.setup_on_node()
        writer.setup()

        task = AudioTask(
            data={"audio_filepath": "a.wav", "duration": 1.0},
            dataset_name="ds",
        )
        writer.process(task)

        lines = out.read_text().strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["audio_filepath"] == "a.wav"

    def test_returns_audio_task(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonl"
        writer = ManifestWriterStage(output_path=str(out))
        writer.setup_on_node()
        writer.setup()

        task = AudioTask(data={"x": 1}, dataset_name="ds")
        result = writer.process(task)

        assert isinstance(result, AudioTask)
        assert result.data == {"x": 1}
        assert result.dataset_name == "ds"

    def test_propagates_metadata_and_stage_perf(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonl"
        writer = ManifestWriterStage(output_path=str(out))
        writer.setup_on_node()
        writer.setup()

        metadata = {"source_files": ["manifest.jsonl"]}
        stage_perf = [{"stage": "some_stage", "process_time": 0.5}]
        task = AudioTask(
            data={"x": 1},
            dataset_name="ds",
            _metadata=metadata,
            _stage_perf=stage_perf,
        )
        result = writer.process(task)

        assert result._metadata == metadata
        assert result._stage_perf == stage_perf

    def test_appends_across_multiple_process_calls(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonl"
        writer = ManifestWriterStage(output_path=str(out))
        writer.setup_on_node()
        writer.setup()

        writer.process(AudioTask(data={"entry": 1}))
        writer.process(AudioTask(data={"entry": 2}))
        writer.process(AudioTask(data={"entry": 3}))

        lines = out.read_text().strip().split("\n")
        assert len(lines) == 3
        assert [json.loads(line)["entry"] for line in lines] == [1, 2, 3]

    def test_setup_on_node_preserves_existing_file(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonl"
        out.write_text('{"old": "data"}\n')

        writer = ManifestWriterStage(output_path=str(out))
        writer.setup_on_node()

        assert out.read_text() == '{"old": "data"}\n'

    def test_worker_setup_never_truncates_committed_rows(self, tmp_path: Path) -> None:
        """``setup()`` runs per worker actor; a replacement actor must not erase earlier rows."""
        out = tmp_path / "output.jsonl"
        writer = ManifestWriterStage(output_path=str(out))
        writer.prepare_on_driver()
        writer.setup_on_node()
        writer.setup()
        writer.process(AudioTask(data={"audio_filepath": "a.wav"}, dataset_name="ds"))
        writer.process(AudioTask(data={"audio_filepath": "b.wav"}, dataset_name="ds"))

        replacement = pickle.loads(pickle.dumps(writer))  # noqa: S301
        replacement.setup_on_node()
        replacement.setup()
        replacement.process(AudioTask(data={"audio_filepath": "c.wav"}, dataset_name="ds"))

        assert [json.loads(line)["audio_filepath"] for line in out.read_text().splitlines()] == [
            "a.wav",
            "b.wav",
            "c.wav",
        ]

    def test_setup_on_node_creates_parent_directories(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "deep" / "output.jsonl"
        writer = ManifestWriterStage(output_path=str(out))
        writer.setup_on_node()

        assert out.parent.exists()

    def test_handles_unicode_content(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonl"
        writer = ManifestWriterStage(output_path=str(out))
        writer.setup_on_node()
        writer.setup()

        task = AudioTask(data={"text": "日本語テスト", "speaker": "Ñoño"})
        writer.process(task)

        loaded = json.loads(out.read_text().strip())
        assert loaded["text"] == "日本語テスト"
        assert loaded["speaker"] == "Ñoño"

    def test_preserves_nested_structures(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonl"
        writer = ManifestWriterStage(output_path=str(out))
        writer.setup_on_node()
        writer.setup()

        entry = {
            "audio_filepath": "a.wav",
            "windows": [
                {"segments": [{"start": 0.0, "end": 5.0, "speaker": "spk_0"}]},
            ],
            "stats": {"lost_bw": 3, "lost_sr": 0},
        }
        task = AudioTask(data=entry)
        writer.process(task)

        loaded = json.loads(out.read_text().strip())
        assert loaded["windows"][0]["segments"][0]["speaker"] == "spk_0"
        assert loaded["stats"]["lost_bw"] == 3

    def test_num_workers_returns_one(self, tmp_path: Path) -> None:
        writer = ManifestWriterStage(output_path=str(tmp_path / "out.jsonl"))
        assert writer.num_workers() == 1

    def test_xenna_stage_spec(self, tmp_path: Path) -> None:
        writer = ManifestWriterStage(output_path=str(tmp_path / "out.jsonl"))
        assert writer.xenna_stage_spec() == {}


def test_agent_manifest_writer_truncates_on_setup_on_node(tmp_path: Path) -> None:
    """A fresh run (setup_on_node) truncates the output so reruns do not accumulate duplicates."""
    out_path = tmp_path / "manifest.jsonl"
    writer = ManifestWriterStage(output_path=str(out_path))
    task = AudioTask(dataset_name="t", data={"audio_filepath": "src.wav", "text": "row"})

    writer.setup_on_node()
    writer.setup()
    writer.process(task)
    writer.process(task)
    assert len(out_path.read_text(encoding="utf-8").strip().splitlines()) == 2  # appends within a run

    writer.finalize()
    writer.setup_on_node()  # new run truncates
    writer.setup()
    writer.process(task)
    assert len(out_path.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_writer_success_removes_only_its_own_initialization_record(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    writer = ManifestWriterStage(str(path))
    writer.prepare_on_driver()
    writer.setup()
    writer.process(AudioTask(data={"row": 1}))
    owner = Path(f"{path}._RUN")
    assert owner.exists()
    writer.finalize()
    assert not owner.exists()
    assert path.read_text() == '{"row": 1}\n'
    other_run = ManifestWriterStage(str(path))
    other_run.prepare_on_driver()
    reservation = owner.read_bytes()
    writer.abort_on_driver()
    assert owner.read_bytes() == reservation


def test_writer_cleanup_failure_does_not_discard_committed_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "manifest.jsonl"
    writer = ManifestWriterStage(str(path))
    writer.prepare_on_driver()
    writer.setup()
    writer.process(AudioTask(data={"row": 1}))

    def refuse_cleanup(_path: str) -> None:
        message = "sidecar cleanup denied"
        raise PermissionError(message)

    monkeypatch.setattr(writer._fs, "rm", refuse_cleanup)
    with pytest.raises(PermissionError, match="sidecar cleanup denied"):
        writer.finalize()
    assert path.read_text() == '{"row": 1}\n'


def test_agent_manifest_writer_truncates_on_driver_preparation(tmp_path: Path) -> None:
    """A fresh run (setup_on_node) truncates the output so reruns do not accumulate duplicates."""
    out_path = tmp_path / "manifest.jsonl"
    writer = ManifestWriterStage(output_path=str(out_path))
    task = AudioTask(dataset_name="t", data={"audio_filepath": "src.wav", "text": "row"})

    writer.setup_on_node()
    writer.setup()
    writer.process(task)
    writer.process(task)
    assert len(out_path.read_text(encoding="utf-8").strip().splitlines()) == 2  # appends within a run

    writer.prepare_on_driver()  # new run truncates
    writer.setup()
    writer.process(task)
    assert len(out_path.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_resumable_writer_preparation_keeps_completed_source_rows(tmp_path: Path) -> None:
    out = tmp_path / "manifest.jsonl"
    checkpoint = tmp_path / "resume-state"
    driver = ManifestWriterStage(str(out))
    driver.prepare_on_driver(checkpoint_path=checkpoint, pipeline_identity="same-pipeline")
    driver.process(AudioTask(data={"audio_filepath": "b.wav"}))
    _write_resume_state(checkpoint)
    driver.finalize()
    driver = ManifestWriterStage(str(out))
    driver.prepare_on_driver(checkpoint_path=checkpoint, pipeline_identity="same-pipeline")
    worker = pickle.loads(pickle.dumps(driver))  # noqa: S301 - only locally serialized stage objects
    worker.setup_on_node()
    worker.setup()
    worker.process(AudioTask(data={"audio_filepath": "a.wav"}))
    driver.finalize()
    assert [json.loads(row)["audio_filepath"] for row in out.read_text().splitlines()] == ["b.wav", "a.wav"]
    assert not Path(f"{out}._RUN").exists()


def _write_resume_state(checkpoint: Path) -> None:
    import lmdb

    from nemo_curator.utils.resumability_actor import METADATA_DIRNAME

    metadata = checkpoint / METADATA_DIRNAME
    metadata.mkdir(parents=True, exist_ok=True)
    with lmdb.open(str(metadata / "worker.mdb"), subdir=False, max_dbs=1) as env:
        completed = env.open_db(b"completed_sources")
        with env.begin(write=True) as txn:
            txn.put(b"source-b", b"1", db=completed)


@pytest.mark.parametrize(
    "change", ["fresh", "pipeline", "checkpoint", "output", "missing-output", "state", "partial", "binding"]
)
def test_writer_rejects_unproven_resume_without_changing_output(tmp_path: Path, change: str) -> None:
    output = tmp_path / "output.jsonl"
    checkpoint = tmp_path / "state"
    identity = "original"
    if change == "fresh":
        output.write_text('{"stale": true}\n')
    else:
        writer = ManifestWriterStage(str(output))
        writer.prepare_on_driver(checkpoint_path=checkpoint, pipeline_identity=identity)
        writer.process(AudioTask(data={"original": True}))
        _write_resume_state(checkpoint)
        if change == "partial":
            writer.abort_on_driver()
        else:
            writer.finalize()
    if change == "pipeline":
        identity = "different"
    elif change == "checkpoint":
        checkpoint = tmp_path / "different-state"
        _write_resume_state(checkpoint)
    elif change == "output":
        output.write_text('{"replaced": true}\n')
    elif change == "missing-output":
        output.unlink()
    elif change == "state":
        (checkpoint / ".nemo_curator_metadata" / "worker.mdb").unlink()
    elif change == "binding":
        (checkpoint / ".nemo_curator_metadata" / "audio_manifest_pipeline.json").unlink()
    before = output.read_bytes() if output.exists() else None
    writer = ManifestWriterStage(str(output))
    with pytest.raises(RuntimeError, match="cannot authenticate"):
        writer.prepare_on_driver(checkpoint_path=checkpoint, pipeline_identity=identity)
    assert (output.read_bytes() if output.exists() else None) == before
    assert not Path(f"{output}._RUN").exists()


def test_fresh_checkpoint_cannot_be_claimed_by_a_different_pipeline(tmp_path: Path) -> None:
    checkpoint = tmp_path / "state"
    first = ManifestWriterStage(str(tmp_path / "first.jsonl"))
    first.prepare_on_driver(checkpoint_path=checkpoint, pipeline_identity="first")
    second = ManifestWriterStage(str(tmp_path / "second.jsonl"))
    with pytest.raises(RuntimeError, match="different pipeline"):
        second.prepare_on_driver(checkpoint_path=checkpoint, pipeline_identity="second")
    assert not Path(second.output_path).exists()
    assert not Path(f"{second.output_path}._RUN").exists()
    first.abort_on_driver()


def test_checkpoint_writer_requires_pipeline_identity_before_altering_output(tmp_path: Path) -> None:
    output = tmp_path / "output.jsonl"
    writer = ManifestWriterStage(str(output))
    with pytest.raises(ValueError, match="pipeline_identity"):
        writer.prepare_on_driver(checkpoint_path=tmp_path / "state")
    assert not output.exists()
    assert not Path(f"{output}._RUN").exists()


@pytest.mark.parametrize("phase", ["worker_setup", "write", "finalize"])
def test_writer_replaced_output_is_rejected(tmp_path: Path, phase: str) -> None:
    output = tmp_path / "output.jsonl"
    driver = ManifestWriterStage(str(output))
    driver.prepare_on_driver()
    worker = pickle.loads(pickle.dumps(driver))  # noqa: S301 - only locally serialized stages
    worker.setup()
    worker.process(AudioTask(data={"owned": True}))
    output.unlink()
    output.write_text('{"replacement": true}\n')
    actions = {
        "worker_setup": worker.setup_on_node,
        "write": lambda: worker.process(AudioTask(data={"must_not_append": True})),
        "finalize": driver.finalize,
    }
    with pytest.raises(RuntimeError, match="driver-prepared"):
        actions[phase]()
    assert output.read_text() == '{"replacement": true}\n'
    driver.abort_on_driver()


def test_writer_same_size_memory_output_replacement_is_rejected(tmp_path: Path) -> None:
    writer = ManifestWriterStage(f"memory://{tmp_path}/manifest.jsonl")
    writer.prepare_on_driver()
    writer.process(AudioTask(data={"a": 1}))
    writer._fs.rm(writer._path)
    with writer._fs.open(writer._path, "w", encoding="utf-8") as output:
        output.write('{"b": 1}\n')
    with pytest.raises(RuntimeError, match="driver-prepared"):
        writer.finalize()
    assert writer._fs.cat(writer._path) == b'{"b": 1}\n'
    writer.abort_on_driver()
    writer._fs.rm(writer._path)


def test_writer_identity_lookup_failure_releases_owned_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "manifest.jsonl"
    writer = ManifestWriterStage(str(output))

    def metadata_unavailable() -> dict[str, str]:
        msg = "metadata unavailable"
        raise OSError(msg)

    monkeypatch.setattr(writer, "_output_identity", metadata_unavailable)
    with pytest.raises(OSError, match="metadata unavailable"):
        writer.prepare_on_driver()
    assert not Path(f"{output}._RUN").exists()
    assert writer._run_token is None


def test_writer_refuses_size_only_filesystem_and_cleans_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = ManifestWriterStage(f"memory://{tmp_path}/manifest.jsonl")
    writer._resolve_output()
    original_info = writer._fs.info

    def size_only_info(path, **kwargs):  # noqa: ANN001, ANN202 - fsspec info surface
        info = original_info(path, **kwargs)
        return {**info, "created": None, "etag": None, "version_id": None}

    monkeypatch.setattr(writer._fs, "info", size_only_info)
    with pytest.raises(RuntimeError, match="lacks artifact identity"):
        writer.prepare_on_driver()
    assert not writer._fs.exists(f"{writer._path}._RUN")
    assert writer._run_token is None
    writer._fs.rm(writer._path)


def test_value_filter_identity_supports_real_checkpoint_writer(tmp_path: Path) -> None:
    import soundfile as sf

    from nemo_curator.stages.audio.agent import pipeline_identity

    source = tmp_path / "clip.wav"
    sf.write(source, np.zeros(1600, dtype=np.float32), 16000)
    output = tmp_path / "output.jsonl"
    checkpoint = tmp_path / "state"
    duration = GetAudioDurationStage()
    selection = PreserveByValueStage("duration", 0.05, operator="gt")
    writer = ManifestWriterStage(str(output))
    identity = pipeline_identity([duration, selection, writer])
    writer.prepare_on_driver(checkpoint_path=checkpoint, pipeline_identity=identity)
    rows = selection.process_batch([duration.process(AudioTask(data={"audio_filepath": str(source)}))])
    assert len(rows) == 1
    writer.process(rows[0])
    _write_resume_state(checkpoint)
    writer.finalize()
    assert json.loads(output.read_text())["duration"] == pytest.approx(0.1)
    resumed = ManifestWriterStage(str(output))
    resumed.prepare_on_driver(checkpoint_path=checkpoint, pipeline_identity=identity)
    resumed.finalize()
    assert len(output.read_text().splitlines()) == 1


def test_prepared_manifest_batch_rejects_replaced_output(tmp_path: Path) -> None:
    output = tmp_path / "manifest.jsonl"
    writer = ManifestWriterStage(str(output))
    writer.prepare_on_driver()
    writer.setup()
    writer.write_jsonl_batch('{"id":1}\n{"id":2}\n')
    output.unlink()
    output.write_text('{"id":"other-owner"}\n')
    with pytest.raises(RuntimeError, match="driver-prepared"):
        writer.write_jsonl_batch('{"id":3}\n')
    assert output.read_text() == '{"id":"other-owner"}\n'
    writer.abort_on_driver()
