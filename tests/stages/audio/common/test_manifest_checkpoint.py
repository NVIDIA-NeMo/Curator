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

import pytest

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.common import (
    CreateInitialManifestAudioFolderStage,
    ManifestCheckpointStage,
    ManifestWriterStage,
)
from nemo_curator.tasks import AudioTask


@pytest.mark.parametrize("rows", [[], [{"row": 1}, {"row": 2}]])
def test_driver_prepared_checkpoint_publishes_exact_shared_rows(tmp_path: Path, rows: list[dict]) -> None:
    path = tmp_path / "checkpoint.jsonl"
    driver = ManifestCheckpointStage(str(path))
    driver.prepare_on_driver()
    original_owner = Path(f"{path}._RETRY_OWNER").read_bytes()
    with pytest.raises(RuntimeError, match="reset_for_retry"):
        driver.prepare_on_driver()
    assert Path(f"{path}._RETRY_OWNER").read_bytes() == original_owner
    worker = pickle.loads(pickle.dumps(driver))  # noqa: S301
    worker.setup_on_node()
    worker.setup(WorkerMetadata())
    for row in rows:
        worker.process(AudioTask(data=row))
    driver.finalize()
    assert [json.loads(line) for line in path.read_text().splitlines()] == rows
    assert Path(f"{path}._COMPLETE").exists()


def test_prepared_checkpoint_worker_refuses_a_replaced_artifact(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.jsonl"
    driver = ManifestCheckpointStage(str(path))
    driver.prepare_on_driver()
    worker = pickle.loads(pickle.dumps(driver))  # noqa: S301
    path.write_text('{"replacement": true}\n')
    with pytest.raises(RuntimeError, match="cannot verify"):
        worker.setup(WorkerMetadata())
    assert path.read_text() == '{"replacement": true}\n'
    assert not Path(f"{path}._COMPLETE").exists()


def test_checkpoint_actor_replacement_adopts_existing_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_curator.backends.base import NodeInfo
    from nemo_curator.backends.ray_data import adapter

    monkeypatch.setattr(adapter, "get_worker_metadata_and_node_id", lambda: (NodeInfo(), WorkerMetadata()))
    path = tmp_path / "checkpoint.jsonl"
    driver = ManifestCheckpointStage(str(path))
    driver.prepare_on_driver()
    for row in (1, 2):
        actor = adapter.create_actor_from_stage(pickle.loads(pickle.dumps(driver)))()  # noqa: S301
        actor.stage.process(AudioTask(data={"row": row}))
    driver.finalize()
    assert [json.loads(line) for line in path.read_text().splitlines()] == [{"row": 1}, {"row": 2}]


def test_standalone_checkpoint_worker_reserves_before_writing(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.jsonl"
    checkpoint = ManifestCheckpointStage(str(path))
    checkpoint.setup(WorkerMetadata())
    checkpoint.process(AudioTask(data={"row": 1}))
    checkpoint.finalize()
    assert path.read_text() == '{"row": 1}\n'
    assert Path(f"{path}._COMPLETE").exists()
    assert not Path(f"{path}._RETRY_OWNER").exists()


class TestManifestCheckpointStage:
    """Focused unit tests for the reusable metadata checkpoint."""

    def test_setup_atomically_refuses_to_overwrite_existing_checkpoint(self, tmp_path: Path) -> None:
        out = tmp_path / "checkpoint.jsonl"
        out.write_bytes(b"retained artifact\n")
        checkpoint = ManifestCheckpointStage(output_path=str(out))

        with pytest.raises(FileExistsError, match="refuses to overwrite"):
            checkpoint.setup()

        assert out.read_bytes() == b"retained artifact\n"
        assert not Path(f"{out}._RETRY_OWNER").exists()

    def test_setup_refuses_stale_completion_marker_without_leaving_output(self, tmp_path: Path) -> None:
        out = tmp_path / "checkpoint.jsonl"
        Path(f"{out}._COMPLETE").write_text("stale", encoding="utf-8")
        checkpoint = ManifestCheckpointStage(output_path=str(out))

        with pytest.raises(FileExistsError, match="completion marker"):
            checkpoint.setup()

        assert not out.exists()

    def test_retry_reset_removes_only_owned_partial_and_reserves_cleanly(
        self,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "checkpoint.jsonl"
        checkpoint = ManifestCheckpointStage(output_path=str(out))
        checkpoint.setup()
        checkpoint.process(AudioTask(data={"attempt": 1}))

        checkpoint.reset_for_retry()

        assert not out.exists()
        assert checkpoint._checkpoint_rows_written == 0
        assert checkpoint._checkpoint_bytes_written == 0
        checkpoint.setup()
        checkpoint.process(AudioTask(data={"attempt": 2}))
        assert out.read_text(encoding="utf-8") == '{"attempt": 2}\n'

    def test_retry_reset_refuses_completed_checkpoint(self, tmp_path: Path) -> None:
        out = tmp_path / "checkpoint.jsonl"
        checkpoint = ManifestCheckpointStage(output_path=str(out))
        checkpoint.setup()
        checkpoint.process(AudioTask(data={"retained": True}))
        before = out.read_bytes()
        Path(f"{out}._COMPLETE").write_text("complete", encoding="utf-8")

        with pytest.raises(FileExistsError, match="completion marker"):
            checkpoint.reset_for_retry()

        assert out.read_bytes() == before

    def test_successful_finalize_publishes_completion_before_removing_retry_owner(self, tmp_path: Path) -> None:
        out = tmp_path / "checkpoint.jsonl"
        owner_path = Path(f"{out}._RETRY_OWNER")
        marker_path = Path(f"{out}._COMPLETE")
        checkpoint = ManifestCheckpointStage(output_path=str(out))
        checkpoint.setup()
        checkpoint.process(AudioTask(data={"retained": True}))

        checkpoint.finalize()

        assert out.read_text(encoding="utf-8") == '{"retained": true}\n'
        assert marker_path.exists()
        assert not owner_path.exists()
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        assert marker["st_size"] == out.stat().st_size
        with pytest.raises(FileExistsError, match="completion marker"):
            checkpoint.reset_for_retry()

    def test_successful_finalize_publishes_an_empty_checkpoint(self, tmp_path: Path) -> None:
        out = tmp_path / "checkpoint.jsonl"
        checkpoint = ManifestCheckpointStage(output_path=str(out))

        checkpoint.setup()
        checkpoint.finalize()

        assert out.read_bytes() == b""
        assert Path(f"{out}._COMPLETE").exists()
        assert not Path(f"{out}._RETRY_OWNER").exists()

    def test_driver_finalize_and_agent_release_are_idempotent(self, tmp_path: Path) -> None:
        path = tmp_path / "checkpoint.jsonl"
        driver = ManifestCheckpointStage(str(path))
        worker = pickle.loads(pickle.dumps(driver))  # noqa: S301
        worker.setup()
        worker.process(AudioTask(data={"row": 1}))
        driver.finalize()
        marker_before = Path(f"{path}._COMPLETE").read_bytes()
        driver.release_retry_reservation()
        assert Path(f"{path}._COMPLETE").read_bytes() == marker_before
        assert path.read_text() == '{"row": 1}\n'

    def test_missing_driver_artifact_is_not_published_as_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import fsspec

        from nemo_curator.stages.audio import common

        namespace = {"name": "worker"}
        filesystem = fsspec.filesystem("file")
        monkeypatch.setattr(
            common,
            "url_to_fs",
            lambda _path: (filesystem, str(tmp_path / namespace["name"] / "checkpoint.jsonl")),
        )
        checkpoint = ManifestCheckpointStage("/logical/checkpoint.jsonl")
        worker = pickle.loads(pickle.dumps(checkpoint))  # noqa: S301
        worker.setup()
        worker.process(AudioTask(data={"row": 1}))
        namespace["name"] = "driver"
        with pytest.raises(RuntimeError, match="cannot verify"):
            checkpoint.finalize()
        assert (tmp_path / "worker" / "checkpoint.jsonl").read_text() == '{"row": 1}\n'
        assert not (tmp_path / "driver" / "checkpoint.jsonl").exists()
        assert not (tmp_path / "driver" / "checkpoint.jsonl._COMPLETE").exists()

    def test_successful_release_refuses_to_complete_a_replaced_checkpoint(self, tmp_path: Path) -> None:
        out = tmp_path / "checkpoint.jsonl"
        checkpoint = ManifestCheckpointStage(output_path=str(out))
        checkpoint.setup()
        checkpoint.process(AudioTask(data={"attempt": 1}))
        out.unlink()
        out.write_text("replacement\n", encoding="utf-8")

        with pytest.raises(RuntimeError, match="cannot verify"):
            checkpoint.release_retry_reservation()

        assert out.read_text(encoding="utf-8") == "replacement\n"
        assert not Path(f"{out}._COMPLETE").exists()
        assert Path(f"{out}._RETRY_OWNER").exists()

    def test_retry_reset_refuses_preexisting_unowned_checkpoint(
        self,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "checkpoint.jsonl"
        out.write_text("user file\n", encoding="utf-8")
        checkpoint = ManifestCheckpointStage(output_path=str(out))

        with pytest.raises(FileExistsError, match="did not reserve"):
            checkpoint.reset_for_retry()

        assert out.read_text(encoding="utf-8") == "user file\n"

    def test_retry_reset_refuses_replaced_reservation(self, tmp_path: Path) -> None:
        out = tmp_path / "checkpoint.jsonl"
        checkpoint = ManifestCheckpointStage(output_path=str(out))
        checkpoint.setup()
        out.unlink()
        out.write_text("replacement\n", encoding="utf-8")

        with pytest.raises(FileExistsError, match="no longer its exact reservation"):
            checkpoint.reset_for_retry()

        assert out.read_text(encoding="utf-8") == "replacement\n"

    def test_configured_contract_is_audio_pass_through_with_checkpoint_gates(self, tmp_path: Path) -> None:
        from nemo_curator.stages.audio._agent._agent_registry import build_contract

        checkpoint = ManifestCheckpointStage(output_path=str(tmp_path / "checkpoint.jsonl"))
        contract = build_contract(checkpoint)
        params = {parameter.name: parameter for parameter in contract.params}

        assert checkpoint.name == "manifest_checkpoint"
        assert checkpoint.name != ManifestWriterStage(output_path=str(tmp_path / "manifest.jsonl")).name
        assert checkpoint.num_workers() == 1
        assert contract.accepts_task_type == "AudioTask"
        assert contract.produces_task_type == "AudioTask"
        assert contract.gates.writes_to_disk is True
        assert contract.gates.output_path_params == ["output_path"]
        assert contract.gates.requires_serializable_input is True
        assert contract.gates.per_row_independent is True
        assert contract.gates.lifecycle_side_effects is True
        assert params["output_path"].required is True
        assert "max_bytes" not in params
        assert params["retention_sec"].default == 0
        assert params["owner"].choices == ["user", "project"]
        assert params["planning_provenance"].default is None

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"retention_sec": -1}, "retention_sec"),
            ({"owner": "nobody"}, "owner"),
            ({"output_path": "s3://bucket/checkpoint.jsonl"}, "plain local path"),
            ({"output_path": "file:///tmp/checkpoint.jsonl"}, "plain local path"),
        ],
    )
    def test_rejects_invalid_checkpoint_policy(
        self,
        tmp_path: Path,
        kwargs: dict[str, object],
        message: str,
    ) -> None:
        params = {"output_path": str(tmp_path / "checkpoint.jsonl"), **kwargs}
        with pytest.raises(ValueError, match=message):
            ManifestCheckpointStage(**params)

    def test_driver_retry_accepts_worker_device_id_for_same_shared_file(self, tmp_path: Path) -> None:
        out = tmp_path / "checkpoint.jsonl"
        driver = ManifestCheckpointStage(output_path=str(out))
        driver.prepare_on_driver()
        worker = pickle.loads(pickle.dumps(driver))  # noqa: S301
        worker.setup()
        worker.process(AudioTask(data={"attempt": 1}))
        owner_path = Path(f"{out}._RETRY_OWNER")
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
        owner["st_dev"] += 1
        owner_path.write_text(json.dumps(owner), encoding="utf-8")

        driver.reset_for_retry()

        assert not out.exists()
        assert not owner_path.exists()
        driver.prepare_on_driver()
        retry_worker = pickle.loads(pickle.dumps(driver))  # noqa: S301
        retry_worker.setup()
        retry_worker.process(AudioTask(data={"attempt": 2}))
        driver.finalize()
        assert out.read_text(encoding="utf-8") == '{"attempt": 2}\n'
        assert Path(f"{out}._COMPLETE").exists()


@pytest.mark.parametrize("with_rows", [False, True])
def test_checkpoint_driver_lifecycle_with_serialized_pipeline(tmp_path: Path, with_rows: bool) -> None:
    from nemo_curator.tasks import EmptyTask

    class SerializedExecutor:
        def execute(self, stages, initial_tasks):  # noqa: ANN001, ANN202
            current = initial_tasks or [EmptyTask()]
            for stage in stages:
                if not current:
                    break
                worker = pickle.loads(pickle.dumps(stage))  # noqa: S301 - only locally serialized stages
                worker.setup_on_node()
                worker.setup()
                current = worker.process_batch(current)
            return current

    source = tmp_path / "source"
    source.mkdir()
    if with_rows:
        (source / "a.wav").touch()
    output = tmp_path / "checkpoint.jsonl"
    checkpoint = ManifestCheckpointStage(str(output))
    checkpoint.prepare_on_driver()
    pipeline = Pipeline(
        name="checkpoint-lifecycle", stages=[CreateInitialManifestAudioFolderStage(str(source)), checkpoint]
    )
    result = pipeline.run(SerializedExecutor())
    assert len(result) == int(with_rows)
    assert len(output.read_text().splitlines()) == int(with_rows)
    assert not Path(f"{output}._COMPLETE").exists()
    checkpoint.finalize()
    assert Path(f"{output}._COMPLETE").exists()
    assert not Path(f"{output}._RETRY_OWNER").exists()


@pytest.mark.parametrize("with_rows", [False, True])
def test_checkpoint_driver_reservation_is_adopted_and_completed(tmp_path: Path, with_rows: bool) -> None:
    out = tmp_path / "checkpoint.jsonl"
    driver = ManifestCheckpointStage(str(out))
    driver.prepare_on_driver()
    worker = pickle.loads(pickle.dumps(driver))  # noqa: S301 - only locally serialized stage objects
    worker.setup_on_node()
    worker.setup()
    if with_rows:
        worker.process(AudioTask(data={"row": 1}))
    replacement = pickle.loads(pickle.dumps(driver))  # noqa: S301 - only locally serialized stage objects
    replacement.setup_on_node()
    replacement.setup()
    assert out.read_text() == ('{"row": 1}\n' if with_rows else "")
    driver.finalize()
    assert json.loads(Path(f"{out}._COMPLETE").read_text())["token"] == driver._reservation_token
    assert not Path(f"{out}._RETRY_OWNER").exists()
    driver.release_retry_reservation()  # idempotent audio-agent success hook


def test_checkpoint_missing_driver_state_cannot_publish_completion(tmp_path: Path) -> None:
    checkpoint = ManifestCheckpointStage(str(tmp_path / "unshared" / "checkpoint.jsonl"))
    with pytest.raises(RuntimeError, match="cannot verify"):
        checkpoint.finalize()
    assert not Path(checkpoint.output_path).exists()
    assert not Path(f"{checkpoint.output_path}._COMPLETE").exists()


def test_checkpoint_replaced_artifact_is_not_marked_complete(tmp_path: Path) -> None:
    out = tmp_path / "checkpoint.jsonl"
    checkpoint = ManifestCheckpointStage(str(out))
    checkpoint.prepare_on_driver()
    out.unlink()
    out.write_text("unowned replacement")
    with pytest.raises(RuntimeError, match="cannot verify"):
        checkpoint.finalize()
    assert not Path(f"{out}._COMPLETE").exists()
    assert out.read_text() == "unowned replacement"


def test_checkpoint_worker_in_another_namespace_fails_without_reserving(tmp_path: Path) -> None:
    driver = ManifestCheckpointStage(str(tmp_path / "driver" / "checkpoint.jsonl"))
    driver.prepare_on_driver()
    worker = pickle.loads(pickle.dumps(driver))  # noqa: S301 - only locally serialized stage objects
    worker.output_path = str(tmp_path / "worker" / "checkpoint.jsonl")
    with pytest.raises(RuntimeError, match="cannot verify"):
        worker.setup_on_node()
    assert not Path(worker.output_path).exists()
    assert not Path(f"{worker.output_path}._COMPLETE").exists()


@pytest.mark.parametrize("prepared", [False, True])
def test_checkpoint_rejects_replaced_artifact_before_append(tmp_path: Path, prepared: bool) -> None:
    from nemo_curator.stages.audio.common import ManifestCheckpointStage

    out = tmp_path / "checkpoint.jsonl"
    stage = ManifestCheckpointStage(str(out))
    if prepared:
        stage.prepare_on_driver()
    stage.setup()
    replacement = tmp_path / "replacement.jsonl"
    unrelated = '{"unrelated": true}\n'
    replacement.write_text(unrelated)
    replacement.replace(out)
    with pytest.raises(RuntimeError):
        stage.process(AudioTask(data={"audio_filepath": "clip.wav"}))
    assert out.read_text() == unrelated
    assert not Path(f"{out}._COMPLETE").exists()


def test_standalone_checkpoint_completes_multiple_appends(tmp_path: Path) -> None:
    from nemo_curator.stages.audio.common import ManifestCheckpointStage

    out = tmp_path / "checkpoint.jsonl"
    stage = ManifestCheckpointStage(str(out))
    stage.setup()
    for name in ("one.wav", "two.wav"):
        stage.process(AudioTask(data={"audio_filepath": name}))
    stage.finalize()
    assert len(out.read_text().splitlines()) == 2
    assert Path(f"{out}._COMPLETE").exists()
