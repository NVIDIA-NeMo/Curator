# modality: audio
#
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

from nemo_curator.backends.base import BaseExecutor
from nemo_curator.checkpointing.audio import AudioCheckpointRunner
from nemo_curator.checkpointing.audio.serialization import (
    deserialize_audio_task,
    dump_audio_task_manifest,
    load_audio_task_manifest,
    serialize_audio_task,
)
from nemo_curator.checkpointing.audio.store import StageCheckpointStore, fingerprint_stage
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, EmptyTask


class LocalExecutor(BaseExecutor):
    def execute(
        self,
        stages: list[ProcessingStage],
        initial_tasks: list[AudioTask] | None = None,
    ) -> list[AudioTask]:
        tasks = initial_tasks or [EmptyTask]
        for stage in stages:
            tasks = stage.process_batch(tasks)
        return tasks


@dataclass
class ReaderStage(ProcessingStage):
    emitted_tasks: list[AudioTask]
    name: str = "reader_stage"

    def process(self, _task: object) -> list[AudioTask]:
        return [
            AudioTask(
                task_id=task.task_id,
                dataset_name=task.dataset_name,
                data=dict(task.data),
                sample_key=task.sample_key,
            )
            for task in self.emitted_tasks
        ]


@dataclass
class CountingStage(ProcessingStage[AudioTask, AudioTask]):
    calls: int = 0
    field_name: str = "counted"
    name: str = "counting_stage"

    def process(self, task: AudioTask) -> AudioTask:
        self.calls += 1
        task.data[self.field_name] = True
        return task


@dataclass
class FailingStage(ProcessingStage[AudioTask, AudioTask]):
    bad_key: str = "bad"
    name: str = "failing_stage"

    def process(self, task: AudioTask) -> AudioTask:
        if task.sample_key == self.bad_key:
            msg = f"boom for {task.sample_key}"
            raise RuntimeError(msg)
        task.data["processed"] = True
        return task


@dataclass
class RetryTempArtifactStage(ProcessingStage[AudioTask, AudioTask]):
    artifact_dir: str
    name: str = "retry_temp_artifact_stage"

    def process(self, task: AudioTask) -> AudioTask:
        return task

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) > 1:
            for index, task in enumerate(tasks):
                temp_path = Path(self.artifact_dir) / f"{task.task_id}_{index}.tmp"
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path.write_text("artifact")
                task.data["_temporary_audio_path"] = str(temp_path)
                task.data["audio_filepath"] = str(temp_path)
            msg = "batch failure after temp materialization"
            raise RuntimeError(msg)

        task = tasks[0]
        if not task.data["audio_filepath"].startswith("manifest::"):
            msg = f"expected pristine manifest path, got {task.data['audio_filepath']}"
            raise RuntimeError(msg)
        task.data["processed"] = True
        return tasks


@dataclass
class SetConfigStage(ProcessingStage[AudioTask, AudioTask]):
    values: set[str]
    name: str = "set_config_stage"

    def process(self, task: AudioTask) -> AudioTask:
        return task

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config["values"] = self.values
        return config


def _make_audio_task(sample_key: str, checkpoint_shard_id: str = "shard_0") -> AudioTask:
    return AudioTask(
        task_id=sample_key,
        dataset_name="dataset",
        data={"audio_filepath": f"{sample_key}.wav"},
        sample_key=sample_key,
        _metadata={"checkpoint_shard_id": checkpoint_shard_id},
    )


def test_serialization_manifest_roundtrip(tmp_path: Path) -> None:
    tasks = [_make_audio_task("sample-a"), _make_audio_task("sample-b")]
    manifest_path = tmp_path / "manifest.jsonl"

    dump_audio_task_manifest(tasks, manifest_path)
    restored = load_audio_task_manifest(manifest_path)

    assert [task.sample_key for task in restored] == ["sample-a", "sample-b"]
    assert restored[0].data["audio_filepath"] == "sample-a.wav"
    assert deserialize_audio_task(serialize_audio_task(tasks[0])).sample_key == "sample-a"


def test_stage_fingerprint_is_stable_for_sets() -> None:
    first = SetConfigStage(values={"alpha", "beta"})
    second = SetConfigStage(values={"beta", "alpha"})

    assert fingerprint_stage(first) == fingerprint_stage(second)


def test_runner_normalizes_pipeline_config_before_writing_metadata(tmp_path: Path) -> None:
    pipeline = Pipeline(
        name="checkpoint_test",
        stages=[],
        config={"link_stages_via_io": True, "values": {"beta", "alpha"}},
    )
    runner = AudioCheckpointRunner(
        pipeline=pipeline,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        executor=LocalExecutor(),
    )

    runner._write_pipeline_metadata([])

    payload = json.loads((tmp_path / "checkpoints" / "pipeline.json").read_text())
    assert payload["config"]["values"] == ["alpha", "beta"]

def test_runner_skips_completed_stages_on_rerun(tmp_path: Path) -> None:
    reader = ReaderStage(emitted_tasks=[_make_audio_task("sample-a"), _make_audio_task("sample-b")])
    counting = CountingStage()

    pipeline = Pipeline(name="checkpoint_test", stages=[reader, counting])
    runner = AudioCheckpointRunner(
        pipeline=pipeline,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        executor=LocalExecutor(),
    )

    first = runner.run()
    second = runner.run()

    assert first is not None
    assert second is not None
    assert [task.data["counted"] for task in first] == [True, True]
    assert [task.data["counted"] for task in second] == [True, True]
    assert counting.calls == 2


def test_runner_records_failed_samples_when_ignore_failed_enabled(tmp_path: Path) -> None:
    reader = ReaderStage(emitted_tasks=[_make_audio_task("good"), _make_audio_task("bad")])
    failing = FailingStage()

    pipeline = Pipeline(name="checkpoint_failures", stages=[reader, failing])
    runner = AudioCheckpointRunner(
        pipeline=pipeline,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        executor=LocalExecutor(),
        ignore_failed=True,
    )

    results = runner.run()

    assert results is not None
    assert [task.sample_key for task in results] == ["good"]
    stage_json = json.loads((tmp_path / "checkpoints" / "01_failing_stage" / "stage.json").read_text())
    assert stage_json["failed_count"] == 1

    records_path = tmp_path / "checkpoints" / "01_failing_stage" / "records" / "shard_0.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    statuses = {record["sample_key"]: record["status"] for record in records}

    assert statuses == {
        "good": "done",
        "bad": "failed_retriable",
    }


def test_runner_retries_from_clean_task_snapshots_and_cleans_temp_artifacts(tmp_path: Path) -> None:
    reader = ReaderStage(
        emitted_tasks=[
            AudioTask(task_id="a", dataset_name="dataset", data={"audio_filepath": "manifest::a"}, sample_key="a"),
            AudioTask(task_id="b", dataset_name="dataset", data={"audio_filepath": "manifest::b"}, sample_key="b"),
        ]
    )
    retry_stage = RetryTempArtifactStage(artifact_dir=str(tmp_path / "artifacts"))

    pipeline = Pipeline(name="checkpoint_retry_cleanup", stages=[reader, retry_stage])
    runner = AudioCheckpointRunner(
        pipeline=pipeline,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        executor=LocalExecutor(),
        ignore_failed=True,
    )

    results = runner.run()

    assert results is not None
    assert [task.sample_key for task in results] == ["a", "b"]
    assert [task.data["processed"] for task in results] == [True, True]
    assert list((tmp_path / "artifacts").glob("*")) == []


def test_store_ensures_sample_keys_before_filtered_accounting(tmp_path: Path) -> None:
    task = AudioTask(task_id="task-a", dataset_name="dataset", data={"audio_filepath": "a.wav"})
    store = StageCheckpointStore(
        checkpoint_dir=tmp_path / "checkpoints",
        stage_index=0,
        stage_name="reader_stage",
        config_fingerprint="fingerprint",
    )

    store.write_stage_result(input_tasks=[task], output_tasks=[])

    records_path = store.records_dir / "partition_unknown.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    assert len(records) == 1
    assert records[0]["status"] == "filtered"
    assert records[0]["sample_key"]


def test_store_writes_one_file_per_checkpoint_shard(tmp_path: Path) -> None:
    store = StageCheckpointStore(
        checkpoint_dir=tmp_path / "checkpoints",
        stage_index=0,
        stage_name="reader_stage",
        config_fingerprint="fingerprint",
    )
    tasks = [
        _make_audio_task("sample-a", checkpoint_shard_id="shard_0"),
        _make_audio_task("sample-b", checkpoint_shard_id="shard_1"),
    ]

    store.write_stage_result(input_tasks=tasks, output_tasks=tasks)

    output_files = sorted(path.name for path in store.outputs_dir.glob("*.jsonl"))
    record_files = sorted(path.name for path in store.records_dir.glob("*.jsonl"))

    assert output_files == ["shard_0.jsonl", "shard_1.jsonl"]
    assert record_files == ["shard_0.jsonl", "shard_1.jsonl"]


def test_store_keeps_existing_shard_files_when_temp_write_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import nemo_curator.checkpointing.audio.store as store_module

    store = StageCheckpointStore(
        checkpoint_dir=tmp_path / "checkpoints",
        stage_index=0,
        stage_name="reader_stage",
        config_fingerprint="fingerprint",
    )
    initial_tasks = [_make_audio_task("sample-a", checkpoint_shard_id="shard_0")]
    store.write_stage_result(input_tasks=initial_tasks, output_tasks=initial_tasks)

    original_output = (store.outputs_dir / "shard_0.jsonl").read_text()
    original_records = (store.records_dir / "shard_0.jsonl").read_text()

    def fail_write_jsonl_atomic(_path: Path, _payloads: list[dict[str, object]]) -> None:
        msg = "simulated records write failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(store_module, "write_jsonl_atomic", fail_write_jsonl_atomic)

    with pytest.raises(RuntimeError, match="simulated records write failure"):
        store.write_stage_result(input_tasks=initial_tasks, output_tasks=initial_tasks)

    assert (store.outputs_dir / "shard_0.jsonl").read_text() == original_output
    assert (store.records_dir / "shard_0.jsonl").read_text() == original_records
