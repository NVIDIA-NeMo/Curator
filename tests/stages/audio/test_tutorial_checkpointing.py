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

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from nemo_curator.tasks import AudioTask, EmptyTask

_REPO_ROOT = Path(__file__).parents[3]


def _load_tutorial(name: str, path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("name", "path"),
    [
        ("single_speaker_filter_tutorial", "tutorials/audio/single_speaker_filter/run.py"),
        ("callhome_diar_tutorial", "tutorials/audio/callhome_diar/run.py"),
    ],
)
def test_incomplete_reader_checkpoint_is_rebuilt(
    name: str, path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tutorial = _load_tutorial(name, path)
    tasks = [AudioTask(data={"audio_filepath": "same.wav", "row": row}) for row in (1, 2)]
    for index, task in enumerate(tasks):
        task._set_task_id("manifest", index)
    assert tutorial._task_hash(tasks[0]) != tutorial._task_hash(tasks[1])

    stage = type("Reader", (), {"_name": "reader"})()
    stage_dir = tutorial._stage_ckpt_dir(tmp_path, 0, stage._name)
    tutorial._save_task(stage_dir, tutorial._task_hash(tasks[0]), tasks[0])
    calls = 0

    class FakePipeline:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self, **_kwargs: object) -> list[AudioTask]:
            nonlocal calls
            calls += 1
            return tasks

    monkeypatch.setattr(tutorial, "Pipeline", FakePipeline)
    monkeypatch.setattr(tutorial, "XennaExecutor", lambda: object())

    assert len(tutorial._run_stages_with_checkpoints([stage], tmp_path)) == 2
    assert tutorial._stage_complete(stage_dir)
    assert len(list(stage_dir.glob("*.json"))) == 2
    resumed = tutorial._run_stages_with_checkpoints([stage], tmp_path)
    assert [task.task_id for task in resumed] == [task.task_id for task in tasks]
    assert calls == 1


def test_manifest_writer_always_rebuilds_from_all_cached_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tutorial = _load_tutorial(
        "single_speaker_filter_writer_tutorial",
        "tutorials/audio/single_speaker_filter/run.py",
    )
    tasks = [AudioTask(data={"audio_filepath": "same.wav", "row": row}) for row in (1, 2)]
    reader = type("Reader", (), {"_name": "reader"})()
    reader_dir = tutorial._stage_ckpt_dir(tmp_path, 0, reader._name)
    for index, task in enumerate(tasks):
        task._set_task_id("manifest", index)
        task._metadata[tutorial.CKPT_HASH_KEY] = tutorial._task_hash(task)
        task._metadata[tutorial.CKPT_ORDER_KEY] = index
        tutorial._save_task(reader_dir, task._metadata[tutorial.CKPT_HASH_KEY], task)
    tutorial._mark_stage_complete(reader_dir)

    output = tmp_path / "filtered.jsonl"
    output.write_text('{"row": 1}\n')
    writer = tutorial.ManifestWriterStage(str(output))
    writer_dir = tutorial._stage_ckpt_dir(tmp_path, 1, writer._name)
    tutorial._save_task(writer_dir, tasks[0]._metadata[tutorial.CKPT_HASH_KEY], tasks[0])
    writer_calls: list[int] = []

    class FakePipeline:
        def __init__(self, *, stages: list[object], **_kwargs: object) -> None:
            self.stage = stages[0]

        def run(self, *, initial_tasks: list[AudioTask] | None = None, **_kwargs: object) -> list[AudioTask]:
            assert initial_tasks is not None
            writer_calls.append(len(initial_tasks))
            self.stage.setup()
            result = [self.stage.process(task) for task in initial_tasks]
            self.stage.finalize()
            return result

    monkeypatch.setattr(tutorial, "Pipeline", FakePipeline)
    monkeypatch.setattr(tutorial, "XennaExecutor", lambda: object())

    tutorial._run_stages_with_checkpoints([reader, writer], tmp_path)
    assert [json.loads(row)["row"] for row in output.read_text().splitlines()] == [1, 2]
    output.write_text('{"corrupt": true}\n')
    tutorial._run_stages_with_checkpoints([reader, writer], tmp_path)
    assert [json.loads(row)["row"] for row in output.read_text().splitlines()] == [1, 2]
    assert writer_calls == [2, 2]


@pytest.mark.parametrize(
    ("name", "path"),
    [
        ("single_speaker_filter_prune_tutorial", "tutorials/audio/single_speaker_filter/run.py"),
        ("callhome_diar_prune_tutorial", "tutorials/audio/callhome_diar/run.py"),
    ],
)
def test_completed_stage_prunes_unmatched_checkpoints(
    name: str, path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tutorial = _load_tutorial(name, path)
    tasks = [AudioTask(data={"row": row}) for row in (1, 2)]
    reader = type("Reader", (), {"_name": "reader"})()
    reader_dir = tutorial._stage_ckpt_dir(tmp_path, 0, reader._name)
    for order, task in enumerate(tasks):
        task._set_task_id("manifest", order)
        task._metadata[tutorial.CKPT_HASH_KEY] = tutorial._task_hash(task)
        task._metadata[tutorial.CKPT_ORDER_KEY] = order
        tutorial._save_task(reader_dir, task._metadata[tutorial.CKPT_HASH_KEY], task)
    tutorial._mark_stage_complete(reader_dir)

    stage = type("Passthrough", (), {"_name": "passthrough"})()
    stage_dir = tutorial._stage_ckpt_dir(tmp_path, 1, stage._name)
    tutorial._save_task(stage_dir, tasks[0]._metadata[tutorial.CKPT_HASH_KEY], tasks[0])
    stale = AudioTask(data={"row": 999})
    stale._set_task_id("manifest", 999)
    tutorial._save_task(stage_dir, tutorial._task_hash(stale), stale)

    class FakePipeline:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(self, *, initial_tasks: list[AudioTask], **_kwargs: object) -> list[AudioTask]:
            return initial_tasks

    monkeypatch.setattr(tutorial, "Pipeline", FakePipeline)
    monkeypatch.setattr(tutorial, "XennaExecutor", lambda: object())

    result = tutorial._run_stages_with_checkpoints([reader, stage], tmp_path)
    assert [task.data["row"] for task in result] == [1, 2]
    assert len(list(stage_dir.glob("*.json"))) == 2
    assert len(tutorial._run_stages_with_checkpoints([reader, stage], tmp_path)) == 2


def test_manifest_writer_truncates_stale_output_for_empty_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tutorial = _load_tutorial(
        "single_speaker_filter_empty_writer_tutorial",
        "tutorials/audio/single_speaker_filter/run.py",
    )
    reader = type("Reader", (), {"_name": "reader"})()
    tutorial._mark_stage_complete(tutorial._stage_ckpt_dir(tmp_path, 0, reader._name))
    output = tmp_path / "filtered.jsonl"
    output.write_text('{"stale": true}\n')

    monkeypatch.setattr(tutorial, "XennaExecutor", lambda: object())
    tutorial._run_stages_with_checkpoints([reader, tutorial.ManifestWriterStage(str(output))], tmp_path)

    assert output.read_text() == ""


def test_callhome_reader_does_not_treat_rttm_as_checkpoint(tmp_path: Path) -> None:
    tutorial = _load_tutorial("callhome_diar_rttm_tutorial", "tutorials/audio/callhome_diar/run.py")
    cha_dir = tmp_path / "eng"
    rttm_dir = tmp_path / "rttm"
    cha_dir.mkdir()
    rttm_dir.mkdir()
    (tmp_path / "0001.wav").touch()
    (cha_dir / "0001.cha").touch()
    (rttm_dir / "0001.rttm").touch()

    tasks = tutorial.CallHomeReaderStage(data_dir=str(tmp_path), cha_dir=str(cha_dir)).process(EmptyTask())

    assert [task.data["session_name"] for task in tasks] == ["0001"]


@pytest.mark.parametrize(
    ("name", "path"),
    [
        ("single_speaker_filter_identity_tutorial", "tutorials/audio/single_speaker_filter/run.py"),
        ("callhome_diar_identity_tutorial", "tutorials/audio/callhome_diar/run.py"),
    ],
)
def test_checkpoint_run_rejects_changed_identity(name: str, path: str, tmp_path: Path) -> None:
    tutorial = _load_tutorial(name, path)
    tutorial._bind_checkpoint_run(tmp_path, {"input": "first"})
    tutorial._bind_checkpoint_run(tmp_path, {"input": "first"})

    with pytest.raises(RuntimeError, match="--clean"):
        tutorial._bind_checkpoint_run(tmp_path, {"input": "changed"})


@pytest.mark.parametrize(
    ("name", "path"),
    [
        ("single_speaker_filter_clean_tutorial", "tutorials/audio/single_speaker_filter/run.py"),
        ("callhome_diar_clean_tutorial", "tutorials/audio/callhome_diar/run.py"),
    ],
)
def test_clean_refuses_to_delete_an_input(name: str, path: str, tmp_path: Path) -> None:
    tutorial = _load_tutorial(name, path)
    output_dir = tmp_path / "output"
    protected = output_dir / "input"

    with pytest.raises(ValueError, match="contains input"):
        tutorial._ensure_safe_clean(output_dir, protected)

    tutorial._ensure_safe_clean(output_dir, tmp_path / "outside")


def test_single_speaker_clean_refuses_manifest_audio_under_output(tmp_path: Path) -> None:
    tutorial = _load_tutorial(
        "single_speaker_filter_manifest_clean_tutorial",
        "tutorials/audio/single_speaker_filter/run.py",
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    audio_path = output_dir / "source.wav"
    audio_path.touch()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"audio_filepath": str(audio_path)}) + "\n")

    with pytest.raises(ValueError, match="contains input"):
        tutorial._ensure_manifest_sources_safe(output_dir, manifest)

    manifest.write_text(json.dumps({"audio_filepath": str(tmp_path / "outside.wav")}) + "\n")
    tutorial._ensure_manifest_sources_safe(output_dir, manifest)
