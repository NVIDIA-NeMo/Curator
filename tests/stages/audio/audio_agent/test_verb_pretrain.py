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

"""ALM output ownership across complete, empty, and failed agent attempts."""

import json
import tarfile
import wave
from pathlib import Path

import pytest

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.stages.audio.audio_agent import _verb_outputs, _verb_run
from nemo_curator.tasks import EmptyTask


class _SerialExecutor:
    def execute(self, stages, initial_tasks):  # noqa: ANN001, ANN202
        tasks = initial_tasks or [EmptyTask()]
        prepared = []
        try:
            for stage in stages:
                stage.setup_on_node()
                stage.setup()
                prepared.append(stage)
                tasks = BaseStageAdapter(stage).process_batch(tasks)
            return tasks
        finally:
            for stage in reversed(prepared):
                stage.teardown()


def _recipe(tmp_path: Path, *, empty: bool = False, dry_run: bool = False) -> dict:
    audio = tmp_path / "source.wav"
    if not audio.exists():
        with wave.open(str(audio), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b"\x00\x00" * 3200)
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "new" if empty else "old",
                "audio_filepath": str(audio),
                "_snippet_plan": [] if empty else [{"start": 0.0, "end": 0.1, "segments": []}],
            }
        )
        + "\n"
    )
    return {
        "stages": [
            {"ref": "ManifestReader", "params": {"manifest_path": str(source)}},
            {
                "ref": "SnippetExtractionStage",
                "params": {
                    "output_dir": str(tmp_path / "clips"),
                    "output_audio_tar_path": str(tmp_path / "audio.tar"),
                    "dry_run": dry_run,
                },
            },
            {"ref": "SnippetManifestWriterStage", "params": {"output_path": str(tmp_path / "manifest.jsonl")}},
            {"ref": "PretrainMetricsAggregatorStage", "params": {"output_path": str(tmp_path / "metrics.json")}},
        ]
    }


@pytest.fixture(autouse=True)
def _private_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.delenv("AUDIO_AGENT_WORKSPACE", raising=False)
    monkeypatch.delenv("AUDIO_AGENT_REQUIRE_SMOKE", raising=False)


@pytest.mark.parametrize("prior_run", [False, True])
def test_zero_snippet_run_publishes_only_its_own_empty_bundle(tmp_path: Path, prior_run: bool) -> None:
    if prior_run:
        previous = _verb_run.run(_recipe(tmp_path), confirm=True, executor=_SerialExecutor())
        assert previous["status"] == "completed", previous
        assert previous["report"]["accepted"] == 1
    result = _verb_run.run(_recipe(tmp_path, empty=True), confirm=True, executor=_SerialExecutor())
    assert result["status"] == "completed", result
    assert result["report"]["accepted"] == 0
    assert (tmp_path / "manifest.jsonl").read_bytes() == b""
    assert json.loads((tmp_path / "metrics.json").read_text())["num_output_snippets"] == 0
    with tarfile.open(tmp_path / "audio.tar") as archive:
        assert archive.getnames() == []
    assert not list(tmp_path.glob("*.audio-agent-*"))


@pytest.mark.parametrize("failure", ["execution", "finalization", "promotion"])
def test_failed_attempt_preserves_the_complete_prior_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    previous = _verb_run.run(_recipe(tmp_path), confirm=True, executor=_SerialExecutor())
    assert previous["status"] == "completed", previous
    paths = [tmp_path / name for name in ("manifest.jsonl", "metrics.json", "audio.tar")]
    expected = {path: path.read_bytes() for path in paths}

    def fail(*_args, **_kwargs):  # noqa: ANN202
        raise OSError("injected attempt failure")  # noqa: EM101

    executor = _SerialExecutor()
    if failure == "execution":
        execute = executor.execute

        def fail_after_execution(*args, **kwargs):  # noqa: ANN202
            execute(*args, **kwargs)
            fail()

        monkeypatch.setattr(executor, "execute", fail_after_execution)
    elif failure == "finalization":
        monkeypatch.setattr("nemo_curator.stages.audio.alm.pretrain.finalize._merge_metrics_shards", fail)
    else:
        move = _verb_outputs._move_staged_output

        def fail_second_promotion(output, source, destination):  # noqa: ANN001, ANN202
            if output.index == 1 and source == output.staged_path and destination == output.final_path:
                fail()
            move(output, source, destination)

        monkeypatch.setattr(_verb_outputs, "_move_staged_output", fail_second_promotion)
    result = _verb_run.run(_recipe(tmp_path, empty=True), confirm=True, executor=executor)
    assert result["status"] == "failed", result
    assert result["reuse"]["published"] == []
    assert {path: path.read_bytes() for path in paths} == expected
    assert not list(tmp_path.glob("*.audio-agent-*"))


def test_dry_run_does_not_reconcile_against_or_replace_a_prior_archive(tmp_path: Path) -> None:
    previous = _verb_run.run(_recipe(tmp_path), confirm=True, executor=_SerialExecutor())
    assert previous["status"] == "completed", previous
    archive = (tmp_path / "audio.tar").read_bytes()
    recipe = _recipe(tmp_path, dry_run=True)
    source = tmp_path / "source.jsonl"
    row = json.loads(source.read_text())
    row["id"] = "different"
    source.write_text(json.dumps(row) + "\n")
    result = _verb_run.run(recipe, confirm=True, executor=_SerialExecutor())
    assert result["status"] == "completed", result
    assert result["report"]["accepted"] == 1
    assert json.loads((tmp_path / "manifest.jsonl").read_text())["id"] == "different"
    assert (tmp_path / "audio.tar").read_bytes() == archive
    assert not list(tmp_path.glob("*.audio-agent-*"))
