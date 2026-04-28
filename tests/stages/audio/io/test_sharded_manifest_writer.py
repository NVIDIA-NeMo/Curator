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

from nemo_curator.stages.audio.io.sharded_manifest_writer import ShardedManifestWriterStage
from nemo_curator.tasks import AudioTask


def test_requires_output_dir() -> None:
    with pytest.raises(ValueError, match="output_dir"):
        ShardedManifestWriterStage(output_dir="")


def test_writes_jsonl(tmp_path: Path) -> None:
    stage = ShardedManifestWriterStage(output_dir=str(tmp_path))
    task = AudioTask(
        task_id="t1",
        data={"text": "hello"},
        _metadata={"_shard_key": "corpus/shard_0", "_shard_total": 0},
    )
    result = stage.process(task)
    out_path = tmp_path / "corpus" / "shard_0.jsonl"
    assert out_path.exists()
    content = json.loads(out_path.read_text().strip())
    assert content["text"] == "hello"
    assert result.data == [str(out_path)]


def test_done_marker_on_completion(tmp_path: Path) -> None:
    stage = ShardedManifestWriterStage(output_dir=str(tmp_path))
    task = AudioTask(
        task_id="t1",
        data={"text": "hello"},
        _metadata={"_shard_key": "corpus/shard_0", "_shard_total": 1},
    )
    stage.process(task)
    done_path = tmp_path / "corpus" / "shard_0.jsonl.done"
    assert done_path.exists()


def test_no_done_marker_when_incomplete(tmp_path: Path) -> None:
    stage = ShardedManifestWriterStage(output_dir=str(tmp_path))
    task = AudioTask(
        task_id="t1",
        data={"text": "hello"},
        _metadata={"_shard_key": "corpus/shard_0", "_shard_total": 2},
    )
    stage.process(task)
    done_path = tmp_path / "corpus" / "shard_0.jsonl.done"
    assert not done_path.exists()


def test_process_batch(tmp_path: Path) -> None:
    stage = ShardedManifestWriterStage(output_dir=str(tmp_path))
    tasks = [
        AudioTask(
            task_id="t1",
            data={"text": "a"},
            _metadata={"_shard_key": "s/0", "_shard_total": 0},
        ),
        AudioTask(
            task_id="t2",
            data={"text": "b"},
            _metadata={"_shard_key": "s/0", "_shard_total": 0},
        ),
    ]
    results = stage.process_batch(tasks)
    assert len(results) == 2
    out_path = tmp_path / "s" / "0.jsonl"
    lines = out_path.read_text().strip().split("\n")
    assert len(lines) == 2


def test_empty_batch(tmp_path: Path) -> None:
    stage = ShardedManifestWriterStage(output_dir=str(tmp_path))
    assert stage.process_batch([]) == []


def test_inputs_outputs(tmp_path: Path) -> None:
    stage = ShardedManifestWriterStage(output_dir=str(tmp_path))
    assert stage.inputs() == ([], [])
    assert stage.outputs() == ([], [])
