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

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.audio.io.sharded_manifest_writer import ShardedManifestWriterStage
from nemo_curator.tasks import AudioTask
from nemo_curator.utils.performance_utils import StagePerfStats


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


def test_write_perf_stats_false_skips_perf_outputs(tmp_path: Path) -> None:
    stage = ShardedManifestWriterStage(output_dir=str(tmp_path), write_perf_stats=False)
    task = AudioTask(
        task_id="t1",
        data={"text": "hello", "duration": 1.0},
        _metadata={"_shard_key": "corpus/shard_0", "_shard_total": 1},
        _stage_perf=[
            StagePerfStats(
                stage_name="upstream",
                process_time=1.0,
                num_items_processed=1,
                custom_metrics={"audio_duration_s": 1.0},
            )
        ],
    )

    stage.process(task)
    stage.teardown()

    assert (tmp_path / "corpus" / "shard_0.jsonl").exists()
    assert (tmp_path / "corpus" / "shard_0.jsonl.done").exists()
    assert not (tmp_path / "corpus" / "shard_0_perf.jsonl").exists()
    assert not (tmp_path / "perf_summary.json").exists()


def test_empty_batch(tmp_path: Path) -> None:
    stage = ShardedManifestWriterStage(output_dir=str(tmp_path))
    assert stage.process_batch([]) == []


def test_inputs_outputs(tmp_path: Path) -> None:
    stage = ShardedManifestWriterStage(output_dir=str(tmp_path))
    assert stage.inputs() == ([], [])
    assert stage.outputs() == ([], [])


def test_backend_worker_specs(tmp_path: Path) -> None:
    stage = ShardedManifestWriterStage(output_dir=str(tmp_path))
    assert stage.num_workers() == 1
    assert stage.xenna_stage_spec() == {"num_workers": 1}
    assert stage.ray_stage_spec() == {RayStageSpecKeys.IS_ACTOR_STAGE: True}


def test_perf_summary_deduplicates_batch_stage_metrics(tmp_path: Path) -> None:
    stage = ShardedManifestWriterStage(output_dir=str(tmp_path))
    perf = StagePerfStats(
        stage_name="QwenOmni_inference",
        process_time=2.0,
        num_items_processed=2,
        custom_metrics={
            "audio_duration_s": 10.0,
            "inference_time_s": 1.0,
            "output_tokens": 50.0,
        },
        invocation_id="same-batch",
    )
    tasks = [
        AudioTask(
            task_id="t1",
            data={"text": "a", "duration": 4.0},
            _metadata={"_shard_key": "s/0", "_shard_total": 0},
            _stage_perf=[perf],
        ),
        AudioTask(
            task_id="t2",
            data={"text": "b", "duration": 6.0},
            _metadata={"_shard_key": "s/0", "_shard_total": 0},
            _stage_perf=[perf],
        ),
    ]

    stage.process_batch(tasks)
    stage.teardown()

    summary = json.loads((tmp_path / "perf_summary.json").read_text())
    omni_summary = summary["stages"]["QwenOmni_inference"]
    assert summary["total_utterances"] == 2
    assert summary["total_audio_seconds"] == 10.0
    assert summary["perf_invocations_counted"] == 1
    assert omni_summary["invocation_count"] == 1
    assert omni_summary["total_items_processed"] == 2
    assert omni_summary["custom_metrics_sum"]["audio_duration_s"] == 10.0
    assert omni_summary["throughput_audio_s_per_inference_s"] == 10.0
    assert omni_summary["throughput_output_tokens_per_process_s"] == 25.0
