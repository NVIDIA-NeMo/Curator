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

import numpy as np

from nemo_curator.stages.audio.io.sharded_manifest_writer import ShardedManifestWriterStage
from nemo_curator.tasks import AudioTask


def test_writer_drops_waveform_and_writes_final_manifest_on_completion(tmp_path: Path) -> None:
    final_manifest = tmp_path / "output.jsonl"
    stage = ShardedManifestWriterStage(
        output_dir=str(tmp_path),
        final_manifest_path=str(final_manifest),
        write_perf_stats=False,
    )
    stage.setup_on_node()

    task = AudioTask(
        task_id="utt-1",
        dataset_name="test",
        data={
            "audio_filepath": "utt-1.wav",
            "duration": 1.0,
            "waveform": np.zeros(4, dtype=np.float32),
            "qwen3_prediction_s1": "hello",
        },
        _metadata={"_shard_key": "corpus/manifest_0", "_shard_total": 1},
    )

    result = stage.process(task)

    shard_path = tmp_path / "corpus" / "manifest_0.jsonl"
    assert result.data == [str(shard_path)]
    shard_row = json.loads(shard_path.read_text(encoding="utf-8").strip())
    assert "waveform" not in shard_row
    assert shard_row["qwen3_prediction_s1"] == "hello"
    assert json.loads(final_manifest.read_text(encoding="utf-8").strip()) == shard_row

    stage.teardown()

    final_row = json.loads(final_manifest.read_text(encoding="utf-8").strip())
    assert shard_row == final_row


def test_writer_appends_each_completed_shard_to_final_manifest_once(tmp_path: Path) -> None:
    final_manifest = tmp_path / "output.jsonl"
    stage = ShardedManifestWriterStage(
        output_dir=str(tmp_path),
        final_manifest_path=str(final_manifest),
        write_perf_stats=False,
    )
    stage.setup_on_node()

    first = AudioTask(
        task_id="utt-1",
        dataset_name="test",
        data={"audio_filepath": "utt-1.wav", "duration": 1.0},
        _metadata={"_shard_key": "corpus/manifest_0", "_shard_total": 1},
    )
    second = AudioTask(
        task_id="utt-2",
        dataset_name="test",
        data={"audio_filepath": "utt-2.wav", "duration": 1.0},
        _metadata={"_shard_key": "corpus/manifest_1", "_shard_total": 1},
    )

    stage.process(first)
    stage.process(second)

    rows = [json.loads(line) for line in final_manifest.read_text(encoding="utf-8").splitlines()]
    assert [row["audio_filepath"] for row in rows] == ["utt-1.wav", "utt-2.wav"]

    stage.teardown()

    rows = [json.loads(line) for line in final_manifest.read_text(encoding="utf-8").splitlines()]
    assert [row["audio_filepath"] for row in rows] == ["utt-1.wav", "utt-2.wav"]


def test_writer_rebuilds_final_manifest_from_completed_shards_on_teardown(tmp_path: Path) -> None:
    final_manifest = tmp_path / "output.jsonl"
    final_manifest.write_text('{"audio_filepath": "stale.wav"}\n', encoding="utf-8")
    shard_path = tmp_path / "corpus" / "manifest_0.jsonl"
    shard_path.parent.mkdir(parents=True)
    shard_path.write_text('{"audio_filepath": "fresh.wav"}\n', encoding="utf-8")
    done_path = tmp_path / "corpus" / "manifest_0.jsonl.done"
    done_path.write_text("1\n", encoding="utf-8")
    stage = ShardedManifestWriterStage(
        output_dir=str(tmp_path),
        final_manifest_path=str(final_manifest),
        write_perf_stats=False,
    )

    stage.setup_on_node()
    stage.teardown()

    assert final_manifest.read_text(encoding="utf-8") == '{"audio_filepath": "fresh.wav"}\n'


def test_writer_perf_summary_splits_invocations_and_items(tmp_path: Path) -> None:
    stage = ShardedManifestWriterStage(output_dir=str(tmp_path), write_perf_stats=True)
    stage.setup_on_node()
    tasks = [
        AudioTask(
            task_id=f"utt-{i}",
            dataset_name="test",
            data={"audio_filepath": f"utt-{i}.wav", "duration": 1.0},
            _metadata={"_shard_key": "corpus/manifest_0", "_shard_total": 2},
        )
        for i in range(2)
    ]

    stage.process_batch(tasks)
    stage.teardown()

    writer_summary = json.loads((tmp_path / "perf_summary.json").read_text(encoding="utf-8"))["stages"][
        "sharded_manifest_writer"
    ]
    assert writer_summary["total_items_processed"] == 2.0
    assert writer_summary["invocation_count"] == 1.0
    assert writer_summary["custom_metrics_sum"]["writer_process_calls"] == 1.0
    assert writer_summary["custom_metrics_sum"]["writer_invocation_count"] == 1.0
    assert writer_summary["custom_metrics_sum"]["writer_items_processed"] == 2.0


def test_writer_preserves_final_manifest_when_done_markers_exist(tmp_path: Path) -> None:
    final_manifest = tmp_path / "output.jsonl"
    final_manifest.write_text('{"audio_filepath": "old.wav"}\n', encoding="utf-8")
    done_path = tmp_path / "corpus" / "manifest_0.jsonl.done"
    done_path.parent.mkdir(parents=True)
    done_path.write_text("1\n", encoding="utf-8")
    stage = ShardedManifestWriterStage(
        output_dir=str(tmp_path),
        final_manifest_path=str(final_manifest),
        write_perf_stats=False,
    )

    stage.setup_on_node()

    assert final_manifest.read_text(encoding="utf-8") == '{"audio_filepath": "old.wav"}\n'


def test_writer_teardown_does_not_overwrite_existing_perf_summary_without_new_tasks(tmp_path: Path) -> None:
    perf_summary = tmp_path / "perf_summary.json"
    perf_summary.write_text('{"existing": true}\n', encoding="utf-8")
    stage = ShardedManifestWriterStage(output_dir=str(tmp_path), write_perf_stats=True)

    stage.setup_on_node()
    stage.teardown()

    assert json.loads(perf_summary.read_text(encoding="utf-8")) == {"existing": True}
