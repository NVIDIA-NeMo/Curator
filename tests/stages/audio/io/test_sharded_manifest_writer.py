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

import numpy as np

from nemo_curator.stages.audio.io.sharded_manifest_writer import ShardedManifestWriterStage
from nemo_curator.tasks import AudioTask


def test_writer_drops_waveform_and_writes_final_manifest(tmp_path) -> None:
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
    final_row = json.loads(final_manifest.read_text(encoding="utf-8").strip())
    assert shard_row == final_row
    assert "waveform" not in shard_row
    assert shard_row["qwen3_prediction_s1"] == "hello"
