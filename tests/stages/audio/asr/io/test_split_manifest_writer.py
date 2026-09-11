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

from nemo_curator.stages.audio.asr.io.split_manifest_writer import SplitAwareManifestWriter
from nemo_curator.tasks import AudioTask


def test_split_manifest_writer_defaults_to_split_jsonl(tmp_path: Path) -> None:
    stage = SplitAwareManifestWriter(output_dir=str(tmp_path), langs=["gu"], splits=["dev"])
    stage.setup()

    task = AudioTask(data={"lang": "gu", "split_type": "dev", "text": "ગુજરાતી"})
    result = stage.process(task)
    stage.teardown()

    assert result is task
    manifest_path = tmp_path / "gu" / "dev.jsonl"
    assert manifest_path.exists()
    with manifest_path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert rows == [task.data]


def test_split_manifest_writer_uses_output_filename_pattern(tmp_path: Path) -> None:
    stage = SplitAwareManifestWriter(
        output_dir=str(tmp_path),
        langs=["gu"],
        splits=["dev", "test"],
        output_filename_pattern="{split}_normalized.jsonl",
    )
    stage.setup()

    dev_task = AudioTask(data={"lang": "gu", "split_type": "dev", "text": "ગુજરાતી"})
    test_task = AudioTask(data={"lang": "gu", "split_type": "test", "text": "વાક્ય"})
    stage.process(dev_task)
    stage.process(test_task)
    stage.teardown()

    assert (tmp_path / "gu" / "dev_normalized.jsonl").exists()
    assert (tmp_path / "gu" / "test_normalized.jsonl").exists()
    assert not (tmp_path / "gu" / "dev.jsonl").exists()
