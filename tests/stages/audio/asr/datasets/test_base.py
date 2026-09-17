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
from dataclasses import dataclass, field
from pathlib import Path

from nemo_curator.stages.audio.asr.datasets.base import BaseASRDatasetHandlerStage
from nemo_curator.stages.audio.asr.metadata import ASRMetadata
from nemo_curator.tasks import AudioTask, _EmptyTask


@dataclass
class _DummyASRDatasetHandler(BaseASRDatasetHandlerStage):
    source_name: str = "dummy"
    manifest_splits: list[str] | None = field(default_factory=lambda: ["dev"])

    def process(self, _: _EmptyTask) -> list[AudioTask]:
        return []


def test_base_asr_dataset_handler_writes_split_manifests(tmp_path: Path) -> None:
    stage = _DummyASRDatasetHandler(
        raw_data_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "out"),
        langs=["gu"],
        write_manifest=True,
    )
    stage.setup_on_node()
    meta = ASRMetadata(
        audio_filepath=str(tmp_path / "out" / "gu" / "dev" / "audio" / "sample.wav"),
        text="ગુજરાતી",
        duration=1.0,
        lang="gu",
        split_type="dev",
        source="dummy",
        sample_rate=16000,
        num_channels=1,
        orig_sample_rate=16000,
        orig_num_channels=1,
    )

    stage.write_manifest_entry(meta)
    stage.teardown()

    manifest_path = tmp_path / "out" / "gu" / "dev.jsonl"
    with manifest_path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert rows == [meta.to_dict()]
