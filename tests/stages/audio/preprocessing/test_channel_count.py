# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from pathlib import Path

import pytest

from nemo_curator.stages.audio import agent
from nemo_curator.stages.audio.preprocessing.channel_count import ChannelCountStage
from nemo_curator.tasks import AudioTask


def test_public_discovery_advertises_all_cardinalities() -> None:
    assert agent.describe_stage("ChannelCountStage").cardinality_options == ["1:1", "filter"]
    entry = next(entry for entry in agent.audio_stage_catalog() if entry["name"] == "ChannelCountStage")
    assert entry["contract"]["cardinality_options"] == ["1:1", "filter"]
    for stage in (ChannelCountStage(), ChannelCountStage(action="filter", allowed_channels=[1])):
        assert agent.build_contract(stage).cardinality_options == ["1:1", "filter"]


@pytest.mark.parametrize("stable_dir", [False, True])
def test_converted_disk_output_is_stable_with_and_without_output_dir(
    wav_filepath: Path, tmp_path: Path, stable_dir: bool
) -> None:
    stage = ChannelCountStage(
        action="convert",
        target_channels=1,
        write_to_disk=True,
        output_dir=str(tmp_path / "out") if stable_dir else None,
    )
    paths = []
    try:
        for _ in range(2):
            result = stage.process(AudioTask(data={"audio_filepath": str(wav_filepath)}))
            paths.append(result.data[stage.output_audio_filepath_key])
        assert paths[0] == paths[1]
        assert stage.describe().gates.per_row_independent
        assert all(Path(path).is_file() for path in paths)
    finally:
        for path in set(paths):
            Path(path).unlink(missing_ok=True)
