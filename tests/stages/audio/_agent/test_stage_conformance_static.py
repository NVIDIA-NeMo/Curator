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
"""Conformance for agent-ready stages whose runtime path needs a driver lifecycle or decomposes."""

from collections.abc import Callable
from pathlib import Path

import pytest

from nemo_curator.stages.audio._agent._conformance import assert_agent_ready
from nemo_curator.stages.audio.advanced_pipelines.audio_data_filter.audio_data_filter import AudioDataFilterStage
from nemo_curator.stages.audio.common import (
    CreateInitialManifestAudioFolderStage,
    ManifestCheckpointStage,
    ManifestReader,
    ManifestWriterStage,
    PreserveByValueConditionsStage,
)
from nemo_curator.stages.audio.io.group_export import ManifestGroupExportStage
from nemo_curator.stages.audio.tagging.split import SplitASRAlignJoinStage
from nemo_curator.tasks import AudioTask


@pytest.mark.parametrize(
    "make",
    [
        lambda _tmp: AudioDataFilterStage(),
        lambda _tmp: SplitASRAlignJoinStage(),
        lambda tmp: ManifestReader(str(tmp / "in.jsonl")),
        lambda tmp: CreateInitialManifestAudioFolderStage(data_dir=str(tmp)),
        lambda tmp: ManifestCheckpointStage(str(tmp / "checkpoint.jsonl")),
        lambda tmp: ManifestWriterStage(str(tmp / "out.jsonl")),
        lambda tmp: ManifestGroupExportStage(output_dir=str(tmp)),
    ],
)
def test_stage_contract_is_agent_ready(make: Callable[[Path], object], tmp_path: Path) -> None:
    assert_agent_ready(make(tmp_path), run=False)


def test_preserve_by_value_conditions_is_agent_ready() -> None:
    assert_agent_ready(
        PreserveByValueConditionsStage({"mos": 3.0}),
        lambda: AudioTask(data={"mos": 4.0}),
        expected_cardinality="filter",
        available_keys={"mos"},
    )
