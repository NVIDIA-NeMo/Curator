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

import pytest

from nemo_curator.stages.audio._agent._agent_ready import AgentReady, ConditionalWrite, IOSpec, StageContract
from nemo_curator.stages.audio._agent._planning import validate_pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


class _ContractStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    def __init__(self, contract: StageContract) -> None:
        self.contract = contract

    def describe(self) -> StageContract:
        return self.contract

    def process(self, task: AudioTask) -> AudioTask:
        return task


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("reemit", [False, True])
def test_rebuild_discards_inherited_conditional_state(nested: bool, reemit: bool) -> None:
    scope = "segment_data_keys" if nested else "data_keys"
    conditional = ConditionalWrite(writes=IOSpec(**{scope: ["probe_key"]}), condition="runtime branch")
    producer = _ContractStage(StageContract(conditional_writes=[conditional]))
    rebuild = _ContractStage(
        StageContract(
            writes=IOSpec(data_keys=["audio_filepath"]),
            conditional_writes=[conditional] if reemit else [],
            preserves_upstream_keys=False,
        )
    )
    consumer = _ContractStage(StageContract(reads=IOSpec(**{scope: ["probe_key"]})))
    report = validate_pipeline([producer, rebuild, consumer], initial_keys={"audio_filepath"})
    assert report.ok is reemit
    expected = "conditional_read" if reemit else "unsatisfied_reads"
    assert any(issue.code == expected for issue in report.issues)
