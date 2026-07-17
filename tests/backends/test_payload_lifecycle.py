# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

import pytest

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.pipeline import payload_refs
from nemo_curator.pipeline.payload_refs import PayloadRef
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


@dataclass
class _DropStage(ProcessingStage[AudioTask, AudioTask]):
    name: str = "drop"
    _curator_tracks_payload_refs = True

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: AudioTask) -> AudioTask:
        return task

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        return []


def test_adapter_releases_payload_when_consumer_drops_task(monkeypatch: pytest.MonkeyPatch) -> None:
    released: list[str] = []
    monkeypatch.setattr(
        payload_refs,
        "release_payload_ref",
        lambda payload_ref: released.append(payload_ref.payload_id),
    )
    task = AudioTask(
        data={
            "waveform_ref": PayloadRef(
                payload_id="payload",
                owner_node_id="node",
                store_actor_name="store",
                admission_actor_name="admission",
                amount_bytes=16,
                metadata={"sample_rate": 16_000, "num_samples": 4},
            )
        }
    )

    adapter = BaseStageAdapter(_DropStage())
    refs = adapter._collect_payload_refs([task])
    adapter._release_dropped_payload_refs(refs, [])
    assert released == ["payload"]
