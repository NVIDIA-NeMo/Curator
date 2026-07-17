# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.dispatch_batch import DispatchBatchUnpackStage
from nemo_curator.tasks import AudioTask, DispatchBatchTask


def test_atomic_dispatch_round_trips_through_explicit_unpack() -> None:
    children = [AudioTask(data={"value": 1}), AudioTask(data={"value": 2})]
    batch = DispatchBatchTask(
        dataset_name="audio",
        data=children,
        batch_id="batch-0",
        owner_stage="owner",
        item_costs=(1.0, 1.0),
        total_cost=2.0,
        cost_unit="seconds",
        policy_signature="signature",
    )
    stage = DispatchBatchUnpackStage()

    assert batch.validate()
    assert stage.process(batch) == children
    assert stage.ray_stage_spec() == {RayStageSpecKeys.IS_FANOUT_STAGE: True}


def test_with_items_preserves_atomic_envelope_structure() -> None:
    original = [AudioTask(data={"value": 1}), AudioTask(data={"value": 2})]
    replacement = [AudioTask(data={"value": 3}), AudioTask(data={"value": 4})]
    batch = DispatchBatchTask(
        dataset_name="audio",
        data=original,
        batch_id="batch-structural",
        owner_stage="owner",
        sequence_index=7,
        bucket_index=2,
        item_costs=(1.5, 2.5),
        total_cost=4.0,
        cost_unit="seconds",
        policy_signature="signature",
    )

    rebuilt = batch.with_items(replacement)

    assert rebuilt.items is rebuilt.data
    assert rebuilt.items == replacement
    assert rebuilt.items is not batch.items
    assert rebuilt.batch_id == batch.batch_id
    assert rebuilt.owner_stage == batch.owner_stage
    assert rebuilt.sequence_index == batch.sequence_index
    assert rebuilt.bucket_index == batch.bucket_index
    assert rebuilt.item_costs == batch.item_costs
    assert rebuilt.total_cost == batch.total_cost
    assert rebuilt.policy_signature == batch.policy_signature
