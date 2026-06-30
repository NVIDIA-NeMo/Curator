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

import pytest

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.dispatch_batch import DispatchBatchUnpackStage
from nemo_curator.tasks import AudioTask, DispatchBatchTask


def _batch(items: list[AudioTask] | None = None) -> DispatchBatchTask:
    children = items or [AudioTask(data={"value": 1}), AudioTask(data={"value": 2})]
    return DispatchBatchTask(
        dataset_name="dataset",
        data=children,
        batch_id="run:dispatch:0",
        owner_stage="gpu_stage",
        sequence_index=0,
        bucket_index=1,
        total_cost=30.0,
        item_costs=(10.0, 20.0),
        cost_unit="seconds",
        policy_signature="signature",
    )


def test_dispatch_batch_validates_and_preserves_metadata_when_items_change() -> None:
    batch = _batch()
    batch.task_id = "source_0"
    replacement = [AudioTask(data={"value": 3}), AudioTask(data={"value": 4})]

    rebuilt = batch.with_items(replacement)

    assert batch.validate()
    assert rebuilt.validate()
    assert rebuilt.items == replacement
    assert rebuilt.task_id == "source_0"
    assert rebuilt.batch_id == batch.batch_id
    assert rebuilt.item_costs == batch.item_costs
    assert rebuilt.num_items == 2


def test_dispatch_batch_rejects_changed_cardinality() -> None:
    with pytest.raises(ValueError, match="expected 2 item"):
        _batch().with_items([AudioTask(data={"value": 3})])


def test_dispatch_batch_unpack_is_one_row_fanout_stage() -> None:
    stage = DispatchBatchUnpackStage()
    batch = _batch()

    assert stage.process(batch) == batch.items
    assert stage.batch_size == 1
    assert stage.ray_stage_spec() == {RayStageSpecKeys.IS_FANOUT_STAGE: True}
    assert stage.xenna_stage_spec() == {}
