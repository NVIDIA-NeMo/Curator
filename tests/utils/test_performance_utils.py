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

import pickle

from nemo_curator.utils.performance_utils import StagePerfStats


def test_batch_identity_survives_serialization_without_changing_metric_exports() -> None:
    stats = StagePerfStats(stage_name="test", process_time=2, custom_metrics={"requests": 3})
    restored = pickle.loads(pickle.dumps(stats))  # noqa: S301
    assert restored is not stats
    assert restored.batch_id == stats.batch_id
    assert restored.to_dict() == {
        "stage_name": "test",
        "process_time": 2,
        "actor_idle_time": 0,
        "input_data_size_mb": 0,
        "num_items_processed": 0,
        "custom_metrics": {"requests": 3},
    }
    assert dict(restored.items())["custom.requests"] == 3
    assert "batch_id" not in dict(restored.items())


def test_distinct_equal_observations_have_distinct_batch_identities() -> None:
    first = StagePerfStats(stage_name="test", process_time=2)
    second = StagePerfStats(stage_name="test", process_time=2)
    assert first == second
    assert first.batch_id != second.batch_id
