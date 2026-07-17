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

from nemo_curator.stages.audio.inference.batch_policy import BatchPolicy


def test_batch_policy_globally_packs_by_bucket_and_caps() -> None:
    policy = BatchPolicy(
        buckets_sec=[0.0, 10.0],
        max_items_per_batch_by_bucket=[3, 2],
        max_audio_sec_per_batch=20.0,
    )
    items = ["short-a", "long-a", "short-b", "long-b", "long-c"]
    costs = [4.0, 12.0, 5.0, 8.0, 15.0]

    batches = policy.bucketize_with_costs(items, costs)

    assert sorted(item for _, batch, _ in batches for item in batch) == sorted(items)
    assert all(len(batch) <= policy.max_items_per_batch_by_bucket[bucket] for bucket, batch, _ in batches)
    assert all(len(batch) == 1 or sum(batch_costs) <= 20 for _, batch, batch_costs in batches)
    assert [sum(batch_costs) for _, _, batch_costs in batches] == sorted(
        (sum(batch_costs) for _, _, batch_costs in batches),
        reverse=True,
    )
