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

import pytest

from nemo_curator.stages.deduplication.semantic.kmeans_utils import plan_kmeans_prediction
from nemo_curator.stages.deduplication.semantic.utils import CUDF_COLUMN_SIZE_LIMIT, ParquetFileInfo


@pytest.mark.gpu
@pytest.mark.parametrize(("memory_budget", "expected_workers"), [(80_000_000_000, 4), (20_000_000_000, 1)])
def test_prediction_groups_respect_memory_and_element_limits(memory_budget: int, expected_workers: int) -> None:
    info = [ParquetFileInfo(str(i), 1000, 1000, embedding_elements=400_000_000) for i in range(12)]
    groups, workers = plan_kmeans_prediction(
        info, memory_budget=memory_budget, max_workers=4, n_clusters=2, max_samples_per_batch=32
    )
    assert workers == expected_workers
    assert [path for group in groups for path in group] == [item.path for item in info]
    assert all(len(group) == 1 for group in groups)


@pytest.mark.gpu
def test_prediction_groups_respect_element_limit_with_spare_memory() -> None:
    info = [ParquetFileInfo(str(i), 1000, 1000, embedding_elements=CUDF_COLUMN_SIZE_LIMIT // 3) for i in range(8)]
    groups, workers = plan_kmeans_prediction(
        info, memory_budget=1_000_000_000_000, max_workers=4, n_clusters=2, max_samples_per_batch=32
    )
    assert [path for group in groups for path in group] == [item.path for item in info]
    assert workers == len(groups) == 3
    assert all(len(group) * info[0].embedding_elements < CUDF_COLUMN_SIZE_LIMIT for group in groups)
