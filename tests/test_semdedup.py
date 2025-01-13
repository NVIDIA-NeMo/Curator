# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import os

import pytest
from dask.dataframe.utils import assert_eq
from distributed import Client

from nemo_curator import SemDedup, SemDedupConfig
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.import_utils import gpu_only_import, gpu_only_import_from

cudf = gpu_only_import("cudf")
dask_cudf = gpu_only_import("dask_cudf")
LocalCUDACluster = gpu_only_import_from("dask_cuda", "LocalCUDACluster")


@pytest.fixture
def dedup_data():
    df = cudf.DataFrame(
        {
            "id": [1, 2, 3, 4, 100, 200, 300],
            "text": [
                "The quick brown fox jumps over the lazy dog",
                "The quick brown foxes jumps over the lazy dog",
                "The quick brown wolf jumps over the lazy dog",
                "The quick black cat jumps over the lazy dog",
                "A test string",
                "Another test string",
                "A different object",
            ],
        }
    )
    df = dask_cudf.from_cudf(df, 2)
    return DocumentDataset(df)


@pytest.mark.gpu
class TestSemDuplicates:
    @pytest.fixture(autouse=True, scope="class")
    def gpu_client(self, request):
        with LocalCUDACluster(n_workers=1) as cluster, Client(cluster) as client:
            request.cls.client = client
            request.cls.cluster = cluster
            yield

    def test_sem_dedup(
        self,
        dedup_data,
        tmpdir,
    ):
        print("client", self.client)
        cache_dir = os.path.join(tmpdir, "test_sem_dedup_cache")
        config = SemDedupConfig(
            cache_dir=cache_dir,
            seed=42,
            n_clusters=3,
            eps_thresholds=[0.10],
            eps_to_extract=0.10,
        )
        sem_duplicates = SemDedup(
            config=config,
            input_column="text",
            id_column="id",
            id_column_type="int",
        )
        result = sem_duplicates(dedup_data)
        result_df = result.df.compute()
        duplicate_docs = [2, 3, 4, 200, 300]
        expected_df = cudf.Series(duplicate_docs, name="id")
        assert_eq(result_df["id"].sort_values(), expected_df, check_index=False)
