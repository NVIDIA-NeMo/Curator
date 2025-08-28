# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# ruff: noqa: E402
cudf = pytest.importorskip("cudf", reason="utils tests require cudf")

import cupy as cp
import torch

from nemo_curator.stages.text.embedders.utils import create_list_series_from_1d_or_2d_ar


@pytest.mark.gpu
class TestCreateListSeriesFrom1dOr2dAr:
    """Test create_list_series_from_1d_or_2d_ar function."""

    def test_create_list_series_from_1d_or_2d_ar_1d(self):
        tensor = torch.tensor([101, 102, 103])
        index = [1, 2, 3]
        series = create_list_series_from_1d_or_2d_ar(tensor, index)
        assert isinstance(series, cudf.Series)
        expected = cudf.Series([[101], [102], [103]], index=index)
        # convert to pandas because cudf.Series.equals doesn't work for list series
        assert series.to_pandas().equals(expected.to_pandas())

    def test_create_list_series_from_1d_or_2d_ar_2d(self):
        tensor = torch.tensor([[101, 102], [103, 104], [105, 106]])
        index = [1, 2, 3]
        series = create_list_series_from_1d_or_2d_ar(tensor, index)
        assert isinstance(series, cudf.Series)
        expected = cudf.Series([[101, 102], [103, 104], [105, 106]], index=index)
        # convert to pandas because cudf.Series.equals doesn't work for list series
        assert series.to_pandas().equals(expected.to_pandas())

    def test_embedding_creation_with_shuffled_index(self):
        """Test the function with a reordered/shuffled index as suggested by colleague."""
        collected_output = cp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])

        # Create a GPU dataframe with reordered index
        df_gpu = cudf.DataFrame({"col_a": [1, 2, 3]})
        df_gpu = df_gpu.iloc[[2, 0, 1]]  # Reorder: [2, 0, 1]

        series = create_list_series_from_1d_or_2d_ar(collected_output, index=df_gpu.index)

        assert isinstance(series, cudf.Series)
        assert series.index.equals(df_gpu.index)

        # The embeddings should still be in the same order as the input array
        # but the index should be shuffled [2, 0, 1]
        expected_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        expected_index = [2, 0, 1]

        # Convert the series (not df_gpu) to pandas to check values
        embedding_series_cpu = series.to_pandas()
        for i, (expected_embedding, expected_idx) in enumerate(zip(expected_embeddings, expected_index, strict=False)):
            assert embedding_series_cpu.iloc[i] == expected_embedding
            assert embedding_series_cpu.index[i] == expected_idx
