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
import pandas as pd
import pytest
from dask.dataframe.utils import assert_eq

from nemo_curator import DocumentSplitter, ToBackend
from nemo_curator.datasets import DocumentDataset


class TestDocumentSplitter:
    @pytest.mark.parametrize(
        "backend", ["pandas", pytest.param("cudf", marks=pytest.mark.gpu)]
    )
    def test_basic_split_default(self, backend):
        # Use default text_field "text" and segment_id_field "segment_id"
        # Four examples:
        #   "a|b|c"    → splits to ["a", "b", "c"]
        #   "nosplit"  → ["nosplit"]
        #   "start|middle" → ["start", "middle"]
        #   "end|"     → ["end", ""]
        docs = ["a|b|c", "nosplit", "start|middle", "end|"]
        pdf = pd.DataFrame({"text": docs})
        dataset = DocumentDataset.from_pandas(pdf, npartitions=1)
        to_backend = ToBackend(backend)
        dataset = to_backend(dataset)

        splitter = DocumentSplitter(separator="|")
        result_dataset = splitter(dataset)

        result_df = result_dataset.df.compute()
        if backend == "cudf":
            result_df = result_df.to_pandas()

        expected_df = pd.DataFrame(
            {
                "text": ["a", "b", "c", "nosplit", "start", "middle", "end", ""],
                "segment_id": [0, 1, 2, 0, 0, 1, 0, 1],
            }
        )
        # Compare without considering the index order.
        assert_eq(
            result_df.reset_index(drop=True),
            expected_df.reset_index(drop=True),
            check_index=False,
        )

    @pytest.mark.parametrize(
        "backend", ["pandas", pytest.param("cudf", marks=pytest.mark.gpu)]
    )
    def test_split_custom_fields(self, backend):
        # Use a custom text field name ("content") and segment id field ("seg_id")
        # with a different separator.
        # Examples:
        #   "x;y"               → ["x", "y"]
        #   "single"            → ["single"]
        #   "first;second;third" → ["first", "second", "third"]
        #   ";leading"          → ["", "leading"]
        docs = ["x;y", "single", "first;second;third", ";leading"]
        pdf = pd.DataFrame({"content": docs})
        dataset = DocumentDataset.from_pandas(pdf, npartitions=1)
        to_backend = ToBackend(backend)
        dataset = to_backend(dataset)

        splitter = DocumentSplitter(
            separator=";", text_field="content", segment_id_field="seg_id"
        )
        result_dataset = splitter(dataset)

        result_df = result_dataset.df.compute()
        if backend == "cudf":
            result_df = result_df.to_pandas()

        expected_df = pd.DataFrame(
            {
                "content": [
                    "x",
                    "y",
                    "single",
                    "first",
                    "second",
                    "third",
                    "",
                    "leading",
                ],
                "seg_id": [0, 1, 0, 0, 1, 2, 0, 1],
            }
        )
        assert_eq(
            result_df.reset_index(drop=True),
            expected_df.reset_index(drop=True),
            check_index=False,
        )
