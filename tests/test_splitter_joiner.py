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
from dask.dataframe.utils import assert_eq

from nemo_curator import DocumentJoiner, DocumentSplitter
from nemo_curator.datasets import DocumentDataset


class TestSplitJoinReconstruction:
    def test_reconstruction_default(self):
        # Create an original dataset with a unique "id" column and text examples.
        # Four examples include edge cases:
        #   "a|b|c"          → multiple splits
        #   "nosplit"        → no separator present
        #   "a||b|"          → consecutive separators yield empty strings
        #   ""               → empty document
        docs = ["a|b|c", "nosplit", "a||b|", ""]
        pdf = pd.DataFrame({"id": [1, 2, 3, 4], "text": docs})
        original_dataset = DocumentDataset.from_pandas(pdf, npartitions=1)

        # First, split using "|" as separator.
        splitter = DocumentSplitter(separator="|")
        split_dataset = splitter(original_dataset)

        # Then, rejoin using the same separator.
        joiner = DocumentJoiner(
            separator="|",
            text_field="text",
            segment_id_field="segment_id",
            document_id_field="id",
            drop_segment_id_field=True,
        )
        reconstructed_dataset = joiner(split_dataset)

        # The reconstructed "text" column should match the original.
        original_sorted = (
            original_dataset.df.compute().sort_values(by="id").reset_index(drop=True)
        )
        reconstructed_sorted = (
            reconstructed_dataset.df.compute()
            .sort_values(by="id")
            .reset_index(drop=True)
        )
        assert_eq(reconstructed_sorted, original_sorted, check_index=False)
