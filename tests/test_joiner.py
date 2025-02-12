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

from nemo_curator import DocumentJoiner
from nemo_curator.datasets import DocumentDataset


class TestDocumentJoiner:
    def test_join_default(self):
        # Input represents documents already split.
        # For example, a document with id=1 split as "a", "b", "c" becomes joined to "a|b|c".
        # Four documents are used.
        data = {
            "id": [1, 1, 1, 2, 3, 3, 4, 4],
            "text": ["a", "b", "c", "nosplit", "start", "middle", "end", ""],
            "segment_id": [0, 1, 2, 0, 0, 1, 0, 1],
        }
        pdf = pd.DataFrame(data)
        dataset = DocumentDataset.from_pandas(pdf, npartitions=1)
        joiner = DocumentJoiner(
            separator="|",
            text_field="text",
            segment_id_field="segment_id",
            document_id_field="id",
            drop_segment_id_field=True,
        )
        result_dataset = joiner(dataset)

        expected_df = pd.DataFrame(
            {"id": [1, 2, 3, 4], "text": ["a|b|c", "nosplit", "start|middle", "end|"]}
        )
        assert_eq(
            result_dataset.df.compute().reset_index(drop=True),
            expected_df,
            check_index=False,
        )

    def test_join_custom_fields(self):
        # Use custom field names:
        #   document id field: "doc"
        #   text field: "content"
        #   segment id field: "s_id"
        # Also keep the segment id field (drop_segment_id_field=False)
        data = {
            "doc": [101, 101, 102, 103, 103, 104, 104],
            "content": ["first", "second", "only", "hello", "world", "baz", ""],
            "s_id": [0, 1, 0, 0, 1, 0, 1],
        }
        pdf = pd.DataFrame(data)
        dataset = DocumentDataset.from_pandas(pdf, npartitions=1)
        joiner = DocumentJoiner(
            separator="~",
            text_field="content",
            segment_id_field="s_id",
            document_id_field="doc",
            drop_segment_id_field=False,
        )
        result_dataset = joiner(dataset)

        # Expected: each document is joined by "~". The segment id becomes the first segment's id.
        expected_df = pd.DataFrame(
            {
                "doc": [101, 102, 103, 104],
                "content": ["first~second", "only", "hello~world", "baz~"],
                "s_id": [0, 0, 0, 0],
            }
        )
        assert_eq(
            result_dataset.df.compute().reset_index(drop=True),
            expected_df,
            check_index=False,
        )

    def test_join_max_length(self):
        # Here we test joining when a maximum length is specified.
        # Each segment carries a precomputed "length" value.
        # The joiner should accumulate segments until adding the next one (plus separator)
        # would exceed max_length=5.
        #
        # For document 1:
        #   segments: "ab"(2), "cd"(2), "ef"(2), "gh"(2)
        #   - "ab" then "cd": 2+2+1 = 5  → join as "ab-cd" (length 5)
        #   - then "ef" then "gh": 2+2+1 = 5 → join as "ef-gh" (length 5)
        #
        # For document 2:
        #   segments: "a"(1), "b"(1) → join as "a-b" (length 3)
        #
        # For document 3:
        #   segment: "hello"(5) → remains "hello"
        #
        # For document 4:
        #   segments: "x"(1), "yz"(2), "0"(1)
        #   - "x" then "yz": 1+2+1 = 4 → "x-yz" (length 4)
        #   - "0" remains alone.
        data = {
            "id": [1, 1, 1, 1, 2, 2, 3, 4, 4, 4],
            "text": ["ab", "cd", "ef", "gh", "a", "b", "hello", "x", "yz", "0"],
            "segment_id": [0, 1, 2, 3, 0, 1, 0, 0, 1, 2],
            "length": [2, 2, 2, 2, 1, 1, 5, 1, 2, 1],
        }
        pdf = pd.DataFrame(data)
        dataset = DocumentDataset.from_pandas(pdf, npartitions=1)
        joiner = DocumentJoiner(
            separator="-",
            text_field="text",
            segment_id_field="segment_id",
            document_id_field="id",
            drop_segment_id_field=True,
            max_length=5,
            length_field="length",
        )
        result_dataset = joiner(dataset)

        expected_df = pd.DataFrame(
            [
                {"id": 1, "text": "ab-cd", "length": 5},
                {"id": 1, "text": "ef-gh", "length": 5},
                {"id": 2, "text": "a-b", "length": 3},
                {"id": 3, "text": "hello", "length": 5},
                {"id": 4, "text": "x-yz", "length": 4},
                {"id": 4, "text": "0", "length": 1},
            ]
        )
        # Sort by id and text to ensure consistent order
        expected_sorted = expected_df.sort_values(by=["id", "text"]).reset_index(
            drop=True
        )
        result_sorted = (
            result_dataset.df.compute()
            .sort_values(by=["id", "text"])
            .reset_index(drop=True)
        )
        assert_eq(result_sorted, expected_sorted, check_index=False)

    def test_join_with_string_ids(self):
        # Test join functionality when document id field is a string.
        data = {
            "doc": ["doc1", "doc1", "doc2", "doc3", "doc3", "doc4", "doc4"],
            "text": ["a", "b", "nosplit", "start", "middle", "end", ""],
            "segment_id": [0, 1, 0, 0, 1, 0, 1],
        }
        pdf = pd.DataFrame(data)
        dataset = DocumentDataset.from_pandas(pdf, npartitions=1)
        joiner = DocumentJoiner(
            separator="|",
            text_field="text",
            segment_id_field="segment_id",
            document_id_field="doc",
            drop_segment_id_field=True,
        )
        result_dataset = joiner(dataset)

        expected_df = pd.DataFrame(
            {
                "doc": ["doc1", "doc2", "doc3", "doc4"],
                "text": ["a|b", "nosplit", "start|middle", "end|"],
            }
        )
        assert_eq(
            result_dataset.df.compute().reset_index(drop=True),
            expected_df,
            check_index=False,
        )
