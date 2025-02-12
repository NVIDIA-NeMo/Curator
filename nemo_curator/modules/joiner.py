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
from typing import List, Optional

import pandas as pd

from nemo_curator.datasets import DocumentDataset
from nemo_curator.modules.base import BaseModule


class DocumentJoiner(BaseModule):
    """
    Joins documents that have a common id back into a single document.
    The order of the documents is dictated by an additional segment_id column.
    A maximum length can be specified to limit the size of the joined documents.

    The joined documents are joined by a separator.
    """

    def __init__(
        self,
        separator: str,
        text_field: str = "text",
        segment_id_field: str = "segment_id",
        document_id_field: str = "id",
        drop_segment_id_field: bool = True,
        max_length: Optional[int] = None,
        length_field: Optional[str] = None,
    ):
        """
        Args:
            separator (str): The separator to join the documents on.
            text_field (str): The name of the column containing the text to join.
            segment_id_field (str): The name of the column containing the segment id.
            document_id_field (str): The name of the column containing the document id.
            drop_segment_id_field (bool): Whether to drop the segment_id_field after joining.
            max_length (int, optional): The maximum length of the joined documents.
                Both max_length and length_field must be specified or neither can be specified.
            length_field (str, optional): The name of the column containing the length of the documents.
                Both max_length and length_field must be specified or neither can be specified.
        """
        if max_length is not None and length_field is None:
            raise ValueError("max_length is specified but length_field is not")
        if max_length is None and length_field is not None:
            raise ValueError("length_field is specified but max_length is not")

        super().__init__(input_backend="pandas")
        self.separator = separator
        self.text_field = text_field
        self.segment_id_field = segment_id_field
        self.document_id_field = document_id_field
        self.drop_segment_id_field = drop_segment_id_field
        self.max_length = max_length
        self.length_field = length_field

    def _join_segments(self, group):
        # Ensure segments are processed in order.
        group = group.sort_values(self.segment_id_field)
        joined_rows = []
        current_seg_id = 0
        accumulator_text = None
        accumulator_length = 0
        accumulator_row = None

        for _, row in group.iterrows():
            if accumulator_row is None:
                # Start a new accumulation with the first segment.
                accumulator_text = row[self.text_field]
                accumulator_length = row[self.length_field]
                accumulator_row = row
            else:
                # Calculate what the new length would be if we joined this segment.
                proposed_length = accumulator_length + row[self.length_field] + 1
                if proposed_length <= self.max_length:
                    accumulator_text = (
                        accumulator_text + self.separator + row[self.text_field]
                    )
                    accumulator_length = proposed_length
                else:
                    # Commit the current accumulation as one joined segment.
                    new_row = accumulator_row.copy()
                    new_row[self.text_field] = accumulator_text
                    new_row[self.length_field] = accumulator_length
                    new_row[self.segment_id_field] = current_seg_id
                    joined_rows.append(new_row)
                    current_seg_id += 1
                    # Start a new accumulation with the current row.
                    accumulator_text = row[self.text_field]
                    accumulator_length = row[self.length_field]
                    accumulator_row = row

        # Commit the last accumulated segment.
        if accumulator_row is not None:
            new_row = accumulator_row.copy()
            new_row[self.text_field] = accumulator_text
            new_row[self.length_field] = accumulator_length
            new_row[self.segment_id_field] = current_seg_id
            joined_rows.append(new_row)
        if joined_rows:
            return pd.concat(
                [group.iloc[0:0], pd.DataFrame(joined_rows)], ignore_index=True
            )
        else:
            return group.iloc[0:0]

    def _join_partition(
        self, df: pd.DataFrame, expected_cols: List[str]
    ) -> pd.DataFrame:
        if df.empty:
            return df

        if self.max_length is None:
            # Sort the segments by the segment_id_field to maintain proper order before aggregating.
            df_sorted = df.sort_values(self.segment_id_field)
            # Build aggregation functions to preserve all original columns:
            # - For self.text_field, join all segments using the separator.
            # - For all other columns (except self.document_id_field, which is our grouping key), take the first occurrence.
            agg_funcs = {}
            for col in df_sorted.columns:
                if col == self.text_field:
                    agg_funcs[col] = lambda texts: self.separator.join(
                        texts.astype(str)
                    )
                elif col != self.document_id_field:
                    agg_funcs[col] = "first"
            # Group by document_id_field while keeping the key as a column.
            joined = df_sorted.groupby(self.document_id_field, as_index=False).agg(
                agg_funcs
            )
        else:
            joined = df.groupby(self.document_id_field, group_keys=False).apply(
                self._join_segments
            )

        if self.drop_segment_id_field:
            joined = joined.drop(columns=self.segment_id_field)
        # Reorder the columns to match the expected metadata order.
        joined = joined[expected_cols]
        return joined

    def call(self, dataset: DocumentDataset) -> DocumentDataset:
        """
        Joins the documents back into a single document while preserving all the original fields.
        """
        # Construct meta information for the transformed dataframe.
        meta = dataset.df._meta.copy()
        if self.text_field not in meta.columns:
            meta[self.text_field] = pd.Series(dtype="object")
        # If dropping the segment id field, remove it from the metadata to prevent mismatches.
        if self.drop_segment_id_field:
            meta = meta.drop(columns=self.segment_id_field)
        expected_cols = list(meta.columns)
        # Apply the join operation partition-wise.
        dataset.df = dataset.df.map_partitions(
            self._join_partition, expected_cols=expected_cols, meta=meta
        )
        return dataset
