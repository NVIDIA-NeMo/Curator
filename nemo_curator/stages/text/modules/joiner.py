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

from dataclasses import dataclass

import pandas as pd

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch


@dataclass
class DocumentJoiner(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    Joins documents that have a common id back into a single document.
    The order of the documents is dictated by an additional segment_id column.
    A maximum length can be specified to limit the size of the joined documents.

    The joined documents are joined by a separator.

    This stage performs the inverse operation of DocumentSplitter, allowing you
    to reconstruct documents from their segments.

    Important:
        This stage assumes that all segments belonging to the same document are
        contained within a single DocumentBatch. Segments from the same document
        split across multiple batches will NOT be joined together. Ensure your
        batching logic keeps all segments of a document together.

    Example:
        If you have segments with document_id=1, segment_id=[0,1] and text=["Hello", "World"],
        they will be joined into a single row with document_id=1 and text="Hello\\n\\nWorld"
        (assuming separator="\\n\\n").

    Args:
        separator (str): The separator to join the documents on.
        text_field (str): The name of the column containing the text to join.
            Defaults to "text".
        segment_id_field (str): The name of the column containing the segment id.
            Defaults to "segment_id".
        document_id_field (str): The name of the column containing the document id.
            Defaults to "id".
        drop_segment_id_field (bool): Whether to drop the segment_id_field after joining.
            Defaults to True.
        max_length (int, optional): The maximum length of the joined documents.
            Both max_length and length_field must be specified or neither can be specified.
        length_field (str, optional): The name of the column containing the length of the documents.
            Both max_length and length_field must be specified or neither can be specified.
    """

    separator: str = "\n\n"
    text_field: str = "text"
    segment_id_field: str = "segment_id"
    document_id_field: str = "id"
    drop_segment_id_field: bool = True
    max_length: int | None = None
    length_field: str | None = None
    name: str = "document_joiner"

    def __post_init__(self):
        if self.max_length is not None and self.length_field is None:
            msg = "max_length is specified but length_field is not"
            raise ValueError(msg)
        if self.max_length is None and self.length_field is not None:
            msg = "length_field is specified but max_length is not"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        """Define stage input requirements."""
        required_cols = [self.text_field, self.segment_id_field, self.document_id_field]
        if self.length_field is not None:
            required_cols.append(self.length_field)
        return ["data"], required_cols

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define stage output specification."""
        output_cols = [self.text_field, self.document_id_field]
        if not self.drop_segment_id_field:
            output_cols.append(self.segment_id_field)
        if self.length_field is not None:
            output_cols.append(self.length_field)
        return ["data"], output_cols

    def _join_segments(self, group: pd.DataFrame) -> pd.DataFrame:
        """Join segments with max_length constraint."""
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
                accumulator_row = row.copy()
            else:
                # Calculate what the new length would be if we joined this segment.
                proposed_length = accumulator_length + row[self.length_field] + len(self.separator)
                if proposed_length <= self.max_length:
                    accumulator_text = accumulator_text + self.separator + row[self.text_field]
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
                    accumulator_row = row.copy()

        # Commit the last accumulated segment.
        if accumulator_row is not None:
            new_row = accumulator_row.copy()
            new_row[self.text_field] = accumulator_text
            new_row[self.length_field] = accumulator_length
            new_row[self.segment_id_field] = current_seg_id
            joined_rows.append(new_row)

        if joined_rows:
            return pd.DataFrame(joined_rows)
        else:
            return pd.DataFrame(columns=group.columns)

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """
        Joins the documents back into a single document while preserving all the original fields.

        Args:
            batch (DocumentBatch): Input batch to process

        Returns:
            DocumentBatch: Batch with documents joined by document_id
        """
        df = batch.to_pandas()

        if df.empty:
            return batch

        if self.max_length is None:
            # Sort the segments by the segment_id_field to maintain proper order before aggregating.
            df_sorted = df.sort_values(self.segment_id_field)

            # Build aggregation functions to preserve all original columns:
            # - For self.text_field, join all segments using the separator.
            # - For all other columns (except self.document_id_field, which is our grouping key), take the first occurrence.
            agg_funcs = {}
            for col in df_sorted.columns:
                if col == self.text_field:
                    agg_funcs[col] = lambda texts: self.separator.join(texts.astype(str))
                elif col != self.document_id_field:
                    agg_funcs[col] = "first"

            # Group by document_id_field while keeping the key as a column.
            joined = df_sorted.groupby(self.document_id_field, as_index=False).agg(agg_funcs)
        else:
            # Use the more complex joining logic with max_length constraint
            joined_groups = []
            for _doc_id, group in df.groupby(self.document_id_field):
                joined_group = self._join_segments(group)
                joined_groups.append(joined_group)

            joined = pd.concat(joined_groups, ignore_index=True) if joined_groups else pd.DataFrame(columns=df.columns)

        if self.drop_segment_id_field and self.segment_id_field in joined.columns:
            joined = joined.drop(columns=self.segment_id_field)

        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=joined,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

