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

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch


@dataclass
class DocumentSplitter(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    Splits documents into segments based on a separator.
    Each segment becomes a new row within the batch with an additional column
    indicating the segment id.

    To restore the original document, ensure that each document
    has a unique id prior to splitting.

    Example:
        If a document has text="Hello\\n\\nWorld", and separator="\\n\\n",
        it will be split into two rows: one with text="Hello" and segment_id=0,
        and another with text="World" and segment_id=1.

    Args:
        separator (str): The separator to split the documents on.
        text_field (str): The name of the column containing the text to split.
            Defaults to "text".
        segment_id_field (str): The name of the column to add to indicate the segment id.
            Defaults to "segment_id".
    """

    separator: str
    text_field: str = "text"
    segment_id_field: str = "segment_id"
    name: str = "document_splitter"

    def inputs(self) -> tuple[list[str], list[str]]:
        """Define stage input requirements."""
        return ["data"], [self.text_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define stage output specification."""
        return ["data"], [self.text_field, self.segment_id_field]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """
        Splits the documents into segments based on the separator and
        adds a column indicating the segment id.

        Args:
            batch (DocumentBatch): Input batch to process

        Returns:
            DocumentBatch: Batch with documents split into segments
        """
        df = batch.to_pandas()

        # Split the text field into segments using the separator.
        df["_split_text"] = df[self.text_field].str.split(self.separator)

        # Explode the list so that each segment becomes a separate row.
        # The index is preserved and duplicated for each segment from the same document
        df = df.explode("_split_text")

        # For each original document (grouped by index level 0), assign a segment id.
        # level=0 refers to the (duplicated) index after explode
        df[self.segment_id_field] = df.groupby(level=0).cumcount()

        # Replace the original text field with the split segment.
        df[self.text_field] = df["_split_text"]

        # Drop the temporary column and reset index to sequential
        df = df.drop(columns=["_split_text"]).reset_index(drop=True)

        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

