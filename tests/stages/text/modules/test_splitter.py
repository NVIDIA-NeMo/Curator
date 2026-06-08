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

from nemo_curator.stages.text.modules.splitter import DocumentSplitter
from nemo_curator.tasks import DocumentBatch


class TestDocumentSplitter:
    def test_basic_split(self):
        """Test basic document splitting functionality."""
        # Create test data
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "text": ["Hello\n\nWorld", "First\n\nSecond\n\nThird"],
            }
        )
        batch = DocumentBatch(
            task_id="test_batch",
            dataset_name="test_dataset",
            data=df,
        )

        # Create splitter and process
        splitter = DocumentSplitter(separator="\n\n")
        result = splitter.process(batch)

        # Check results
        result_df = result.to_pandas()
        assert len(result_df) == 5  # 2 + 3 segments
        assert "segment_id" in result_df.columns

        # Check first document segments
        doc1_segments = result_df[result_df["id"] == 1].reset_index(drop=True)
        assert len(doc1_segments) == 2
        assert doc1_segments.loc[0, "text"] == "Hello"
        assert doc1_segments.loc[0, "segment_id"] == 0
        assert doc1_segments.loc[1, "text"] == "World"
        assert doc1_segments.loc[1, "segment_id"] == 1

        # Check second document segments
        doc2_segments = result_df[result_df["id"] == 2].reset_index(drop=True)
        assert len(doc2_segments) == 3
        assert doc2_segments.loc[0, "text"] == "First"
        assert doc2_segments.loc[0, "segment_id"] == 0
        assert doc2_segments.loc[1, "text"] == "Second"
        assert doc2_segments.loc[1, "segment_id"] == 1
        assert doc2_segments.loc[2, "text"] == "Third"
        assert doc2_segments.loc[2, "segment_id"] == 2

    def test_custom_separator(self):
        """Test splitting with a custom separator."""
        df = pd.DataFrame(
            {
                "text": ["apple|banana|cherry"],
            }
        )
        batch = DocumentBatch(
            task_id="test_batch",
            dataset_name="test_dataset",
            data=df,
        )

        splitter = DocumentSplitter(separator="|")
        result = splitter.process(batch)

        result_df = result.to_pandas()
        assert len(result_df) == 3
        assert result_df.loc[0, "text"] == "apple"
        assert result_df.loc[1, "text"] == "banana"
        assert result_df.loc[2, "text"] == "cherry"

    def test_custom_text_field(self):
        """Test splitting with a custom text field."""
        df = pd.DataFrame(
            {
                "content": ["Part1\n\nPart2"],
                "metadata": ["some_metadata"],
            }
        )
        batch = DocumentBatch(
            task_id="test_batch",
            dataset_name="test_dataset",
            data=df,
        )

        splitter = DocumentSplitter(separator="\n\n", text_field="content")
        result = splitter.process(batch)

        result_df = result.to_pandas()
        assert len(result_df) == 2
        assert result_df.loc[0, "content"] == "Part1"
        assert result_df.loc[1, "content"] == "Part2"
        assert result_df.loc[0, "metadata"] == "some_metadata"
        assert result_df.loc[1, "metadata"] == "some_metadata"

    def test_custom_segment_id_field(self):
        """Test splitting with a custom segment ID field name."""
        df = pd.DataFrame(
            {
                "text": ["A\n\nB"],
            }
        )
        batch = DocumentBatch(
            task_id="test_batch",
            dataset_name="test_dataset",
            data=df,
        )

        splitter = DocumentSplitter(separator="\n\n", segment_id_field="chunk_id")
        result = splitter.process(batch)

        result_df = result.to_pandas()
        assert "chunk_id" in result_df.columns
        assert "segment_id" not in result_df.columns
        assert result_df.loc[0, "chunk_id"] == 0
        assert result_df.loc[1, "chunk_id"] == 1

    def test_no_split_needed(self):
        """Test documents that don't contain the separator."""
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "text": ["NoSeparatorHere", "AlsoNoSeparator"],
            }
        )
        batch = DocumentBatch(
            task_id="test_batch",
            dataset_name="test_dataset",
            data=df,
        )

        splitter = DocumentSplitter(separator="\n\n")
        result = splitter.process(batch)

        result_df = result.to_pandas()
        assert len(result_df) == 2  # Same as input
        assert all(result_df["segment_id"] == 0)  # All are segment 0

    def test_empty_segments(self):
        """Test handling of empty segments from consecutive separators."""
        df = pd.DataFrame(
            {
                "text": ["A\n\n\n\nB"],  # Double separator creates empty segment
            }
        )
        batch = DocumentBatch(
            task_id="test_batch",
            dataset_name="test_dataset",
            data=df,
        )

        splitter = DocumentSplitter(separator="\n\n")
        result = splitter.process(batch)

        result_df = result.to_pandas()
        # This will create 3 segments: "A", "", "B"
        assert len(result_df) == 3
        assert result_df.loc[0, "text"] == "A"
        assert result_df.loc[1, "text"] == ""
        assert result_df.loc[2, "text"] == "B"

    def test_metadata_preservation(self):
        """Test that metadata is preserved through the split."""
        df = pd.DataFrame(
            {
                "id": [1],
                "text": ["Part1\n\nPart2"],
                "author": ["John Doe"],
            }
        )
        batch = DocumentBatch(
            task_id="test_batch",
            dataset_name="test_dataset",
            data=df,
            _metadata={"source": "test_source"},
        )

        splitter = DocumentSplitter(separator="\n\n")
        result = splitter.process(batch)

        # Check metadata is preserved
        assert result._metadata == {"source": "test_source"}

        # Check all columns are preserved
        result_df = result.to_pandas()
        assert "author" in result_df.columns
        assert all(result_df["author"] == "John Doe")

    def test_inputs_outputs(self):
        """Test the inputs and outputs specification."""
        splitter = DocumentSplitter(separator="\n\n", text_field="content", segment_id_field="chunk_id")

        top_level, data_attrs = splitter.inputs()
        assert "data" in top_level
        assert "content" in data_attrs

        top_level, data_attrs = splitter.outputs()
        assert "data" in top_level
        assert "content" in data_attrs
        assert "chunk_id" in data_attrs

    def test_validate_input(self):
        """Test input validation."""
        df = pd.DataFrame(
            {
                "text": ["Hello World"],
            }
        )
        batch = DocumentBatch(
            task_id="test_batch",
            dataset_name="test_dataset",
            data=df,
        )

        splitter = DocumentSplitter(separator="\n\n")
        assert splitter.validate_input(batch) is True

        # Test with missing text field
        df_missing = pd.DataFrame(
            {
                "other_field": ["Hello"],
            }
        )
        batch_missing = DocumentBatch(
            task_id="test_batch",
            dataset_name="test_dataset",
            data=df_missing,
        )
        assert splitter.validate_input(batch_missing) is False

    def test_reconstruction_with_unique_ids(self):
        """Test that documents can be reconstructed using unique IDs."""
        df = pd.DataFrame(
            {
                "doc_id": ["doc1", "doc2"],
                "text": ["Hello\n\nWorld", "Foo\n\nBar"],
            }
        )
        batch = DocumentBatch(
            task_id="test_batch",
            dataset_name="test_dataset",
            data=df,
        )

        splitter = DocumentSplitter(separator="\n\n")
        result = splitter.process(batch)

        result_df = result.to_pandas()

        # Reconstruct first document
        doc1_segments = result_df[result_df["doc_id"] == "doc1"].sort_values("segment_id")
        reconstructed_doc1 = "\n\n".join(doc1_segments["text"].tolist())
        assert reconstructed_doc1 == "Hello\n\nWorld"

        # Reconstruct second document
        doc2_segments = result_df[result_df["doc_id"] == "doc2"].sort_values("segment_id")
        reconstructed_doc2 = "\n\n".join(doc2_segments["text"].tolist())
        assert reconstructed_doc2 == "Foo\n\nBar"
