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

from nemo_curator.stages.text.modules.joiner import DocumentJoiner
from nemo_curator.tasks import DocumentBatch


class TestDocumentJoiner:
    def test_basic_join(self):
        """Test basic document joining functionality."""
        # Create test data with split segments
        df = pd.DataFrame(
            {
                "id": [1, 1, 2, 2, 2],
                "segment_id": [0, 1, 0, 1, 2],
                "text": ["Hello", "World", "First", "Second", "Third"],
            }
        )
        batch = DocumentBatch(
            task_id="test_batch",
            dataset_name="test_dataset",
            data=df,
        )

        # Create joiner and process
        joiner = DocumentJoiner(separator="\n\n")
        result = joiner.process(batch)

        # Check results
        result_df = result.to_pandas()
        assert len(result_df) == 2  # Two documents

        # Check first document
        doc1 = result_df[result_df["id"] == 1].iloc[0]
        assert doc1["text"] == "Hello\n\nWorld"
        assert "segment_id" not in result_df.columns  # Should be dropped by default

        # Check second document
        doc2 = result_df[result_df["id"] == 2].iloc[0]
        assert doc2["text"] == "First\n\nSecond\n\nThird"

    def test_custom_separator(self):
        """Test joining with a custom separator."""
        df = pd.DataFrame(
            {
                "id": [1, 1, 1],
                "segment_id": [0, 1, 2],
                "text": ["apple", "banana", "cherry"],
            }
        )
        batch = DocumentBatch(
            task_id="test_batch",
            dataset_name="test_dataset",
            data=df,
        )

        joiner = DocumentJoiner(separator="|")
        result = joiner.process(batch)

        result_df = result.to_pandas()
        assert len(result_df) == 1
        assert result_df.iloc[0]["text"] == "apple|banana|cherry"

    def test_custom_fields(self):
        """Test joining with custom field names."""
        df = pd.DataFrame(
            {
                "doc_id": [1, 1],
                "chunk_id": [0, 1],
                "content": ["Part1", "Part2"],
            }
        )
        batch = DocumentBatch(
            task_id="test_batch",
            dataset_name="test_dataset",
            data=df,
        )

        joiner = DocumentJoiner(
            separator="\n\n",
            text_field="content",
            segment_id_field="chunk_id",
            document_id_field="doc_id",
        )
        result = joiner.process(batch)

        result_df = result.to_pandas()
        assert len(result_df) == 1
        assert result_df.iloc[0]["content"] == "Part1\n\nPart2"

    def test_keep_segment_id_field(self):
        """Test keeping the segment_id field after joining."""
        df = pd.DataFrame(
            {
                "id": [1, 1],
                "segment_id": [0, 1],
                "text": ["A", "B"],
            }
        )
        batch = DocumentBatch(
            task_id="test_batch",
            dataset_name="test_dataset",
            data=df,
        )

        joiner = DocumentJoiner(separator="\n\n", drop_segment_id_field=False)
        result = joiner.process(batch)

        result_df = result.to_pandas()
        assert "segment_id" in result_df.columns
        # When not using max_length, segment_id should be the first occurrence
        assert result_df.iloc[0]["segment_id"] == 0

    def test_max_length_single_segment_exceeds(self):
        """Test max_length when a single segment exceeds the limit."""
        df = pd.DataFrame(
            {
                "id": [1, 1, 1],
                "segment_id": [0, 1, 2],
                "text": ["Short", "VeryLongSegmentThatExceedsLimit", "AlsoShort"],
                "length": [5, 30, 9],
            }
        )
        batch = DocumentBatch(
            task_id="test_batch",
            dataset_name="test_dataset",
            data=df,
        )

        joiner = DocumentJoiner(
            separator="\n\n",
            max_length=20,
            length_field="length",
        )
        result = joiner.process(batch)

        result_df = result.to_pandas()
        # Should create 3 joined documents (each segment on its own)
        assert len(result_df) == 3
        assert result_df.iloc[0]["text"] == "Short"
        assert result_df.iloc[1]["text"] == "VeryLongSegmentThatExceedsLimit"
        assert result_df.iloc[2]["text"] == "AlsoShort"

    def test_multiple_documents(self):
        """Test joining multiple documents in one batch."""
        df = pd.DataFrame(
            {
                "id": [1, 1, 2, 2, 3, 3, 3],
                "segment_id": [0, 1, 0, 1, 0, 1, 2],
                "text": ["A", "B", "C", "D", "E", "F", "G"],
            }
        )
        batch = DocumentBatch(
            task_id="test_batch",
            dataset_name="test_dataset",
            data=df,
        )

        joiner = DocumentJoiner(separator=" ")
        result = joiner.process(batch)

        result_df = result.to_pandas().sort_values("id").reset_index(drop=True)
        assert len(result_df) == 3
        assert result_df.iloc[0]["text"] == "A B"
        assert result_df.iloc[1]["text"] == "C D"
        assert result_df.iloc[2]["text"] == "E F G"

    def test_preserve_additional_columns(self):
        """Test that additional columns are preserved during joining."""
        df = pd.DataFrame(
            {
                "id": [1, 1, 2, 2],
                "segment_id": [0, 1, 0, 1],
                "text": ["Hello", "World", "Foo", "Bar"],
                "author": ["John", "John", "Jane", "Jane"],
                "date": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"],
            }
        )
        batch = DocumentBatch(
            task_id="test_batch",
            dataset_name="test_dataset",
            data=df,
        )

        joiner = DocumentJoiner(separator="\n\n")
        result = joiner.process(batch)

        result_df = result.to_pandas().sort_values("id").reset_index(drop=True)
        assert len(result_df) == 2
        assert "author" in result_df.columns
        assert "date" in result_df.columns
        assert result_df.iloc[0]["author"] == "John"
        assert result_df.iloc[0]["date"] == "2023-01-01"
        assert result_df.iloc[1]["author"] == "Jane"
        assert result_df.iloc[1]["date"] == "2023-01-02"

    def test_out_of_order_segments(self):
        """Test that segments are correctly ordered even if input is out of order."""
        df = pd.DataFrame(
            {
                "id": [1, 1, 1],
                "segment_id": [2, 0, 1],  # Out of order
                "text": ["Third", "First", "Second"],
            }
        )
        batch = DocumentBatch(
            task_id="test_batch",
            dataset_name="test_dataset",
            data=df,
        )

        joiner = DocumentJoiner(separator=" ")
        result = joiner.process(batch)

        result_df = result.to_pandas()
        # Should be joined in the correct order
        assert result_df.iloc[0]["text"] == "First Second Third"

    def test_empty_batch(self):
        """Test handling of empty batch."""
        df = pd.DataFrame(columns=["id", "segment_id", "text"])
        batch = DocumentBatch(
            task_id="test_batch",
            dataset_name="test_dataset",
            data=df,
        )

        joiner = DocumentJoiner(separator="\n\n")
        result = joiner.process(batch)

        assert result.to_pandas().empty

    def test_single_segment_document(self):
        """Test document with only one segment."""
        df = pd.DataFrame(
            {
                "id": [1],
                "segment_id": [0],
                "text": ["OnlySegment"],
            }
        )
        batch = DocumentBatch(
            task_id="test_batch",
            dataset_name="test_dataset",
            data=df,
        )

        joiner = DocumentJoiner(separator="\n\n")
        result = joiner.process(batch)

        result_df = result.to_pandas()
        assert len(result_df) == 1
        assert result_df.iloc[0]["text"] == "OnlySegment"

    def test_validation_errors(self):
        """Test validation of max_length and length_field parameters."""
        with pytest.raises(ValueError, match="max_length is specified but length_field is not"):
            DocumentJoiner(separator="\n\n", max_length=100)

        with pytest.raises(ValueError, match="length_field is specified but max_length is not"):
            DocumentJoiner(separator="\n\n", length_field="length")

    def test_inputs_outputs(self):
        """Test the inputs and outputs specification."""
        joiner = DocumentJoiner(
            separator="\n\n",
            text_field="content",
            segment_id_field="chunk_id",
            document_id_field="doc_id",
        )

        top_level, data_attrs = joiner.inputs()
        assert "data" in top_level
        assert "content" in data_attrs
        assert "chunk_id" in data_attrs
        assert "doc_id" in data_attrs

        top_level, data_attrs = joiner.outputs()
        assert "data" in top_level
        assert "content" in data_attrs
        assert "doc_id" in data_attrs
        # segment_id should not be in outputs if drop_segment_id_field=True (default)
        assert "chunk_id" not in data_attrs

    def test_inputs_outputs_with_length(self):
        """Test the inputs and outputs specification with length field."""
        joiner = DocumentJoiner(
            separator="\n\n",
            max_length=100,
            length_field="length",
        )

        _top_level, data_attrs = joiner.inputs()
        assert "length" in data_attrs

        _top_level, data_attrs = joiner.outputs()
        assert "length" in data_attrs

    def test_metadata_preservation(self):
        """Test that metadata is preserved through the join."""
        df = pd.DataFrame(
            {
                "id": [1, 1],
                "segment_id": [0, 1],
                "text": ["Part1", "Part2"],
            }
        )
        batch = DocumentBatch(
            task_id="test_batch",
            dataset_name="test_dataset",
            data=df,
            _metadata={"source": "test_source", "version": "1.0"},
        )

        joiner = DocumentJoiner(separator="\n\n")
        result = joiner.process(batch)

        assert result._metadata == {"source": "test_source", "version": "1.0"}

    def test_roundtrip_with_splitter(self):
        """Test that DocumentJoiner can reverse DocumentSplitter."""
        from nemo_curator.stages.text.modules.splitter import DocumentSplitter

        # Original data
        original_df = pd.DataFrame(
            {
                "doc_id": ["doc1", "doc2"],
                "text": ["Hello\n\nWorld", "Foo\n\nBar\n\nBaz"],
                "author": ["Alice", "Bob"],
            }
        )
        original_batch = DocumentBatch(
            task_id="test_batch",
            dataset_name="test_dataset",
            data=original_df,
        )

        # Split
        splitter = DocumentSplitter(separator="\n\n", text_field="text")
        split_batch = splitter.process(original_batch)

        # Join back
        joiner = DocumentJoiner(
            separator="\n\n",
            text_field="text",
            segment_id_field="segment_id",
            document_id_field="doc_id",
        )
        joined_batch = joiner.process(split_batch)

        # Compare
        joined_df = joined_batch.to_pandas().sort_values("doc_id").reset_index(drop=True)
        original_sorted = original_df.sort_values("doc_id").reset_index(drop=True)

        assert len(joined_df) == len(original_sorted)
        assert list(joined_df["text"]) == list(original_sorted["text"])
        assert list(joined_df["author"]) == list(original_sorted["author"])
