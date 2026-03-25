# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_curator.stages.math.modifiers.merge_chunks import ChunkMergeStage
from nemo_curator.tasks import DocumentBatch


def _make_batch(df: pd.DataFrame) -> DocumentBatch:
    return DocumentBatch(data=df, task_id="test", dataset_name="test")


class TestChunkMergeStage:
    """Test the ChunkMergeStage class."""

    def test_process_basic_merge(self):
        """Two documents with 3 chunks each merge into 2 rows."""
        df = pd.DataFrame(
            {
                "url": ["a.com"] * 3 + ["b.com"] * 3,
                "chunk_id": [0, 1, 2, 0, 1, 2],
                "cleaned_text": ["A0", "A1", "A2", "B0", "B1", "B2"],
                "text": ["rA0", "rA1", "rA2", "rB0", "rB1", "rB2"],
                "type": ["html"] * 6,
            }
        )
        stage = ChunkMergeStage()
        result = stage.process(_make_batch(df))

        assert len(result.data) == 2
        urls = set(result.data["url"])
        assert urls == {"a.com", "b.com"}

        row_a = result.data[result.data["url"] == "a.com"].iloc[0]
        assert row_a["cleaned_text"] == "A0\nA1\nA2"
        assert row_a["text"] == "rA0\nrA1\nrA2"

    def test_process_chunk_ordering(self):
        """Chunks with out-of-order chunk_ids are sorted correctly before merge."""
        df = pd.DataFrame(
            {
                "url": ["a.com"] * 3,
                "chunk_id": [2, 0, 1],
                "cleaned_text": ["C2", "C0", "C1"],
                "text": ["r2", "r0", "r1"],
            }
        )
        stage = ChunkMergeStage()
        result = stage.process(_make_batch(df))

        assert len(result.data) == 1
        assert result.data["cleaned_text"].iloc[0] == "C0\nC1\nC2"
        assert result.data["text"].iloc[0] == "r0\nr1\nr2"

    def test_process_filter_no_useful_content(self):
        """Chunks with 'NO USEFUL CONTENT' markers are dropped before merge."""
        df = pd.DataFrame(
            {
                "url": ["a.com"] * 3,
                "chunk_id": [0, 1, 2],
                "cleaned_text": ["Good text", "NO USEFUL CONTENT", '"NO USEFUL CONTENT"'],
                "text": ["raw0", "raw1", "raw2"],
            }
        )
        stage = ChunkMergeStage()
        result = stage.process(_make_batch(df))

        assert len(result.data) == 1
        assert result.data["cleaned_text"].iloc[0] == "Good text"

    def test_process_filter_empty_text(self):
        """Null, empty, and newline-only chunks are filtered out."""
        df = pd.DataFrame(
            {
                "url": ["a.com"] * 4,
                "chunk_id": [0, 1, 2, 3],
                "cleaned_text": ["Keep this", None, "", "\n"],
                "text": ["raw0", "raw1", "raw2", "raw3"],
            }
        )
        stage = ChunkMergeStage()
        result = stage.process(_make_batch(df))

        assert len(result.data) == 1
        assert result.data["cleaned_text"].iloc[0] == "Keep this"

    def test_process_dedup_chunks(self):
        """Duplicate (url, chunk_id) rows are deduplicated, keeping first."""
        df = pd.DataFrame(
            {
                "url": ["a.com"] * 4,
                "chunk_id": [0, 1, 1, 2],
                "cleaned_text": ["C0", "C1-first", "C1-dup", "C2"],
                "text": ["r0", "r1-first", "r1-dup", "r2"],
            }
        )
        stage = ChunkMergeStage()
        result = stage.process(_make_batch(df))

        assert len(result.data) == 1
        assert result.data["cleaned_text"].iloc[0] == "C0\nC1-first\nC2"

    def test_process_max_text_length(self):
        """Merged text exceeding max_text_length is dropped."""
        long_text = "x" * 500
        df = pd.DataFrame(
            {
                "url": ["a.com"] * 3 + ["b.com"],
                "chunk_id": [0, 1, 2, 0],
                "cleaned_text": [long_text, long_text, long_text, "short"],
                "text": ["r0", "r1", "r2", "r3"],
            }
        )
        stage = ChunkMergeStage(max_text_length=1000)
        result = stage.process(_make_batch(df))

        assert len(result.data) == 1
        assert result.data["url"].iloc[0] == "b.com"

    def test_process_metadata_preserved(self):
        """Metadata columns (url, finemath_scores, type) survive merge via first()."""
        df = pd.DataFrame(
            {
                "url": ["a.com"] * 2,
                "chunk_id": [0, 1],
                "cleaned_text": ["C0", "C1"],
                "text": ["r0", "r1"],
                "type": ["html", "html"],
                "finemath_scores": [4.5, 4.5],
            }
        )
        stage = ChunkMergeStage()
        result = stage.process(_make_batch(df))

        assert len(result.data) == 1
        row = result.data.iloc[0]
        assert row["type"] == "html"
        assert row["finemath_scores"] == 4.5
        assert row["url"] == "a.com"

    def test_process_all_chunks_filtered(self):
        """If all chunks of a document are invalid, the document is dropped entirely."""
        df = pd.DataFrame(
            {
                "url": ["a.com"] * 3 + ["b.com"],
                "chunk_id": [0, 1, 2, 0],
                "cleaned_text": ["NO USEFUL CONTENT", "", None, "Valid"],
                "text": ["r0", "r1", "r2", "r3"],
            }
        )
        stage = ChunkMergeStage()
        result = stage.process(_make_batch(df))

        assert len(result.data) == 1
        assert result.data["url"].iloc[0] == "b.com"

    def test_process_custom_groupby_columns(self):
        """Custom groupby columns (e.g., warc_filename + url) produce separate documents."""
        df = pd.DataFrame(
            {
                "warc_filename": ["w1", "w1", "w2", "w2"],
                "url": ["a.com", "a.com", "a.com", "a.com"],
                "chunk_id": [0, 1, 0, 1],
                "cleaned_text": ["W1C0", "W1C1", "W2C0", "W2C1"],
                "text": ["r0", "r1", "r2", "r3"],
            }
        )
        stage = ChunkMergeStage(groupby_columns=["warc_filename", "url"])
        result = stage.process(_make_batch(df))

        # Same URL but different warc_filename -> 2 separate documents
        assert len(result.data) == 2

    def test_process_custom_separator(self):
        """Custom separator is used when concatenating text."""
        df = pd.DataFrame(
            {
                "url": ["a.com"] * 2,
                "chunk_id": [0, 1],
                "cleaned_text": ["Part1", "Part2"],
                "text": ["r0", "r1"],
            }
        )
        stage = ChunkMergeStage(separator="\n\n")
        result = stage.process(_make_batch(df))

        assert result.data["cleaned_text"].iloc[0] == "Part1\n\nPart2"

    def test_process_no_raw_text_field(self):
        """When raw_text_field is None, only cleaned_text is concatenated."""
        df = pd.DataFrame(
            {
                "url": ["a.com"] * 2,
                "chunk_id": [0, 1],
                "cleaned_text": ["C0", "C1"],
            }
        )
        stage = ChunkMergeStage(raw_text_field=None)
        result = stage.process(_make_batch(df))

        assert len(result.data) == 1
        assert result.data["cleaned_text"].iloc[0] == "C0\nC1"

    def test_process_token_counts_summed(self):
        """Token count columns (num_generated_tokens, num_input_tokens) are summed, not first()."""
        df = pd.DataFrame(
            {
                "url": ["a.com"] * 3,
                "chunk_id": [0, 1, 2],
                "cleaned_text": ["C0", "C1", "C2"],
                "text": ["r0", "r1", "r2"],
                "num_generated_tokens": [100, 200, 150],
                "num_input_tokens": [50, 80, 70],
                "type": ["html", "html", "html"],
            }
        )
        stage = ChunkMergeStage()
        result = stage.process(_make_batch(df))

        assert len(result.data) == 1
        row = result.data.iloc[0]
        assert row["num_generated_tokens"] == 450
        assert row["num_input_tokens"] == 200
        assert row["type"] == "html"  # metadata still uses first()
