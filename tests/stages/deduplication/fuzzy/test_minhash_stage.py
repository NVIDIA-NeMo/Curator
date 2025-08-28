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

# ruff: noqa: E402
import os
from pathlib import Path

import pandas as pd
import pytest

cudf = pytest.importorskip("cudf", reason="MinHashStage tests require cudf")

from nemo_curator.stages.deduplication.fuzzy.minhash import MinHashStage
from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from nemo_curator.tasks import FileGroupTask


@pytest.fixture
def sample_data_with_duplicates() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create sample data that includes duplicates for testing."""
    # First dataset with some duplicates
    data1 = pd.DataFrame(
        {
            "text": [
                "The quick brown fox jumps over the lazy dog",  # Will appear again in data2
                "A test string for deduplication",
                "Another test string that is similar",
                "This is an exact duplicate",  # Will appear again in this file
                "This is an exact duplicate",  # Duplicate
            ],
            "content": [  # Alternative column name for testing
                "The quick brown fox jumps over the lazy dog",
                "A test string for deduplication",
                "Another test string that is similar",
                "This is an exact duplicate",
                "This is an exact duplicate",
            ],
            "meta": ["doc1", "doc2", "doc3", "doc4", "doc5"],
        }
    )

    # Second dataset with some duplicates from first
    data2 = pd.DataFrame(
        {
            "text": [
                "The quick brown fox jumps over the lazy dog",  # Duplicate from data1
                "A different test string",
                "Completely different content here",
                "Yet another unique document",
            ],
            "content": [
                "The quick brown fox jumps over the lazy dog",
                "A different test string",
                "Completely different content here",
                "Yet another unique document",
            ],
            "meta": ["doc6", "doc7", "doc8", "doc9"],
        }
    )

    return data1, data2


@pytest.fixture
def sample_files(
    tmp_path: Path, sample_data_with_duplicates: tuple[pd.DataFrame, pd.DataFrame], request: "pytest.FixtureRequest"
) -> tuple[list[str], str]:
    """Create sample files in the requested format."""
    data1, data2 = sample_data_with_duplicates
    format_type = request.param  # Will be either 'jsonl' or 'parquet'

    if format_type == "jsonl":
        file1 = tmp_path / "data1.jsonl"
        file2 = tmp_path / "data2.jsonl"
        data1.to_json(file1, orient="records", lines=True)
        data2.to_json(file2, orient="records", lines=True)
    else:  # parquet
        file1 = tmp_path / "data1.parquet"
        file2 = tmp_path / "data2.parquet"
        data1.to_parquet(file1)
        data2.to_parquet(file2)

    return [str(file1), str(file2)], format_type


@pytest.fixture
def input_task(sample_files: tuple[list[str], str]) -> FileGroupTask:
    """Create a FileGroupTask from sample files."""
    files, format_type = sample_files
    return FileGroupTask(
        task_id=f"test_task_{format_type}",
        dataset_name="test_dataset",
        data=files,
        _metadata={"batch_id": 0, "total_batches": 1, "format": format_type},
    )


@pytest.mark.gpu
class TestMinHashStage:
    """Test suite for MinHashStage ProcessingStage."""

    @pytest.mark.parametrize("sample_files", ["jsonl", "parquet"], indirect=True)
    @pytest.mark.parametrize("use_64bit_hash", [False, True])
    @pytest.mark.parametrize(
        ("num_hashes", "char_ngrams", "text_field"),
        [
            (64, 3, "text"),
            (128, 5, "text"),
            (256, 10, "content"),  # Test alternative column name
        ],
    )
    @pytest.mark.usefixtures("ray_client_with_id_generator")
    def test_minhash_processing(  # noqa: PLR0913
        self,
        input_task: FileGroupTask,
        tmp_path: Path,
        use_64bit_hash: bool,
        num_hashes: int,
        char_ngrams: int,
        text_field: str,
    ) -> None:
        """Test minhash processing with various parameter combinations."""
        # Get format from task metadata
        read_format = input_task._metadata["format"]

        # Create stage
        stage = MinHashStage(
            output_path=str(tmp_path / f"output_{read_format}_{use_64bit_hash}_{num_hashes}"),
            text_field=text_field,
            minhash_field="_minhash_signature",
            char_ngrams=char_ngrams,
            num_hashes=num_hashes,
            seed=42,
            use_64bit_hash=use_64bit_hash,
            read_format=read_format,
            pool=False,
        )

        # Setup and process
        stage.setup()
        output_task = stage.process(input_task)

        # Verify output task structure
        assert isinstance(output_task, FileGroupTask)
        assert len(output_task.data) == 1
        assert output_task._metadata["minhash_field"] == "_minhash_signature"
        assert output_task._metadata["num_hashes"] == num_hashes

        # Verify output file exists
        output_file = output_task.data[0]
        assert os.path.exists(output_file)

        # Read and verify the output
        result_df = cudf.read_parquet(output_file)

        # Check required columns exist
        assert CURATOR_DEDUP_ID_STR in result_df.columns
        assert "_minhash_signature" in result_df.columns

        assert len(result_df) == 9

        # Verify minhash signatures have correct length
        sig_lengths = result_df["_minhash_signature"].list.len()
        assert (sig_lengths == num_hashes).all()

        # Verify IDs are unique
        ids = result_df[CURATOR_DEDUP_ID_STR].to_pandas()
        assert len(ids) == len(set(ids))

        # Get minhashes for duplicate detection test
        minhashes = result_df["_minhash_signature"].to_pandas().tolist()

        # Test duplicate detection:
        # Documents at indices 3 and 4 in first file are exact duplicates
        # Document at index 0 in first file is duplicate of index 0 in second file
        # (In combined output: indices 3,4 are duplicates and 0,5 are duplicates)
        assert minhashes[3] == minhashes[4], "Exact duplicates should have identical minhashes"
        assert minhashes[0] == minhashes[5], "Cross-file duplicates should have identical minhashes"

        # Verify different texts have different minhashes
        assert minhashes[0] != minhashes[1], "Different texts should have different minhashes"
        assert minhashes[1] != minhashes[2], "Different texts should have different minhashes"

        # Verify hash value ranges
        assert (
            result_df["_minhash_signature"].dtype == cudf.core.dtypes.ListDtype("uint64")
            if use_64bit_hash
            else cudf.core.dtypes.ListDtype("uint32")
        )

    @pytest.mark.usefixtures("ray_client_with_id_generator")
    def test_error_handling_missing_column(self, tmp_path: Path) -> None:
        """Test error handling when text column is missing."""
        # Create data without the expected column
        data = pd.DataFrame({"wrong_column": ["text1", "text2"], "meta": ["a", "b"]})

        input_file = tmp_path / "bad_schema.jsonl"
        data.to_json(input_file, orient="records", lines=True)

        input_task = FileGroupTask(
            task_id="bad_test", dataset_name="bad_dataset", data=[str(input_file)], _metadata={}
        )

        stage = MinHashStage(
            output_path=str(tmp_path / "output"),
            text_field="text",  # This column doesn't exist
            pool=False,
        )

        stage.setup()

        # Should raise KeyError for missing column
        with pytest.raises(KeyError):
            stage.process(input_task)

    @pytest.mark.usefixtures("ray_client_with_id_generator")
    def test_empty_input_handling(self, tmp_path: Path) -> None:
        """Test handling of empty input files."""
        # Create empty dataframe
        data = pd.DataFrame({"text": []})

        input_file = tmp_path / "empty.jsonl"
        data.to_json(input_file, orient="records", lines=True)

        input_task = FileGroupTask(
            task_id="empty_test", dataset_name="empty_dataset", data=[str(input_file)], _metadata={}
        )

        stage = MinHashStage(
            output_path=str(tmp_path / "output"),
            text_field="text",
            pool=False,
        )

        stage.setup()
        with pytest.raises(KeyError):
            stage.process(input_task)

    def test_process_without_setup(self, tmp_path: Path) -> None:
        """Test that process raises error if setup wasn't called."""
        stage = MinHashStage(
            output_path=str(tmp_path),
            text_field="text",
        )

        input_task = FileGroupTask(
            task_id="test_task", dataset_name="test_dataset", data=["dummy.jsonl"], _metadata={}
        )

        # Should raise error because setup wasn't called
        with pytest.raises(RuntimeError, match="MinHash processor or ID generator not initialized"):
            stage.process(input_task)

    @pytest.mark.usefixtures("ray_client_with_id_generator")
    def test_large_text_handling(self, tmp_path: Path) -> None:
        """Test handling of large text documents."""
        # Create data with varying text sizes
        data = pd.DataFrame(
            {
                "text": [
                    "short text",
                    "medium " * 100,  # ~700 chars
                    "long " * 1000,  # ~5000 chars
                    "very " * 5000,  # ~25000 chars
                ]
            }
        )

        input_file = tmp_path / "large_texts.jsonl"
        data.to_json(input_file, orient="records", lines=True)

        input_task = FileGroupTask(
            task_id="large_test", dataset_name="large_dataset", data=[str(input_file)], _metadata={}
        )

        stage = MinHashStage(
            output_path=str(tmp_path / "output"),
            text_field="text",
            num_hashes=128,
            char_ngrams=5,
            pool=False,
        )

        stage.setup()
        output_task = stage.process(input_task)

        # Verify all documents were processed
        result_df = cudf.read_parquet(output_task.data[0])
        assert len(result_df) == 4

        # All should have valid minhashes regardless of text size
        sig_lengths = result_df["_minhash_signature"].list.len()
        assert (sig_lengths == 128).all()

    @pytest.mark.usefixtures("ray_client_with_id_generator")
    def test_special_characters_and_unicode(self, tmp_path: Path) -> None:
        """Test handling of special characters and unicode text."""
        # Create data with special characters and unicode
        data = pd.DataFrame(
            {
                "text": [
                    "Hello, world! 123 #test @user",  # ASCII with special chars
                    "Привет мир! 测试文本",  # Russian and Chinese
                    "🚀 Emoji test 🎉 with symbols ♠♣♥♦",  # Emojis and symbols
                    "Mixed: café, naïve, résumé",  # Accented characters
                    "\n\t  Whitespace  \r\n  test  ",  # Various whitespace
                ]
            }
        )

        input_file = tmp_path / "special_chars.jsonl"
        data.to_json(input_file, orient="records", lines=True)

        input_task = FileGroupTask(
            task_id="special_test", dataset_name="special_dataset", data=[str(input_file)], _metadata={}
        )

        stage = MinHashStage(
            output_path=str(tmp_path / "output"),
            text_field="text",
            num_hashes=64,
            char_ngrams=3,
            pool=False,
        )

        stage.setup()
        output_task = stage.process(input_task)

        # Verify all documents were processed
        result_df = cudf.read_parquet(output_task.data[0])
        assert len(result_df) == 5

        # All should have valid minhashes
        sig_lengths = result_df["_minhash_signature"].list.len()
        assert (sig_lengths == 64).all()

        # Different texts should produce different minhashes
        minhashes = result_df["_minhash_signature"].to_pandas().tolist()
        # Check that all minhashes are different (no duplicates in this test)
        for i in range(len(minhashes)):
            for j in range(i + 1, len(minhashes)):
                assert minhashes[i] != minhashes[j], (
                    f"Different texts at indices {i} and {j} should have different minhashes"
                )

    @pytest.mark.usefixtures("ray_client_with_id_generator")
    def test_setup_idempotency(self, tmp_path: Path) -> None:
        """Test that calling setup multiple times doesn't cause issues and IDs continue from where they left off."""
        # Create first stage
        stage1 = MinHashStage(
            output_path=str(tmp_path / "output1"),
            text_field="text",
            pool=False,
        )

        input_file1 = tmp_path / "test1.jsonl"
        input_file2 = tmp_path / "test2.jsonl"
        data = pd.DataFrame({"text": ["Document 1", "Document 2", "Document 3"]})
        data.to_json(input_file1, orient="records", lines=True)
        data.to_json(input_file2, orient="records", lines=True)
        input_task1 = FileGroupTask(
            task_id="setup_test_1", dataset_name="setup_dataset_1", data=[str(input_file1)], _metadata={}
        )
        input_task2 = FileGroupTask(
            task_id="setup_test_2", dataset_name="setup_dataset_2", data=[str(input_file2)], _metadata={}
        )

        # Setup and process first batch
        stage1.setup()
        first_id_generator = stage1.id_generator
        output_task1 = stage1.process(input_task1)

        # Read first batch results and get IDs
        result_df1 = cudf.read_parquet(output_task1.data[0])
        ids_batch1 = sorted(result_df1[CURATOR_DEDUP_ID_STR].to_pandas().tolist())
        assert len(ids_batch1) == 3

        # Create second stage (different instance)
        stage2 = MinHashStage(
            output_path=str(tmp_path / "output2"),
            text_field="text",
            pool=False,
        )

        # Setup second stage - should reuse the same ID generator actor
        stage2.setup()
        second_id_generator = stage2.id_generator
        output_task2 = stage2.process(input_task2)

        # ID generators should be the same Ray actor
        assert first_id_generator == second_id_generator

        # Read second batch results and get IDs
        result_df2 = cudf.read_parquet(output_task2.data[0])
        ids_batch2 = sorted(result_df2[CURATOR_DEDUP_ID_STR].to_pandas().tolist())
        assert len(ids_batch2) == 3

        # Verify IDs continued from where batch 1 left off
        # IDs should be sequential integers
        assert ids_batch1 == [0, 1, 2]
        assert ids_batch2 == [3, 4, 5]
