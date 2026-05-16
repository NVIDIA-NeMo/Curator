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

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from nemo_curator.stages.text.deduplication.removal import TextDuplicatesRemovalStage
from nemo_curator.tasks import DocumentBatch


class TestTextDuplicatesRemovalStage:
    @pytest.fixture
    def sample_document_batch(self) -> DocumentBatch:
        """Create a sample DocumentBatch for testing."""
        df = pd.DataFrame(
            {
                "text": [
                    "First document text",
                    "Second document text",
                    "Third document text",
                    "Fourth document text",
                    "Fifth document text",
                ],
                CURATOR_DEDUP_ID_STR: [1, 2, 3, 4, 5],
            }
        )
        return DocumentBatch(
            task_id="test_batch",
            dataset_name="test_dataset",
            data=df,
            _metadata={"source": "test"},
        )

    @pytest.fixture
    def removal_parquet_file(self, tmp_path: Path) -> str:
        """Create a temporary parquet file with IDs to remove."""
        removal_df = pd.DataFrame({"id": [2, 4, 6]})  # Remove IDs 2 and 4 (6 doesn't exist)
        removal_file = tmp_path / "removal_ids.parquet"
        removal_df.to_parquet(removal_file)
        return str(removal_file)

    @pytest.fixture
    def empty_removal_parquet_file(self, tmp_path: Path) -> str:
        """Create a temporary parquet file with no IDs to remove."""
        removal_df = pd.DataFrame({"id": []})  # Empty removal list
        removal_file = tmp_path / "empty_removal_ids.parquet"
        removal_df.to_parquet(removal_file)
        return str(removal_file)

    @patch("pandas.read_parquet")
    def test_process_removes_duplicates(
        self, mock_read_parquet: MagicMock, sample_document_batch: DocumentBatch, removal_parquet_file: str
    ):
        """Test that process method removes documents with IDs in the removal list."""
        # Mock the parquet read to return IDs 2 and 4
        mock_read_parquet.return_value = pd.DataFrame({"id": [2, 4]})

        stage = TextDuplicatesRemovalStage(ids_to_remove_path=removal_parquet_file)
        result = stage.process(sample_document_batch)

        # Verify mock was called with correct parameters
        mock_read_parquet.assert_called_once()
        call_args = mock_read_parquet.call_args
        assert call_args[0][0] == removal_parquet_file
        assert call_args[1]["filters"] == [("id", ">=", 1), ("id", "<=", 5)]
        assert call_args[1]["columns"] == ["id"]

        # Verify result
        assert isinstance(result, DocumentBatch)
        assert result.task_id == "removal_test_batch"
        assert result.dataset_name == "test_dataset"

        result_df = result.to_pandas()
        # Should have 3 documents remaining (IDs 1, 3, 5)
        assert len(result_df) == 3
        assert sorted(result_df[CURATOR_DEDUP_ID_STR].tolist()) == [1, 3, 5]
        assert result._metadata["num_removed"] == 2

    @patch("pandas.read_parquet")
    def test_process_with_empty_removal_list(
        self, mock_read_parquet: MagicMock, sample_document_batch: DocumentBatch, removal_parquet_file: str
    ):
        """Test process method with empty removal list."""
        # Mock the parquet read to return empty DataFrame
        mock_read_parquet.return_value = pd.DataFrame({"id": []})

        stage = TextDuplicatesRemovalStage(ids_to_remove_path=removal_parquet_file)
        result = stage.process(sample_document_batch)

        result_df = result.to_pandas()
        # Should have all 5 documents remaining
        assert len(result_df) == 5
        assert sorted(result_df[CURATOR_DEDUP_ID_STR].tolist()) == [1, 2, 3, 4, 5]
        assert result._metadata["num_removed"] == 0
        # Stage should have custom metrics
        assert all(
            stage._custom_metrics[k] > 0 for k in ["input_df_min_max_time", "read_dupes_time", "id_removal_time"]
        )

    @patch("pandas.read_parquet")
    def test_process_with_no_matching_ids(
        self, mock_read_parquet: MagicMock, sample_document_batch: DocumentBatch, removal_parquet_file: str
    ):
        """Test process method when removal list has no matching IDs."""
        # Mock the parquet read to return IDs not in the batch (6, 7, 8)
        mock_read_parquet.return_value = pd.DataFrame({"id": [6, 7, 8]})

        stage = TextDuplicatesRemovalStage(ids_to_remove_path=removal_parquet_file)
        result = stage.process(sample_document_batch)

        result_df = result.to_pandas()
        # Should have all 5 documents remaining since no IDs match
        assert len(result_df) == 5
        assert sorted(result_df[CURATOR_DEDUP_ID_STR].tolist()) == [1, 2, 3, 4, 5]
        assert result._metadata["num_removed"] == 3  # 3 IDs were in removal list but didn't match
        assert all(
            stage._custom_metrics[k] > 0 for k in ["input_df_min_max_time", "read_dupes_time", "id_removal_time"]
        )

    @patch("pandas.read_parquet")
    def test_process_removes_all_documents(
        self, mock_read_parquet: MagicMock, sample_document_batch: DocumentBatch, removal_parquet_file: str
    ):
        """Test process method when all documents should be removed."""
        # Mock the parquet read to return all IDs in the batch
        mock_read_parquet.return_value = pd.DataFrame({"id": [1, 2, 3, 4, 5]})

        stage = TextDuplicatesRemovalStage(ids_to_remove_path=removal_parquet_file)
        result = stage.process(sample_document_batch)

        result_df = result.to_pandas()
        # Should have 0 documents remaining
        assert len(result_df) == 0
        assert result._metadata["num_removed"] == 5
        assert all(
            stage._custom_metrics[k] > 0 for k in ["input_df_min_max_time", "read_dupes_time", "id_removal_time"]
        )

    @patch("pandas.read_parquet")
    def test_process_with_custom_id_fields(self, mock_read_parquet: MagicMock, removal_parquet_file: str):
        """Test process method with custom ID field."""
        # Create document batch with custom ID field
        df = pd.DataFrame(
            {
                "text": ["Doc 1", "Doc 2", "Doc 3"],
                "custom_id": ["id_1", "id_2", "id_3"],
            }
        )
        doc_batch = DocumentBatch(
            task_id="custom_test",
            dataset_name="test",
            data=df,
        )

        # Mock the parquet read to return ID 20
        mock_read_parquet.return_value = pd.DataFrame({"dedup_id": ["id_2"]})

        stage = TextDuplicatesRemovalStage(
            ids_to_remove_path=removal_parquet_file,
            id_field="custom_id",
            duplicate_id_field="dedup_id",
        )
        result = stage.process(doc_batch)

        # Verify correct filters were used (min=10, max=30)
        call_args = mock_read_parquet.call_args
        assert call_args[1]["filters"] == [("dedup_id", ">=", "id_1"), ("dedup_id", "<=", "id_3")]

        result_df = result.to_pandas()
        # Should have 2 documents remaining (IDs "id_1", "id_3")
        assert len(result_df) == 2
        assert sorted(result_df["custom_id"].tolist()) == ["id_1", "id_3"]
        assert result._metadata["num_removed"] == 1
        assert all(
            stage._custom_metrics[k] > 0 for k in ["input_df_min_max_time", "read_dupes_time", "id_removal_time"]
        )
