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
from typing import Any

import pandas as pd
import pytest

from nemo_curator.backends.experimental.ray_data import RayDataExecutor
from nemo_curator.backends.xenna import XennaExecutor

_ = pytest.importorskip("cudf")
from nemo_curator.stages.text.deduplication.semantic import TextSemanticDeduplicationWorkflow


def create_data_with_duplicates(input_dir: Path) -> pd.DataFrame:
    """Create test parquet files with text data for semantic deduplication testing."""
    input_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 100, 200, 300],
            "text": [
                "The quick brown fox jumps over the lazy dog",
                "The quick brown foxes jumps over the lazy dog",
                "The quick brown wolf jumps over the lazy dog",
                "The quick black cat jumps over the lazy dog",
                "A test string",
                "Another test string",
                "A different object",
            ],
        }
    )
    # Write to parquet files (one file per record for testing)
    for i in range(len(df)):
        df.iloc[i : i + 1].to_parquet(input_dir / f"test_file_{i}.parquet")
    return df


@pytest.mark.gpu
@pytest.mark.parametrize(
    "test_config",
    [
        pytest.param((XennaExecutor, {}), id="xenna"),
        pytest.param((RayDataExecutor, {}), id="ray_data"),
    ],
    indirect=True,
)
class TestTextSemanticDeduplicationWorkflow:
    """Integration tests for TextSemanticDeduplicationWorkflow."""

    # Class attributes for shared test data
    executor_cls: type | None = None
    config: dict[str, Any] | None = None
    input_dir: Path | None = None
    output_dir: Path | None = None
    expected_df: pd.DataFrame | None = None
    output_tasks: list[Any] | None = None

    # Attributes needed for the fixture
    perform_removal: bool = False  # Will be overridden in individual tests
    use_id_generator: bool = False  # Will be overridden in individual tests

    @pytest.fixture(scope="class", autouse=True)
    def test_config(
        self, request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
    ) -> "TestTextSemanticDeduplicationWorkflow":
        """Set up test environment and execute workflow."""
        executor_cls, config = request.param

        request.cls.executor_cls = executor_cls
        request.cls.config = config

        # Create test data
        tmp_path = tmp_path_factory.mktemp("semantic_workflow_test")
        self.input_dir = tmp_path / "input"
        self.output_dir = tmp_path / "output"

    def test_semantic_dedup_with_duplicates_and_removal_standalone(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """Test semantic deduplication with duplicate removal on dataset with known duplicates."""
        # Create test data with duplicates
        tmp_path = tmp_path_factory.mktemp("semantic_dedup_test")
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        # Create test data with duplicates
        create_data_with_duplicates(input_dir)

        # Run workflow with duplicate removal enabled using the configured executor
        workflow = TextSemanticDeduplicationWorkflow(
            input_path=str(input_dir),
            output_path=str(output_dir),
            perform_removal=True,
            n_clusters=3,  # Use fewer clusters to group similar documents
            eps=0.1,  # Set epsilon to identify duplicates
            which_to_keep="hard",  # Keep harder examples (less similar to others)
            use_id_generator=False,  # Use existing IDs
            id_field="id",
            input_filetype="parquet",
            output_filetype="parquet",
            verbose=True,
            clear_output=True,  # Avoid potential race conditions
        )

        # Run the workflow
        results = workflow.run(self.executor_cls(self.config))

        # Verify the workflow completed successfully
        assert "total_execution_time" in results
        assert results["total_execution_time"] > 0

        # Check that final output directory exists
        final_output_path = results["final_output_path"]
        assert final_output_path is not None
        assert os.path.exists(final_output_path)

        # Read the deduplicated output
        output_files = list(Path(final_output_path).glob("*.parquet"))
        assert len(output_files) > 0, "No output files found"

        # Read all output data
        final_df = pd.read_parquet(output_files)

        # Extract the IDs from the final deduplicated dataset
        final_ids = set(final_df["id"].tolist()) if not final_df.empty else set()

        # Expected behavior based on user's requirements:
        # First group (1, 2, 3, 4): should keep exactly 3 records
        # Second group (100, 200, 300): should keep exactly 2 records
        first_group_ids = {1, 2, 3, 4}
        second_group_ids = {100, 200, 300}

        first_group_kept = final_ids.intersection(first_group_ids)
        second_group_kept = final_ids.intersection(second_group_ids)

        # Verify the exact counts as specified
        assert len(first_group_kept) == 3, (
            f"Expected 3 records from first group {first_group_ids}, got {len(first_group_kept)}: {sorted(first_group_kept)}"
        )
        assert len(second_group_kept) == 2, (
            f"Expected 2 records from second group {second_group_ids}, got {len(second_group_kept)}: {sorted(second_group_kept)}"
        )

        # Verify total records (should be 3 + 2 = 5)
        expected_total = 5
        actual_total = len(final_df)
        assert actual_total == expected_total, f"Expected {expected_total} total records, got {actual_total}"

        # Check directory structure
        # embeddings
        assert (output_dir / "embeddings").exists()

        # semantic dedup
        assert (output_dir / "semantic_dedup").exists()

        assert (output_dir / "duplicates").exists()
        assert (output_dir / "deduplicated").exists()
        if self.use_id_generator:
            assert (output_dir / "semantic_id_generator.json").exists()
        else:
            assert not (output_dir / "semantic_id_generator.json").exists()

        # Check that the embeddings are saved
        embeddings_files = list(Path(output_dir / "embeddings").glob("*.parquet"))
        assert len(embeddings_files) > 0, "No embeddings files found"
