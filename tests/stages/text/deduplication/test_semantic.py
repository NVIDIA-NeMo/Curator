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
from huggingface_hub import snapshot_download

from nemo_curator.stages.text.deduplication.semantic import TextSemanticDeduplicationWorkflow

# Pre-download the model to avoid rate limiting in CI. If it fails, skip the test.
try:
    snapshot_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir=None,
        token=None,
        local_files_only=False,
    )
except Exception as e:  # noqa: BLE001
    msg = f"Failed to download sentence-transformers/all-MiniLM-L6-v2 due to {e}"
    pytest.skip(msg)


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
        pytest.param((XennaExecutor, {}, True), id="xenna_with_id_generator"),
        # TODO: Uncomment this when we are able to figure out how to run Xenna again after Dedup
        # pytest.param((XennaExecutor, {}, False), id="xenna_without_id_generator"),  # noqa: ERA001
        pytest.param((RayDataExecutor, {}, False), id="ray_data_without_id_generator"),
    ],
    indirect=True,
)
class TestTextSemanticDeduplicationWorkflow:
    """Integration tests for TextSemanticDeduplicationWorkflow."""

    # Class attributes for shared test data
    executor_cls: type | None = None
    config: dict[str, Any] | None = None
    use_id_generator: bool | None = None
    input_dir: Path | None = None
    output_dir: Path | None = None
    cache_dir: Path | None = None
    expected_df: pd.DataFrame | None = None
    results: dict[str, Any] | None = None
    final_df: pd.DataFrame | None = None

    @pytest.fixture(scope="class", autouse=True)
    def test_config(
        self, request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
    ) -> "TestTextSemanticDeduplicationWorkflow":
        """Set up test environment and execute workflow."""
        executor_cls, config, use_id_generator = request.param

        request.cls.executor_cls = executor_cls
        request.cls.config = config
        request.cls.use_id_generator = use_id_generator

        # Create test data
        tmp_path = tmp_path_factory.mktemp("semantic_workflow_test")
        request.cls.input_dir = tmp_path / "input"
        request.cls.output_dir = tmp_path / "output"
        request.cls.cache_dir = tmp_path / "cache"

        # Create test data with duplicates
        request.cls.expected_df = create_data_with_duplicates(request.cls.input_dir)
        # Run workflow with duplicate removal enabled using the configured executor
        workflow = TextSemanticDeduplicationWorkflow(
            input_path=str(request.cls.input_dir),
            output_path=str(request.cls.output_dir),
            cache_path=str(request.cls.cache_dir),
            perform_removal=True,
            n_clusters=3,  # Use fewer clusters to group similar documents
            eps=0.1,  # Set epsilon to identify duplicates
            which_to_keep="hard",  # Keep harder examples (less similar to others)
            use_id_generator=use_id_generator,
            id_field="id" if not use_id_generator else "_curator_dedup_id",
            input_filetype="parquet",
            output_filetype="parquet",
            verbose=True,
            clear_output=True,
        )

        # Run the workflow
        request.cls.results = workflow.run(executor_cls(config))

        # Read the final deduplicated output for use in tests
        final_output_path = request.cls.results["final_output_path"]
        output_files = list(Path(final_output_path).glob("*.parquet"))
        if output_files:
            request.cls.final_df = pd.read_parquet(output_files)
        else:
            request.cls.final_df = pd.DataFrame()

        return

    def test_semantic_deduplication_correctness(self) -> None:
        """Test that semantic deduplication produces the correct number of records from each group."""
        # Verify the workflow completed successfully
        assert self.results is not None, "Workflow results should be available"
        assert "total_execution_time" in self.results
        assert self.results["total_execution_time"] > 0

        # Check that final output directory exists
        final_output_path = self.results["final_output_path"]
        assert final_output_path is not None
        assert os.path.exists(final_output_path)

        # Verify we have output files and data
        output_files = list(Path(final_output_path).glob("*.parquet"))
        assert len(output_files) > 0, "No output files found"
        assert self.final_df is not None, "Final dataframe should not be None"
        assert not self.final_df.empty, "Final dataframe should not be empty"

        # Extract the IDs from the final deduplicated dataset
        final_ids = set(self.final_df["id"].tolist())

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
        actual_total = len(self.final_df)
        assert actual_total == expected_total, f"Expected {expected_total} total records, got {actual_total}"

    def test_directory_structure(self) -> None:
        """Test that all expected directories are present."""
        assert self.cache_dir is not None, "Cache directory should be set"
        assert self.output_dir is not None, "Output directory should be set"
        assert self.use_id_generator is not None, "ID generator flag should be set"

        # Check cache directories
        assert (self.cache_dir / "embeddings").exists(), "Embeddings cache directory should exist"
        assert (self.cache_dir / "semantic_dedup").exists(), "Semantic dedup cache directory should exist"
        assert (self.cache_dir / "semantic_dedup" / "kmeans_results").exists(), "KMeans results directory should exist"
        assert (self.cache_dir / "semantic_dedup" / "pairwise_results").exists(), (
            "Pairwise results directory should exist"
        )

        # Check output directories
        assert (self.output_dir / "duplicates").exists(), "Duplicates output directory should exist"
        assert (self.output_dir / "deduplicated").exists(), "Deduplicated output directory should exist"

        # Check ID generator file based on configuration
        id_generator_file = self.output_dir / "semantic_id_generator.json"
        if self.use_id_generator:
            assert id_generator_file.exists(), "ID generator file should exist when use_id_generator=True"
        else:
            assert not id_generator_file.exists(), "ID generator file should not exist when use_id_generator=False"

    def test_directory_schemas_and_counts(self) -> None:
        """Test that all directories have the expected schema and number of rows."""
        assert self.cache_dir is not None, "Cache directory should be set"
        assert self.output_dir is not None, "Output directory should be set"
        assert self.use_id_generator is not None, "ID generator flag should be set"

        # 1. Check embeddings data
        embeddings_df = pd.read_parquet(self.cache_dir / "embeddings")
        # Embeddings always has the ID field used by the workflow and embeddings
        if self.use_id_generator:
            expected_embedding_cols = {"_curator_dedup_id", "embeddings"}
        else:
            expected_embedding_cols = {"id", "embeddings"}
        assert set(embeddings_df.columns) >= expected_embedding_cols, (
            f"Embeddings missing columns: {expected_embedding_cols - set(embeddings_df.columns)}"
        )
        assert len(embeddings_df) == 7, f"Expected 7 embedding records, got {len(embeddings_df)}"

        # 2. Check kmeans results data - preserves all embedding columns plus centroid
        kmeans_df = pd.read_parquet(self.cache_dir / "semantic_dedup" / "kmeans_results")
        if self.use_id_generator:
            expected_kmeans_cols = {"_curator_dedup_id", "embeddings", "centroid"}
        else:
            expected_kmeans_cols = {"id", "embeddings", "centroid"}
        assert set(kmeans_df.columns) >= expected_kmeans_cols, (
            f"KMeans missing columns: {expected_kmeans_cols - set(kmeans_df.columns)}"
        )
        assert len(kmeans_df) == 7, f"Expected 7 kmeans records, got {len(kmeans_df)}"

        # 3. Check pairwise results data
        pairwise_df = pd.read_parquet(self.cache_dir / "semantic_dedup" / "pairwise_results")
        # Pairwise always has id, max_id, cosine_sim_score
        expected_pairwise_cols = {"id", "max_id", "cosine_sim_score"}
        assert set(pairwise_df.columns) >= expected_pairwise_cols, (
            f"Pairwise missing columns: {expected_pairwise_cols - set(pairwise_df.columns)}"
        )
        assert len(pairwise_df) == 7, f"Expected 7 pairwise records, got {len(pairwise_df)}"

        # 4. Check duplicates data (in output directory only)
        duplicates_output_df = pd.read_parquet(self.output_dir / "duplicates")
        expected_duplicates_cols = {"id"}
        assert set(duplicates_output_df.columns) >= expected_duplicates_cols, (
            f"Output duplicates missing columns: {expected_duplicates_cols - set(duplicates_output_df.columns)}"
        )
        assert len(duplicates_output_df) == 2, (
            f"Expected 2 duplicate records in output path, got {len(duplicates_output_df)}"
        )

        # 5. Check final deduplicated data
        deduplicated_df = pd.read_parquet(self.output_dir / "deduplicated")
        expected_dedup_cols = {"id", "text"}
        assert set(deduplicated_df.columns) >= expected_dedup_cols, (
            f"Deduplicated missing columns: {expected_dedup_cols - set(deduplicated_df.columns)}"
        )
        assert len(deduplicated_df) == 5, f"Expected 5 deduplicated records, got {len(deduplicated_df)}"
