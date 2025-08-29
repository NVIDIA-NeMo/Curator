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

# ruff: noqa:E402
import glob
import os
import random
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.datasets import make_blobs

cupy = pytest.importorskip("cupy")
from nemo_curator.stages.deduplication.semantic import SemanticDeduplicationWorkflow


@pytest.mark.gpu
class TestSemanticDeduplicationWorkflow:
    """Test the SemanticDeduplicationWorkflow against the same data and expectations as the original Dask-based test."""

    def setup_method(self) -> None:
        """Setup method that creates the same synthetic data as the original test."""
        self.n_clusters = 5
        self.n_samples_per_cluster = [100 * (i + 1) for i in range(self.n_clusters)]
        self.n_features = 3

        # Reset all random state to get deterministic results
        random.seed(42)
        np.random.seed(42)  # noqa: NPY002
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.X, _ = make_blobs(
            n_samples=self.n_samples_per_cluster,
            centers=None,
            n_features=self.n_features,
            random_state=42,
        )

        # Create DataFrame with same structure as original test
        self.df = pd.DataFrame({"id": np.arange(len(self.X)), "embeddings": self.X.tolist()})

    def _create_input_parquet_files(self, input_dir: str, npartitions: int = 2) -> None:
        """Create parquet files from the synthetic data for pipeline input."""
        os.makedirs(input_dir, exist_ok=True)

        # Split dataframe into partitions (similar to how Dask would partition)
        chunk_size = len(self.df) // npartitions
        for i in range(npartitions):
            start_idx = i * chunk_size
            if i == npartitions - 1:
                # Last partition gets remaining data
                chunk_df = self.df.iloc[start_idx:]
            else:
                end_idx = (i + 1) * chunk_size
                chunk_df = self.df.iloc[start_idx:end_idx]

            # Save each chunk as a parquet file
            chunk_path = os.path.join(input_dir, f"part_{i:04d}.parquet")
            chunk_df.to_parquet(chunk_path, index=False)

    @pytest.mark.parametrize("which_to_keep", ["hard", "random", "easy"])
    @pytest.mark.parametrize("distance_metric", ["cosine", "l2"])
    @pytest.mark.parametrize("executor_type", ["ray_data", "xenna"])
    def test_semantic_deduplication_with_duplicate_identification(
        self,
        tmpdir: Path,
        which_to_keep: Literal["hard", "random", "easy"],
        distance_metric: Literal["cosine", "l2"],
        executor_type: Literal["ray_data", "xenna"],
    ) -> None:
        """Test semantic deduplication with duplicate identification to match original test results."""

        # Skip Ray Data tests except for one specific configuration to reduce test time
        if executor_type == "ray_data" and not (which_to_keep == "hard" and distance_metric == "cosine"):
            pytest.skip("Ray Data executor only tested on hard+cosine configuration")

        input_dir = os.path.join(tmpdir, "input")
        output_dir = os.path.join(tmpdir, "output")

        # Create input parquet files
        self._create_input_parquet_files(input_dir)

        # Create and run workflow with duplicate identification
        # Using eps=0.01 to match the original test
        pipeline = SemanticDeduplicationWorkflow(
            input_path=input_dir,
            output_path=output_dir,
            n_clusters=self.n_clusters,
            id_field="id",
            embedding_field="embeddings",
            distance_metric=distance_metric,
            which_to_keep=which_to_keep,
            eps=0.01,
            random_state=42,
            verbose=False,
        )

        # Run pipeline
        # Create executor based on type
        if executor_type == "xenna":
            from nemo_curator.backends.xenna import XennaExecutor

            executor = XennaExecutor()
        elif executor_type == "ray_data":
            from nemo_curator.backends.experimental.ray_data import RayDataExecutor

            executor = RayDataExecutor()

        results = pipeline.run(pairwise_executor=executor)

        # Validate basic execution
        assert results["total_execution_time"] > 0
        assert results["kmeans_execution_time"] > 0
        assert results["pairwise_execution_time"] > 0
        assert len(results["kmeans_results"]) > 0
        assert len(results["pairwise_results"]) > 0

        # Check that output directories were created (now automatically created)
        kmeans_output_dir = os.path.join(output_dir, "kmeans_results")
        pairwise_output_dir = os.path.join(output_dir, "pairwise_results")
        duplicates_dir = os.path.join(output_dir, "duplicates")

        assert os.path.exists(kmeans_output_dir)
        assert os.path.exists(pairwise_output_dir)
        assert os.path.exists(duplicates_dir)

        # Check duplicate identification results
        duplicates_identified = results["total_duplicates_identified"]
        assert duplicates_identified > 0

        # Validate against expected results from original test
        # These are the same expected values from the original test
        if which_to_keep == "hard":
            expected_removed = 1471
        elif which_to_keep == "easy":
            expected_removed = 1495
        else:  # random
            expected_removed = 1483

        assert duplicates_identified == expected_removed, (
            f"Expected duplicates: {expected_removed}, got {duplicates_identified}"
        )

        # Check that duplicates directory contains files
        duplicate_files = os.listdir(duplicates_dir)
        assert len(duplicate_files) > 0

        validations = [
            (
                {"centroid", "id", "embeddings", "l2_dist_to_cent", "cosine_dist_to_cent"},
                1500,
                kmeans_output_dir,
                "K-means",
            ),
            ({"id", "max_id", "cosine_sim_score"}, 1500, pairwise_output_dir, "Pairwise"),
            ({"id"}, duplicates_identified, duplicates_dir, "Duplicates"),
        ]

        for expected_schema, expected_rows, directory, description in validations:
            # Find all parquet files and read at once
            parquet_files = glob.glob(os.path.join(directory, "**/*.parquet"), recursive=True)
            assert len(parquet_files) > 0, f"No parquet files for {description} ({directory})"

            # Read all parquet files at once
            df = pd.read_parquet(parquet_files)
            # Check schema
            assert expected_schema == (set(df.columns)), (
                f"{description}: Missing columns {set(df.columns) - expected_schema}"
            )
            # Check row count
            assert len(df) == expected_rows, f"{description}: Expected {expected_rows} rows, got {len(df)}"
