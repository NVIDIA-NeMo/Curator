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
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

cudf = pytest.importorskip("cudf")
cuml = pytest.importorskip("cuml")
cp = pytest.importorskip("cupy")

from sklearn.metrics import adjusted_rand_score

from nemo_curator.backends.experimental.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.deduplication.semantic.kmeans import KMeansReadFitWriteStage, KMeansStage
from nemo_curator.stages.deduplication.semantic.utils import get_array_from_df
from nemo_curator.stages.text.embedders.utils import create_list_series_from_1d_or_2d_ar
from nemo_curator.tasks import FileGroupTask

N_CLUSTERS = 4
N_SAMPLES_PER_CLUSTER = 10_000
EMBEDDING_DIM = 1024
RANDOM_STATE = 42


def create_clustered_dataset(  # noqa: PLR0913
    tmp_path: Path,
    n_clusters: int = N_CLUSTERS,
    n_samples_per_cluster: int = N_SAMPLES_PER_CLUSTER,
    embedding_dim: int = EMBEDDING_DIM,
    random_state: int = RANDOM_STATE,
    file_format: str = "parquet",
) -> tuple[Path, np.ndarray, np.ndarray]:
    """Create a synthetic clustered dataset using sklearn make_blobs.

    Args:
        tmp_path: Temporary directory path
        n_clusters: Number of clusters to create
        n_samples_per_cluster: Number of samples per cluster
        embedding_dim: Dimensionality of embeddings
        random_state: Random seed for reproducibility
        file_format: Output file format ('parquet' or 'jsonl')

    Returns:
        Tuple of (input_dir_path, embeddings_array, true_labels_array)
    """
    # Create clustered data using sklearn
    X, y_true = make_blobs(  # noqa: N806
        n_samples=n_clusters * n_samples_per_cluster,
        centers=n_clusters,
        n_features=embedding_dim,
        random_state=random_state,
        cluster_std=0.5,  # Reduced cluster standard deviation for tighter clusters
    )

    # Normalize embeddings (same as KMeans stage will do)
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)  # noqa: N806

    # Create input directory
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Create dataframe with embeddings and IDs
    num_files = 20  # Create multiple files to test file partitioning
    samples_per_file = len(X_normalized) // num_files
    rng = np.random.default_rng(random_state)

    for file_idx in range(num_files):
        start_idx = file_idx * samples_per_file
        end_idx = (file_idx + 1) * samples_per_file if file_idx < num_files - 1 else len(X_normalized)
        df = pd.DataFrame(
            {
                "id": np.arange(start_idx, end_idx),
                "embeddings": X_normalized[start_idx:end_idx].tolist(),
                "true_cluster": y_true[start_idx:end_idx].tolist(),
            }
        )
        df["random_col"] = rng.integers(0, 100, size=len(df))

        if file_format == "parquet":
            file_path = input_dir / f"data_part_{file_idx:02d}.parquet"
            df.to_parquet(file_path, index=False)
        elif file_format == "jsonl":
            file_path = input_dir / f"data_part_{file_idx:02d}.jsonl"
            df.to_json(file_path, orient="records", lines=True)
        else:
            msg = f"Unsupported file format: {file_format}"
            raise ValueError(msg)

    return input_dir, y_true


def run_single_gpu_baseline(
    input_dir: Path,
    n_clusters: int = N_CLUSTERS,
    file_format: str = "parquet",
) -> np.ndarray:
    single_gpu_kmeans = cuml.KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        max_iter=300,
        tol=1e-4,
        random_state=RANDOM_STATE,
        output_type="numpy",  # Use numpy output for easier comparison
    )

    # Read data based on file format
    if file_format == "parquet":
        df = cudf.read_parquet(str(input_dir / "*.parquet"))
    elif file_format == "jsonl":
        # For JSONL files, we need to use a glob pattern to read all files in the directory
        df = cudf.read_json(str(input_dir / "*.jsonl"), lines=True)
    else:
        msg = f"Unsupported file format: {file_format}"
        raise ValueError(msg)

    embeddings = get_array_from_df(df, "embeddings")
    single_gpu_kmeans.fit(embeddings)
    df["centroid"] = single_gpu_kmeans.predict(embeddings)

    return df.sort_values("id", ignore_index=True)["centroid"].to_numpy()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "file_format_config",
    ["parquet", "jsonl"],
    indirect=True,
)
class TestKMeansStageIntegration:
    """Integration tests for KMeansStage comparing multi-GPU vs single-GPU results."""

    # Class attributes for shared test data - set by fixture
    file_format = None
    input_dir = None
    output_dir = None
    true_labels = None
    pipeline_results = None

    @pytest.fixture(scope="class", autouse=True)
    def file_format_config(self, request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> None:
        """Setup fixture that runs pipeline once per class and file format combination."""
        # Get file_format from the parametrized values
        file_format = request.param

        # Store as class attributes using request.cls
        request.cls.file_format = file_format

        # Create fresh directories using tmp_path_factory for class-scoped fixture
        tmp_path = tmp_path_factory.mktemp("kmeans_test_data")
        request.cls.input_dir = tmp_path / "input"
        request.cls.output_dir = tmp_path / "output"

        # Generate synthetic clustered dataset
        input_dir, true_labels = create_clustered_dataset(tmp_path, file_format=file_format)
        request.cls.input_dir = input_dir
        request.cls.true_labels = true_labels

        # Create output directory
        request.cls.output_dir.mkdir(parents=True, exist_ok=True)

        pipeline = Pipeline(
            name="kmeans_integration_test",
            stages=[
                KMeansStage(
                    id_field="id",
                    embedding_field="embeddings",
                    n_clusters=N_CLUSTERS,
                    input_path=str(request.cls.input_dir),
                    output_path=str(request.cls.output_dir),
                    metadata_fields=["random_col", "true_cluster"],
                    embedding_dim=EMBEDDING_DIM,
                    input_filetype=request.cls.file_format,
                    verbose=True,
                    random_state=RANDOM_STATE,
                    max_iter=300,
                    tol=1e-4,
                )
            ],
        )
        request.cls.pipeline_results = pipeline.run(RayActorPoolExecutor())

    def test_multi_gpu_vs_single_gpu_consistency(self) -> None:
        """Test that multi-GPU KMeans produces consistent results with single-GPU baseline."""
        # Verify pipeline execution
        assert len(self.pipeline_results) > 0, "Pipeline should produce results"

        # Run single-GPU baseline for this test
        single_gpu_assignments = run_single_gpu_baseline(self.input_dir, file_format=self.file_format)
        # Read the multi-gpu output data
        multi_gpu_assignments = (
            cudf.read_parquet(self.output_dir).sort_values("id", ignore_index=True)["centroid"].to_numpy()
        )

        # Compare results with multi-GPU baseline
        multi_gpu_ari = adjusted_rand_score(multi_gpu_assignments, self.true_labels)
        single_gpu_ari = adjusted_rand_score(single_gpu_assignments, self.true_labels)

        # Both should produce reasonable clustering (not random)
        assert multi_gpu_ari > 0.99, f"Multi-GPU clustering should be better than random (got {multi_gpu_ari:.3f})"
        assert single_gpu_ari > 0.99, f"Single-GPU clustering should be better than random (got {single_gpu_ari:.3f})"

        # Both single-gpu and multi-gpu methods should produce similar quality results
        quality_diff = abs(multi_gpu_ari - single_gpu_ari)
        assert quality_diff < 0.01, (
            f"Multi-GPU and single-GPU should produce similar quality results (difference: {quality_diff:.3f})"
        )

    def test_output_columns(self) -> None:
        """Test that the output contains the expected columns."""
        expected_columns = {"id", "embeddings", "random_col", "centroid", "l2_dist_to_cent", "cosine_dist_to_cent"}
        output_df = cudf.read_parquet(self.output_dir)
        actual_columns = set(output_df.columns)
        assert expected_columns.issubset(actual_columns), f"Missing columns: {expected_columns - actual_columns}"

        # Verify data types
        assert output_df["id"].dtype == np.int64, "ID column should be integer"
        # Check if centroid column is categorical (as written by partitioning)
        centroid_dtype = output_df["centroid"].dtype
        assert isinstance(output_df["centroid"].dtype, cudf.CategoricalDtype), (
            f"Centroid column should be categorical, got {centroid_dtype}"
        )
        # Distance columns can be float32
        l2_dtype = output_df["l2_dist_to_cent"].dtype
        cosine_dtype = output_df["cosine_dist_to_cent"].dtype
        assert l2_dtype == np.float32, f"L2 distance should be float, got {l2_dtype}"
        assert cosine_dtype == np.float32, f"Cosine distance should be float, got {cosine_dtype}"

    def test_output_filenames_and_structure(self) -> None:
        """Test that the output files are created with exact expected filenames and partitioning.

        Each actor (we should have two GPU actors) writes files with predictable names: {tasks[0]._uuid}_{subgroup_index}.parquet
        Since our test data is small, each actor creates 1 subgroup, so files are named {uuid}_0.parquet
        """
        # Get the expected filenames from pipeline results
        # The pipeline returns EmptyTasks with task_id = output_filename = f"{tasks[0]._uuid}_{i}"
        expected_filenames = set()
        for result_task in self.pipeline_results:
            expected_filename = f"{result_task.task_id}.parquet"
            expected_filenames.add(expected_filename)

        # Should have exactly 2 result tasks (one per actor)
        assert len(expected_filenames) == 2, f"Expected 2 result tasks/filenames, got {len(expected_filenames)}"

        # Collect all actual filenames across all partitions
        actual_filenames = set()
        centroid_dirs = list(self.output_dir.glob("centroid=*"))

        # Collect filenames from all centroid partitions
        for centroid_dir in centroid_dirs:
            partition_files = list(centroid_dir.glob("*.parquet"))
            for file in partition_files:
                actual_filenames.add(file.name)

        # Verify that all expected filenames are present
        assert actual_filenames == expected_filenames, (
            f"Expected filenames {expected_filenames}, but found {actual_filenames}. "
            f"Missing: {expected_filenames - actual_filenames}, "
            f"Extra: {actual_filenames - expected_filenames}"
        )

        # Verify we have the expected number of centroid partitions (should be exactly N_CLUSTERS)
        assert len(centroid_dirs) == N_CLUSTERS, (
            f"Expected exactly {N_CLUSTERS} centroid partitions, got {len(centroid_dirs)}"
        )


@pytest.mark.gpu
class TestKMeansReadFitWriteStage:
    """Unit tests for KMeansReadFitWriteStage methods."""

    def test_assign_distances(self):
        """Test _assign_distances method computes L2 and cosine distances correctly."""
        df = cudf.DataFrame(
            {
                "centroid": [0, 1, 0],
                "embedding": [
                    [1, 0],
                    [0, 1],
                    [0.6, 0.8],
                ],
            }
        )
        centroids = cp.array([[1, 0], [0, 1]])

        # Call _assign_distances
        df_with_distances = KMeansReadFitWriteStage._assign_distances(df, "embedding", centroids)

        # Assert the distances match the expected values
        np.testing.assert_almost_equal(
            df_with_distances["l2_dist_to_cent"].to_arrow().to_pylist(),
            [0.0, 0.0, (0.16 + 0.64) ** 0.5],
            decimal=4,
        )
        np.testing.assert_almost_equal(
            df_with_distances["cosine_dist_to_cent"].to_arrow().to_pylist(),
            [0.0, 0.0, 0.4],
            decimal=4,
        )

    def test_normalize_embeddings_col_in_df(self):
        """Test normalize_embeddings_col_in_df method normalizes embeddings correctly."""
        df = cudf.DataFrame(
            {
                "embedding": [[3, 4, 5], [1, 2, 2], [1, 0, 0]],
            }
        )
        expected_normalized = cp.array(
            [
                [0.42426407, 0.565685, 0.707107],
                [0.33333334, 0.6666667, 0.6666667],
                [1.0, 0.0, 0.0],
            ]
        )

        # Call the function
        normalized_embeddings = KMeansReadFitWriteStage.normalize_embeddings_col_in_df(df, "embedding")

        # Assert the normalized embeddings match the expected values
        cp.testing.assert_allclose(
            get_array_from_df(normalized_embeddings, "embedding"),
            expected_normalized,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_process_batch_multiple_groups(self, tmp_path: Path):  # noqa: PLR0915
        """Test process_batch method with multiple groups from break_parquet_partition_into_groups."""
        # Create test parquet files with real embeddings
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create test files with realistic data
        for i in range(4):
            # Create normalized embeddings (as the real code expects)
            embeddings = cp.random.random((10, 32), dtype=cp.float32)
            embeddings = embeddings / cp.linalg.norm(embeddings, axis=1, keepdims=True)

            df = cudf.DataFrame(
                {
                    "id": range(i * 10, (i + 1) * 10),
                    "embeddings": create_list_series_from_1d_or_2d_ar(embeddings, index=cudf.RangeIndex(10)),
                    "metadata_col": [f"meta_{j}" for j in range(10)],
                }
            )
            df.to_parquet(input_dir / f"file_{i}.parquet", index=False)

        # Create stage instance
        stage = KMeansReadFitWriteStage(
            id_field="id",
            embedding_field="embeddings",
            output_path=str(output_dir),
            filetype="parquet",
            n_clusters=2,
            metadata_fields=["metadata_col"],
            embedding_dim=32,
        )

        # Only mock the essential parts that can't run without RAFT setup
        mock_kmeans = Mock()
        mock_kmeans.fit = Mock()
        mock_kmeans.predict = Mock(return_value=cp.zeros(40, dtype=cp.int32))
        mock_kmeans.cluster_centers_ = cp.random.random((2, 32), dtype=cp.float32)
        stage.kmeans = mock_kmeans
        stage._raft_handle = Mock()

        # Create task
        all_files = [str(input_dir / f"file_{i}.parquet") for i in range(4)]

        all_tasks = [
            FileGroupTask(
                task_id=f"test_task_{i}",
                dataset_name="test_dataset",
                data=[file],
            )
            for i, file in enumerate(all_files)
        ]

        # Track method calls using spy pattern instead of full mocking
        original_read = stage.read_parquet
        original_write = stage.write_parquet
        read_calls = []
        write_calls = []

        def spy_read(*args, **kwargs) -> cudf.DataFrame:
            read_calls.append((args, kwargs))
            return original_read(*args, **kwargs)

        def spy_write(*args, **kwargs) -> None:
            write_calls.append((args, kwargs))
            return original_write(*args, **kwargs)

        # Only mock break_parquet_partition_into_groups to force multiple groups
        with (
            patch(
                "nemo_curator.stages.deduplication.semantic.kmeans.break_parquet_partition_into_groups"
            ) as mock_break,
            patch.object(stage, "read_parquet", side_effect=spy_read),
            patch.object(stage, "write_parquet", side_effect=spy_write),
        ):
            # Force breaking into 2 groups
            mock_break.return_value = [all_files[:2], all_files[2:]]

            results = stage.process_batch(all_tasks)

            # Verify break function was called
            mock_break.assert_called_once_with(all_files, embedding_dim=32)

            # Verify read operations
            assert len(read_calls) == 2, "Should have called read_parquet twice (once per group)"

            # Verify the files passed to each read call
            read_files = [call[0][0] for call in read_calls]  # First positional arg
            assert read_files[0] == all_files[:2], "First read should get first 2 files"
            assert read_files[1] == all_files[2:], "Second read should get last 2 files"

            # Verify read parameters
            for _, call_kwargs in read_calls:
                assert call_kwargs["columns"] == ["id", "embeddings", "metadata_col"]
                assert call_kwargs["assign_id"] is False

            # Verify KMeans operations
            mock_kmeans.fit.assert_called_once()
            mock_kmeans.predict.assert_called_once()

            # Check the concatenated embeddings shape
            fit_call_args = mock_kmeans.fit.call_args[0]
            embeddings_passed_to_fit = fit_call_args[0]
            assert embeddings_passed_to_fit.shape == (40, 32), "Should concatenate embeddings from all groups"

            # Verify write operations
            assert len(write_calls) == 2, "Should have called write_parquet twice (once per group)"

            # Verify write parameters
            for i, (call_args, call_kwargs) in enumerate(write_calls):
                assert call_args[1] == str(output_dir), "Should write to correct output directory"
                # The actual implementation uses tasks[0]._uuid for all output files
                assert call_kwargs["partition_file_name"] == f"{all_tasks[0]._uuid}_{i}.parquet"
                assert call_kwargs["partition_cols"] == ["centroid"]
                assert call_kwargs["index"] is False

            # Verify results
            assert len(results) == 2, "Should return 2 results (one per group)"
            for i, result in enumerate(results):
                # The actual implementation uses tasks[0]._uuid for all result task_ids
                assert result.task_id == f"{all_tasks[0]._uuid}_{i}"
                assert result.dataset_name == f"kmeans_group_{i}"

            # Verify actual output files were created (integration test aspect)
            output_files = list(output_dir.rglob("*.parquet"))
            assert len(output_files) == 2, "Should have created 2 output files"
