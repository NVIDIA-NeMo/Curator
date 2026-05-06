# modality: text

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

from contextlib import suppress
from pathlib import Path
from typing import Literal
from unittest.mock import Mock, patch

# Suppress GPU-related import errors when running pytest -m "not gpu"
with suppress(ImportError):
    import cudf
    import cuml
    import cupy as cp

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

# Suppress GPU-related import errors when running pytest -m "not gpu"
with suppress(ImportError):
    from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.deduplication.semantic.kmeans import KMeansReadFitWriteStage, KMeansStage
    from nemo_curator.stages.deduplication.semantic.utils import get_array_from_df
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
        init="k-means||",
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
        """Setup fixture that runs pipeline once per class."""
        # Use parquet for the end-to-end integration run (JSONL read is tested in test_process_batch_read_paths).
        request.cls.file_format = "parquet"

        # Create fresh directories using tmp_path_factory for class-scoped fixture
        tmp_path = tmp_path_factory.mktemp("kmeans_test_data")
        request.cls.input_dir = tmp_path / "input"
        request.cls.output_dir = tmp_path / "output"

        # Generate synthetic clustered dataset
        input_dir, true_labels = create_clustered_dataset(tmp_path, file_format=request.cls.file_format)
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

    @pytest.mark.parametrize(
        # expect_break: Whether to expect a call to break_parquet_partition_into_groups
        # expect_multiple_groups: Whether to expect multiple groups to be returned
        ("filetype", "expect_break", "expect_multiple_groups"),
        [
            ("parquet", True, True),
            ("jsonl", False, False),
        ],
    )
    def test_process_batch_read_paths(
        self,
        tmp_path: Path,
        filetype: Literal["parquet", "jsonl"],
        expect_break: bool,
        expect_multiple_groups: bool,
    ) -> None:
        """Ensure process_batch routes reads and grouping by filetype."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        stage = KMeansReadFitWriteStage(
            id_field="id",
            embedding_field="embeddings",
            output_path=str(output_dir),
            filetype=filetype,
            n_clusters=2,
            metadata_fields=["metadata_col"],
            embedding_dim=32,
        )

        stage._raft_handle = Mock()

        if filetype == "parquet":
            all_files = [str(input_dir / f"file_{i}.parquet") for i in range(4)]
            all_tasks = [
                FileGroupTask(
                    task_id=f"test_task_{i}",
                    dataset_name="test_dataset",
                    data=[file],
                )
                for i, file in enumerate(all_files)
            ]
            df = cudf.DataFrame(
                {
                    "id": list(range(20)),
                    "embeddings": [[1.0, 0.0]] * 20,
                    "metadata_col": ["meta"] * 20,
                }
            )
            expected_groups = [all_files[:2], all_files[2:]]
        else:
            input_file = input_dir / "data.jsonl"
            all_files = [str(input_file)]
            all_tasks = [
                FileGroupTask(
                    task_id="test_task_jsonl",
                    dataset_name="test_dataset",
                    data=[str(input_file)],
                )
            ]
            df = cudf.DataFrame(
                {
                    "id": [0, 1],
                    "embeddings": [[1.0, 0.0], [0.0, 1.0]],
                    "metadata_col": ["a", "b"],
                }
            )
            expected_groups = [all_files]

        total_rows = len(df) * len(expected_groups)
        stage.kmeans = Mock()
        stage.kmeans._fit = Mock()
        stage.kmeans.predict = Mock(return_value=cp.zeros(total_rows, dtype=cp.int32))
        stage.kmeans.cluster_centers_ = cp.random.random((2, 2), dtype=cp.float32)

        with (
            patch(
                "nemo_curator.stages.deduplication.semantic.kmeans.break_parquet_partition_into_groups"
            ) as mock_break,
            patch.object(stage, "read_parquet", return_value=df) as mock_read_parquet,
            patch.object(stage, "read_jsonl", return_value=df) as mock_read_jsonl,
            patch.object(stage, "write_parquet") as mock_write,
        ):
            mock_break.return_value = expected_groups

            results = stage.process_batch(all_tasks)

            if expect_break:
                mock_break.assert_called_once_with(all_files, embedding_dim=32)
            else:
                mock_break.assert_not_called()

            if filetype == "parquet":
                assert mock_read_jsonl.call_count == 0
                assert mock_read_parquet.call_count == len(expected_groups)
                assert [call.args[0] for call in mock_read_parquet.call_args_list] == expected_groups
            else:
                mock_read_jsonl.assert_called_once()
                mock_read_parquet.assert_not_called()
                assert mock_read_jsonl.call_args[0][0] == all_files

            for call in mock_read_parquet.call_args_list or [mock_read_jsonl.call_args]:
                assert call.kwargs["columns"] == ["id", "embeddings", "metadata_col"]
                assert call.kwargs["assign_id"] is False

            stage.kmeans._fit.assert_called_once()
            stage.kmeans.predict.assert_called_once()

            assert mock_write.call_count == len(expected_groups)
            if expect_multiple_groups:
                assert len(results) == len(expected_groups), "Should return one result per group"

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


def _build_inner_stage(
    tmp_path: Path,
    *,
    fit_data_fraction: float | None = None,
    cache_path: str | None = None,
    filetype: Literal["parquet", "jsonl"] = "parquet",
    random_state: int = 42,
    n_clusters: int = 2,
    embedding_dim: int = 2,
) -> "KMeansReadFitWriteStage":
    """Build a minimally-mocked KMeansReadFitWriteStage for unit tests."""
    output_dir = tmp_path / "out"
    output_dir.mkdir(exist_ok=True)
    stage = KMeansReadFitWriteStage(
        id_field="id",
        embedding_field="embeddings",
        output_path=str(output_dir),
        filetype=filetype,
        n_clusters=n_clusters,
        embedding_dim=embedding_dim,
        random_state=random_state,
        fit_data_fraction=fit_data_fraction,
        cache_path=cache_path,
    )
    stage._raft_handle = Mock()
    stage.kmeans = Mock()
    stage.kmeans._fit = Mock()
    stage.kmeans.predict = Mock(return_value=cp.zeros(2, dtype=cp.int32))
    stage.kmeans.cluster_centers_ = cp.array(
        [[1.0, 0.0], [0.0, 1.0]], dtype=cp.float32
    )
    return stage


@pytest.mark.gpu
class TestFitDataFractionValidation:
    """Eager validation of fit_data_fraction at construction time."""

    @pytest.mark.parametrize("bad_fraction", [0.0, 1.0, -0.001, 1.001, -1.0, 2.0])
    def test_kmeans_stage_rejects_out_of_range(self, tmp_path: Path, bad_fraction: float) -> None:
        with pytest.raises(ValueError, match="fit_data_fraction must be in"):
            KMeansStage(
                n_clusters=2,
                id_field="id",
                embedding_field="embeddings",
                input_path=str(tmp_path / "in"),
                output_path=str(tmp_path / "out"),
                fit_data_fraction=bad_fraction,
            )

        with pytest.raises(ValueError, match="fit_data_fraction must be in"):
            KMeansReadFitWriteStage(
                id_field="id",
                embedding_field="embeddings",
                output_path=str(tmp_path / "out"),
                filetype="parquet",
                n_clusters=2,
                fit_data_fraction=bad_fraction,
            )


@pytest.mark.gpu
class TestFitDataFractionDispatch:
    """process_batch routes between single-pass and two-pass based on fit_data_fraction."""

    @staticmethod
    def _jsonl_task() -> "FileGroupTask":
        # Use jsonl so process_batch skips break_parquet_partition_into_groups (which reads metadata).
        return FileGroupTask(
            task_id="t",
            dataset_name="d",
            data=["x.jsonl"],
        )

    def test_none_dispatches_to_single_pass(self, tmp_path: Path) -> None:
        stage = _build_inner_stage(tmp_path, fit_data_fraction=None, filetype="jsonl")
        with (
            patch.object(stage, "_process_batch_single_pass", return_value=[]) as sp,
            patch.object(stage, "_process_batch_two_pass", return_value=[]) as tp,
        ):
            stage.process_batch([self._jsonl_task()])
            sp.assert_called_once()
            tp.assert_not_called()

    def test_set_fraction_dispatches_to_two_pass(self, tmp_path: Path) -> None:
        stage = _build_inner_stage(tmp_path, fit_data_fraction=0.5, filetype="jsonl")
        with (
            patch.object(stage, "_process_batch_single_pass", return_value=[]) as sp,
            patch.object(stage, "_process_batch_two_pass", return_value=[]) as tp,
        ):
            stage.process_batch([self._jsonl_task()])
            tp.assert_called_once()
            sp.assert_not_called()


@pytest.mark.gpu
class TestFitPassFileSampling:
    """_fit_pass samples files at the actor level (across groups)."""

    @staticmethod
    def _embedding_df(n_rows: int = 4) -> "cudf.DataFrame":
        return cudf.DataFrame({"embeddings": [[1.0, 0.0]] * n_rows})

    @pytest.mark.parametrize(
        ("n_files", "fraction", "expected"),
        [
            (20, 0.5, 10),
            (20, 0.25, 5),
            (10, 0.3, 3),
            (10, 0.35, 4),  # round half-to-even -> 4 with banker's rounding
        ],
    )
    def test_samples_expected_file_count(
        self, tmp_path: Path, n_files: int, fraction: float, expected: int
    ) -> None:
        stage = _build_inner_stage(tmp_path, fit_data_fraction=fraction)
        groups = [[f"f{i}.parquet" for i in range(n_files)]]
        with (
            patch(
                "nemo_curator.stages.deduplication.semantic.kmeans.break_parquet_partition_into_groups"
            ) as mock_break,
            patch.object(stage, "_read_group", return_value=self._embedding_df()),
        ):
            mock_break.side_effect = lambda files, **kwargs: [list(files)]
            stage._fit_pass(groups)
            (sampled_files,) = mock_break.call_args.args
            assert len(sampled_files) == expected, (
                f"fraction={fraction} on {n_files} files: expected {expected}, got {len(sampled_files)}"
            )
            # All sampled files come from the input, no duplicates
            assert set(sampled_files).issubset(set(groups[0]))
            assert len(set(sampled_files)) == len(sampled_files)

    def test_floors_at_one_file_and_warns(self, tmp_path: Path) -> None:
        """Tiny fractions still pick >= 1 file (RAFT cooperative fit needs every
        actor to contribute), but emit a warning since the realized sample exceeds
        the requested fraction."""
        stage = _build_inner_stage(tmp_path, fit_data_fraction=0.001)
        groups = [["only.parquet"]]
        with (
            patch(
                "nemo_curator.stages.deduplication.semantic.kmeans.break_parquet_partition_into_groups"
            ) as mock_break,
            patch.object(stage, "_read_group", return_value=self._embedding_df(1)),
            patch("nemo_curator.stages.deduplication.semantic.kmeans.logger") as mock_logger,
        ):
            mock_break.side_effect = lambda files, **kwargs: [list(files)]
            stage._fit_pass(groups)
            (sampled_files,) = mock_break.call_args.args
            assert sampled_files == ["only.parquet"]
            mock_logger.warning.assert_called_once()
            assert "fit_data_fraction" in mock_logger.warning.call_args.args[0]

    def test_flattens_across_groups(self, tmp_path: Path) -> None:
        """Sampling happens at the actor level, not within each group."""
        stage = _build_inner_stage(tmp_path, fit_data_fraction=0.5)
        # Two groups of 5 files each → 10 total → sample 5
        group_a = [f"a{i}.parquet" for i in range(5)]
        group_b = [f"b{i}.parquet" for i in range(5)]
        groups = [group_a, group_b]
        with (
            patch(
                "nemo_curator.stages.deduplication.semantic.kmeans.break_parquet_partition_into_groups"
            ) as mock_break,
            patch.object(stage, "_read_group", return_value=self._embedding_df()),
        ):
            mock_break.side_effect = lambda files, **kwargs: [list(files)]
            stage._fit_pass(groups)
            (sampled_files,) = mock_break.call_args.args
            assert len(sampled_files) == 5
            # Sample is drawn from the union — should not be confined to one group
            # (with n=5 from a 10-file pool, this holds with overwhelming probability for any seed)
            assert set(sampled_files).issubset(set(group_a) | set(group_b))

    def test_jsonl_skips_parquet_grouper(self, tmp_path: Path) -> None:
        """JSONL filetype routes sampled files into a single fit_group, no grouping."""
        stage = _build_inner_stage(tmp_path, fit_data_fraction=0.5, filetype="jsonl")
        groups = [[f"f{i}.jsonl" for i in range(10)]]
        with (
            patch(
                "nemo_curator.stages.deduplication.semantic.kmeans.break_parquet_partition_into_groups"
            ) as mock_break,
            patch.object(stage, "_read_group", return_value=self._embedding_df()) as mock_read,
        ):
            stage._fit_pass(groups)
            mock_break.assert_not_called()
            # _read_group called once with the sampled files as one fit_group
            mock_read.assert_called_once()
            assert len(mock_read.call_args.args[0]) == 5

    def test_reads_every_group_regardless_of_fraction(self, tmp_path: Path) -> None:
        stage = _build_inner_stage(tmp_path, fit_data_fraction=0.1)
        groups = [
            ["g0_f0.parquet", "g0_f1.parquet"],
            ["g1_f0.parquet", "g1_f1.parquet"],
            ["g2_f0.parquet"],
        ]
        df = cudf.DataFrame(
            {
                "id": [0, 1],
                "embeddings": [[1.0, 0.0], [0.0, 1.0]],
            }
        )
        # predict mock must size labels to match df length per call
        stage.kmeans.predict = Mock(return_value=cp.zeros(len(df), dtype=cp.int32))
        tasks = [FileGroupTask(task_id="t0", dataset_name="d", data=["any.parquet"])]
        with (
            patch.object(stage, "_read_group", return_value=df) as mock_read,
            patch.object(stage, "write_parquet"),
        ):
            results, _, total_rows = stage._predict_write_pass(tasks, groups)

        assert mock_read.call_count == len(groups), "Pass 2 must read every group"
        assert [call.args[0] for call in mock_read.call_args_list] == groups
        assert len(results) == len(groups)
        assert total_rows == len(df) * len(groups)


@pytest.mark.gpu
class TestCachePath:
    """Centroid persistence via cache_path, in both single-pass and two-pass."""

    @staticmethod
    def _embedding_df() -> "cudf.DataFrame":
        return cudf.DataFrame({"embeddings": [[1.0, 0.0]] * 4})

    def test_two_pass_saves_centroids_on_actor_zero(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "centroids"
        stage = _build_inner_stage(
            tmp_path, fit_data_fraction=0.5, cache_path=str(cache_path)
        )
        # _actor_index defaults to 0 via getattr fallback
        groups = [[f"f{i}.parquet" for i in range(4)]]
        with (
            patch(
                "nemo_curator.stages.deduplication.semantic.kmeans.break_parquet_partition_into_groups",
                side_effect=lambda files, **kwargs: [list(files)],
            ),
            patch.object(stage, "_read_group", return_value=self._embedding_df()),
        ):
            stage._fit_pass(groups)

        npy = cache_path / "kmeans_centroids.npy"
        assert npy.exists(), "centroids file should be saved at cache_path"
        loaded = np.load(npy)
        assert loaded.shape == (2, 2), f"unexpected centroids shape: {loaded.shape}"

    def test_two_pass_skips_save_for_non_zero_actor(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "centroids"
        stage = _build_inner_stage(
            tmp_path, fit_data_fraction=0.5, cache_path=str(cache_path)
        )
        stage._actor_index = 1  # non-zero actor
        groups = [[f"f{i}.parquet" for i in range(4)]]
        with (
            patch(
                "nemo_curator.stages.deduplication.semantic.kmeans.break_parquet_partition_into_groups",
                side_effect=lambda files, **kwargs: [list(files)],
            ),
            patch.object(stage, "_read_group", return_value=self._embedding_df()),
        ):
            stage._fit_pass(groups)

        assert not (cache_path / "kmeans_centroids.npy").exists()
        # And the directory itself wasn't created
        assert not cache_path.exists()

    def test_two_pass_skips_save_when_cache_path_is_none(self, tmp_path: Path) -> None:
        stage = _build_inner_stage(tmp_path, fit_data_fraction=0.5, cache_path=None)
        groups = [[f"f{i}.parquet" for i in range(4)]]
        with (
            patch(
                "nemo_curator.stages.deduplication.semantic.kmeans.break_parquet_partition_into_groups",
                side_effect=lambda files, **kwargs: [list(files)],
            ),
            patch.object(stage, "_read_group", return_value=self._embedding_df()),
        ):
            stage._fit_pass(groups)

        # No .npy files anywhere under tmp_path
        assert not list(tmp_path.rglob("*.npy"))

    def test_creates_missing_cache_directory(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "deeply" / "nested" / "centroids"
        assert not cache_path.exists()
        stage = _build_inner_stage(
            tmp_path, fit_data_fraction=0.5, cache_path=str(cache_path)
        )
        groups = [[f"f{i}.parquet" for i in range(4)]]
        with (
            patch(
                "nemo_curator.stages.deduplication.semantic.kmeans.break_parquet_partition_into_groups",
                side_effect=lambda files, **kwargs: [list(files)],
            ),
            patch.object(stage, "_read_group", return_value=self._embedding_df()),
        ):
            stage._fit_pass(groups)
        assert (cache_path / "kmeans_centroids.npy").exists()

    def test_single_pass_saves_centroids_on_actor_zero(self, tmp_path: Path) -> None:
        """The single-pass path also persists centroids when cache_path is set."""
        cache_path = tmp_path / "centroids_sp"
        stage = _build_inner_stage(
            tmp_path, fit_data_fraction=None, cache_path=str(cache_path)
        )
        df = cudf.DataFrame(
            {
                "id": [0, 1],
                "embeddings": [[1.0, 0.0], [0.0, 1.0]],
            }
        )
        stage.kmeans.predict = Mock(return_value=cp.zeros(len(df), dtype=cp.int32))
        tasks = [FileGroupTask(task_id="t", dataset_name="d", data=["any.parquet"])]
        groups = [["any.parquet"]]
        with (
            patch.object(stage, "_read_group", return_value=df),
            patch.object(stage, "write_parquet"),
        ):
            stage._process_batch_single_pass(tasks, groups)

        npy = cache_path / "kmeans_centroids.npy"
        assert npy.exists()
        loaded = np.load(npy)
        assert loaded.shape == (2, 2)


@pytest.mark.gpu
class TestKMeansFitDataFractionIntegration:
    """End-to-end pipeline runs exercising fit_data_fraction and cache_path."""

    def test_pipeline_with_fit_data_fraction_predicts_all_rows(
        self, tmp_path: Path
    ) -> None:
        """Pipeline with fit_data_fraction=0.5 still labels every row and clusters well."""
        input_dir, true_labels = create_clustered_dataset(tmp_path)
        output_dir = tmp_path / "output"
        output_dir.mkdir(exist_ok=True)
        cache_path = tmp_path / "centroids_cache"

        pipeline = Pipeline(
            name="kmeans_fdf_integration",
            stages=[
                KMeansStage(
                    id_field="id",
                    embedding_field="embeddings",
                    n_clusters=N_CLUSTERS,
                    input_path=str(input_dir),
                    output_path=str(output_dir),
                    metadata_fields=["random_col", "true_cluster"],
                    embedding_dim=EMBEDDING_DIM,
                    input_filetype="parquet",
                    random_state=RANDOM_STATE,
                    fit_data_fraction=0.5,
                    cache_path=str(cache_path),
                )
            ],
        )
        results = pipeline.run(RayActorPoolExecutor())
        assert len(results) > 0

        npy = cache_path / "kmeans_centroids.npy"
        assert npy.exists(), f"centroids file should be saved at {npy}"
        centroids = np.load(npy)
        assert centroids.shape == (N_CLUSTERS, EMBEDDING_DIM)

        df = cudf.read_parquet(output_dir).sort_values("id", ignore_index=True)
        # Pass 2 must have predicted every row, even though fit only saw half the files
        assert len(df) == len(true_labels), (
            f"every row should be labeled (got {len(df)}, expected {len(true_labels)})"
        )
        ari = adjusted_rand_score(df["centroid"].to_numpy(), true_labels)
        # Well-separated blobs should still cluster cleanly with half the data
        assert ari > 0.95, f"ARI too low at fit_data_fraction=0.5: {ari:.3f}"
