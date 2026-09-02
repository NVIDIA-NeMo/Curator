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

import re
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
    from nemo_curator.stages.deduplication.semantic.utils import ParquetFileInfo, get_array_from_df
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
class TestKMeansStage:
    """Unit tests for KMeansStage decomposition."""

    @pytest.mark.parametrize(
        ("input_filetype", "expected_extensions"),
        [
            ("parquet", [".parquet"]),
            ("jsonl", [".jsonl", ".json"]),
        ],
    )
    def test_input_file_extensions_default_to_input_filetype(
        self,
        tmp_path: Path,
        input_filetype: Literal["parquet", "jsonl"],
        expected_extensions: list[str],
    ) -> None:
        stage = KMeansStage(
            id_field="id",
            embedding_field="embeddings",
            n_clusters=2,
            input_path=str(tmp_path / "input"),
            output_path=str(tmp_path / "output"),
            input_filetype=input_filetype,
        )

        stages = stage.decompose()

        assert stages[0].file_extensions == expected_extensions

    def test_input_file_extensions_override_default(self, tmp_path: Path) -> None:
        stage = KMeansStage(
            id_field="id",
            embedding_field="embeddings",
            n_clusters=2,
            input_path=str(tmp_path / "input"),
            output_path=str(tmp_path / "output"),
            input_filetype="parquet",
            input_file_extensions=[".pq"],
        )

        stages = stage.decompose()

        assert stages[0].file_extensions == [".pq"]

    def test_unsupported_input_filetype_raises(self, tmp_path: Path) -> None:
        stage = KMeansStage(
            id_field="id",
            embedding_field="embeddings",
            n_clusters=2,
            input_path=str(tmp_path / "input"),
            output_path=str(tmp_path / "output"),
            input_filetype="csv",  # type: ignore[arg-type]
        )

        with pytest.raises(ValueError, match="Unsupported filetype: csv"):
            stage.decompose()


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
        """Output files are written with deterministic, input-derived names and
        partitioned by centroid.

        Each frame uses ``{input_task_id}_{frame_index}.parquet`` where
        the input task id is the FilePartitioning id (``0_<file_hash>``).
        We assert the names match that deterministic pattern (never a random
        ``r<uuid>`` fallback) and that the centroid partitioning is correct.

        Note: the pipeline's result tasks are terminal ``EmptyTask`` signals
        whose ids are framework-assigned (and, for this aggregating stage, the
        non-deterministic ``r<uuid>`` fallback) — they are intentionally NOT
        tied to the output filenames, which are derived from the input ids.
        """
        # Each actor emits a terminal result task; the actor count is controlled by the executor.
        assert self.pipeline_results

        # Collect all output filenames across centroid partitions. (The same
        # file name appears under each centroid=* dir, so dedupe into a set.)
        centroid_dirs = list(self.output_dir.glob("centroid=*"))
        actual_filenames = {f.name for d in centroid_dirs for f in d.glob("*.parquet")}

        # Bounded reads may produce multiple frames per actor, but every name must retain input ancestry.
        assert actual_filenames
        deterministic_name = re.compile(r"^0_[0-9a-f]+_\d+\.parquet$")
        for name in actual_filenames:
            assert deterministic_name.match(name), (
                f"Output filename {name!r} is not deterministic/input-derived "
                f"(an 'r<uuid>' name would mean ancestry was lost)"
            )

        # Exactly N_CLUSTERS centroid partitions.
        assert len(centroid_dirs) == N_CLUSTERS, (
            f"Expected exactly {N_CLUSTERS} centroid partitions, got {len(centroid_dirs)}"
        )

    def test_pipeline_with_fit_data_fraction_predicts_all_rows(self, tmp_path: Path) -> None:
        """fit_data_fraction=0.5 still labels every row and clusters well end-to-end."""
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
        assert np.load(npy).shape == (N_CLUSTERS, EMBEDDING_DIM)

        df = cudf.read_parquet(output_dir).sort_values("id", ignore_index=True)
        # Pass 2 must label every row even though fit only saw half the files
        assert len(df) == len(true_labels)
        ari = adjusted_rand_score(df["centroid"].to_numpy(), true_labels)
        assert ari > 0.95, f"ARI too low at fit_data_fraction=0.5: {ari:.3f}"


@pytest.mark.gpu
class TestKMeansReadFitWriteStage:
    """Unit tests for KMeansReadFitWriteStage methods."""

    @pytest.fixture
    def make_stage(self, tmp_path: Path):
        """Factory: minimally-mocked KMeansReadFitWriteStage; kwargs override defaults."""

        def _make(**kwargs) -> "KMeansReadFitWriteStage":
            stage = KMeansReadFitWriteStage(
                **{
                    "id_field": "id",
                    "embedding_field": "embeddings",
                    "output_path": str(tmp_path),
                    "filetype": "parquet",
                    "n_clusters": 2,
                    "random_state": 42,
                    **kwargs,
                }
            )
            stage.kmeans = Mock()
            stage.kmeans.cluster_centers_ = cp.array([[1.0, 0.0], [0.0, 1.0]], dtype=cp.float32)
            return stage

        return _make

    @pytest.mark.parametrize(
        ("filetype", "fit_data_fraction", "expected_path"),
        [
            ("parquet", None, "parquet"),
            ("parquet", 0.5, "parquet"),
            ("jsonl", None, "single_pass"),
            ("jsonl", 0.5, "two_pass"),
        ],
    )
    def test_process_batch_routes_by_filetype_and_fit_fraction(
        self,
        make_stage: "KMeansReadFitWriteStage",
        filetype: Literal["parquet", "jsonl"],
        fit_data_fraction: float | None,
        expected_path: str,
    ) -> None:
        """Parquet has one bounded path; JSONL retains its single- and two-pass paths."""
        stage = make_stage(filetype=filetype, fit_data_fraction=fit_data_fraction)
        suffix = "parquet" if filetype == "parquet" else "jsonl"
        task = FileGroupTask(dataset_name="test", data=[f"input.{suffix}"])

        with (
            patch.object(stage, "_process_parquet", return_value=[]) as parquet,
            patch.object(stage, "_process_batch_single_pass", return_value=[]) as single_pass,
            patch.object(stage, "_process_batch_two_pass", return_value=[]) as two_pass,
        ):
            stage.process_batch([task])

        assert parquet.called is (expected_path == "parquet")
        assert single_pass.called is (expected_path == "single_pass")
        assert two_pass.called is (expected_path == "two_pass")

    def test_process_batch_with_no_tasks_is_a_noop(self, make_stage: "KMeansReadFitWriteStage") -> None:
        stage = make_stage()

        assert stage.process_batch([]) == []

    def test_process_batch_rejects_unknown_filetype(self, make_stage: "KMeansReadFitWriteStage") -> None:
        stage = make_stage(filetype="csv")

        with pytest.raises(ValueError, match="Only jsonl and parquet are supported"):
            stage.process_batch([FileGroupTask(dataset_name="test", data=["input.csv"])])

    @pytest.mark.parametrize("filetype", ["parquet", "jsonl"])
    def test_read_group_uses_the_matching_reader(
        self, make_stage: "KMeansReadFitWriteStage", filetype: Literal["parquet", "jsonl"]
    ) -> None:
        """Both legacy JSONL paths and the bounded Parquet path share this reader boundary."""
        stage = make_stage(filetype=filetype)
        files = [f"input.{filetype}"]
        columns = ["id", "embeddings", "metadata"]

        with (
            patch.object(stage, "read_parquet", return_value=Mock()) as read_parquet,
            patch.object(stage, "read_jsonl", return_value=Mock()) as read_jsonl,
        ):
            stage._read_group(files, columns)

        expected_reader = read_parquet if filetype == "parquet" else read_jsonl
        other_reader = read_jsonl if filetype == "parquet" else read_parquet
        expected_reader.assert_called_once_with(
            files,
            columns=columns,
            storage_options=None,
            assign_id=False,
        )
        other_reader.assert_not_called()

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

    @pytest.mark.parametrize("bad_fraction", [0.0, -0.001, 1.001])
    def test_fit_data_fraction_validation(self, tmp_path: Path, bad_fraction: float) -> None:
        """Both KMeansStage and KMeansReadFitWriteStage reject out-of-range values at construction."""
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

    def test_fit_sample_selects_complete_files(self, make_stage: "KMeansReadFitWriteStage") -> None:
        stage = make_stage(fit_data_fraction=0.5)
        file_info = [ParquetFileInfo(f"file-{i}.parquet", i + 1, 10) for i in range(5)]

        fit, remaining = stage._sample_fit_files(file_info)

        assert len(fit) == 2
        assert {info.path for info in fit}.isdisjoint(info.path for info in remaining)
        assert {info.path for info in [*fit, *remaining]} == {info.path for info in file_info}

    def test_full_fit_samples_every_file(self, make_stage: "KMeansReadFitWriteStage") -> None:
        stage = make_stage(fit_data_fraction=1.0)
        file_info = [ParquetFileInfo(f"file-{i}.parquet", 1, 0) for i in range(5)]

        fit, remaining = stage._sample_fit_files(file_info)

        assert {info.path for info in fit} == {info.path for info in file_info}
        assert remaining == []

    def test_auto_fit_budget_includes_metadata(self, make_stage: "KMeansReadFitWriteStage") -> None:
        stage = make_stage(fit_data_fraction=None)
        file_info = [
            ParquetFileInfo("metadata-heavy.parquet", 1, 1_000, embedding_elements=2),
            ParquetFileInfo("fits.parquet", 10, 0, embedding_elements=20),
        ]

        with patch("cupy.cuda.runtime.memGetInfo", return_value=(200, 1_000)):
            fit, remaining = stage._sample_fit_files(file_info)

        assert [info.path for info in fit] == ["fits.parquet"]
        assert [info.path for info in remaining] == ["metadata-heavy.parquet"]

    def test_local_parquet_groups_legal_files(self, make_stage: "KMeansReadFitWriteStage") -> None:
        stage = make_stage()
        frame = cudf.DataFrame({"id": [1], "embeddings": [[1.0, 0.0]]})
        file_info = [ParquetFileInfo("file.parquet", 1, 0, embedding_elements=2)]

        with patch.object(stage, "_read_group", return_value=frame) as reader:
            assert list(stage._iter_parquet_frames(file_info, ["id", "embeddings"])) == [frame]

        reader.assert_called_once_with(["file.parquet"], ["id", "embeddings"])

    def test_process_parquet_fits_sample_and_predicts_remaining_rows(
        self, make_stage: "KMeansReadFitWriteStage"
    ) -> None:
        """Sampled rows reuse fit labels; only unread files go through predict."""
        stage = make_stage(fit_data_fraction=0.5)
        fit_info = ParquetFileInfo("fit.parquet", 2, 0)
        remaining_info = ParquetFileInfo("remaining.parquet", 2, 0)
        frames = {
            "fit.parquet": cudf.DataFrame({"id": [0, 1], "embeddings": [[1.0, 0.0], [0.0, 1.0]]}),
            "remaining.parquet": cudf.DataFrame({"id": [2, 3], "embeddings": [[1.0, 0.0], [0.0, 1.0]]}),
        }
        stage.kmeans.labels_ = cp.array([0, 1], dtype=cp.int32)
        stage.kmeans.predict.return_value = cp.array([0, 1], dtype=cp.int32)
        kmeans = stage.kmeans
        with (
            patch(
                "nemo_curator.stages.deduplication.semantic.kmeans.read_parquet_file_info",
                return_value=[fit_info, remaining_info],
            ),
            patch.object(stage, "_sample_fit_files", return_value=([fit_info], [remaining_info])),
            patch.object(
                stage,
                "_iter_parquet_frames",
                side_effect=lambda info, _columns: iter([frames[info[0].path]]),
            ),
            patch.object(stage, "_write_output_frame") as write,
            patch.object(stage, "_log_metrics") as log_metrics,
            patch.object(cudf.DataFrame, "drop", side_effect=AssertionError("deep frame copy")),
        ):
            stage._process_parquet([FileGroupTask(dataset_name="d", data=list(frames))], list(frames))

        kmeans.fit.assert_called_once()
        assert write.call_count == 2
        kmeans.predict.assert_called_once()
        assert kmeans.predict.call_args.kwargs == {"convert_dtype": False}
        # The embedding column is held in the exact-size CuPy buffer, not duplicated in metadata frames.
        assert all("embeddings" not in call.args[1].columns for call in write.call_args_list)
        final_metrics = log_metrics.call_args_list[-1].args[0]
        assert final_metrics["num_rows"] == 4
        assert {"kmeans_read_time", "kmeans_predict_time", "kmeans_write_time"} <= final_metrics.keys()

    def test_process_parquet_full_fit_writes_every_row_without_predict(
        self, make_stage: "KMeansReadFitWriteStage"
    ) -> None:
        """A full fit already has labels for every row, so no second read or predict is needed."""
        stage = make_stage(fit_data_fraction=1.0)
        file_info = [
            ParquetFileInfo("first.parquet", 2, 0),
            ParquetFileInfo("second.parquet", 1, 0),
        ]
        frames = [
            cudf.DataFrame({"id": [0, 1], "embeddings": [[1.0, 0.0], [0.0, 1.0]]}),
            cudf.DataFrame({"id": [2], "embeddings": [[1.0, 0.0]]}),
        ]
        stage.kmeans.labels_ = cp.array([0, 1, 0], dtype=cp.int32)
        kmeans = stage.kmeans

        with (
            patch(
                "nemo_curator.stages.deduplication.semantic.kmeans.read_parquet_file_info",
                return_value=file_info,
            ),
            patch.object(stage, "_sample_fit_files", return_value=(file_info, [])),
            patch.object(stage, "_iter_parquet_frames", return_value=iter(frames)) as read_frames,
            patch.object(stage, "_write_output_frame") as write,
            patch.object(stage, "_log_metrics") as log_metrics,
        ):
            stage._process_parquet(
                [FileGroupTask(dataset_name="d", data=[info.path for info in file_info])],
                [info.path for info in file_info],
            )

        kmeans.fit.assert_called_once()
        kmeans.predict.assert_not_called()
        read_frames.assert_called_once()
        assert write.call_count == 2
        assert sum(len(call.args[1]) for call in write.call_args_list) == 3
        final_metrics = log_metrics.call_args_list[-1].args[0]
        assert final_metrics["num_rows"] == 3
        assert final_metrics["kmeans_predict_time"] == 0

    @pytest.mark.parametrize(
        ("groups", "fraction", "expected_count"),
        [
            ([[f"f{i}.parquet" for i in range(20)]], 0.5, 10),
            ([[f"f{i}.parquet" for i in range(20)]], 0.25, 5),
            ([[f"f{i}.parquet" for i in range(10)]], 0.35, 4),  # banker's rounding
            # multi-group: sampling flattens at the actor level, not within groups
            (
                [[f"a{i}.parquet" for i in range(5)], [f"b{i}.parquet" for i in range(5)]],
                0.5,
                5,
            ),
        ],
    )
    def test_fit_pass_samples_files_at_actor_level(
        self, make_stage: "KMeansReadFitWriteStage", groups: list[list[str]], fraction: float, expected_count: int
    ) -> None:
        """_fit_pass samples round(fraction * total_files) across all groups, no duplicates."""
        stage = make_stage(fit_data_fraction=fraction, filetype="jsonl")
        df = cudf.DataFrame({"embeddings": [[1.0, 0.0]] * 4})
        all_input = {f for g in groups for f in g}
        with patch.object(stage, "_read_group", return_value=df) as read_group:
            stage._fit_pass(groups)
        sampled_files = read_group.call_args.args[0]
        assert len(sampled_files) == expected_count
        assert set(sampled_files).issubset(all_input)
        assert len(set(sampled_files)) == len(sampled_files)

    def test_fit_pass_floors_at_one_file_and_warns(self, make_stage: "KMeansReadFitWriteStage") -> None:
        """Tiny fractions still pick >= 1 file (RAFT cooperative fit needs every actor to
        contribute), but emit a warning since the realized sample exceeds the request."""
        stage = make_stage(fit_data_fraction=0.001, filetype="jsonl")
        df = cudf.DataFrame({"embeddings": [[1.0, 0.0]]})
        with (
            patch.object(stage, "_read_group", return_value=df) as read_group,
            patch("nemo_curator.stages.deduplication.semantic.kmeans.logger") as mock_logger,
        ):
            stage._fit_pass([["only.parquet"]])
        sampled_files = read_group.call_args.args[0]
        assert sampled_files == ["only.parquet"]
        mock_logger.warning.assert_called_once()
        assert "fit_data_fraction" in mock_logger.warning.call_args.args[0]

    def test_fit_pass_jsonl_skips_parquet_grouper(self, make_stage: "KMeansReadFitWriteStage") -> None:
        """JSONL filetype routes sampled files into a single fit_group, no grouping."""
        stage = make_stage(fit_data_fraction=0.5, filetype="jsonl")
        df = cudf.DataFrame({"embeddings": [[1.0, 0.0]] * 4})
        with (
            patch(
                "nemo_curator.stages.deduplication.semantic.kmeans.break_parquet_partition_into_groups"
            ) as mock_break,
            patch.object(stage, "_read_group", return_value=df) as mock_read,
        ):
            stage._fit_pass([[f"f{i}.jsonl" for i in range(10)]])
        mock_break.assert_not_called()
        mock_read.assert_called_once()
        assert len(mock_read.call_args.args[0]) == 5

    def test_predict_write_pass_reads_every_group(self, make_stage: "KMeansReadFitWriteStage") -> None:
        """Pass 2 must load every original group, regardless of fit_data_fraction."""
        stage = make_stage(fit_data_fraction=0.1)
        groups = [
            ["g0_f0.parquet", "g0_f1.parquet"],
            ["g1_f0.parquet", "g1_f1.parquet"],
            ["g2_f0.parquet"],
        ]
        df = cudf.DataFrame({"id": [0, 1], "embeddings": [[1.0, 0.0], [0.0, 1.0]]})
        stage.kmeans.predict = Mock(return_value=cp.zeros(len(df), dtype=cp.int32))
        tasks = [FileGroupTask(dataset_name="d", data=["any.parquet"])]
        with (
            patch.object(stage, "_read_group", return_value=df) as mock_read,
            patch.object(stage, "write_parquet"),
        ):
            results, _, total_rows = stage._predict_write_pass(tasks, groups)

        assert mock_read.call_count == len(groups)
        assert [call.args[0] for call in mock_read.call_args_list] == groups
        assert len(results) == len(groups)
        assert total_rows == len(df) * len(groups)

    def test_two_pass_combines_fit_and_predict_read_times(self, make_stage: "KMeansReadFitWriteStage") -> None:
        """The two-pass wrapper reports both reads while preserving the predict/write results."""
        stage = make_stage(fit_data_fraction=0.5, filetype="jsonl")
        tasks = [FileGroupTask(dataset_name="d", data=["input.jsonl"])]
        results = [Mock()]

        with (
            patch.object(stage, "_fit_pass", return_value=1.25) as fit_pass,
            patch.object(stage, "_predict_write_pass", return_value=(results, 2.5, 7)) as predict_write,
            patch.object(stage, "_log_metrics") as log_metrics,
        ):
            actual = stage._process_batch_two_pass(tasks, [["input.jsonl"]])

        assert actual == results
        fit_pass.assert_called_once_with([["input.jsonl"]])
        predict_write.assert_called_once_with(tasks, [["input.jsonl"]])
        log_metrics.assert_called_once_with({"kmeans_read_time": 3.75, "num_rows": 7})

    @pytest.mark.parametrize(
        ("actor_index", "cache_subpath", "expect_saved"),
        [
            (0, "centroids", True),
            (1, "centroids", False),  # non-zero actors don't write
            (0, None, False),  # no cache_path -> don't write
            (0, "deeply/nested/centroids", True),  # creates missing dirs
        ],
    )
    def test_two_pass_cache_path(
        self,
        tmp_path: Path,
        make_stage: "KMeansReadFitWriteStage",
        actor_index: int,
        cache_subpath: str | None,
        expect_saved: bool,
    ) -> None:
        """_fit_pass saves centroids only on actor 0 when cache_path is set."""
        cache_path = tmp_path / cache_subpath if cache_subpath else None
        stage = make_stage(
            fit_data_fraction=0.5,
            filetype="jsonl",
            cache_path=str(cache_path) if cache_path else None,
        )
        stage._actor_index = actor_index
        df = cudf.DataFrame({"embeddings": [[1.0, 0.0]] * 4})
        with patch.object(stage, "_read_group", return_value=df):
            stage._fit_pass([[f"f{i}.parquet" for i in range(4)]])

        if expect_saved:
            npy = cache_path / "kmeans_centroids.npy"
            assert npy.exists()
            assert np.load(npy).shape == (2, 2)
        else:
            assert not list(tmp_path.rglob("*.npy"))

    def test_single_pass_reads_fits_predicts_and_writes_all_groups(
        self, tmp_path: Path, make_stage: "KMeansReadFitWriteStage"
    ) -> None:
        """The JSONL single-pass path fits once, predicts once, and writes every input group."""
        cache_path = tmp_path / "centroids"
        stage = make_stage(fit_data_fraction=None, filetype="jsonl", cache_path=str(cache_path))
        df = cudf.DataFrame({"id": [0, 1], "embeddings": [[1.0, 0.0], [0.0, 1.0]]})
        stage.kmeans.predict = Mock(return_value=cp.zeros(2 * len(df), dtype=cp.int32))
        tasks = [FileGroupTask(dataset_name="d", data=["first.jsonl", "second.jsonl"])]
        groups = [["first.jsonl"], ["second.jsonl"]]
        with (
            patch.object(stage, "_read_group", return_value=df) as read_group,
            patch.object(stage, "write_parquet") as write,
        ):
            results = stage._process_batch_single_pass(tasks, groups)

        assert [call.args[0] for call in read_group.call_args_list] == groups
        stage.kmeans.fit.assert_called_once()
        stage.kmeans.predict.assert_called_once()
        assert write.call_count == len(groups)
        assert len(results) == len(groups)
        # Only actor zero persists the shared model; the cache assertion keeps that contract visible.
        npy = cache_path / "kmeans_centroids.npy"
        assert npy.exists()
        assert np.load(npy).shape == (2, 2)
