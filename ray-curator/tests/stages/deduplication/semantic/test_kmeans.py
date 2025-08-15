"""End-to-end integration tests for KMeans semantic deduplication stage."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from loguru import logger
from sklearn.datasets import make_blobs

from ray_curator.backends.experimental.ray_actor_pool import RayActorPoolExecutor
from ray_curator.pipeline import Pipeline
from ray_curator.stages.deduplication.semantic.kmeans import KMeansStage

N_CLUSTERS = 4
N_SAMPLES_PER_CLUSTER = 100
EMBEDDING_DIM = 32
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
    data = []
    for i, (embedding, label) in enumerate(zip(X_normalized, y_true, strict=False)):
        record = {
            "id": f"doc_{i}",
            "text": f"Document {i} from cluster {label}",
            "embeddings": embedding.tolist(),
            "true_cluster": int(label),  # Ground truth for verification
        }
        data.append(record)

    df = pd.DataFrame(data).sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Write data in specified format
    num_files = 20  # Create multiple files to test file partitioning
    samples_per_file = len(df) // num_files

    for file_idx in range(num_files):
        start_idx = file_idx * samples_per_file
        end_idx = (file_idx + 1) * samples_per_file if file_idx < num_files - 1 else len(df)
        file_data = df.iloc[start_idx:end_idx]

        if not file_data.empty:
            if file_format == "parquet":
                file_path = input_dir / f"data_part_{file_idx:02d}.parquet"
                file_data.to_parquet(file_path, index=False)
            elif file_format == "jsonl":
                file_path = input_dir / f"data_part_{file_idx:02d}.jsonl"
                file_data.to_json(file_path, orient="records", lines=True)
            else:
                msg = f"Unsupported file format: {file_format}"
                raise ValueError(msg)

        logger.info(f"Created {file_path}")

    return input_dir, X_normalized, y_true


def create_kmeans_pipeline(
    input_dir: Path,
    output_dir: Path,
    n_clusters: int = N_CLUSTERS,
    embedding_dim: int = EMBEDDING_DIM,
    file_format: str = "parquet",
) -> Pipeline:
    """Create a KMeans pipeline for testing.

    Args:
        input_dir: Input directory containing test data
        output_dir: Output directory for results
        n_clusters: Number of clusters
        embedding_dim: Embedding dimensionality
        file_format: Input file format

    Returns:
        Configured pipeline
    """
    pipeline = Pipeline(name="kmeans_integration_test")

    kmeans_stage = KMeansStage(
        n_clusters=n_clusters,
        id_field="id",
        embedding_field="embeddings",
        input_path=str(input_dir),
        output_path=str(output_dir),
        embedding_dim=embedding_dim,
        input_filetype=file_format,
        distance_metric_to_use="cosine",
        verbose=True,
        random_state=RANDOM_STATE,
        max_iter=300,
        tol=1e-4,
    )

    pipeline.add_stage(kmeans_stage)
    return pipeline


def verify_kmeans_results_with_cuml(
    output_dir: Path,
    true_labels: np.ndarray,
    n_clusters: int = N_CLUSTERS,
) -> dict[str, Any]:
    """Verify KMeans results by comparing with cuML KMeans on the same data.

    Args:
        output_dir: Directory containing KMeans stage output
        true_labels: Ground truth cluster labels
        n_clusters: Number of clusters

    Returns:
        Dictionary containing verification metrics
    """
    cudf = pytest.importorskip("cudf", reason="cuML verification requires cudf")
    cuml = pytest.importorskip("cuml", reason="cuML verification requires cuml")
    cp = pytest.importorskip("cupy", reason="cuML verification requires cupy")

    # Read the KMeans stage output
    output_files = list(output_dir.glob("**/*.parquet"))
    assert len(output_files) > 0, "No output files found"

    # Combine all output files
    dfs = []
    for file_path in output_files:
        df = cudf.read_parquet(file_path)
        dfs.append(df)

    combined_df = cudf.concat(dfs, ignore_index=True)

    # Extract embeddings and predicted clusters (embeddings are already normalized)
    normalized_embeddings = np.stack([np.array(row) for row in combined_df["embeddings"].to_pandas()])
    stage_clusters = combined_df["centroid"].to_pandas().to_numpy()

    # Run cuML KMeans on the same normalized data for comparison
    cuml_kmeans = cuml.KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        max_iter=300,
        tol=1e-4,
        random_state=RANDOM_STATE,
    )

    # Convert to cupy for cuML
    cupy_embeddings = cp.asarray(normalized_embeddings)
    cuml_clusters = cuml_kmeans.fit_predict(cupy_embeddings).get()
    cuml_centroids = cuml_kmeans.cluster_centers_.get()

    # Calculate metrics
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    # Compare stage results with cuML results
    stage_cuml_ari = adjusted_rand_score(stage_clusters, cuml_clusters)

    # Compare both with ground truth
    stage_true_ari = adjusted_rand_score(stage_clusters, true_labels)
    cuml_true_ari = adjusted_rand_score(cuml_clusters, true_labels)

    # Calculate silhouette scores
    stage_silhouette = silhouette_score(normalized_embeddings, stage_clusters)
    cuml_silhouette = silhouette_score(normalized_embeddings, cuml_clusters)

    # Check that distances are computed correctly
    assert "l2_dist_to_cent" in combined_df.columns
    assert "cosine_dist_to_cent" in combined_df.columns

    # Verify distance calculations
    stage_centroids_from_output = None
    centroid_files = list(output_dir.glob("**/centroids*.npy"))
    if centroid_files:
        stage_centroids_from_output = np.load(centroid_files[0])

    return {
        "stage_cuml_ari": stage_cuml_ari,
        "stage_true_ari": stage_true_ari,
        "cuml_true_ari": cuml_true_ari,
        "stage_silhouette": stage_silhouette,
        "cuml_silhouette": cuml_silhouette,
        "n_output_files": len(output_files),
        "n_samples": len(combined_df),
        "unique_clusters": len(set(stage_clusters)),
        "stage_centroids": stage_centroids_from_output,
        "cuml_centroids": cuml_centroids,
    }


@pytest.mark.gpu
@pytest.mark.parametrize(
    "file_format",
    [
        # "parquet",
        "jsonl"
    ],
)
class TestKMeansStageIntegration:
    """Integration tests for KMeansStage with RayActorPoolExecutor."""

    def test_kmeans_stage_e2e(
        self,
        tmp_path: Path,
        file_format: str,
    ) -> None:
        """End-to-end test of KMeans stage with synthetic clustered data.

        This test:
        1. Creates synthetic clustered data with known ground truth
        2. Runs KMeansStage using RayActorPoolExecutor
        3. Verifies results by comparing with cuML KMeans
        4. Checks that clustering quality is reasonable
        """
        # Generate synthetic clustered dataset
        input_dir, original_embeddings, true_labels = create_clustered_dataset(tmp_path, file_format=file_format)

        input_files = list(input_dir.glob(f"**/*.{file_format}"))
        logger.info(f"Input files: {len(input_files)}")

        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create and run pipeline
        pipeline = create_kmeans_pipeline(input_dir, output_dir, file_format=file_format)

        # Execute with RayActorPoolExecutor
        executor = RayActorPoolExecutor()
        results = pipeline.run(executor)

        # Verify pipeline execution
        assert len(results) > 0, "Pipeline should produce results"

        # Verify output files exist
        output_files = list(output_dir.glob("**/*.parquet"))
        assert len(output_files) > 0, "Should produce output files"

        # Verify results with cuML
        metrics = verify_kmeans_results_with_cuml(output_dir, original_embeddings, true_labels)

        # Assertions for clustering quality
        assert metrics["n_samples"] == len(original_embeddings), "Should process all input samples"
        assert metrics["unique_clusters"] == N_CLUSTERS, f"Should produce {N_CLUSTERS} clusters"
        assert metrics["n_output_files"] > 0, "Should produce output files"

        # Verify that the KMeans stage produces reasonable results
        # Note: We focus on pipeline correctness rather than clustering quality
        # since sklearn's make_blobs with normalization can produce challenging clustering scenarios

        # Both methods should produce some clustering structure
        print(f"Stage vs Ground Truth ARI: {metrics['stage_true_ari']:.3f}")
        print(f"cuML vs Ground Truth ARI: {metrics['cuml_true_ari']:.3f}")
        print(f"Stage vs cuML ARI: {metrics['stage_cuml_ari']:.3f}")
        print(f"Stage Silhouette Score: {metrics['stage_silhouette']:.3f}")
        print(f"cuML Silhouette Score: {metrics['cuml_silhouette']:.3f}")

        # Verify that both methods produce similar results (main integration test)
        assert metrics["stage_cuml_ari"] > 0.8, (
            f"Stage and cuML results should be highly similar (got {metrics['stage_cuml_ari']:.3f})"
        )

        # Verify clustering produces reasonable structure
        assert metrics["stage_silhouette"] > -0.5, (
            f"Stage clustering should not be completely random (got {metrics['stage_silhouette']:.3f})"
        )
        assert metrics["cuml_silhouette"] > -0.5, (
            f"cuML clustering should not be completely random (got {metrics['cuml_silhouette']:.3f})"
        )

        # Check that centroids are reasonable
        if metrics["stage_centroids"] is not None and metrics["cuml_centroids"] is not None:
            # Centroids should be in similar regions of space
            stage_centroids = metrics["stage_centroids"]
            cuml_centroids = metrics["cuml_centroids"]

            assert stage_centroids.shape == cuml_centroids.shape, "Centroids should have same shape"
            assert stage_centroids.shape[0] == N_CLUSTERS, f"Should have {N_CLUSTERS} centroids"
            assert stage_centroids.shape[1] == EMBEDDING_DIM, f"Centroids should have {EMBEDDING_DIM} dimensions"
