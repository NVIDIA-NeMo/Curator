from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ray_curator.backends.base import WorkerMetadata
from ray_curator.stages.base import CompositeStage, ProcessingStage
from ray_curator.stages.deduplication.id_generator import CURATOR_ID_GENERATOR_ACTOR_NAME, IdGenerator
from ray_curator.stages.deduplication.io_utils import DeduplicationIO
from ray_curator.stages.resources import Resources
from ray_curator.tasks import FileGroupTask, _EmptyTask

from .kmeans_io import BatchedFileGroupTask, KMeansFilePartitioningStage
from .utils import get_array_from_df

if TYPE_CHECKING:
    import cudf
    import cupy as cp


# Column names
L2_DIST_TO_CENT_COL = "l2_dist_to_cent"
COSINE_DIST_TO_CENT_COL = "cosine_dist_to_cent"


class KMeansReadFitWriteStage(ProcessingStage[BatchedFileGroupTask, _EmptyTask], DeduplicationIO):
    """KMeans clustering stage that requires RAFT for distributed processing."""

    def __init__(
        self,
        id_field: str,
        embedding_field: str,
        output_path: str,
        n_clusters: int,
        distance_metric_to_use: Literal["l2", "cosine"] | None = "cosine",
        verbose: bool = False,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int = 1,
        init: Literal["k-means||", "random"] | np.ndarray = "k-means||",
        n_init: int | Literal["auto"] = 1,
        oversampling_factor: float = 2.0,
        max_samples_per_batch: int = 1 << 15,
        input_storage_options: dict[str, Any] | None = None,
        output_storage_options: dict[str, Any] | None = None,
    ):
        """KMeans clustering stage that requires RAFT for distributed processing.

        Args:
            id_field (str): The column name of the id column.
            embedding_field (str): The column name of the embedding column.
            output_path (str): The path to the output directory.
            n_clusters (int): The number of clusters to create.
                # id_generator (IdGenerator): The id generator to use.
            which_to_keep (Literal["hard", "easy", "random"]): TODO: Add explanation
            distance_metric_to_use (Literal["l2", "cosine"] | None): TODO: Add explanation
            verbose (bool): Whether to print verbose output.
            max_iter (int): The maximum number of iterations to run.
            tol (float): Tolerance for stopping criteria of the kmeans algorithm.
            random_state (int): Seed for the random number generator. Unseeded by default. Does not currently fully guarantee the exact same results.
            init (Literal["k-means||", "random"] | np.ndarray): 'scalable-k-means++' or 'k-means||': Uses fast and stable scalable kmeans++ initialization. 'random': Choose 'n_cluster' observations (rows) at random from data for the initial centroids. If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
            n_init (int | Literal["auto"]): Number of times the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
            oversampling_factor (float): The amount of points to sample in scalable k-means++ initialization for potential centroids. Increasing this value can lead to better initial centroids at the cost of memory. The total number of centroids sampled in scalable k-means++ is oversampling_factor * n_clusters * 8.
            max_samples_per_batch (int): The number of data samples to use for batches of the pairwise distance computation. This computation is done throughout both fit predict. The default should suit most cases. The total number of elements in the batched pairwise distance computation is max_samples_per_batch * n_clusters. It might become necessary to lower this number when n_clusters becomes prohibitively large.
        """
        self.id_field = id_field
        self.embedding_field = embedding_field
        self.output_path = output_path
        self.n_clusters = n_clusters
        if distance_metric_to_use == "l2":
            self.distance_col = L2_DIST_TO_CENT_COL
        elif distance_metric_to_use == "cosine":
            self.distance_col = COSINE_DIST_TO_CENT_COL
        else:
            msg = "Distance metric must be either 'l2' or 'cosine'"
            raise ValueError(msg)
        self.verbose = verbose
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init
        self.n_init = n_init
        self.oversampling_factor = oversampling_factor
        self.max_samples_per_batch = max_samples_per_batch

        self.input_storage_options = input_storage_options
        self.output_storage_options = output_storage_options

        self._name = "KMeansStage"
        self._resources = Resources(cpus=1.0, gpus=1.0)
        # TODO: Do we need this?
        self.id_generator: IdGenerator | None = None

    # TODO: Use KMeansFileGroupTask
    def process(self, task: FileGroupTask) -> FileGroupTask:
        import cupy as cp

        num_rows, dfs = 0, []

        # Read in all the data
        for file_paths in task.data:
            if task.filetype == "parquet":
                df = self.read_parquet(
                    file_paths, columns=[self.id_field, self.embedding_field], storage_options=self.input_storage_options
                )
            elif task.filetype == "jsonl":
                df = self.read_jsonl(
                    file_paths, columns=[self.id_field, self.embedding_field], storage_options=self.input_storage_options
                )
            else:
                msg = f"Unsupported data type: {task.filetype}"
                raise ValueError(msg)

            # normalize the embeddings
            df = self.normalize_embeddings_col_in_df(df, self.embedding_field)
            dfs.append(df)
            num_rows += len(df)

        # Concatenate all embeddings in cupy to avoid the 2bn limit in cudf
        cupy_arr = cp.concatenate([get_array_from_df(df, self.embedding_field) for df in dfs], axis=0)
        current_row_idx = 0
        centroids = self.kmeans.fit_predict(cupy_arr).astype(cp.int32)
        for df in dfs:
            df["centroid"] = centroids[current_row_idx : current_row_idx + len(df)]
            current_row_idx += len(df)
        del cupy_arr

        # Assign distance and write to parquet
        for i in range(len(dfs)):
            dfs[i] = self._assign_distances(dfs[i], self.embedding_field, self.kmeans.cluster_centers_)
            self.write_parquet(
                dfs[i],
                self.output_path,
                partition_file_name=f"{task.task_id}_{i}.parquet",
                partition_cols=["centroid"],
                index=False,
                storage_options=self.output_storage_options,
            )

        return _EmptyTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
            data=None,
        )

    def setup(self, _: WorkerMetadata | None = None) -> None:
        from cuml.cluster.kmeans_mg import KMeansMG as cumlKMeans

        if not hasattr(self, "_raft_handle"):
            msg = "RAFT handle not found. Make sure the stage is initialized with RAFT"
            raise ValueError(msg)

        self.kmeans = cumlKMeans(
            handle=self._raft_handle,
            output_type="cupy",
            init=self.init,
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
            n_init=self.n_init,
            oversampling_factor=self.oversampling_factor,
            max_samples_per_batch=self.max_samples_per_batch,
            convert_dtype=False,
        )
        self.id_generator = IdGenerator.options(
            name=CURATOR_ID_GENERATOR_ACTOR_NAME, get_if_exists=True, lifetime="detached"
        ).remote()

    @staticmethod
    def normalize_embeddings_col_in_df(df: "cudf.DataFrame", embedding_col: str) -> "cudf.DataFrame":
        import cupy as cp
        import torch
        from crossfit.backend.cudf.series import create_list_series_from_1d_or_2d_ar  # TODO: move to ray_curator

        tensor = torch.Tensor(get_array_from_df(df, embedding_col))
        normalized_tensor = tensor / torch.norm(tensor, dim=1, keepdim=True)
        df[embedding_col] = create_list_series_from_1d_or_2d_ar(cp.asarray(normalized_tensor), index=df.index)
        return df

    @staticmethod
    def _assign_distances(df: "cudf.DataFrame", embedding_col: str, centroids: "cp.ndarray") -> "cudf.DataFrame":
        import cupy as cp

        """
        Computes the L2 distance to nearest centroid to each embedding in the DataFrame.
        Embeddings are normalized. For cosine we'll need to normalize the centroids as well.
        """
        normalized_embeddings = get_array_from_df(df, embedding_col)
        # We normalize the centroids as well for cosine distance
        normalized_centroids = centroids / cp.linalg.norm(centroids, axis=1, keepdims=True)

        df[L2_DIST_TO_CENT_COL] = cp.sqrt(
            cp.sum((normalized_embeddings - centroids[df["centroid"].values]) ** 2, axis=1)
        )
        df[COSINE_DIST_TO_CENT_COL] = 1 - (
            cp.sum(
                normalized_embeddings * normalized_centroids[df["centroid"].values],
                axis=1,
            )
        )
        return df

    def ray_stage_spec(self) -> dict[str, Any]:
        return {
            "is_raft_actor": True,
        }


@dataclass
class KMeansStage(CompositeStage[_EmptyTask, _EmptyTask]):
    """KMeans clustering stage that requires RAFT for distributed processing."""

    n_clusters: int
    id_field: str
    embedding_field: str
    output_path: str
    # Read args
    input_path: str | list[str]
    embedding_dim: int | None = None
    input_filetype: Literal["jsonl", "parquet"] = "parquet"
    input_file_extensions: list[str] | None = None
    input_storage_options: dict[str, Any] | None = None
    input_file_limit: int | None = None
    # KMeans args
    distance_metric_to_use: Literal["l2", "cosine"] | None = "cosine"
    verbose: bool = False
    max_iter: int = 300
    tol: float = 1e-4
    random_state: int = 1
    init: Literal["k-means||", "random"] | np.ndarray = "k-means||"
    n_init: int | Literal["auto"] = 1
    oversampling_factor: float = 2.0
    max_samples_per_batch: int = 1 << 15
    # Write args
    output_storage_options: dict[str, Any] | None = None

    """KMeans clustering stage that requires RAFT for distributed processing.

    Args:
        id_field (str): The column name of the id column.
        embedding_field (str): The column name of the embedding column.
        output_path (str): The path to the output directory.
        n_clusters (int): The number of clusters to create.
            # id_generator (IdGenerator): The id generator to use.
        which_to_keep (Literal["hard", "easy", "random"]): TODO: Add explanation
        distance_metric_to_use (Literal["l2", "cosine"] | None): TODO: Add explanation
        verbose (bool): Whether to print verbose output.
        max_iter (int): The maximum number of iterations to run.
        tol (float): Tolerance for stopping criteria of the kmeans algorithm.
        random_state (int): Seed for the random number generator. Unseeded by default. Does not currently fully guarantee the exact same results.
        init (Literal["k-means||", "random"] | np.ndarray): 'scalable-k-means++' or 'k-means||': Uses fast and stable scalable kmeans++ initialization. 'random': Choose 'n_cluster' observations (rows) at random from data for the initial centroids. If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
        n_init (int | Literal["auto"]): Number of times the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
        oversampling_factor (float): The amount of points to sample in scalable k-means++ initialization for potential centroids. Increasing this value can lead to better initial centroids at the cost of memory. The total number of centroids sampled in scalable k-means++ is oversampling_factor * n_clusters * 8.
        max_samples_per_batch (int): The number of data samples to use for batches of the pairwise distance computation. This computation is done throughout both fit predict. The default should suit most cases. The total number of elements in the batched pairwise distance computation is max_samples_per_batch * n_clusters. It might become necessary to lower this number when n_clusters becomes prohibitively large.
    """

    def __post_init__(self):
        """Initialize parent class after dataclass initialization."""
        super().__init__()

    def decompose(self) -> list[ProcessingStage]:
        return [
            KMeansFilePartitioningStage(
                file_paths=self.input_path,
                filetype=self.input_filetype,
                embedding_dim=self.embedding_dim,
                num_output_partitions=None,
                file_extensions=self.input_file_extensions,
                storage_options=self.input_storage_options,
                limit=self.input_file_limit,
            ),
            KMeansReadFitWriteStage(
                id_field=self.id_field,
                embedding_field=self.embedding_field,
                output_path=self.output_path,
                n_clusters=self.n_clusters,
                distance_metric_to_use=self.distance_metric_to_use,
                verbose=self.verbose,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
                init=self.init,
                n_init=self.n_init,
                oversampling_factor=self.oversampling_factor,
                max_samples_per_batch=self.max_samples_per_batch,
                input_storage_options=self.input_storage_options,
                output_storage_options=self.output_storage_options,
            ),
        ]
