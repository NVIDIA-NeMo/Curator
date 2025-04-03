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
import os
import random
from typing import TYPE_CHECKING, Literal

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from dask.dataframe.utils import assert_eq
from sklearn.datasets import make_blobs
from transformers import AutoConfig, AutoModel, AutoTokenizer

from nemo_curator import SemDedup, SemDedupConfig
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.import_utils import gpu_only_import, gpu_only_import_from

cudf = gpu_only_import("cudf")
dask_cudf = gpu_only_import("dask_cudf")
cp = gpu_only_import("cupy")
EmbeddingCreator = gpu_only_import_from(
    "nemo_curator.modules.semantic_dedup.embeddings", "EmbeddingCreator"
)
ClusteringModel = gpu_only_import_from(
    "nemo_curator.modules.semantic_dedup.clusteringmodel", "ClusteringModel"
)
SemanticClusterLevelDedup = gpu_only_import_from(
    "nemo_curator.modules.semantic_dedup.semanticclusterleveldedup",
    "SemanticClusterLevelDedup",
)
add_l2_cosine_dist_to_centroid = gpu_only_import_from(
    "nemo_curator.utils.semdedup_utils",
    "add_l2_cosine_dist_to_centroid",
)
normalize_embeddings_col_in_df = gpu_only_import_from(
    "nemo_curator.utils.semdedup_utils", "normalize_embeddings_col_in_df"
)
get_array_from_df = gpu_only_import_from(
    "nemo_curator.utils.semdedup_utils", "get_array_from_df"
)
pairwise_cosine_similarity = gpu_only_import_from(
    "nemo_curator.utils.semdedup_utils", "pairwise_cosine_similarity"
)
pairwise_cosine_similarity_batched = gpu_only_import_from(
    "nemo_curator.utils.semdedup_utils", "pairwise_cosine_similarity_batched"
)
get_semantic_matches_per_cluster = gpu_only_import_from(
    "nemo_curator.utils.semdedup_utils", "get_semantic_matches_per_cluster"
)

if TYPE_CHECKING:
    import cupy as cp

    from nemo_curator.modules.semantic_dedup.clusteringmodel import ClusteringModel
    from nemo_curator.modules.semantic_dedup.semanticclusterleveldedup import (
        SemanticClusterLevelDedup,
    )
    from nemo_curator.modules.semantic_dedup.semdedup import SemDedup
    from nemo_curator.utils.semdedup_utils import (
        add_l2_cosine_dist_to_centroid,
        get_array_from_df,
        get_semantic_matches_per_cluster,
        normalize_embeddings_col_in_df,
        pairwise_cosine_similarity,
        pairwise_cosine_similarity_batched,
    )


@pytest.fixture
def dedup_data():
    df = cudf.DataFrame(
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
    df = dask_cudf.from_cudf(df, 2)
    return DocumentDataset(df)


@pytest.fixture
def non_dedup_data():
    df = cudf.DataFrame(
        {
            "doc_id": ["doc_1", "doc_2"],
            "text": [
                "The quick brown fox jumps over the lazy dog",
                "A test string",
            ],
        }
    )
    df = dask_cudf.from_cudf(df, 2)
    return DocumentDataset(df)


@pytest.mark.gpu
class TestSemDuplicates:
    @pytest.mark.parametrize("n_clusters", [3, 10])
    @pytest.mark.parametrize("id_col_type", ["int", "str"])
    @pytest.mark.parametrize("perform_removal", [True, False])
    def test_sem_dedup(
        self,
        dedup_data,
        tmpdir,
        n_clusters,
        id_col_type,
        perform_removal,
        gpu_client,
    ):
        cache_dir = os.path.join(tmpdir, "test_sem_dedup_cache")
        config = SemDedupConfig(
            cache_dir=cache_dir,
            n_clusters=n_clusters,
            eps_to_extract=0.10,
        )

        sem_duplicates = SemDedup(
            config=config,
            input_column="text",
            id_column="id",
            perform_removal=perform_removal,
        )
        # Convert id column to the specified type
        dedup_data.df["id"] = dedup_data.df["id"].astype(id_col_type)

        dedup_data_len = dedup_data.df.shape[0].compute()
        if n_clusters > dedup_data_len:
            # Number of records in the dataset should never be less than n_clusters
            with pytest.raises(ValueError):
                result = sem_duplicates(dedup_data)
        else:
            # Correctly returns the original dataset with no duplicates removed
            result = sem_duplicates(dedup_data)
            result_df = result.df.compute()
            docs_to_remove = [1, 100]
            if id_col_type == "str":
                docs_to_remove = list(map(str, docs_to_remove))

            if not perform_removal:
                expected_df = cudf.Series(docs_to_remove, name="id", dtype=id_col_type)
                assert_eq(result_df["id"].sort_values(), expected_df, check_index=False)
            else:
                assert_eq(
                    result_df,
                    dedup_data.df[~dedup_data.df["id"].isin(docs_to_remove)],
                    check_index=False,
                )

    @pytest.mark.parametrize("n_clusters", [2, 3])
    @pytest.mark.parametrize("perform_removal", [True, False])
    def test_no_sem_dedup(
        self,
        non_dedup_data,
        tmpdir,
        n_clusters,
        perform_removal,
        gpu_client,
    ):
        cache_dir = os.path.join(tmpdir, "test_no_sem_dedup")
        config = SemDedupConfig(
            cache_dir=cache_dir,
            n_clusters=n_clusters,
            eps_to_extract=0.10,
        )

        sem_duplicates = SemDedup(
            config=config,
            input_column="text",
            id_column="doc_id",
            perform_removal=perform_removal,
        )

        non_dedup_data_len = non_dedup_data.df.shape[0].compute()
        if n_clusters > non_dedup_data_len:
            # Number of records in the dataset should never be less than n_clusters
            with pytest.raises(ValueError):
                result = sem_duplicates(non_dedup_data)
        else:
            # Correctly returns the original dataset with no duplicates removed
            result = sem_duplicates(non_dedup_data)
            result_df = result.df.compute()
            if not perform_removal:
                assert len(result_df) == 0
            else:
                assert_eq(result_df, non_dedup_data.df, check_index=False)

    @pytest.mark.parametrize("pooling_strategy", ["last_token", "mean_pooling"])
    def test_embedding_creator_pooling_strategies(self, tmpdir, pooling_strategy):
        test_text_1 = "The quick brown fox jumps over the lazy dog"
        test_text_2 = "The brown fox jumps over the dog"
        test_texts = [test_text_1, test_text_2] * 32
        df = cudf.DataFrame({"text": test_texts})
        ddf = dask_cudf.from_cudf(df, 1)

        cache_dir = os.path.join(tmpdir, "test_embeddings_cache")

        embedding_creator = EmbeddingCreator(
            embedding_model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            embedding_batch_size=32,
            embedding_pooling_strategy=pooling_strategy,
            input_column="text",
            embedding_output_dir=os.path.join(cache_dir, "mean_embeddings"),
        )

        embeddings = embedding_creator.create_embeddings(ddf).compute()
        embeddings = embeddings["embeddings"].to_arrow().to_pylist()
        embeddings = np.array(embeddings)

        reference_embeddings = get_reference_embeddings(
            test_texts, pooling_strategy=pooling_strategy
        )

        assert np.allclose(
            embeddings, reference_embeddings, atol=1e-3
        ), "Embeddings should match reference embeddings"


def get_reference_embeddings(
    texts,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    pooling_strategy="last_token",
):
    """
    Get embeddings using either last token or mean pooling strategy.

    Args:
        texts: List of input texts
        model_name: Name or path of the model to use
        pooling_strategy: Either "last_token" for last token or "mean" for mean pooling
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to("cuda")
    model.eval()
    max_len_to_use = tokenizer.model_max_length
    if max_len_to_use > 1e5:
        max_len_to_use = AutoConfig.from_pretrained(model_name).max_position_embeddings
    max_seq_length: int = max_len_to_use

    embs = []
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                outputs = model(**inputs)

        if pooling_strategy == "last_token":
            embeddings = outputs.last_hidden_state[:, -1, :]
        elif pooling_strategy == "mean_pooling":
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        else:
            raise ValueError(
                "pooling_strategy must be either 'last_token' or 'mean_pooling'"
            )

        normed_emb = F.normalize(embeddings, dim=1).cpu()
        normed_emb = normed_emb.squeeze(0)
        embs.append(normed_emb)

    return np.array(embs)


@pytest.mark.gpu
class TestSemDedupUtils:
    def setup_method(self):
        # We create a 6x3 array where each row is a unit vector
        # The second and last two rows are the same
        input_embeddings = torch.tensor(
            np.asarray(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [1, 2, 3], [1, 2, 3]],
            ),
            dtype=torch.float32,
        )
        # Normalize the input array
        self.input_embeddings = input_embeddings / torch.norm(
            input_embeddings, dim=1, keepdim=True
        )
        self.expected_pairwise_similarity = np.array(
            [0.0000, 0.974631, 0.998190, 0.999618, 1.0000, 1.0000]
        )
        self.expected_indices = np.array([0, 0, 1, 2, 0, 0])

    def test_pairwise_cosine_similarity(self):
        max_similarity, max_indices = pairwise_cosine_similarity(
            self.input_embeddings, "cuda"
        )
        np.testing.assert_allclose(
            max_similarity.tolist(),
            self.expected_pairwise_similarity,
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_array_equal(max_indices.tolist(), self.expected_indices)

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 5, 6])
    def test_pairwise_cosine_similarity_batched(self, batch_size: int):
        max_similarity, max_indices = pairwise_cosine_similarity_batched(
            self.input_embeddings, "cuda", batch_size
        )
        np.testing.assert_allclose(
            max_similarity.tolist(),
            self.expected_pairwise_similarity,
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_array_equal(max_indices.tolist(), self.expected_indices)

    @pytest.mark.parametrize("batch_size", [100, 512, 1024, 2048])
    def test_pairwise_cosine_similarity_batched_rand_array(self, batch_size: int):
        N = 1024
        D = 512
        rand_arr = torch.randn(N, D, device="cuda")
        max_similarity, max_indices = pairwise_cosine_similarity(rand_arr, "cuda")
        max_similarity_batched, max_indices_batched = (
            pairwise_cosine_similarity_batched(rand_arr, "cuda", batch_size=batch_size)
        )
        np.testing.assert_allclose(
            max_similarity.tolist(),
            max_similarity_batched.tolist(),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_array_equal(
            max_indices.tolist(), max_indices_batched.tolist()
        )

    def test_get_array_from_df(self):
        df = cudf.DataFrame(
            {
                "embedding": [[3, 4, 5], [1, 2, 2], [1, 0, 0]],
            }
        )
        expected_array = cp.array(
            [
                [3, 4, 5],
                [1, 2, 2],
                [1, 0, 0],
            ]
        )
        cp.testing.assert_allclose(
            get_array_from_df(df, "embedding"), expected_array, rtol=1e-5, atol=1e-5
        )

    def test_normalize_embeddings_col_in_df(self):
        # Mock data setup
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
        normalized_embeddings = normalize_embeddings_col_in_df(df, "embedding")

        # Assert the normalized embeddings match the expected values
        cp.testing.assert_allclose(
            get_array_from_df(normalized_embeddings, "embedding"),
            expected_normalized,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_add_l2_cosine_dist_to_centroid(self):
        # Mock data setup
        df = cudf.DataFrame(
            {
                "nearest_cent": [0, 1, 0],
                "embedding": [
                    [1, 0],
                    [0, 1],
                    [0.6, 0.8],
                ],
            }
        )
        centroids = cp.array([[1, 0], [0, 1]])
        # Call the function
        df_with_distances = add_l2_cosine_dist_to_centroid(df, "embedding", centroids)

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

    @pytest.mark.parametrize("which_to_keep", ["hard", "easy"])
    @pytest.mark.parametrize("sim_metric", ["cosine", "l2"])
    @pytest.mark.parametrize("id_col_type", ["int", "str"])
    def test_get_semantic_matches_per_cluster(
        self,
        which_to_keep: Literal["hard", "easy"],
        sim_metric: Literal["cosine", "l2"],
        id_col_type: Literal["int", "str"],
        tmpdir,
    ):
        cluster_c = 0
        self.centroid = self.input_embeddings[:1]
        os.makedirs(os.path.join(tmpdir, f"nearest_cent={cluster_c}"), exist_ok=True)
        # Step 1) Simulate rank_within_cluster
        embeddings_df = cudf.DataFrame(
            {
                "embedding": self.input_embeddings.tolist(),
                "id": list(range(self.input_embeddings.shape[0])),
            }
        )
        embeddings_df["id"] = embeddings_df["id"].astype(id_col_type)
        # Step 1) Call add_l2_cosine_dist_to_centroid
        embeddings_df = add_l2_cosine_dist_to_centroid(
            df=embeddings_df.assign(nearest_cent=cluster_c),
            embedding_col="embedding",
            centroids=cp.asarray(self.centroid),
        )
        embeddings_df.to_parquet(
            os.path.join(tmpdir, f"nearest_cent={cluster_c}/file.parquet")
        )
        # Step 2) Call get_semantic_matches_per_cluster
        # this internally calls read_cluster_embeddings_and_sort_by_id and pairwise_cosine_similarity
        get_semantic_matches_per_cluster(
            cluster_id=cluster_c,
            emb_by_clust_dir=tmpdir,
            id_col="id",
            output_dir=tmpdir,
            embedding_col="embedding",
            which_to_keep=which_to_keep,
            sim_metric=sim_metric,
            batched_cosine_similarity=1024,
        )

        # Read the output
        output_df = pd.read_parquet(
            os.path.join(tmpdir, f"cluster_{cluster_c}.parquet")
        )
        # See https://github.com/NVIDIA/NeMo-Curator/issues/610 to understand the expected output and the walkthrough
        if which_to_keep == "hard":
            expected_ids = [3, 2, 1, 5, 4, 0]
            expected_max_ids = [3, 3, 2, 1, 5, 5]
            expected_cosine_sim_scores = np.array(
                [
                    0.0000,
                    0.99961,
                    0.99819,
                    0.974631,
                    1.0000,
                    1.0000,
                ],
                dtype=np.float32,
            )
        else:
            expected_ids = [0, 4, 5, 1, 2, 3]
            expected_max_ids = [0, 0, 0, 0, 1, 2]
            expected_cosine_sim_scores = np.array(
                [
                    0.0000,
                    1.0000,
                    1.0000,
                    0.97464,
                    0.99819,
                    0.999618,
                ],
                dtype=np.float32,
            )
        expected_df = pd.DataFrame(
            {
                "id": expected_ids,
                "max_id": expected_max_ids,
                "cosine_sim_score": expected_cosine_sim_scores,
            }
        )
        expected_df["id"] = expected_df["id"].astype(id_col_type)
        expected_df["max_id"] = expected_df["max_id"].astype(id_col_type)
        pd.testing.assert_frame_equal(
            output_df,
            expected_df,
            check_exact=False,
            rtol=1e-5,
            atol=1e-5,
        )


@pytest.mark.gpu
class TestSemanticDedupWithoutEmbeddingCreation:
    def setup_method(self):
        self.n_clusters = 5
        self.n_samples_per_cluster = [100 * (i + 1) for i in range(self.n_clusters)]
        self.n_features = 3
        # reset all random state here to undeterministic results
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.X, _ = make_blobs(
            n_samples=self.n_samples_per_cluster,
            centers=None,
            n_features=self.n_features,
            random_state=42,
        )
        # Convert to Dask DataFrame and then to cuDF
        df = pd.DataFrame({"id": np.arange(len(self.X)), "embeddings": self.X.tolist()})
        ddf = dd.from_pandas(df, npartitions=2)
        self.ddf = ddf.to_backend("cudf")

    def test_clustering_model(self, tmpdir):
        clustering_output_dir = os.path.join(tmpdir, "clustering_output")
        # Initialize ClusteringModel
        clustering_model = ClusteringModel(
            id_column="id",
            n_clusters=self.n_clusters,
            clustering_output_dir=clustering_output_dir,
            embedding_column="embeddings",
            random_state=42,
        )
        # TODO : remove this once we figure out why fusing is causing issues
        with dask.config.set({"optimization.fuse.active": False}):
            _ = clustering_model(DocumentDataset(self.ddf))

        # Check Directory Structure
        files = os.listdir(clustering_output_dir)
        expected_files = ["embs_by_nearest_center", "kmeans_centroids.npy"]
        for expected_file in expected_files:
            assert expected_file in files, f"The {expected_file} file should exist."

        # Check the results of embs_by_nearest_center
        for i in range(self.n_clusters):
            assert os.path.exists(
                os.path.join(
                    clustering_output_dir, "embs_by_nearest_center", f"nearest_cent={i}"
                )
            ), f"The nearest centroid file for cluster {i} should exist."
        embss_by_nearest_center = pd.read_parquet(
            os.path.join(clustering_output_dir, "embs_by_nearest_center")
        )
        # Check that embeddings are normalized
        np.testing.assert_almost_equal(
            sorted(np.stack(embss_by_nearest_center["embeddings"]).tolist()),
            sorted((self.X / np.linalg.norm(self.X, axis=1, keepdims=True)).tolist()),
        )
        num_samples_per_cluster = (
            embss_by_nearest_center["nearest_cent"].value_counts().to_dict()
        )
        np.testing.assert_allclose(
            sorted(num_samples_per_cluster.values()),
            sorted(self.n_samples_per_cluster),
            rtol=0.02,  # be within 2%
        )
        assert embss_by_nearest_center.shape[0] == len(self.X)
        assert embss_by_nearest_center.columns.tolist() == [
            "id",
            "embeddings",
            "l2_dist_to_cent",
            "cosine_dist_to_cent",
            "nearest_cent",
        ]

        # Check the results of kmeans_centroids.npy
        centroids = np.load(os.path.join(clustering_output_dir, "kmeans_centroids.npy"))
        assert centroids.shape == (self.n_clusters, self.n_features)
        # Centroids won't be normalized since they're just the mean of the embeddings
        assert not np.allclose(
            np.linalg.norm(centroids, axis=1, keepdims=True),
            np.ones_like(centroids),
        )

    def test_clustering_model_keep_all_columns(self, tmpdir):
        """Test that extra columns in the input are preserved when keep_all_columns is True."""
        clustering_output_dir = os.path.join(tmpdir, "clustering_output")
        # Create a new DataFrame with an extra column called "extra"
        df = pd.DataFrame(
            {
                "id": np.arange(len(self.X)),
                "embeddings": self.X.tolist(),
                "extra": ["foo"] * len(self.X),
            }
        )
        # Convert to Dask DataFrame and then to cuDF backend
        ddf_with_extra = dd.from_pandas(df, npartitions=2).to_backend("cudf")

        # Initialize ClusteringModel with keep_all_columns set to True.
        # We use the default sort_clusters=True to have consistent post-processing.
        clustering_model = ClusteringModel(
            id_column="id",
            n_clusters=self.n_clusters,
            clustering_output_dir=clustering_output_dir,
            embedding_column="embeddings",
            random_state=42,
            keep_all_columns=True,
        )
        # Run the clustering model on the dataset with the extra column.
        _ = clustering_model(DocumentDataset(ddf_with_extra))

        # Read the output parquet data produced after sorting clusters.
        embss_by_nearest_center = pd.read_parquet(
            os.path.join(clustering_output_dir, "embs_by_nearest_center")
        )

        # Verify that the extra column is present and its values are as expected.
        assert (
            "extra" in embss_by_nearest_center.columns
        ), "The extra column should be present when keep_all_columns is True."
        assert (
            embss_by_nearest_center["extra"].eq("foo").all()
        ), "The extra column should contain 'foo' for all rows."

    @pytest.mark.parametrize("which_to_keep", ["hard", "random", "easy"])
    @pytest.mark.parametrize("sim_metric", ["cosine", "l2"])
    def test_semantic_cluster_level_dedup(self, tmpdir, which_to_keep, sim_metric):
        clustering_output_dir = os.path.join(tmpdir, "clustering_output")
        semantic_extraction_output_dir = os.path.join(tmpdir, "extraction")
        # Initialize ClusteringModel
        clustering_model = ClusteringModel(
            id_column="id",
            n_clusters=self.n_clusters,
            clustering_output_dir=clustering_output_dir,
            embedding_column="embeddings",
            random_state=42,
        )
        with dask.config.set({"optimization.fuse.active": False}):
            _ = clustering_model(DocumentDataset(self.ddf))

        semantic_cluster_level_dedup = SemanticClusterLevelDedup(
            n_clusters=self.n_clusters,
            emb_by_clust_dir=os.path.join(
                clustering_output_dir, "embs_by_nearest_center"
            ),
            id_column="id",
            which_to_keep=which_to_keep,
            sim_metric=sim_metric,
            output_dir=semantic_extraction_output_dir,
            embedding_column="embeddings",
            batched_cosine_similarity=20,
        )

        # Call compute_semantic_match_dfs
        semantic_cluster_level_dedup.compute_semantic_match_dfs()

        output_samples_per_cluster = []
        semdedup_pruning_tables_df = []
        # Check content of semdedup_pruning_tables
        for i in range(self.n_clusters):
            cluster_i_path = os.path.join(
                semantic_extraction_output_dir,
                "semdedup_pruning_tables",
                f"cluster_{i}.parquet",
            )
            assert os.path.exists(cluster_i_path)
            df = pd.read_parquet(cluster_i_path)
            output_samples_per_cluster.append(df.shape[0])
            assert df.columns.tolist() == [
                "id",
                "max_id",
                "cosine_sim_score",
            ]
            semdedup_pruning_tables_df.append(df)

        semdedup_pruning_tables_df = pd.concat(semdedup_pruning_tables_df)
        np.testing.assert_allclose(
            sorted(output_samples_per_cluster),
            self.n_samples_per_cluster,
            rtol=0.02,  # be within 2%
        )

        # Call extract_dedup_data
        semantic_cluster_level_dedup.extract_dedup_data(eps_to_extract=0.01)
        # Check content of unique_ids
        unique_ids_path = os.path.join(
            semantic_extraction_output_dir, "unique_ids_0.01.parquet"
        )
        assert os.path.exists(unique_ids_path)
        unique_ids_df = pd.read_parquet(unique_ids_path)
        assert unique_ids_df.columns.tolist() == [
            "id",
            "cosine_dist_to_cent",
            "cluster",
        ]

        # Check content of semdedup_pruning_table with the filter matches the unique_ids
        semdedup_pruning_tables_df_filtered = semdedup_pruning_tables_df[
            semdedup_pruning_tables_df["cosine_sim_score"] >= 1 - 0.01
        ]
        assert len(semdedup_pruning_tables_df_filtered) == len(unique_ids_df)
        assert set(semdedup_pruning_tables_df_filtered["id"].to_list()) == set(
            unique_ids_df["id"].to_list()
        )

        # Check content of summary file
        summary_path = os.path.join(
            semantic_extraction_output_dir, "dedup_summary_0.01.csv"
        )
        assert os.path.exists(summary_path)
        df = pd.read_csv(summary_path)
        if which_to_keep == "hard":
            _kept, _removed = 29, 1471
        elif which_to_keep == "easy":
            _kept, _removed = 5, 1495
        else:
            _kept, _removed = 17, 1483

        pd.testing.assert_frame_equal(
            df,
            pd.DataFrame(
                {
                    "eps": [0.01],
                    "kept": [_kept],
                    "removed": [_removed],
                    "total": [len(self.X)],
                }
            ),
        )
        # Ensure that the unique_ids are also correct (this implicitly checks for semdedup_pruning_tables output)
        assert len(unique_ids_df) == _removed
        assert len(set(unique_ids_df["id"].to_list())) == _removed
