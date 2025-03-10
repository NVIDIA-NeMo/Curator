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
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from dask.dataframe.utils import assert_eq
from transformers import AutoConfig, AutoModel, AutoTokenizer

from nemo_curator import SemDedup, SemDedupConfig
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.import_utils import gpu_only_import, gpu_only_import_from

cudf = gpu_only_import("cudf")
dask_cudf = gpu_only_import("dask_cudf")
EmbeddingCreator = gpu_only_import_from(
    "nemo_curator.modules.semantic_dedup.embeddings", "EmbeddingCreator"
)
pairwise_cosine_similarity = gpu_only_import_from(
    "nemo_curator.utils.semdedup_utils", "pairwise_cosine_similarity"
)
pairwise_cosine_similarity_batched = gpu_only_import_from(
    "nemo_curator.utils.semdedup_utils", "pairwise_cosine_similarity_batched"
)
if TYPE_CHECKING:
    from nemo_curator.utils.semdedup_utils import (
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
    def test_sem_dedup(
        self,
        dedup_data,
        tmpdir,
        n_clusters,
        gpu_client,
    ):
        print("client", gpu_client)

        cache_dir = os.path.join(tmpdir, "test_sem_dedup_cache")
        config = SemDedupConfig(
            cache_dir=cache_dir,
            n_clusters=n_clusters,
            eps_thresholds=[0.10],
            eps_to_extract=0.10,
        )

        sem_duplicates = SemDedup(
            config=config,
            input_column="text",
            id_column="id",
            id_column_type="int",
        )

        dedup_data_len = dedup_data.df.shape[0].compute()
        if n_clusters > dedup_data_len:
            # Number of records in the dataset should never be less than n_clusters
            with pytest.raises(ValueError):
                result = sem_duplicates(dedup_data)
        else:
            # Correctly returns the original dataset with no duplicates removed
            result = sem_duplicates(dedup_data)
            result_df = result.df.compute()
            duplicate_docs = [2, 3, 4, 200, 300]
            expected_df = cudf.Series(duplicate_docs, name="id")
            assert_eq(result_df["id"].sort_values(), expected_df, check_index=False)

    @pytest.mark.parametrize("n_clusters", [2, 3])
    def test_no_sem_dedup(
        self,
        non_dedup_data,
        tmpdir,
        n_clusters,
        gpu_client,
    ):
        print("client", gpu_client)

        cache_dir = os.path.join(tmpdir, "test_no_sem_dedup")
        config = SemDedupConfig(
            cache_dir=cache_dir,
            n_clusters=n_clusters,
            eps_thresholds=[0.10],
            eps_to_extract=0.10,
        )

        sem_duplicates = SemDedup(
            config=config,
            input_column="text",
            id_column="doc_id",
            id_column_type="str",
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
            duplicate_docs = ["doc_1", "doc_2"]
            expected_df = cudf.Series(duplicate_docs, name="doc_id")
            assert_eq(result_df["doc_id"].sort_values(), expected_df, check_index=False)

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


class TestPairwiseCosineSimilarity:
    def setup_method(self):
        # We create a 5x3 array where each row is a unit vector
        # The second and last two rows are the same
        input_arr = torch.tensor(
            np.asarray(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [1, 2, 3], [1, 2, 3]],
            ),
            dtype=torch.float32,
        )
        # Normalize the input array
        self.input_arr = input_arr / torch.norm(input_arr, dim=1, keepdim=True)
        self.expected_similarity = torch.tensor(
            [0.0000, 0.974631, 0.998190, 0.999618, 1.0000, 1.0000]
        )
        self.expected_indices = [0, 0, 1, 2, 0, 0]

    @pytest.mark.parametrize("device", [pytest.param("cuda", marks=pytest.mark.gpu)])
    def test_pairwise_cosine_similarity(self, device: Literal["cpu", "cuda"]):
        max_similarity, max_indices = pairwise_cosine_similarity(
            self.input_arr.to(device), device
        )
        torch.testing.assert_close(
            max_similarity, self.expected_similarity, rtol=1e-6, atol=1e-6
        )
        assert max_indices == self.expected_indices

    @pytest.mark.parametrize("device", [pytest.param("cuda", marks=pytest.mark.gpu)])
    @pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 5, 6])
    def test_pairwise_cosine_similarity_batched(
        self, device: Literal["cpu", "cuda"], batch_size: int
    ):
        max_similarity, max_indices = pairwise_cosine_similarity_batched(
            self.input_arr.to(device), device, batch_size
        )
        torch.testing.assert_close(max_similarity, self.expected_similarity)
        assert max_indices == self.expected_indices

    @pytest.mark.parametrize("device", [pytest.param("cuda", marks=pytest.mark.gpu)])
    @pytest.mark.parametrize("batch_size", [100, 512, 1024, 2048])
    def test_pairwise_cosine_similarity_batched_rand_array(
        self, device: Literal["cpu", "cuda"], batch_size: int
    ):
        N = 1024
        D = 512
        rand_arr = torch.randn(N, D, device=device)
        max_similarity, max_indices = pairwise_cosine_similarity(rand_arr, device)
        max_similarity_batched, max_indices_batched = (
            pairwise_cosine_similarity_batched(rand_arr, device, batch_size=batch_size)
        )
        torch.testing.assert_close(
            max_similarity, max_similarity_batched, rtol=1e-5, atol=1e-5
        )
        assert max_indices == max_indices_batched
