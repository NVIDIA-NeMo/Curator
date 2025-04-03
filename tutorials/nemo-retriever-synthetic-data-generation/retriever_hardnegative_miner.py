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

import importlib
import itertools

import numpy as np
import pandas as pd
from dask.base import normalize_token
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from nemo_curator import ClusteringModel
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import load_object_on_worker

config = importlib.import_module(
    "tutorials.nemo-retriever-synthetic-data-generation.config.config"
)
RetrieverHardNegativeMiningConfig = config.RetrieverHardNegativeMiningConfig


def create_nim_client(base_url, api_key):
    openai_client = OpenAI(base_url=base_url, api_key=api_key)
    return openai_client


def create_hf_model(model_name_or_path):
    return SentenceTransformer(model_name_or_path, trust_remote_code=True)


class HardNegativeMiner:
    """
    Main class that generates annotated training datasets for retriever customization
    This the main class that performs hard negative mining, it takes in
    (questions, documents) tuples as inputs and generates
    (questions, positive documents, negative documents) triplets as outputs

    """

    def __init__(
        self,
        cfg: RetrieverHardNegativeMiningConfig,
    ):

        self.model_name = cfg.model_name
        self.model_type = cfg.model_type
        self.base_url = cfg.base_url
        self.api_key = cfg.api_key
        self.truncate = cfg.truncate
        self.n_hard_negatives = cfg.hard_negatives_to_mine

        if cfg.passage_prefix:
            self.passage_prefix = cfg.passage_prefix
        if cfg.query_prefix:
            self.query_prefix = cfg.query_prefix
        if cfg.hard_neg_mining_algorithm:
            self.hard_neg_mining_algorithm = cfg.hard_neg_mining_algorithm
        else:
            print(
                "hard negative mining algorithm not mentioned in config, using default"
            )
            self.hard_neg_mining_algorithm = "topk_percpos"
        if self.hard_neg_mining_algorithm == "topk_percpos":
            if cfg.percpos:
                self.percpos = cfg.percpos
            else:
                self.percpos = 0.95
        elif self.hard_neg_mining_algorithm == "topk_abs":
            if cfg.max_hardness_threshold:
                self.max_neg_score_threshold = cfg.max_hardness_threshold
            else:
                raise ValueError("Hard negative threshold is required!")
            if cfg.min_hardness_threshold:
                self.min_neg_score_threshold = cfg.min_hardness_threshold
            else:
                self.min_neg_score_threshold = 0.0

        if cfg.min_number_clusters:
            self.min_number_clusters = cfg.min_number_clusters
        if cfg.cluster_output_dir:
            self.cluster_output_dir = cfg.cluster_output_dir
        if cfg.logger_output_dir:
            self.logger_output_dir = cfg.logger_output_dir

    def __dask_tokenize__(self):
        return normalize_token(HardNegativeMiner)

    def assign_ids(self, partition):
        return partition.assign(doc_id=np.arange(len(partition)) + partition.index[0])

    def repartition_semantic_similarity(
        self, dataset: DocumentDataset
    ) -> DocumentDataset:
        df = dataset.df
        n_data = df.shape[0].compute()  # number of row items
        print(f"number of documents in the datasets = {n_data}")
        if self.min_number_clusters >= n_data:
            print("Using too many clusters not recommended!")
            print("Using 1/10th number of datapoints as number of clusters instead.")
            n_clusters = min(1, np.floor(n_data / 10))
        else:
            n_clusters = self.min_number_clusters

        print("Number of clusters used = {}".format(n_clusters))
        assert "doc_id" not in df.columns
        df["embeddings"] = ""  # refers to document embeddings
        df = df.explode("documents")
        df = df.map_partitions(self._get_doc_embeddings, meta=df)
        # df = dd.from_pandas(pdf)
        df = df.map_partitions(self.assign_ids)
        embeddings_dataset = DocumentDataset(df)
        self.clustering_model = ClusteringModel(
            id_column="doc_id",
            max_iter=100,
            n_clusters=n_clusters,
            clustering_output_dir=self.cluster_output_dir,
            logger=self.logger_output_dir,
            keep_all_columns=True,
        )
        clustered_dataset = self.clustering_model(embeddings_dataset)
        df_c = clustered_dataset.df
        df_c = df_c[["documents", "question"]]

        return DocumentDataset(df_c)

    def _get_doc_embeddings(self, p_df: pd.DataFrame):

        if self.model_type == "nvidia":
            self.client = load_object_on_worker(
                attr="nim_embedding_model",
                load_object_function=create_nim_client,
                load_object_kwargs={"base_url": self.base_url, "api_key": self.api_key},
            )

            p_df["embeddings"] = p_df["documents"].map(
                lambda t: self._get_nim_embedding(t, "passage")
            )
        elif self.model_type == "hf":
            self.hf_model = load_object_on_worker(
                attr="hf_embedding_model",
                load_object_function=create_hf_model,
                load_object_kwargs={"model_name_or_path": self.model_name},
            )

            p_df["embeddings"] = p_df["documents"].map(
                lambda t: self._get_hf_embedding(t, self.passage_prefix)
            )
        return p_df

    def _groupby_question(self, pdf):
        pdf2 = pdf.groupby("question").agg({"documents": set})
        pdf2["documents"] = pdf2["documents"].map(lambda x: list(x))
        return pdf2

    def __call__(self, dataset: DocumentDataset) -> DocumentDataset:

        df = dataset.df
        df = df.to_backend("pandas")
        df = df[["question", "documents"]]
        df = df.map_partitions(self._groupby_question).reset_index()
        print("Number partitions in dataset = {}".format(df.npartitions))

        df["neg_doc_scores"] = ""
        df["neg_doc"] = ""
        df["doc_embed"] = ""
        df["query_embed"] = ""
        df["min_pos_score"] = ""

        df = df.map_partitions(self._process_partition, meta=df)

        df = df.rename(columns={"documents": "pos_doc"})
        df = df[["question", "pos_doc", "neg_doc"]]

        return DocumentDataset(df)

    def _process_partition(self, df_p: pd.DataFrame):

        if self.model_type == "nvidia":
            self.client = load_object_on_worker(
                attr="nim_embedding_model",
                load_object_function=create_nim_client,
                load_object_kwargs={"base_url": self.base_url, "api_key": self.api_key},
            )
            df_p["doc_embed"] = df_p["documents"].map(
                lambda pgs: [self._get_nim_embedding(t, "passage") for t in pgs]
            )
            df_p["query_embed"] = df_p["question"].map(
                lambda x: self._get_nim_embedding(x, "query")
            )
        elif self.model_type == "hf":
            self.hf_model = load_object_on_worker(
                attr="hf_embedding_model",
                load_object_function=create_hf_model,
                load_object_kwargs={"model_name_or_path": self.model_name},
            )
            df_p["doc_embed"] = df_p["documents"].map(
                lambda pgs: [
                    self._get_hf_embedding(t, self.passage_prefix) for t in pgs
                ]
            )
            df_p["query_embed"] = df_p["question"].map(
                lambda x: self._get_hf_embedding(x, self.query_prefix)
            )

        doc_embeds = list(itertools.chain(*df_p["doc_embed"].to_list()))
        docs = list(itertools.chain(*df_p["documents"].to_list()))

        if self.hard_neg_mining_algorithm == "topk_abs":
            df_p["neg_doc_scores"] = df_p[["query_embed", "documents"]].apply(
                lambda row: self._get_scores_topk_abs(
                    row["query_embed"], doc_embeds, docs, row["documents"]
                ),
                axis=1,
            )

        elif self.hard_neg_mining_algorithm == "topk_percpos":
            df_p["min_pos_score"] = df_p[["query_embed", "doc_embed"]].apply(
                lambda row: self._get_min_pos_score(row), axis=1
            )
            df_p["neg_doc_scores"] = df_p[["query_embed", "min_pos_score"]].apply(
                lambda row: self._get_scores_topk_percpos(row, doc_embeds, docs), axis=1
            )

        df_p["neg_doc"] = df_p["neg_doc_scores"].map(
            lambda x: [doc for doc, score in x]
        )
        return df_p

    def _get_min_pos_score(self, row):
        x_ = np.array(row["query_embed"])
        y_ = np.array(row["doc_embed"])  # document embeddings
        scores = np.dot(x_, y_.T)
        return np.min(scores)

    def _get_scores_topk_percpos(self, row, docs_embed, docs):
        x_ = np.array(row["query_embed"])
        y_ = np.array(docs_embed)
        scores = np.dot(x_, y_.T)
        neg_docs = []
        neg_docs_scores = []
        max_neg_score_threshold = row["min_pos_score"] * self.percpos
        for idx, s in enumerate(scores):
            if s <= max_neg_score_threshold:
                if docs[idx] not in neg_docs:
                    neg_docs.append(docs[idx])
                    neg_docs_scores.append((docs[idx], s))
        del neg_docs, scores
        return sorted(neg_docs_scores, reverse=True, key=lambda x: x[1])[
            : self.n_hard_negatives
        ]

    def _get_scores_topk_abs(self, x, docs_embed, docs, pos_docs):
        x_ = np.array(x)
        y_ = np.array(docs_embed)
        scores = np.dot(x_, y_.T)
        neg_docs = []
        neg_docs_scores = []
        for idx, s in enumerate(scores):
            if s <= self.max_neg_score_threshold:
                if docs[idx] not in pos_docs:
                    if docs[idx] not in neg_docs:
                        neg_docs.append(docs[idx])
                        neg_docs_scores.append((docs[idx], s))
        del neg_docs, scores
        return sorted(neg_docs_scores, reverse=True, key=lambda x: x[1])[
            : self.n_hard_negatives
        ]

    def _get_hf_embedding(self, text, prefix="query"):
        embeddings = self.hf_model.encode(prefix + text)
        return embeddings

    def _get_nim_embedding(self, text, input_type):
        # Obtain embeddings from nim model
        if isinstance(text, list):
            input_ = text
        elif isinstance(text, str):
            input_ = [text]

        try:
            response = self.client.embeddings.create(
                input=input_,
                model=self.model_name,
                encoding_format="float",
                extra_body={"input_type": input_type, "truncate": self.truncate},
            )
        except Exception as e:
            print(f"Error: {e}")
            response = None

        if response:
            if isinstance(text, list):
                embeddings = [r.embedding for r in response.data]
            elif isinstance(text, str):
                embeddings = response.data[0].embedding
            return embeddings
        else:
            return []
