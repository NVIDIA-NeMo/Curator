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


import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Union

import dask_cudf
import torch
import torch.nn as nn
from crossfit import op
from crossfit.backend.torch.hf.model import HFModel
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from nemo_curator.classifiers.base import _get_suggest_memory_for_classifier
from nemo_curator.datasets import DocumentDataset
from nemo_curator.log import create_logger
from nemo_curator.utils.distributed_utils import (
    performance_report_if_with_ts_suffix,
    write_to_disk,
)


# Embedding creation module
@dataclass
class EmbeddingConfig:
    model_name_or_path: str
    max_seq_length: int = None
    pooling_strategy: str = "mean_pooling"  # Options: "mean_pooling" or "last_token"

    def __post_init__(self):
        self.max_seq_length = AutoTokenizer.from_pretrained(
            self.model_name_or_path
        ).model_max_length
        # Guard against Hugging Face bug
        # which sets max_seq_length to max(int) for some models
        if self.max_seq_length > 1e5:
            self.max_seq_length = AutoConfig.from_pretrained(
                self.model_name_or_path
            ).max_position_embeddings
        if self.pooling_strategy not in ["mean_pooling", "last_token"]:
            raise ValueError(
                "pooling_strategy must be either 'mean_pooling' or 'last_token'"
            )


class EmbeddingPytorchModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = AutoModel.from_pretrained(
            config.model_name_or_path, config=self.config, force_download=False
        )

    def feature(self, input_ids, attention_mask):
        with torch.autocast(device_type=input_ids.device.type):
            embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return embeddings

    @torch.no_grad()
    def forward(self, batch):
        feature = self.feature(batch["input_ids"], batch["attention_mask"])
        if self.config.pooling_strategy == "mean_pooling":
            return self._mean_pooling(feature, batch["attention_mask"])
        else:
            return self._get_last_token(feature, batch["attention_mask"])

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return F.normalize(sum_embeddings / sum_mask, dim=1)

    def _get_last_token(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        # Get indices of last non-padded tokens for each sequence in batch
        last_token_indices = attention_mask.sum(dim=1) - 1  # -1 for 0-based indexing
        last_token_indices = last_token_indices.to(
            torch.long
        )  # Ensure indices are of type long
        batch_size = attention_mask.size(0)
        batch_indices = torch.arange(batch_size, device=attention_mask.device)
        # Get embeddings of last non-padded tokens
        last_token_embeddings = token_embeddings[batch_indices, last_token_indices]
        return F.normalize(last_token_embeddings, dim=1)


class EmbeddingCrossFitModel(HFModel):
    def __init__(
        self,
        config: EmbeddingConfig,
        max_mem_gb: Optional[int] = None,
    ):
        self.config = config
        if max_mem_gb is None:
            max_mem_gb = _get_suggest_memory_for_classifier()
        super().__init__(self.config.model_name_or_path, max_mem_gb=max_mem_gb)

    def load_model(self, device="cuda"):
        model = EmbeddingPytorchModel(self.config)
        model = model.to(device)
        model.eval()
        return model

    def max_seq_length(self):
        return self.config.max_seq_length

    def load_config(self):
        return AutoConfig.from_pretrained(self.config.model_name_or_path)

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.config.model_name_or_path)


class EmbeddingCreator:
    def __init__(
        self,
        embedding_model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_batch_size: int = 128,
        embedding_output_dir: str = "./embeddings",
        embedding_max_mem_gb: Optional[int] = None,
        embedding_pooling_strategy: str = "mean_pooling",
        input_column: str = "text",
        embedding_column: str = "embeddings",
        write_embeddings_to_disk: bool = True,
        write_to_filename: bool = False,
        logger: Union[logging.Logger, str] = "./",
        profile_dir: Optional[str] = None,
    ):
        """
        Initializes an EmbeddingCreator for generating embeddings using the specified model configurations.

        Args:
            embedding_model_name_or_path (str): Model name or path for embeddings.
                Default is "sentence-transformers/all-MiniLM-L6-v2".
            embedding_batch_size (int): Initial batch size for processing embeddings.
                Default is 128.
            embedding_output_dir (str): Location to save embeddings.
                Default is "./embeddings".
            embedding_max_mem_gb (int, optional): Maximum memory usage in GB for the embedding process.
                If None, it defaults to the available GPU memory minus 4 GB.
            embedding_pooling_strategy: Strategy for pooling embeddings, either "mean_pooling" or "last_token".
                Default is "mean_pooling".
            input_column (str): Column name from the data to be used for embedding generation.
                Default is "text".
            embedding_column (str): The column name that stores the embeddings. Default is "embeddings".
            write_embeddings_to_disk (bool): If True, saves the embeddings to disk.
                We recommend setting this to False when you have a delayed pipeline.
                Setting it to False can lead to more memory overhead. Default is True.
            write_to_filename (bool): If True, saves the embeddings to the same filename as input files.
                Default False.
            logger (Union[logging.Logger, str]): Existing logger to log to, or a path to a log directory.
                Default is "./".
            profile_dir (Optional[str]): If specified, directory to write Dask profile.
                Default is None.

        """

        self.embeddings_config = EmbeddingConfig(
            model_name_or_path=embedding_model_name_or_path,
            pooling_strategy=embedding_pooling_strategy,
        )
        self.batch_size = embedding_batch_size
        self.logger = self._setup_logger(logger)
        self.embedding_output_dir = embedding_output_dir
        self.input_column = input_column
        self.embedding_column = embedding_column
        self.model = EmbeddingCrossFitModel(
            self.embeddings_config, max_mem_gb=embedding_max_mem_gb
        )
        self.write_embeddings_to_disk = write_embeddings_to_disk
        self.write_to_filename = write_to_filename
        self.profile_dir = profile_dir

    def _setup_logger(self, logger):
        if isinstance(logger, str):
            return create_logger(
                rank=0,
                name="compute-embeddings",
                log_file=os.path.join(logger, "compute_embeddings.log"),
                log_level=logging.INFO,
                stdout=True,
            )
        else:
            return logger

    def create_embeddings(
        self, ddf: dask_cudf.DataFrame, input_column="text"
    ) -> dask_cudf.DataFrame:
        pipe = op.Sequential(
            op.Tokenizer(
                self.model,
                cols=[input_column],
                tokenizer_type="default",
                max_length=self.embeddings_config.max_seq_length,
            ),
            op.Predictor(
                self.model,
                sorted_data_loader=True,
                batch_size=self.batch_size,
                pred_output_col=self.embedding_column,
            ),
            keep_cols=ddf.columns.tolist(),
        )
        return pipe(ddf)

    def __call__(self, dataset: DocumentDataset) -> DocumentDataset:
        t0 = time.time()
        if self.write_embeddings_to_disk:
            with performance_report_if_with_ts_suffix(
                self.profile_dir, "embedding-creator"
            ):
                embedding_ddf = self.create_embeddings(dataset.df, self.input_column)

                # category column dtypes are not supported by the GPU-accelerated Parquet writer
                for col in embedding_ddf.columns:
                    if embedding_ddf[col].dtype.name == "category":
                        embedding_ddf[col] = embedding_ddf[col].astype("str")

                write_to_disk(
                    embedding_ddf,
                    self.embedding_output_dir,
                    write_to_filename=self.write_to_filename,
                    output_type="parquet",
                )

            ddf = DocumentDataset(
                dask_cudf.read_parquet(
                    self.embedding_output_dir, blocksize="2GB", aggregate_files=True
                )
            )
        else:
            embedding_ddf = self.create_embeddings(dataset.df, self.input_column)
            ddf = DocumentDataset(embedding_ddf)

        self.logger.info(
            f"Time taken for Creating Embeddings : {time.time() - t0}"
            + (
                f" and output written at {self.embedding_output_dir}"
                if self.write_embeddings_to_disk
                else ""
            )
        )

        return ddf
