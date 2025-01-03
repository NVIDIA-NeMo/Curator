# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from dataclasses import dataclass

os.environ["RAPIDS_NO_INITIALIZE"] = "1"
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import torch
import torch.nn as nn
from crossfit import op
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoModel

from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_gpu_memory_info


class DistributedDataClassifier(ABC):
    """Abstract class for running multi-node multi-GPU data classification"""

    def __init__(
        self,
        model: str,
        labels: Optional[List[str]],
        filter_by: Optional[List[str]],
        batch_size: int,
        out_dim: Optional[int],
        pred_column: Union[str, List[str]],
        max_chars: int,
        device_type: str,
        autocast: bool,
    ):
        self.model = model
        self.labels = labels
        self.filter_by = filter_by
        self.batch_size = batch_size
        self.out_dim = out_dim
        self.pred_column = pred_column
        self.max_chars = max_chars
        self.device_type = device_type
        self.autocast = autocast

    def __call__(self, dataset: DocumentDataset) -> DocumentDataset:
        result_doc_dataset = self._run_classifier(dataset)
        if self.filter_by is not None:
            return self._filter_documents(result_doc_dataset)

        return result_doc_dataset

    @abstractmethod
    def _run_classifier(self) -> DocumentDataset:
        pass

    def _filter_documents(
        self,
        dataset: DocumentDataset,
    ) -> DocumentDataset:
        df = dataset.df

        filter_by = self.filter_by
        if type(filter_by) == str:
            filtered_df = df[df[self.pred_column].astype(str) == filter_by]
            return DocumentDataset(filtered_df)
        elif type(filter_by) == list:
            filtered_df = df[df[self.pred_column].isin(filter_by)]
            return DocumentDataset(filtered_df)

        raise TypeError("filter_by must be a string or list type")

    def get_labels(self) -> List[str]:
        return self.labels


class HFDeberta(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: dataclass):
        super(HFDeberta, self).__init__()
        self.model = AutoModel.from_pretrained(config["base_model"])
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))

    def _forward(self, batch):
        features = self.model(
            batch["input_ids"], batch["attention_mask"]
        ).last_hidden_state
        dropped = self.dropout(features)
        outputs = self.fc(dropped)
        return torch.softmax(outputs[:, 0, :], dim=1)

    def forward(self, batch):
        if self.autocast:
            with torch.autocast(device_type="cuda"):
                return self._forward(batch)
        else:
            return self._forward(batch)

    def set_autocast(self, autocast: bool):
        self.autocast = autocast


def _run_classifier_helper(
    df: "dask_cudf.DataFrame",
    model: "HFModel",
    labels: list[str],
    max_chars: int,
    batch_size: int,
    label_col: str,
    text_field: str = "text",
    prob_col: str = None,
) -> "dask_cudf.DataFrame":

    if prob_col:
        df[prob_col] = 0
    else:
        prob_col = "_prob"

    columns_to_keep_list = df.columns.to_list()

    classifier_pipe = op.Sequential(
        op.Tokenizer(
            model, cols=[text_field], tokenizer_type="default", max_chars=max_chars
        ),
        op.Predictor(
            model,
            sorted_data_loader=True,
            batch_size=batch_size,
            pred_output_col=prob_col,
        ),
        op.Labeler(labels, cols=[prob_col], suffix=label_col),
        repartition=df.npartitions,
        keep_cols=columns_to_keep_list,
    )
    df = classifier_pipe(df)

    return df


def _get_suggest_memory_for_classifier() -> int:
    gpu_memory_info = get_gpu_memory_info()
    min_gpu_memory = min(gpu_memory_info.values())
    # Convert memory from bytes to GB
    min_gpu_memory_gb = min_gpu_memory / (1024**3)
    # Subtract 4GB from the minimum
    # to leave room for other operations
    # like cuDF operations
    min_gpu_memory_gb = min_gpu_memory_gb - 4
    return int(min_gpu_memory_gb)
