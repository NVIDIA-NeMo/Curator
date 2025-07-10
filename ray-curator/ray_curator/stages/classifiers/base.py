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
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crossfit.backend.torch.hf.model import HFModel

os.environ["RAPIDS_NO_INITIALIZE"] = "1"
from abc import abstractmethod

import cudf
import pandas as pd
import torch
from crossfit import op
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from transformers import AutoModel

from ray_curator.stages.base import CompositeStage, ProcessingStage
from ray_curator.stages.resources import Resources
from ray_curator.tasks import DocumentBatch

from .crossfit_wrappers import CrossFitLabelerWrapper, CrossFitPredictorWrapper, CrossFitTokenizerWrapper
from .utils import _get_suggest_memory_for_classifier


@dataclass(kw_only=True)
class DistributedDataClassifier(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Abstract class for running multi-node multi-GPU data classification"""

    labels: list[str]
    filter_by: list[str]
    model_batch_size: int
    out_dim: int
    pred_column: str | list[str]
    max_chars: int
    device_type: str
    autocast: bool

    @property
    def name(self) -> str:
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        # TODO: Check for prob_column too, if there is one
        return ["data"], [self.pred_column] if isinstance(self.pred_column, str) else self.pred_column

    @property
    def resources(self) -> Resources:
        """Resource requirements for this stage."""
        # TODO: Check this
        return Resources(gpu_memory_gb=_get_suggest_memory_for_classifier())

    def process(self, batch: DocumentBatch) -> DocumentBatch | None:
        """Run classifier on documents and filter if desired."""

        df = batch.to_pandas()
        result_df = self._run_classifier(df)

        if self.filter_by is not None:
            result_df = self._filter_documents(result_df)

        result_df = result_df.to_pandas()

        if len(result_df) == 0:
            print(f"All documents filtered out for batch {batch.task_id}")
            return None

        # Create output batch
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=result_df,
        )

    @abstractmethod
    def _run_classifier(self, df: pd.DataFrame | cudf.DataFrame) -> pd.DataFrame | cudf.DataFrame:
        pass

    def _filter_documents(
        self,
        df: pd.DataFrame | cudf.DataFrame,
    ) -> pd.DataFrame | cudf.DataFrame:
        filter_by = self.filter_by
        if isinstance(filter_by, str):
            return df[df[self.pred_column].astype(str) == filter_by]
        elif isinstance(filter_by, list):
            return df[df[self.pred_column].isin(filter_by)]

        msg = "filter_by must be a string or list type"
        raise TypeError(msg)

    def get_labels(self) -> list[str]:
        return self.labels


@dataclass(kw_only=True)
class StreamingDataClassifier(CompositeStage[DocumentBatch, DocumentBatch]):
    """Abstract class for running multi-node multi-GPU data classification"""

    labels: list[str]
    filter_by: list[str]
    model_batch_size: int
    out_dim: int
    pred_column: str | list[str]
    max_chars: int
    device_type: str
    autocast: bool
    gpu_tokenizer: bool = True

    @property
    def name(self) -> str:
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        # TODO: Check for prob_column too, if there is one
        return ["data"], [self.pred_column] if isinstance(self.pred_column, str) else self.pred_column

    def decompose(self) -> list[ProcessingStage]:
        """Decompose into tokenizer, predictor, and labeler stages."""
        # TODO: Add filter_by

        if self.prob_column is None:
            prob_col = "_prob"
            keep_prob_col = False
        else:
            prob_col = self.prob_column
            keep_prob_col = True

        return [
            CrossFitTokenizerWrapper(
                model=self.model,
                cols=[self.text_field],
                tokenizer_type="default",
                max_chars=self.max_chars,
                use_gpu=self.gpu_tokenizer,
            ),
            CrossFitPredictorWrapper(
                model=self.model,
                sorted_data_loader=True,
                model_batch_size=self.model_batch_size,
                pred_output_col=prob_col,
                progress_bar=False,
            ),
            CrossFitLabelerWrapper(
                labels=self.labels,
                cols=[prob_col],
                suffix=self.pred_column,
                prob_col=prob_col if keep_prob_col else None,
            ),
        ]

    def get_labels(self) -> list[str]:
        return self.labels


class HFDeberta(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: dataclass):
        super().__init__()
        self.model = AutoModel.from_pretrained(config["base_model"])
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))

    def _forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.model(batch["input_ids"], batch["attention_mask"]).last_hidden_state
        dropped = self.dropout(features)
        outputs = self.fc(dropped)
        return torch.softmax(outputs[:, 0, :], dim=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.autocast:
            with torch.autocast(device_type="cuda"):
                return self._forward(batch)
        else:
            return self._forward(batch)

    def set_autocast(self, autocast: bool) -> None:
        self.autocast = autocast


def _run_classifier_helper(  # noqa: PLR0913
    df: pd.DataFrame | cudf.DataFrame,
    model: "HFModel",
    labels: list[str],
    max_chars: int,
    model_batch_size: int,
    label_col: str,
    text_field: str = "text",
    prob_col: str | None = None,
) -> pd.DataFrame | cudf.DataFrame:
    columns_to_keep_list = df.columns.to_list()

    if prob_col is None:
        prob_col = "_prob"
        labeler = op.Labeler(labels, cols=[prob_col], suffix=label_col)
    else:
        labeler = op.Labeler(labels, cols=[prob_col], keep_cols=[prob_col], suffix=label_col)

    classifier_pipe = op.Sequential(
        op.Tokenizer(model, cols=[text_field], tokenizer_type="default", max_chars=max_chars),
        op.Predictor(
            model,
            sorted_data_loader=True,
            batch_size=model_batch_size,
            pred_output_col=prob_col,
            progress_bar=False,
        ),
        labeler,
        keep_cols=columns_to_keep_list,
    )

    return classifier_pipe(df)
