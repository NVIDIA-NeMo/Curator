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
from typing import Literal

os.environ["RAPIDS_NO_INITIALIZE"] = "1"

import numpy as np
import pandas as pd
import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from transformers import AutoConfig, AutoModel

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.text.models.model import ModelStage
from nemo_curator.stages.text.models.tokenizer import TokenizerStage
from nemo_curator.stages.text.models.utils import ATTENTION_MASK_COLUMN, INPUT_ID_COLUMN
from nemo_curator.stages.text.modules.score_filter import Filter
from nemo_curator.tasks import DocumentBatch


class Deberta(nn.Module, PyTorchModelHubMixin):
    """
    Base PyTorch model where we add a classification head.

    Args:
        config: The configuration of the model.

    """

    def __init__(self, config: dataclass):
        super().__init__()
        self.model = AutoModel.from_pretrained(config["base_model"])
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.model(batch[INPUT_ID_COLUMN], batch[ATTENTION_MASK_COLUMN]).last_hidden_state
        dropped = self.dropout(features)
        outputs = self.fc(dropped)

        del batch, features, dropped

        return torch.softmax(outputs[:, 0, :], dim=1)


class ClassifierModelStage(ModelStage):
    """
    Stage for Hugging Face model inference.

    Args:
        model_identifier: The identifier of the Hugging Face model.
        pred_column: The name of the prediction column.
        prob_column: The name of the probability column. Defaults to None.
        model_inference_batch_size: The size of the batch for model inference. Defaults to 256.
        has_seq_order: Whether to sort the input data by the length of the input tokens.
            Sorting is encouraged to improve the performance of the inference model. Defaults to True.
        padding_side: The side to pad the input tokens. Defaults to "right".
        autocast: Whether to use autocast. When True, we trade off minor accuracy for faster inference.
            Defaults to True.

    """

    def __init__(  # noqa: PLR0913
        self,
        model_identifier: str,
        cache_dir: str | None = None,
        pred_column: str = "preds",
        prob_column: str | None = None,
        model_inference_batch_size: int = 256,
        has_seq_order: bool = True,
        padding_side: Literal["left", "right"] = "right",
        autocast: bool = True,
    ):
        super().__init__(
            model_identifier=model_identifier,
            cache_dir=cache_dir,
            has_seq_order=has_seq_order,
            model_inference_batch_size=model_inference_batch_size,
            padding_side=padding_side,
            unpack_inference_batch=False,
            autocast=autocast,
        )

        self.pred_column = pred_column
        if prob_column is not None:
            self.prob_column = prob_column
            self.keep_prob_column = True
        else:
            self.prob_column = "probs"
            self.keep_prob_column = False

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.pred_column] + ([self.prob_column] if self.keep_prob_column else [])

    def _setup(self, local_files_only: bool = True) -> None:
        self.model = (
            Deberta.from_pretrained(self.model_identifier, cache_dir=self.cache_dir, local_files_only=local_files_only)
            .cuda()
            .eval()
        )

        config = AutoConfig.from_pretrained(
            self.model_identifier, cache_dir=self.cache_dir, local_files_only=local_files_only
        )
        self.labels = list(config.label2id.keys())
        self.labels.sort(key=lambda x: config.label2id[x])

    def process_model_output(
        self, outputs: torch.Tensor, _: dict[str, torch.Tensor] | None = None
    ) -> dict[str, np.ndarray]:
        probs = outputs.cpu().numpy()
        preds = np.argmax(probs, axis=1)

        pred_labels = [self.labels[idx] for idx in preds]

        return {
            self.prob_column: probs,
            self.pred_column: np.array(pred_labels),
        }

    def create_output_dataframe(self, df_cpu: pd.DataFrame, collected_output: dict[str, np.ndarray]) -> pd.DataFrame:
        df_cpu = df_cpu.drop(columns=[INPUT_ID_COLUMN, ATTENTION_MASK_COLUMN])
        df_cpu[self.pred_column] = collected_output[self.pred_column]

        if self.keep_prob_column:
            df_cpu[self.prob_column] = collected_output[self.prob_column].tolist()

        return df_cpu


@dataclass(kw_only=True)
class DistributedDataClassifier(CompositeStage[DocumentBatch, DocumentBatch]):
    """
    Base composite stage for distributed data classification.

    It decomposes into a tokenizer stage and a model stage.

    Args:
        model_identifier: The identifier of the Hugging Face model.
        cache_dir: The Hugging Face cache directory. Defaults to None.
        pred_column: The name of the prediction column. Defaults to "preds".
        prob_column: The name of the probability column. Defaults to None.
        text_field: The name of the text field in the input data. Defaults to "text".
        filter_by: For categorical classifiers, the list of labels to filter the data by. Defaults to None.
        max_chars: Limits the total number of characters that can be fed to the tokenizer.
            If None, text will not be truncated. Defaults to None.
        max_seq_length: Limits the total sequence returned by the tokenizer so that it has a maximum length.
            If None, the tokenizer's model_max_length is used. Defaults to 512.
        padding_side: The side to pad the input tokens. Defaults to "right".
        sort_by_length: Whether to sort the input data by the length of the input tokens.
            Sorting is encouraged to improve the performance of the inference model. Defaults to True.
        model_inference_batch_size: The size of the batch for model inference. Defaults to 256.
        autocast: Whether to use autocast. When True, we trade off minor accuracy for faster inference.
            Defaults to True.

    """

    model_identifier: str
    cache_dir: str | None = None
    pred_column: str = "preds"
    prob_column: str | None = None
    text_field: str = "text"
    filter_by: list[str] | None = None
    max_chars: int | None = None
    max_seq_length: int | None = None
    padding_side: Literal["left", "right"] = "right"
    sort_by_length: bool = True
    model_inference_batch_size: int = 256
    autocast: bool = True

    def __post_init__(self) -> None:
        super().__init__()

        self.stages = [
            TokenizerStage(
                model_identifier=self.model_identifier,
                cache_dir=self.cache_dir,
                text_field=self.text_field,
                max_chars=self.max_chars,
                max_seq_length=self.max_seq_length,
                padding_side=self.padding_side,
                sort_by_length=self.sort_by_length,
            ),
            ClassifierModelStage(
                model_identifier=self.model_identifier,
                cache_dir=self.cache_dir,
                pred_column=self.pred_column,
                prob_column=self.prob_column,
                model_inference_batch_size=self.model_inference_batch_size,
                has_seq_order=self.sort_by_length,
                padding_side=self.padding_side,
                autocast=self.autocast,
            ),
        ]

        if self.filter_by is not None and len(self.filter_by) > 0:
            self.stages.append(Filter(filter_fn=self.filter_by_category, filter_field=self.pred_column))

    def inputs(self) -> tuple[list[str], list[str]]:
        return self.stages[0].inputs()

    def outputs(self) -> tuple[list[str], list[str]]:
        return self.stages[1].outputs()

    def filter_by_category(self, value: str) -> bool:
        return value in self.filter_by

    def decompose(self) -> list[ProcessingStage]:
        return self.stages
