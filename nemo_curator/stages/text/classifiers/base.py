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
from nemo_curator.stages.text.models.utils import ATTENTION_MASK_FIELD, INPUT_ID_FIELD
from nemo_curator.stages.text.modules.score_filter import Filter
from nemo_curator.tasks import DocumentBatch


class Deberta(nn.Module, PyTorchModelHubMixin):
    """
    Base PyTorch model where we add a classification head.

    Args:
        config: The configuration of the model.

    """

    def __init__(
        self,
        config: dataclass,
        input_id_field: str = INPUT_ID_FIELD,
        attention_mask_field: str = ATTENTION_MASK_FIELD
    ):
        super().__init__()
        self.input_id_field = input_id_field
        self.attention_mask_field = attention_mask_field

        self.model = AutoModel.from_pretrained(config["base_model"])
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.model(batch[self.input_id_field], batch[self.attention_mask_field]).last_hidden_state
        dropped = self.dropout(features)
        outputs = self.fc(dropped)

        del batch, features, dropped

        return torch.softmax(outputs[:, 0, :], dim=1)


class ClassifierModelStage(ModelStage):
    """
    Stage for Hugging Face model inference.

    Args:
        model_identifier: The identifier of the Hugging Face model.
        label_field: The name of the prediction column.
        score_field: The name of the probability column. Defaults to None.
        model_inference_batch_size: The size of the batch for model inference. Defaults to 256.
        has_seq_order: Whether to sort the input data by the length of the input tokens.
            Sorting is encouraged to improve the performance of the inference model. Defaults to True.
        padding_side: The side to pad the input tokens. Defaults to "right".
        autocast: Whether to use autocast. When True, we trade off minor accuracy for faster inference.
            Defaults to True.
        drop_tokens: Whether to drop the input tokens from the output dataframe. Defaults to True.
        token_fields: The fields to use for the input tokens. Defaults to ["input_ids", "attention_mask"].

    """

    def __init__(  # noqa: PLR0913
        self,
        model_identifier: str,
        cache_dir: str | None = None,
        label_field: str = "preds",
        score_field: str | None = None,
        model_inference_batch_size: int = 256,
        has_seq_order: bool = True,
        padding_side: Literal["left", "right"] = "right",
        autocast: bool = True,
        drop_tokens: bool = True,
        token_fields: list[str] | None = None,
    ):
        if token_fields is None:
            token_fields = [INPUT_ID_FIELD, ATTENTION_MASK_FIELD]

        super().__init__(
            model_identifier=model_identifier,
            cache_dir=cache_dir,
            has_seq_order=has_seq_order,
            model_inference_batch_size=model_inference_batch_size,
            padding_side=padding_side,
            unpack_inference_batch=False,
            autocast=autocast,
            token_fields=token_fields,
        )

        self.label_field = label_field
        if score_field is not None:
            self.score_field = score_field
            self.keep_score_field = True
        else:
            self.score_field = "probs"
            self.keep_score_field = False

        self.drop_tokens = drop_tokens

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.label_field] + ([self.score_field] if self.keep_score_field else [])

    def _setup(self, local_files_only: bool = True) -> None:
        self.model = (
            Deberta.from_pretrained(
                self.model_identifier,
                cache_dir=self.cache_dir,
                local_files_only=local_files_only,
                input_id_field=self.input_id_field,
                attention_mask_field=self.attention_mask_field,
            )
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
            self.score_field: probs,
            self.label_field: np.array(pred_labels),
        }

    def create_output_dataframe(self, df_cpu: pd.DataFrame, collected_output: dict[str, np.ndarray]) -> pd.DataFrame:
        if self.drop_tokens:
            df_cpu = df_cpu.drop(columns=[self.input_id_field, self.attention_mask_field])

        df_cpu[self.label_field] = collected_output[self.label_field]

        if self.keep_score_field:
            df_cpu[self.score_field] = collected_output[self.score_field].tolist()

        return df_cpu


@dataclass(kw_only=True)
class DistributedDataClassifier(CompositeStage[DocumentBatch, DocumentBatch]):
    """
    Base composite stage for distributed data classification.

    It decomposes into a tokenizer stage and a model stage.

    Args:
        model_identifier: The identifier of the Hugging Face model.
        cache_dir: The Hugging Face cache directory. Defaults to None.
        label_field: The name of the prediction column. Defaults to "preds".
        score_field: The name of the probability column. Defaults to None.
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
        drop_tokens: Whether to drop the input tokens from the output dataframe. Defaults to True.
        use_existing_tokens: Whether to use the existing tokens from the input dataframe.
            If True, assume the relevant token fields are ["input_ids", "attention_mask"] and skip tokenization.
            The use_existing_tokens field can be either a boolean or a list of strings representing the token fields.
            Defaults to False.

    """

    model_identifier: str
    cache_dir: str | None = None
    label_field: str = "preds"
    score_field: str | None = None
    text_field: str = "text"
    filter_by: list[str] | None = None
    max_chars: int | None = None
    max_seq_length: int | None = None
    padding_side: Literal["left", "right"] = "right"
    sort_by_length: bool = True
    model_inference_batch_size: int = 256
    autocast: bool = True
    drop_tokens: bool = True
    use_existing_tokens: bool | list[str] = False

    def __post_init__(self) -> None:
        super().__init__()

        self.stages = []

        if not self.use_existing_tokens:
            tokenizer_stage = TokenizerStage(
                model_identifier=self.model_identifier,
                cache_dir=self.cache_dir,
                text_field=self.text_field,
                max_chars=self.max_chars,
                max_seq_length=self.max_seq_length,
                padding_side=self.padding_side,
                sort_by_length=self.sort_by_length,
            )
            self.stages.append(tokenizer_stage)

        if isinstance(self.use_existing_tokens, list):
            if len(self.use_existing_tokens) != 2:  # noqa: PLR2004
                msg = "use_existing_tokens must be a list of two strings representing the [input_ids, attention_mask] fields"
                raise ValueError(msg)
            token_fields = self.use_existing_tokens
        else:
            token_fields = [INPUT_ID_FIELD, ATTENTION_MASK_FIELD]

        model_stage = ClassifierModelStage(
            model_identifier=self.model_identifier,
            cache_dir=self.cache_dir,
            label_field=self.label_field,
            score_field=self.score_field,
            model_inference_batch_size=self.model_inference_batch_size,
            has_seq_order=self.sort_by_length,
            padding_side=self.padding_side,
            autocast=self.autocast,
            drop_tokens=self.drop_tokens,
            token_fields=token_fields,
        )
        self.stages.append(model_stage)

        if self.filter_by is not None and len(self.filter_by) > 0:
            self.stages.append(Filter(filter_fn=self.filter_by_category, filter_field=self.label_field))

    def inputs(self) -> tuple[list[str], list[str]]:
        return self.stages[0].inputs()

    def outputs(self) -> tuple[list[str], list[str]]:
        return self.stages[-1].outputs()

    def filter_by_category(self, value: str) -> bool:
        return value in self.filter_by

    def decompose(self) -> list[ProcessingStage]:
        return self.stages
