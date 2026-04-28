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

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

os.environ["RAPIDS_NO_INITIALIZE"] = "1"

# Heavy deps (torch, transformers, numpy, huggingface_hub) are NOT imported at
# module level.  They are deferred to _make_deberta() / _setup() so that
# `from nemo_curator.stages.text.classifiers import SomeClassifier` is fast
# even on cold-start aarch64 (GB200) where these libraries are slow to load.

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.text.filters import Filter
from nemo_curator.stages.text.models.model import ModelStage
from nemo_curator.stages.text.models.tokenizer import TokenizerStage
from nemo_curator.stages.text.models.utils import ATTENTION_MASK_FIELD, INPUT_ID_FIELD
from nemo_curator.tasks import DocumentBatch

from .utils import SortByLengthStage

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import torch


# ---------------------------------------------------------------------------
# Lazy factory: defines Deberta (nn.Module subclass) on first call so that
# torch is never imported at module-parse time.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _make_deberta() -> type:
    import torch
    from huggingface_hub import PyTorchModelHubMixin
    from torch import nn
    from transformers import AutoModel

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
            features = self.model(batch[INPUT_ID_FIELD], batch[ATTENTION_MASK_FIELD]).last_hidden_state
            dropped = self.dropout(features)
            outputs = self.fc(dropped)

            del batch, features, dropped

            return torch.softmax(outputs[:, 0, :], dim=1)

    return Deberta


# ---------------------------------------------------------------------------
# ClassifierModelStage
# ---------------------------------------------------------------------------


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
        max_seq_length: If provided, clips the input tokens before the forward pass. Defaults to None.
        autocast: Whether to use autocast. When True, we trade off minor accuracy for faster inference.
            Defaults to True.
        keep_tokens: Whether to keep the input tokens in the output dataframe. Defaults to False.

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
        max_seq_length: int | None = None,
        autocast: bool = True,
        keep_tokens: bool = False,
    ):
        super().__init__(
            model_identifier=model_identifier,
            cache_dir=cache_dir,
            has_seq_order=has_seq_order,
            model_inference_batch_size=model_inference_batch_size,
            padding_side=padding_side,
            max_seq_length=max_seq_length,
            unpack_inference_batch=False,
            autocast=autocast,
        )

        self.label_field = label_field
        if score_field is not None:
            self.score_field = score_field
            self.keep_score_field = True
        else:
            self.score_field = "probs"
            self.keep_score_field = False

        self.keep_tokens = keep_tokens

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.label_field] + ([self.score_field] if self.keep_score_field else [])

    def _setup(self, local_files_only: bool = True) -> None:
        from transformers import AutoConfig

        Deberta = _make_deberta()  # noqa: N806
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
        import numpy as np

        probs = outputs.cpu().numpy()
        preds = np.argmax(probs, axis=1)

        pred_labels = [self.labels[idx] for idx in preds]

        return {
            self.score_field: probs,
            self.label_field: np.array(pred_labels),
        }

    def create_output_dataframe(self, df_cpu: pd.DataFrame, collected_output: dict[str, np.ndarray]) -> pd.DataFrame:
        if not self.keep_tokens:
            df_cpu = df_cpu.drop(columns=[INPUT_ID_FIELD, ATTENTION_MASK_FIELD])

        df_cpu[self.label_field] = collected_output[self.label_field]

        if self.keep_score_field:
            df_cpu[self.score_field] = collected_output[self.score_field].tolist()

        return df_cpu


# ---------------------------------------------------------------------------
# DistributedDataClassifier
# ---------------------------------------------------------------------------


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
        keep_tokens: Whether to keep the input tokens in the output dataframe. Defaults to False.
        use_existing_tokens: Whether to use the existing tokens from the input dataframe.
            If True, assume the relevant token fields are ["input_ids", "attention_mask"] and skip tokenization.
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
    keep_tokens: bool = False
    use_existing_tokens: bool = False

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
            # The TokenizerStage already truncates to the max_seq_length, so the ModelStage does not need to do it again
            model_max_seq_length = None
        else:
            # The ModelStage will truncate to the max_seq_length before the forward pass
            model_max_seq_length = self.max_seq_length

        # Ensure that the data is sorted by length if the tokens are already present and sort_by_length is True
        if self.use_existing_tokens and self.sort_by_length:
            sort_by_length_stage = SortByLengthStage()
            self.stages.append(sort_by_length_stage)

        model_stage = ClassifierModelStage(
            model_identifier=self.model_identifier,
            cache_dir=self.cache_dir,
            label_field=self.label_field,
            score_field=self.score_field,
            model_inference_batch_size=self.model_inference_batch_size,
            has_seq_order=self.sort_by_length,
            padding_side=self.padding_side,
            max_seq_length=model_max_seq_length,
            autocast=self.autocast,
            keep_tokens=self.keep_tokens,
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
