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

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

import cudf
import numpy as np
import pandas as pd
import torch
from crossfit import op
from crossfit.backend.torch.model import Model

from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.resources import Resources
from ray_curator.tasks import DocumentBatch

from .utils import _get_suggest_memory_for_classifier, _get_suggest_memory_for_tokenizer

if TYPE_CHECKING:
    from ray_curator.stages.classifiers.base import HFModel


# Copied from CrossFit
class TokenizerType(Enum):
    SUBWORD = 1
    SENTENCE_PIECE = 2
    DEFAULT = 3


# Copied from Praateek's PR: https://github.com/ayushdg/ray-curator/pull/16
def clip_tokens(token_o: dict, max_length: int | None, padding_side: Literal["left", "right"] = "right") -> dict:
    clip_len = token_o["attention_mask"].sum(axis=1).max()
    clip_len = min(clip_len, max_length) if max_length is not None else clip_len
    if padding_side == "right":
        token_o["input_ids"] = token_o["input_ids"][:, :clip_len]
        token_o["attention_mask"] = token_o["attention_mask"][:, :clip_len]
    else:
        token_o["input_ids"] = token_o["input_ids"][:, -clip_len:]
        token_o["attention_mask"] = token_o["attention_mask"][:, -clip_len:]

    token_o.pop("metadata", None)

    return token_o


# TODO: Check this function for correctness
# Seems to be producing correct results in benchmarks
def create_list_series_from_1d_or_2d_ar(ar: np.ndarray, index: pd.Index | None = None) -> pd.Series:
    if len(ar.shape) == 1:
        n_rows = ar.shape[0]
        list_data = [[x] for x in ar]
    elif len(ar.shape) == 2:  # noqa: PLR2004
        n_rows, n_cols = ar.shape
        list_data = [list(ar[i]) for i in range(n_rows)]
    else:
        msg = f"Unexpected input shape: {ar.shape}"
        raise RuntimeError(msg)

    return pd.Series(list_data, index=index, dtype="object")


# TODO: CrossFit needs to be more flexible to enable this as a CPU-only stage
# For now, we use a custom class here that is (almost) identical to CrossFit's Tokenizer class
class Tokenizer:
    def __init__(  # noqa: PLR0913
        self,
        model: Model,
        tokenizer_type: str = "default",
        cols: list[str] | None = None,
        keep_cols: list[str] | None = None,
        pre: Callable | None = None,
        max_length: int | None = None,
        max_chars: int | None = None,
    ):
        self.pre = pre
        self.cols = cols
        self.keep_cols = keep_cols or []
        self.model = model

        # For now, we only support the default tokenizer
        assert tokenizer_type.lower() == "default"  # noqa: S101
        self.tokenizer_type = TokenizerType.DEFAULT

        self.max_length = max_length or model.max_seq_length()
        self.max_chars = max_chars

    def tokenize_strings(self, sentences: pd.Series, max_length: int | None = None) -> dict:
        tokenizer = self.model.load_tokenizer()
        self.padding_side = tokenizer.padding_side

        if isinstance(sentences, pd.Series):
            sentences = sentences.to_list()

        with torch.no_grad():
            return tokenizer.batch_encode_plus(
                sentences,
                max_length=max_length or self.max_length,
                padding="max_length",
                return_tensors="pt",
                truncation=True,
                add_special_tokens=True,
                return_token_type_ids=False,
            )

    def call_column(self, data: pd.Series) -> tuple[pd.Series, pd.Series]:
        text = data.replace("", "unknown")

        if self.max_chars:
            text = text.str.slice(0, self.max_chars)

        tokenized_data = self.tokenize_strings(text).copy()
        tokenized_data = clip_tokens(
            tokenized_data,
            max_length=self.max_length,
            padding_side=self.padding_side,
        )

        # TODO: Could move this into the CrossFitPredictorWrapper stage and run it before calling op.Predictor
        input_ids = create_list_series_from_1d_or_2d_ar(
            tokenized_data["input_ids"].numpy().astype("int32"), data.index
        )
        attention_mask = create_list_series_from_1d_or_2d_ar(
            tokenized_data["attention_mask"].numpy().astype("int32"), data.index
        )

        return input_ids, attention_mask

    def call(self, data: pd.Series | pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame()

        if self.cols is None:
            input_ids, attention_mask = self.call_column(data)
            output["input_ids"] = input_ids
            output["attention_mask"] = attention_mask

            return output

        for col in self.cols:
            if col not in data.columns:
                msg = f"Column {col} not found in data"
                raise ValueError(msg)

            input_ids, attention_mask = self.call_column(data[col])
            output[self._construct_name(col, "input_ids")] = input_ids
            output[self._construct_name(col, "attention_mask")] = attention_mask

        return output

    def meta(self) -> dict[str, str]:
        tokenized = {
            "input_ids": "int32",
            "attention_mask": "int32",
        }

        if len(self.cols) > 1:
            tokenized = {
                self._construct_name(col, suffix): dtype for col in self.cols for suffix, dtype in tokenized.items()
            }

        return tokenized

    def _construct_name(self, col_name: str, suffix: str) -> str:
        if len(self.cols) == 1:
            return suffix

        return f"{col_name}_{suffix}"

    # Copied from CrossFit's Op class
    def add_keep_cols(self, data: pd.Series | pd.DataFrame, output: pd.DataFrame) -> pd.DataFrame:
        if not self.keep_cols:
            return output

        for col in self.keep_cols:
            if col not in output.columns:
                output[col] = data[col]

        columns = list(output.columns)
        # we use dict.fromkeys to remove duplicates and preserve order
        return output[list(dict.fromkeys(self.keep_cols + columns))]

    # Modified from CrossFit's Op class
    def __call__(self, data: pd.Series | pd.DataFrame, *args, **kwargs):
        if self.pre is not None:
            data = self.pre(data)

        output = self.call(data, *args, **kwargs)

        if self.keep_cols:
            output = self.add_keep_cols(data, output)

        return output


# TODO: Unclear if a CPU-only stage offers any benefits over just making this a GPU stage
@dataclass
class CrossFitTokenizerWrapper(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Wrapper for CrossFit tokenizer"""

    model: "HFModel"
    cols: list[str]
    tokenizer_type: str
    max_chars: int
    use_gpu: bool = True

    @property
    def name(self) -> str:
        return "crossfit_tokenizer"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    @property
    def resources(self) -> Resources:
        """Resource requirements for this stage."""
        if self.use_gpu:
            return Resources(gpu_memory_gb=_get_suggest_memory_for_tokenizer())
        else:
            # Default CPU resources
            return Resources(cpus=1.0)

    def process(self, batch: DocumentBatch) -> DocumentBatch | None:
        df = batch.to_pandas()

        # TODO: Tokenizer or op.Tokenizer
        if self.use_gpu:
            result_df = op.Tokenizer(
                self.model,
                cols=self.cols,
                tokenizer_type=self.tokenizer_type,
                max_chars=self.max_chars,
                keep_cols=df.columns.to_list(),
            )(df).to_pandas()
        else:
            result_df = Tokenizer(
                self.model,
                cols=self.cols,
                tokenizer_type=self.tokenizer_type,
                max_chars=self.max_chars,
                keep_cols=df.columns.to_list(),
            )(df)

        # Create output batch
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=result_df,
        )


@dataclass
class CrossFitPredictorWrapper(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Wrapper for CrossFit predictor"""

    model: "HFModel"
    sorted_data_loader: bool
    model_batch_size: int
    pred_output_col: str
    progress_bar: bool

    @property
    def name(self) -> str:
        return "crossfit_predictor"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    @property
    def resources(self) -> Resources:
        """Resource requirements for this stage."""
        # TODO: Check this
        return Resources(gpu_memory_gb=_get_suggest_memory_for_classifier())

    def process(self, batch: DocumentBatch) -> DocumentBatch | None:
        df = batch.to_pandas()
        # Explicitly convert to cuDF to avoid conversion error
        df = cudf.from_pandas(df)

        keep_cols = df.columns.to_list()
        # Need to explicitly remove token-related columns
        keep_cols.remove("input_ids")
        keep_cols.remove("attention_mask")

        result_df = op.Predictor(
            model=self.model,
            sorted_data_loader=self.sorted_data_loader,
            batch_size=self.model_batch_size,
            pred_output_col=self.pred_output_col,
            progress_bar=self.progress_bar,
            keep_cols=keep_cols,
        )(df).to_pandas()

        # Create output batch
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=result_df,
        )


# TODO: CrossFit needs to be more flexible to enable this as a CPU-only stage
# For now, we use a custom class here that is (almost) identical to CrossFit's Labeler class
class Labeler:
    def __init__(  # noqa: PLR0913
        self,
        labels: list[str],
        cols: list[str] | None = None,
        keep_cols: list[str] | None = None,
        pre: Callable | None = None,
        suffix: str = "labels",
        axis: int = -1,
    ):
        if keep_cols is not None and suffix in keep_cols:
            # suffix is already kept as a column
            # and will raise an error if it is in keep_cols
            keep_cols.remove(suffix)

        self.pre = pre
        self.cols = cols
        self.keep_cols = keep_cols or []
        self.labels = labels
        self.suffix = suffix
        self.axis = axis

    def call_column(self, data: pd.Series) -> pd.Series:
        shape = (data.size, *np.asarray(data.iloc[0]).shape)
        # TODO: Check shape logic here, compared to the original GPU logic
        # scores = data.list.leaves.values.reshape(shape)  # noqa: ERA001
        scores = np.array(data.tolist()).reshape(shape)
        classes = scores.argmax(self.axis)

        if len(classes.shape) > 1:
            msg = f"Max category of the axis {self.axis} of data is not a 1-d array."
            raise RuntimeError(msg)

        labels_map = {i: self.labels[i] for i in range(len(self.labels))}

        return pd.Series(classes).map(labels_map)

    def call(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        output = pd.DataFrame()

        if self.cols is None:
            return self.call_column(data)

        for col in self.cols:
            if col not in data.columns:
                msg = f"Column {col} not found in data"
                raise ValueError(msg)

            labels = self.call_column(data[col])
            output[self._construct_name(col, self.suffix)] = labels

        return output

    def meta(self) -> dict[str, str]:
        labeled = {self.suffix: "string"}

        if self.cols and len(self.cols) > 1:
            labeled = {
                self._construct_name(col, suffix): dtype for col in self.cols for suffix, dtype in labeled.items()
            }

        return labeled

    def _construct_name(self, col_name: str, suffix: str) -> str:
        if len(self.cols) == 1:
            return suffix

        return f"{col_name}_{suffix}"

    # Copied from CrossFit's Op class
    def add_keep_cols(self, data: pd.Series | pd.DataFrame, output: pd.DataFrame) -> pd.DataFrame:
        if not self.keep_cols:
            return output

        for col in self.keep_cols:
            if col not in output.columns:
                output[col] = data[col]

        columns = list(output.columns)
        # we use dict.fromkeys to remove duplicates and preserve order
        return output[list(dict.fromkeys(self.keep_cols + columns))]

    # Modified from CrossFit's Op class
    def __call__(self, data: pd.Series | pd.DataFrame, *args, **kwargs):
        if self.pre is not None:
            data = self.pre(data)

        output = self.call(data, *args, **kwargs)

        if self.keep_cols:
            output = self.add_keep_cols(data, output)

        return output


@dataclass
class CrossFitLabelerWrapper(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Wrapper for CrossFit labeler"""

    labels: list[str]
    cols: list[str]
    suffix: str
    prob_col: str | None = None

    @property
    def name(self) -> str:
        return "crossfit_labeler"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, batch: DocumentBatch) -> DocumentBatch | None:
        df = batch.to_pandas()

        result_df = Labeler(
            labels=self.labels,
            cols=self.cols,
            keep_cols=[*df.columns.to_list(), self.prob_col] if self.prob_col else df.columns.to_list(),
            suffix=self.suffix,
        )(df)

        # Create output batch
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=result_df,
        )
