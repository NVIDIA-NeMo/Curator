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
from typing import TYPE_CHECKING, Literal

import cudf
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from crossfit import op
from transformers import AutoTokenizer

from ray_curator.backends.base import WorkerMetadata
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.resources import Resources
from ray_curator.tasks import DocumentBatch

from .utils import _get_suggest_memory_for_classifier, _get_suggest_memory_for_tokenizer

if TYPE_CHECKING:
    from ray_curator.stages.classifiers.base import HFModel


def _trim_tokenized(batch_tokenized: dict, padding_side: Literal["left", "right"] = "right") -> dict:
    """
    Trim tokenized data to the max non-padded length.
    """
    max_len = batch_tokenized["attention_mask"].sum(axis=1).max()
    for key, val in batch_tokenized.items():
        if padding_side == "left":
            batch_tokenized[key] = val[:, -max_len:]
        else:
            batch_tokenized[key] = val[:, :max_len]
    return batch_tokenized


class BasicTokenizer:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, df: pd.DataFrame) -> list[pd.DataFrame]:
        # TODO: Do not hardcode text column
        # TODO: Slice text
        sentences = df.column("text").to_pylist()

        with torch.no_grad():
            inputs = self.tokenizer.batch_encode_plus(
                sentences,
                # TODO: Do not hardcode this
                max_length=1024,
                padding="max_length",
                # TODO: "pt" or "np"
                return_tensors="np",
                truncation=True,
                add_special_tokens=True,
                return_token_type_ids=False,
            )

        # TODO: Clip tokens here
        inputs = _trim_tokenized(inputs)

        # Assuming a large batch size here
        attention_sum = inputs["attention_mask"].sum(axis=1)
        # TODO: Need to retain all columns here, not just the ones we need
        table = pa.table(
            {
                "text": sentences,
                "input_ids": pa.array(inputs["input_ids"].tolist(), type=pa.list_(pa.int32())),
                "attention_mask": pa.array(inputs["attention_mask"].tolist(), type=pa.list_(pa.int32())),
                "attention_sum": pa.array(attention_sum),
            }
        )
        # TODO: Add ID column here, to help with unsorting

        table = table.sort_by([("attention_sum", "ascending")])

        # Split into chunks of 1024 rows each
        # TODO: Need to check if this actually improves performance
        return [table.slice(i, 1024) for i in range(0, table.num_rows, 1024)]


@dataclass
class CrossFitTokenizerWrapper(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Wrapper for CrossFit tokenizer"""

    model: "HFModel"
    cols: list[str]
    tokenizer_type: str
    max_chars: int
    use_gpu: bool = True
    _name: str = "crossfit_tokenizer"

    def __post_init__(self):
        if self.use_gpu:
            # TODO: Check this
            self._resources = Resources(gpu_memory_gb=_get_suggest_memory_for_tokenizer())
        else:
            # Default CPU resources
            self._resources = Resources(cpus=1.0)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup(self, _: WorkerMetadata | None = None) -> None:
        # TODO: Do not hardcode this
        self.tokenizer = AutoTokenizer.from_pretrained("nvidia/quality-classifier-deberta")

    def process(self, batch: DocumentBatch) -> DocumentBatch | list[DocumentBatch]:
        df = batch.to_pyarrow()

        # TODO: Tokenizer or op.Tokenizer
        if self.use_gpu:
            result_df = op.Tokenizer(
                self.model,
                cols=self.cols,
                tokenizer_type=self.tokenizer_type,
                max_chars=self.max_chars,
                keep_cols=df.columns.to_list(),
            )(df).to_pandas()

            return DocumentBatch(
                task_id=f"{batch.task_id}_{self.name}",
                dataset_name=batch.dataset_name,
                data=result_df,
            )
        else:
            chunks = BasicTokenizer(self.tokenizer)(df)

            # TODO: This version is a fanout stage. Does it help?
            batches = []

            for i, chunk in enumerate(chunks):
                batches.append(
                    DocumentBatch(
                        task_id=f"{batch.task_id}_{self.name}_{i}",
                        dataset_name=f"{batch.dataset_name}_{i}",
                        data=chunk,
                    )
                )

            return batches


@dataclass
class CrossFitPredictorWrapper(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Wrapper for CrossFit predictor"""

    model: "HFModel"
    sorted_data_loader: bool
    model_batch_size: int
    pred_output_col: str
    progress_bar: bool
    _name: str = "crossfit_predictor"

    def __post_init__(self):
        # TODO: Check this
        self._resources = Resources(gpu_memory_gb=_get_suggest_memory_for_classifier() + 3)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

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
    _name: str = "crossfit_labeler"

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
