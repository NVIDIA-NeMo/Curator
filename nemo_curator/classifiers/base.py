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

os.environ["RAPIDS_NO_INITIALIZE"] = "1"
from abc import ABC, abstractmethod
from typing import List

from crossfit import op

from nemo_curator.datasets import DocumentDataset


class DistributedDataClassifier(ABC):
    """Abstract class for running multi-node multi-GPU data classification"""

    def __init__(
        self,
        model,
        labels,
        filter_by,
        batch_size,
        out_dim,
        pred_column,
        max_chars,
        device_type,
        autocast,
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

    def __call__(self, dataset: DocumentDataset):
        result_doc_dataset = self._run_classifier(dataset)
        if self.filter_by is not None:
            return self._filter_documents(result_doc_dataset)

        return result_doc_dataset

    @abstractmethod
    def _run_classifier(self):
        pass

    def _filter_documents(
        self,
        dataset: DocumentDataset,
    ):
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


def _run_classifier_helper(
    df: "dask_cudf.DataFrame",
    model: "HFModel",
    labels: list[str],
    max_chars: int,
    batch_size: int,
    label_col: str,
    prob_col: str = None,
) -> "dask_cudf.DataFrame":

    keep_prob = prob_col is not None
    prob_internal_col = "_prob"
    # TODO: Make crossfit handle this cleanly
    pred_internal_col = "labels"
    df["sliced_text"] = df["text"].str.slice(0, max_chars)
    columns_to_keep_list = df.columns.to_list()
    columns_to_keep_list.remove("sliced_text")

    classifier_pipe = op.Sequential(
        op.Tokenizer(model, cols=["sliced_text"], tokenizer_type="default"),
        op.Predictor(
            model,
            sorted_data_loader=True,
            batch_size=batch_size,
            pred_output_col=prob_internal_col,
        ),
        repartition=df.npartitions,
        keep_cols=columns_to_keep_list,
    )
    df = classifier_pipe(df)

    # TODO: Make crossfit handle this cleanly
    # to prevent the labeler from dropping the prob_internal_col
    # and combine it into a single step
    labeling_pipe = op.Sequential(
        op.Labeler(labels, cols=[prob_internal_col]),
        keep_cols=columns_to_keep_list + [prob_internal_col],
    )
    df = labeling_pipe(df)

    if keep_prob:
        df = df.rename(
            columns={prob_internal_col: prob_col, pred_internal_col: label_col},
        )
    else:
        df = df.rename(columns={pred_internal_col: label_col})
        df = df.drop(columns=[prob_internal_col])

    return df
