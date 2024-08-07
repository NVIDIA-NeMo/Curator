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
from typing import Callable, Optional, Union

import pandas as pd
from dask.dataframe.extensions import make_array_nonempty
from dask.typing import no_default

from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import DocumentFilter
from nemo_curator.utils.module_utils import is_batched

# Override so that pd.NA is not passed during the metadata inference
make_array_nonempty.register(
    pd.StringDtype,
    lambda x: pd.array(["a", "b"], dtype=x),
)


class Score:
    """
    The module responsible for adding metadata to records based on statistics about the text.
    It accepts an arbitrary scoring function that accepts a text field and returns a score.

    Unlike ScoreFilter, it does not filter based on the computed score.
    It only adds metadata to the record.
    """

    def __init__(
        self,
        score_fn: Callable,
        score_field: str,
        text_field: str = "text",
        score_type: Union[type, str] = None,
    ):
        """
        Constructs a Score module.

        Args:
          score_fn (Callable): The score function that takes in a document string and outputs a score for the document.
          score_field (str): The field the score will be stored in.
          text_field (str): The field the documents will be read from.
          score_type (Union[type, str]): The datatype of the score that will be made for each document.
        """
        self.score_fn = score_fn
        self.score_field = score_field
        self.text_field = text_field
        self.score_type = score_type

    def __call__(self, dataset: DocumentDataset) -> DocumentDataset:
        """
        Applies the scoring to a dataset

        Args:
            dataset (DocumentDataset): The dataset to apply the module to

        Returns:
            DocumentDataset: A dataset with the new score
        """
        # Set the metadata for the function calls if provided
        if self.score_type:
            meta = (None, self.score_type)
        else:
            meta = no_default

        if is_batched(self.score_fn):
            dataset.df[self.score_field] = dataset.df[self.text_field].map_partitions(
                self.score_fn, meta=meta
            )
        else:
            dataset.df[self.score_field] = dataset.df[self.text_field].apply(
                self.score_fn, meta=meta
            )

        return dataset


class Filter:
    """
    The module responsible for filtering records based on a metadata field.
    It accepts an arbitrary filter function that accepts a metadata field and returns True if the field should be kept.

    Unlike ScoreFilter, it does not compute the metadata based on a document.
    It only filters using existing metadata.
    """

    def __init__(self, filter_fn: Callable, filter_field: str, invert: bool = False):
        """
        Constructs a Filter module

        Args:
          filter_fn (Callable): A function that returns True if the document is to be kept.
          filter_field (str): The field(s) to be passed into the filter function.
          invert (bool): Whether to invert the filter condition.
        """
        self.filter_fn = filter_fn
        self.filter_field = filter_field
        self.invert = invert

    def __call__(self, dataset: DocumentDataset) -> DocumentDataset:
        """
        Applies the filtering to a dataset

        Args:
            dataset (DocumentDataset): The dataset to apply the module to

        Returns:
            DocumentDataset: A dataset with entries removed according to the filter
        """
        if is_batched(self.filter_fn):
            bool_mask = dataset.df[self.filter_field].map_partitions(
                self.filter_fn, meta=(None, bool)
            )
        else:
            bool_mask = dataset.df[self.filter_field].apply(
                self.filter_fn, meta=(None, bool)
            )

        if self.invert:
            bool_mask = ~bool_mask

        return DocumentDataset(dataset.df[bool_mask])


class ScoreFilter:
    """
    The module responsible for applying a filter to all documents in a DocumentDataset.
    It accepts an arbitrary DocumentFilter and first computes the score for a document.
    Then, determines whether to keep the document based on the criteria in the DocumentFilter.

    The filter can be applied to any field in the dataset, and the score can be logged for later.
    Also, the filter can be inverted such that "rejected" documents are kept.
    """

    def __init__(
        self,
        filter_obj: DocumentFilter,
        text_field: str = "text",
        score_field: Optional[str] = None,
        score_type: Union[type, str] = None,
        invert: bool = False,
    ):
        """
        Constructs a ScoreFilter module.

        Args:
          filter_obj (DocumentFilter): The score function that takes in a document string and outputs a score for the document.
          text_field (str): The field the documents will be read from.
          score_field: The field to which the scores will be written. If None, scores will be immediately discarded after use.
          score_type (Union[type, str]): The datatype of the score that will be made for each document.
          invert (bool): If True, will keep all documents that are normally discarded.
        """
        self.filter_obj = filter_obj
        self.text_field = text_field
        self.score_field = score_field
        self.score_type = score_type
        self.invert = invert

    def __call__(self, dataset: DocumentDataset) -> DocumentDataset:
        """
        Scores and filters all records in the dataset

        Args:
            dataset (DocumentDataset): The dataset to apply the module to

        Returns:
            DocumentDataset: A dataset with the score and filter applied
        """
        if self.score_type:
            meta = (None, self.score_type)
        else:
            meta = no_default

        if is_batched(self.filter_obj.score_document):
            scores = dataset.df[self.text_field].map_partitions(
                self.filter_obj.score_document, meta=meta
            )
        else:
            scores = dataset.df[self.text_field].apply(
                self.filter_obj.score_document, meta=meta
            )

        if self.score_field is not None:
            dataset.df[self.score_field] = scores

        if is_batched(self.filter_obj.keep_document):
            bool_mask = scores.map_partitions(
                self.filter_obj.keep_document, meta=(None, bool)
            )
        else:
            bool_mask = scores.apply(self.filter_obj.keep_document, meta=(None, bool))
        if self.invert:
            bool_mask = ~bool_mask

        return DocumentDataset(dataset.df[bool_mask])
