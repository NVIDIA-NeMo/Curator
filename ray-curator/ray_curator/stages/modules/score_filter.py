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

import pandas as pd
from loguru import logger

from ray_curator.backends.base import NodeInfo, WorkerMetadata
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.filters.doc_filter import DocumentFilter
from ray_curator.tasks import DocumentBatch


@dataclass
class Score(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    The module responsible for adding metadata to records based on statistics about the text.
    It accepts an arbitrary scoring function that accepts a text field and returns a score.
    It also accepts a DocumentFilter object, in which case the score_fn will be the score_document method of the DocumentFilter.

    Unlike ScoreFilter, it does not filter based on the computed score.
    It only adds metadata to the record.

    If a list of DocumentFilters is provided, the filters are applied in order.
    In this case, the score_field parameter should be a list of strings corresponding to the filters.
    If different filters should be applied to different text fields, then text_field should be a list of strings corresponding to the filters.

    Args:
        score_fn (Callable | DocumentFilter | list[DocumentFilter]): The score function or the DocumentFilter object (or list of DocumentFilters). If it is a DocumentFilter object, the score_fn will be the score_document method of the DocumentFilter.
        score_field (str | list[str]): The field (or list of fields) the score will be stored in.
        text_field (str | list[str]): The field (or list of fields) the documents will be read from.

    """

    score_fn: Callable[[str], float | str] | DocumentFilter | list[DocumentFilter]
    score_field: str | list[str]
    text_field: str | list[str] = "text"
    _name: str = "score_fn"

    def __post_init__(self):
        if self.score_field is None:
            msg = "Score field cannot be None"
            raise ValueError(msg)

        if isinstance(self.score_fn, DocumentFilter):
            self._name = self.score_fn.name

        if isinstance(self.score_fn, (DocumentFilter, Callable)):
            if isinstance(self.score_field, list) and len(self.score_field) > 1:
                msg = f"More score fields than filters provided: {self.score_field}"
                raise ValueError(msg)
            if isinstance(self.text_field, list) and len(self.text_field) > 1:
                msg = f"More text fields than filters provided: {self.text_field}"
                raise ValueError(msg)

            self.score_fn = [self.score_fn]
            self.score_field = [self.score_field]
            self.text_field = [self.text_field]

        elif isinstance(self.score_fn, list):
            self._name = "score_fn_chain"

            if isinstance(self.score_field, str):
                msg = f"Score field must be a list of strings if multiple filters are used: {self.score_field}"
                raise TypeError(msg)
            if isinstance(self.text_field, str):
                logger.info(f"Using the same text field for all filters: {self.text_field}")
                self.text_field = [self.text_field] * len(self.score_fn)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], self.text_field

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], self.score_field

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        for score_fn in self.score_fn:
            if isinstance(score_fn, DocumentFilter) and hasattr(score_fn, "model_check_or_download"):
                score_fn.model_check_or_download()

    def setup(self, _: WorkerMetadata | None = None) -> None:
        for score_fn in self.score_fn:
            if isinstance(score_fn, DocumentFilter):
                if hasattr(score_fn, "load_model"):
                    score_fn.load_model()
                elif hasattr(score_fn, "load_tokenizer"):
                    score_fn.load_tokenizer()

    def process(self, batch: DocumentBatch) -> DocumentBatch | None:
        """
        Applies the scoring to a dataset

        Args:
            batch (DocumentBatch): The batch to apply the module to

        Returns:
            DocumentBatch: A batch with the new score

        """

        df = batch.to_pandas()

        for score_fn_i, text_field_i, score_field_i in zip(
            self.score_fn, self.text_field, self.score_field, strict=True
        ):
            inner_score_fn = score_fn_i.score_document if isinstance(score_fn_i, DocumentFilter) else score_fn_i
            df[score_field_i] = df[text_field_i].apply(inner_score_fn)

        # Create output batch
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


@dataclass
class Filter(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    The module responsible for filtering records based on a metadata field.
    It accepts an arbitrary filter function that accepts a metadata field and returns True if the field should be kept.
    It also accepts a DocumentFilter object, in which case the filter_fn will be the keep_document method of the DocumentFilter.
    Unlike ScoreFilter, it does not compute the metadata based on a document.
    It only filters using existing metadata.

    If a list of DocumentFilters is provided, the filters are applied in order.
    In this case, the filter_field parameter should be a list of strings corresponding to the filters.
    If some filters should be inverted and others not, then invert should be a list of booleans corresponding to the filters.

    Args:
        filter_fn (Callable | DocumentFilter | list[DocumentFilter]): A function (or list of functions) that returns True if the document is to be kept or a DocumentFilter object,
            in which case the filter_fn will be the keep_document method of the DocumentFilter.
        filter_field (str | list[str]): The field (or list of fields) to be passed into the filter function.
        invert (bool | list[bool]): Whether to invert the filter condition.

    """

    filter_fn: Callable | DocumentFilter | list[DocumentFilter]
    filter_field: str | list[str]
    invert: bool | list[bool] = False
    _name: str = "filter_fn"

    def __post_init__(self):
        if self.filter_field is None:
            msg = "Filter field cannot be None"
            raise ValueError(msg)

        if isinstance(self.filter_fn, DocumentFilter):
            self._name = self.filter_fn.name

        if isinstance(self.filter_fn, (DocumentFilter, Callable)):
            if isinstance(self.filter_field, list) and len(self.filter_field) > 1:
                msg = f"More filter fields than filters provided: {self.filter_field}"
                raise ValueError(msg)
            if isinstance(self.invert, list) and len(self.invert) > 1:
                msg = f"More invert flags than filters provided: {self.invert}"
                raise ValueError(msg)

            self.filter_fn = [self.filter_fn]
            self.filter_field = [self.filter_field]
            self.invert = [self.invert]

        elif isinstance(self.filter_fn, list):
            self._name = "filter_fn_chain"

            if isinstance(self.filter_field, str):
                msg = f"Filter field must be a list of strings if multiple filters are used: {self.filter_field}"
                raise TypeError(msg)
            if isinstance(self.invert, bool):
                logger.info(f"Using the same invert flag for all filters: {self.invert}")
                self.invert = [self.invert] * len(self.filter_fn)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], self.filter_field

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def compute_filter_mask(
        self, df: pd.DataFrame, filter_fn: Callable | DocumentFilter, filter_field: str, invert: bool
    ) -> pd.Series:
        """Compute the bool mask to filter the dataset.

        Args:
            df (pd.DataFrame): The dataset to compute filter mask on.
            filter_fn (Callable | DocumentFilter): The filter function to use.
            filter_field (str): The field to read the filter from.
            invert (bool): Whether to invert the filter condition.

        Returns:
            Series: A mask corresponding to each data instance indicating whether it will be retained.

        """

        if isinstance(filter_fn, DocumentFilter):
            filter_fn = filter_fn.keep_document

        bool_mask = df[filter_field].apply(filter_fn)

        if invert:
            bool_mask = ~bool_mask

        return bool_mask

    def process(self, batch: DocumentBatch) -> DocumentBatch | None:
        """
        Applies the filtering to a dataset

        Args:
            batch (DocumentBatch): The batch to apply the module to

        Returns:
            DocumentBatch: A batch with entries removed according to the filter

        """
        df = batch.to_pandas()

        for filter_fn_i, filter_field_i, invert_i in zip(self.filter_fn, self.filter_field, self.invert, strict=True):
            bool_mask = self.compute_filter_mask(df, filter_fn_i, filter_field_i, invert_i)
            df = df[bool_mask]

        if len(df) == 0:
            logger.info(f"All documents filtered out for batch {batch.task_id}")
            return None

        # Create output batch
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


@dataclass
class ScoreFilter(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    The module responsible for applying a filter (or chain of filters) to all documents in a dataset.
    It accepts an arbitrary DocumentFilter and first computes the score for a document.
    Then, determines whether to keep the document based on the criteria in the DocumentFilter.

    The filter can be applied to any field in the dataset, and the score can be logged for later.
    Also, the filter can be inverted such that "rejected" documents are kept.

    If a list of DocumentFilters is provided, the filters are applied in order.
    If different filters should be applied to different text fields, then text_field should be a list of strings corresponding to the filters.
    If different score fields should be created for each filter, then score_field should be a list of strings corresponding to the filters.
    If some filters should be inverted and others not, then invert should be a list of booleans corresponding to the filters.

    Args:
        filter_obj (DocumentFilter | list[DocumentFilter]): The score function (or list of score functions) that takes in a document string and outputs a score for the document.
        text_field (str | list[str]): The field (or list of fields) the documents will be read from.
        score_field (str | list[str] | None): The field (or list of fields) to which the scores will be written. If None, scores will be immediately discarded after use.
        invert (bool | list[bool]): If True, will keep all documents that are normally discarded.

    """

    filter_obj: DocumentFilter | list[DocumentFilter]
    text_field: str | list[str] = "text"
    score_field: str | list[str] | None = None
    invert: bool | list[bool] = False
    _name: str = "score_filter"

    def __post_init__(self):
        if isinstance(self.filter_obj, list):
            self._name = "score_filter_chain"

            num_filters = len(self.filter_obj)

            # Okay to assume same text field and invert flag for all filters
            if isinstance(self.text_field, str):
                logger.info(f"Using the same text field for all filters: {self.text_field}")
                self.text_field = [self.text_field] * num_filters
            if self.invert is not None and isinstance(self.invert, bool):
                logger.info(f"Using the same invert flag for all filters: {self.invert}")
                self.invert = [self.invert] * num_filters

            # Score field must be a list of strings if multiple filters are used
            # Otherwise, the field will be overwritten for each filter
            if self.score_field is not None and isinstance(self.score_field, str):
                msg = f"Score field must be a list of strings if multiple filters are used: {self.score_field}"
                raise ValueError(msg)
            elif self.score_field is None:
                self.score_field = [None] * num_filters

        else:
            self._name = self.filter_obj.name

            if isinstance(self.text_field, list) and len(self.text_field) > 1:
                msg = f"More text fields than filters provided: {self.text_field}"
                raise ValueError(msg)
            if self.score_field is not None and isinstance(self.score_field, list) and len(self.score_field) > 1:
                msg = f"More score fields than filters provided: {self.score_field}"
                raise ValueError(msg)
            if isinstance(self.invert, list) and len(self.invert) > 1:
                msg = f"More invert flags than filters provided: {self.invert}"
                raise ValueError(msg)

            self.filter_obj = [self.filter_obj]
            self.text_field = [self.text_field]
            self.score_field = [self.score_field]
            self.invert = [self.invert]

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], self.text_field

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], self.score_field if self.score_field is not None else []

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        for filter_obj in self.filter_obj:
            if isinstance(filter_obj, DocumentFilter) and hasattr(filter_obj, "model_check_or_download"):
                filter_obj.model_check_or_download()

    def setup(self, _: WorkerMetadata | None = None) -> None:
        for filter_obj in self.filter_obj:
            if isinstance(filter_obj, DocumentFilter):
                if hasattr(filter_obj, "load_model"):
                    filter_obj.load_model()
                elif hasattr(filter_obj, "load_tokenizer"):
                    filter_obj.load_tokenizer()

    def compute_filter_mask(
        self, df: pd.DataFrame, filter_obj: DocumentFilter, text_field: str, score_field: str | None, invert: bool
    ) -> pd.Series:
        """Compute the bool mask to filter the dataset.

        Args:
            df (pd.DataFrame): The dataset to compute filter mask on.
            filter_obj (DocumentFilter): The filter object to use.
            text_field (str): The field to read the text from.
            score_field (str | None): The field to write the scores to.
            invert (bool): Whether to invert the filter condition.

        Returns:
            Series: A mask corresponding to each data instance indicating whether it will be retained.

        """

        scores = df[text_field].apply(filter_obj.score_document)

        if score_field is not None:
            df[score_field] = scores

        bool_mask = scores.apply(filter_obj.keep_document)

        if invert:
            bool_mask = ~bool_mask

        return bool_mask

    def process(self, batch: DocumentBatch) -> DocumentBatch | None:
        """
        Scores and filters all records in the dataset

        Args:
            batch (DocumentBatch): The batch to apply the module to

        Returns:
            DocumentBatch: A batch with the score and filter applied

        """
        df = batch.to_pandas()

        for filter_obj_i, text_field_i, score_field_i, invert_i in zip(
            self.filter_obj, self.text_field, self.score_field, self.invert, strict=True
        ):
            bool_mask = self.compute_filter_mask(df, filter_obj_i, text_field_i, score_field_i, invert_i)
            df = df[bool_mask]

        if len(df) == 0:
            logger.info(f"All documents filtered out for batch {batch.task_id}")
            return None

        # Create output batch
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
