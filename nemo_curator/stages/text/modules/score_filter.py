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
from typing import Literal

import pandas as pd
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.filters.doc_filter import DocumentFilter
from nemo_curator.tasks import DocumentBatch


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
        self._name, self.score_fn, self.text_field, _, self.score_field = _validate_and_normalize_filters(
            self.score_fn, self.text_field, None, self.score_field, "score"
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], self.text_field

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], self.text_field + self.score_field

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
                if hasattr(score_fn, "load_tokenizer"):
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
        self._name, self.filter_fn, self.filter_field, self.invert, _ = _validate_and_normalize_filters(
            self.filter_fn, self.filter_field, self.invert, None, "filter"
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], self.filter_field

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], self.filter_field

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
        self._name, self.filter_obj, self.text_field, self.invert, self.score_field = _validate_and_normalize_filters(
            self.filter_obj, self.text_field, self.invert, self.score_field, "score_filter"
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], self.text_field

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], self.text_field + self.score_field if self.score_field is not None else []

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
                if hasattr(filter_obj, "load_tokenizer"):
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

        # Create output batch
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


def _filter_name(x: DocumentFilter | Callable) -> str:
    return x.name if isinstance(x, DocumentFilter) else x.__name__


def _get_filter_stage_name(filters: list[DocumentFilter | Callable], prefix: str) -> str:
    """
    Derive the stage name from the provided score/filter functions.

    """
    return (
        _filter_name(filters[0])
        if len(filters) == 1
        else f"{prefix}_chain_of_" + "_".join(_filter_name(f) for f in filters)
    )


def _format_single_field_list(
    _field: str | list[str] | None, field_name: str, field_type: type = str
) -> list[str] | list[bool]:
    """
    In the case of a single DocumentFilter or Callable, format the relevant field
    (filter_field, score_field, text_field, invert) to a list of length 1.

    Args:
        _field (str | list[str] | None): The field to check and format.
        field_name (str): The name of the field, which is used in error messages.
        field_type (type): The type of the field, which is used in an isinstance check.

    Returns:
        list[str] | list[bool]: The reformatted field.

    """
    if isinstance(_field, list):
        if len(_field) > 1:
            msg = f"More {field_name} fields than functions provided: {_field}"
            raise ValueError(msg)
    elif isinstance(_field, field_type):
        _field = [_field]
    else:
        msg = f"{field_name} field must be a {field_type} or list of {field_type}: {_field}"
        raise TypeError(msg)

    return _field


def _format_field_list(
    _field: str | list[str] | None, filter_count: int, field_name: str, field_type: type = str
) -> list[str] | list[bool]:
    """
    In the case of a list of DocumentFilters or Callables, format the relevant field
    (filter_field, score_field, text_field, invert) to a list of length equal to the number of filters.

    Args:
        _field (str | list[str] | None): The field to check and format.
        filter_count (int): The number of filters. This will be the length of the output list.
        field_name (str): The name of the field, which is used in error messages.
        field_type (type): The type of the field, which is used in an isinstance check.

    Returns:
        list[str] | list[bool]: The reformatted field.

    """
    if isinstance(_field, list):
        if len(_field) == 1:
            logger.info(f"Using the same {field_name} field for all functions: {_field}")
            _field = [_field] * filter_count
        if len(_field) != filter_count:
            msg = f"Number of {field_name} fields must match number of functions: {_field}"
            raise ValueError(msg)
    elif isinstance(_field, field_type):
        logger.info(f"Using the same {field_name} field for all functions: {_field}")
        _field = [_field] * filter_count
    else:
        msg = f"{field_name} field must be a {field_type} or list of {field_type}: {_field}"
        raise TypeError(msg)

    return _field


def _validate_and_normalize_filters(  # noqa: C901, PLR0912
    _filter: DocumentFilter | Callable | list[DocumentFilter | Callable],
    input_field: str | list[str] | None,
    invert: bool | list[bool] | None,
    output_field: str | list[str] | None,
    fn_type: Literal["score", "filter", "score_filter"],
) -> tuple[str, list[DocumentFilter | Callable], list[str] | None, list[bool] | None, list[str] | None]:
    """
    Validate and normalize all parameters needed for the Score, Filter, and ScoreFilter modules.
    "Normalize" means to reformat all parameters to a list of length equal to the number of filters.

    Args:
        _filter (DocumentFilter | Callable | list[DocumentFilter | Callable]): The filter object or list of filter objects.
        input_field (str | list[str] | None): The input field. For Score and ScoreFilter, this is the text field. For Filter, this is the filter field.
        invert (bool | list[bool] | None): The invert flag. This is used for Filter and ScoreFilter.
        output_field (str | list[str] | None): The output field. For Score and ScoreFilter, this is the score field. For Filter, this is not used.
        fn_type (Literal["score", "filter", "score_filter"]): The type of the module.

    Returns:
        tuple[str, list[DocumentFilter | Callable], list[str] | None, list[bool] | None, list[str] | None]:
            The first string returned corresponds to the name given to the DocumentFilter or Callable.
            The normalized filters, input fields, invert flags, and output fields make up the rest of the tuple.

    """

    # For Score and ScoreFilter, the input_field is the text field
    # For Filter, the input_field is the filter field
    input_field_name = "filter" if fn_type == "filter" else "text"
    if input_field is None:
        msg = f"{input_field_name}_field cannot be None"
        raise ValueError(msg)

    # Score is the only module that explicitly requires an output field,
    # i.e., a score_field that is calculated by the DocumentFilter or Callable.
    if output_field is None and fn_type == "score":
        msg = "score_field cannot be None"
        raise ValueError(msg)

    if isinstance(_filter, DocumentFilter):
        _name = _filter.name
    elif isinstance(_filter, Callable):
        _name = f"{fn_type}_fn"

    if isinstance(_filter, (DocumentFilter, Callable)):
        _normalized_filter = [_filter]
        _input_field = _format_single_field_list(input_field, input_field_name, field_type=str)

        if fn_type in ["filter", "score_filter"]:
            _invert = _format_single_field_list(invert, "invert", field_type=bool)
        else:
            # Score does not use an invert flag
            _invert = None

        if fn_type in ["score", "score_filter"]:
            # ScoreFilter is allowed to have no output fields, but Score is not
            if output_field is None and fn_type == "score_filter":
                _output_field = [None]
            else:
                _output_field = _format_single_field_list(output_field, "score", field_type=str)
        else:
            # Filter does not use an output field
            _output_field = None

    elif isinstance(_filter, list):
        _name = _get_filter_stage_name(_filter, prefix=fn_type)
        _normalized_filter = _filter

        # Technically, you could run a list of filters on the same filter_field.
        # However, prefer to use a list of fields to avoid confusion.
        if fn_type == "filter" and (
            isinstance(input_field, str) or (isinstance(input_field, list) and len(input_field) == 1)
        ):
            msg = f"filter_field must be a list of strings if multiple filters are used: {input_field}"
            raise ValueError(msg)

        _input_field = _format_field_list(input_field, len(_filter), input_field_name, field_type=str)

        if fn_type in ["filter", "score_filter"]:
            _invert = _format_field_list(invert, len(_filter), "invert", field_type=bool)
        else:
            # Score does not use an invert flag
            _invert = None

        if fn_type in ["score", "score_filter"]:
            # ScoreFilter is allowed to have no output fields, but Score is not
            if output_field is None and fn_type == "score_filter":
                _output_field = [None] * len(_filter)
            # Output fields are always required to be a (unique) list of strings.
            # We check that here.
            elif isinstance(output_field, str) or (isinstance(output_field, list) and len(output_field) == 1):
                msg = f"score_field must be a list of strings if multiple filters are used: {output_field}"
                raise ValueError(msg)
            else:
                _output_field = _format_field_list(output_field, len(_filter), "score", field_type=str)
        else:
            # Filter does not use an output field
            _output_field = None

    return _name, _normalized_filter, _input_field, _invert, _output_field
