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
from collections.abc import Callable
from dataclasses import dataclass

import fasttext
import huggingface_hub
import pandas as pd
from transformers import AutoTokenizer

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

    Args:
        score_fn (Callable | DocumentFilter): The score function or the DocumentFilter object. If it is a DocumentFilter object, the score_fn will be the score_document method of the DocumentFilter.
        score_field (str): The field the score will be stored in.
        text_field (str): The field the documents will be read from.
        processing_batch_size (int): The number of tasks to process in a batch.

    """

    score_fn: Callable | DocumentFilter
    score_field: str
    text_field: str = "text"
    processing_batch_size: int = 1

    @property
    def name(self) -> str:
        return self.score_fn.name if isinstance(self.score_fn, DocumentFilter) else "score_fn"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.score_field]

    @property
    def batch_size(self) -> int:
        """Number of tasks to process in a batch."""
        return self.processing_batch_size

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        if self.name in ["lang_id", "fasttext_quality_filter"]:
            if not os.path.exists(self.score_fn._model_path):
                msg = f"Model file {self.score_fn._model_path} not found"
                raise FileNotFoundError(msg)
        elif self.name in ["token_count"]:
            # Use snapshot_download to download all files without loading the model into memory.
            huggingface_hub.snapshot_download(
                repo_id=self.score_fn._hf_model_name,
                token=self.score_fn._hf_token,
                local_files_only=False,  # Download if not cached
                resume_download=True,  # Resume interrupted downloads
            )

    def setup(self, _: WorkerMetadata | None = None) -> None:
        if self.name in ["lang_id", "fasttext_quality_filter"]:
            self.score_fn.model = fasttext.load_model(self.score_fn._model_path)
        elif self.name in ["token_count"] and self.score_fn._tokenizer is None:
            self.score_fn._tokenizer = AutoTokenizer.from_pretrained(
                self.score_fn._hf_model_name,
                local_files_only=True,  # Fail if not cached
            )

    def process_batch(self, tasks: list[DocumentBatch]) -> list[DocumentBatch]:
        """
        Scores all records in multiple dataset batches

        Args:
            tasks (list[DocumentBatch]): The dataset batches to apply the module to

        Returns:
            list[DocumentBatch]: The dataset batches with the score applied

        """

        if self.processing_batch_size == 1:
            return [self.process(tasks[0])]

        # Convert and collect all Pandas DataFrames
        dfs = [task.to_pandas() for task in tasks]
        lengths = [len(df) for df in dfs]
        task_ids = [task.task_id for task in tasks]
        dataset_names = [task.dataset_name for task in tasks]

        # Combine and process
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_batch = DocumentBatch(data=combined_df, task_id="batch_list", dataset_name="batch_list")
        processed_batch = self.process(combined_batch)
        processed_df = processed_batch.to_pandas()

        # Use original lengths to rebuild the DocumentBatch objects
        result_batches = []
        offset = 0
        for length, task_id, dataset_name in zip(lengths, task_ids, dataset_names, strict=True):
            chunk = processed_df.iloc[offset : offset + length].reset_index(drop=True)
            result_batches.append(
                DocumentBatch(
                    task_id=f"{task_id}_{self.name}",
                    dataset_name=dataset_name,
                    data=chunk,
                )
            )
            offset += length

        return result_batches

    def process(self, batch: DocumentBatch) -> DocumentBatch | None:
        """
        Applies the scoring to a dataset

        Args:
            batch (DocumentBatch): The batch to apply the module to

        Returns:
            DocumentBatch: A batch with the new score

        """

        if isinstance(self.score_fn, DocumentFilter):
            self.score_fn = self.score_fn.score_document

        df = batch.to_pandas()
        df[self.score_field] = df[self.text_field].apply(self.score_fn)

        # Create output batch
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=df,
        )


@dataclass
class Filter(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    The module responsible for filtering records based on a metadata field.
    It accepts an arbitrary filter function that accepts a metadata field and returns True if the field should be kept.
    It also accepts a DocumentFilter object, in which case the filter_fn will be the keep_document method of the DocumentFilter.
    Unlike ScoreFilter, it does not compute the metadata based on a document.
    It only filters using existing metadata.

    Args:
        filter_fn (Callable | DocumentFilter): A function that returns True if the document is to be kept or a DocumentFilter object,
            in which case the filter_fn will be the keep_document method of the DocumentFilter.
        filter_field (str): The field(s) to be passed into the filter function.
        id_field (str | None): The field to use as the document ID. Required for batch_size > 1.
        invert (bool): Whether to invert the filter condition.
        processing_batch_size (int): The number of tasks to process in a batch.

    """

    filter_fn: Callable | DocumentFilter
    filter_field: str
    id_field: str | None = None
    invert: bool = False
    processing_batch_size: int = 1

    @property
    def name(self) -> str:
        return self.filter_fn.name if isinstance(self.filter_fn, DocumentFilter) else "filter_fn"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.filter_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    @property
    def batch_size(self) -> int:
        """Number of tasks to process in a batch."""
        return self.processing_batch_size

    def compute_filter_mask(self, df: pd.DataFrame) -> pd.Series:
        """Compute the bool mask to filter the dataset.

        Args:
            df (pd.DataFrame): The dataset to compute filter mask on.

        Returns:
            Series: A mask corresponding to each data instance indicating whether it will be retained.

        """

        if isinstance(self.filter_fn, DocumentFilter):
            self.filter_fn = self.filter_fn.keep_document

        bool_mask = df[self.filter_field].apply(self.filter_fn)

        if self.invert:
            bool_mask = ~bool_mask

        return bool_mask

    def process_batch(self, tasks: list[DocumentBatch]) -> list[DocumentBatch]:
        """
        Filters records in multiple dataset batches

        Args:
            tasks (list[DocumentBatch]): The dataset batches to apply the module to

        Returns:
            list[DocumentBatch]: The dataset batches with the filter applied

        """

        if self.processing_batch_size == 1:
            return [self.process(tasks[0])]

        if self.id_field is None:
            msg = "id_field must be provided for batch_size > 1"
            raise ValueError(msg)

        id_field = self.id_field
        id_to_batch = {}
        dfs = []

        for _, task in enumerate(tasks):
            df = task.to_pandas().copy()
            if id_field not in df.columns:
                msg = f"Expected id_field '{id_field}' not found in batch {task.task_id}"
                raise ValueError(msg)

            for doc_id in df[id_field]:
                id_to_batch[doc_id] = (task.task_id, task.dataset_name)

            dfs.append(df)

        # Combine into a single Pandas DataFrame
        combined_df = pd.concat(dfs, ignore_index=True)

        # Process the all DocumentBatch objects together as a single DocumentBatch
        combined_batch = DocumentBatch(data=combined_df, task_id="batch_list", dataset_name="batch_list")
        processed_batch = self.process(combined_batch)

        if processed_batch is None:
            return []

        filtered_df = processed_batch.to_pandas()

        # Group rows back into batches by id_field
        grouped = {}
        for _, row in filtered_df.iterrows():
            doc_id = row[id_field]
            if doc_id not in id_to_batch:
                msg = f"Filtered document ID '{doc_id}' not found in original batches"
                raise ValueError(msg)

            task_id, dataset_name = id_to_batch[doc_id]
            grouped.setdefault((task_id, dataset_name), []).append(row)

        # Rebuild original DocumentBatch objects after filtering
        result_batches = []
        for (task_id, dataset_name), rows in grouped.items():
            batch_df = pd.DataFrame(rows)
            result_batches.append(
                DocumentBatch(
                    task_id=f"{task_id}_{self.name}", dataset_name=dataset_name, data=batch_df.reset_index(drop=True)
                )
            )

        return result_batches

    def process(self, batch: DocumentBatch) -> DocumentBatch | None:
        """
        Applies the filtering to a dataset

        Args:
            batch (DocumentBatch): The batch to apply the module to

        Returns:
            DocumentBatch: A batch with entries removed according to the filter

        """
        df = batch.to_pandas()
        bool_mask = self.compute_filter_mask(df)
        result_df = df[bool_mask]

        if len(result_df) == 0:
            print(f"All documents filtered out for batch {batch.task_id}")
            return None

        # Create output batch
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=result_df,
        )


@dataclass
class ScoreFilter(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    The module responsible for applying a filter to all documents in a dataset.
    It accepts an arbitrary DocumentFilter and first computes the score for a document.
    Then, determines whether to keep the document based on the criteria in the DocumentFilter.

    The filter can be applied to any field in the dataset, and the score can be logged for later.
    Also, the filter can be inverted such that "rejected" documents are kept.

    Args:
        filter_obj (DocumentFilter): The score function that takes in a document string and outputs a score for the document.
        text_field (str): The field the documents will be read from.
        score_field: The field to which the scores will be written. If None, scores will be immediately discarded after use.
        id_field (str | None): The field to use as the document ID. Required for batch_size > 1.
        invert (bool): If True, will keep all documents that are normally discarded.
        processing_batch_size (int): The number of tasks to process in a batch.

    """

    filter_obj: DocumentFilter
    text_field: str = "text"
    score_field: str | None = None
    id_field: str | None = None
    invert: bool = False
    processing_batch_size: int = 1

    @property
    def name(self) -> str:
        return self.filter_obj.name

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.score_field] if self.score_field else []

    @property
    def batch_size(self) -> int:
        """Number of tasks to process in a batch."""
        return self.processing_batch_size

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        if self.name in ["lang_id", "fasttext_quality_filter"]:
            if not os.path.exists(self.filter_obj._model_path):
                msg = f"Model file {self.filter_obj._model_path} not found"
                raise FileNotFoundError(msg)
        elif self.name in ["token_count"]:
            # Use snapshot_download to download all files without loading the model into memory.
            huggingface_hub.snapshot_download(
                repo_id=self.filter_obj._hf_model_name,
                token=self.filter_obj._hf_token,
                local_files_only=False,  # Download if not cached
                resume_download=True,  # Resume interrupted downloads
            )

    def setup(self, _: WorkerMetadata | None = None) -> None:
        if self.name in ["lang_id", "fasttext_quality_filter"]:
            self.filter_obj.model = fasttext.load_model(self.filter_obj._model_path)
        elif self.name in ["token_count"] and self.filter_obj._tokenizer is None:
            self.filter_obj._tokenizer = AutoTokenizer.from_pretrained(
                self.filter_obj._hf_model_name,
                local_files_only=True,  # Fail if not cached
            )

    def compute_filter_mask(self, df: pd.DataFrame) -> pd.Series:
        """Compute the bool mask to filter the dataset.

        Args:
            df (pd.DataFrame): The dataset to compute filter mask on.

        Returns:
            Series: A mask corresponding to each data instance indicating whether it will be retained.

        """

        scores = df[self.text_field].apply(self.filter_obj.score_document)

        if self.score_field is not None:
            df[self.score_field] = scores

        bool_mask = scores.apply(self.filter_obj.keep_document)

        if self.invert:
            bool_mask = ~bool_mask

        return bool_mask

    def process_batch(self, tasks: list[DocumentBatch]) -> list[DocumentBatch]:
        """
        Scores and filters all records in multiple dataset batches

        Args:
            tasks (list[DocumentBatch]): The dataset batches to apply the module to

        Returns:
            list[DocumentBatch]: The dataset batches with the score and filter applied

        """

        if self.processing_batch_size == 1:
            return [self.process(tasks[0])]

        if self.id_field is None:
            msg = "id_field must be provided for batch_size > 1"
            raise ValueError(msg)

        id_field = self.id_field
        id_to_batch = {}
        dfs = []

        for _, task in enumerate(tasks):
            df = task.to_pandas().copy()
            if id_field not in df.columns:
                msg = f"Expected id_field '{id_field}' not found in batch {task.task_id}"
                raise ValueError(msg)

            for doc_id in df[id_field]:
                id_to_batch[doc_id] = (task.task_id, task.dataset_name)

            dfs.append(df)

        # Combine into a single Pandas DataFrame
        combined_df = pd.concat(dfs, ignore_index=True)

        # Process the all DocumentBatch objects together as a single DocumentBatch
        combined_batch = DocumentBatch(data=combined_df, task_id="batch_list", dataset_name="batch_list")
        processed_batch = self.process(combined_batch)

        if processed_batch is None:
            return []

        filtered_df = processed_batch.to_pandas()

        # Group rows back into batches by id_field
        grouped = {}
        for _, row in filtered_df.iterrows():
            doc_id = row[id_field]
            if doc_id not in id_to_batch:
                msg = f"Filtered document ID '{doc_id}' not found in original batches"
                raise ValueError(msg)

            task_id, dataset_name = id_to_batch[doc_id]
            grouped.setdefault((task_id, dataset_name), []).append(row)

        # Rebuild original DocumentBatch objects with the filtered data and/or score columns
        result_batches = []
        for (task_id, dataset_name), rows in grouped.items():
            batch_df = pd.DataFrame(rows)
            result_batches.append(
                DocumentBatch(
                    task_id=f"{task_id}_{self.name}", dataset_name=dataset_name, data=batch_df.reset_index(drop=True)
                )
            )

        return result_batches

    def process(self, batch: DocumentBatch) -> DocumentBatch | None:
        """
        Scores and filters all records in the dataset

        Args:
            batch (DocumentBatch): The batch to apply the module to

        Returns:
            DocumentBatch: A batch with the score and filter applied

        """
        df = batch.to_pandas()
        bool_mask = self.compute_filter_mask(df)
        result_df = df[bool_mask]

        if len(result_df) == 0:
            print(f"All documents filtered out for batch {batch.task_id}")
            return None

        # DEBUGGING
        print(f"[Stage finished] - ScoreFilter - Number of samples - {len(result_df)}")

        # Create output batch
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=result_df,
        )
