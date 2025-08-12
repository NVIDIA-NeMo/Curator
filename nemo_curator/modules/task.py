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

from collections import defaultdict
from collections.abc import Iterable
from functools import partial, reduce
from typing import Any, Union

import dask.dataframe as dd
import pandas as pd
from dask import delayed

from nemo_curator.datasets import DocumentDataset
from nemo_curator.modules.base import BaseModule
from nemo_curator.tasks.downstream_task import DownstreamTask
from nemo_curator.utils.distributed_utils import single_partition_write_with_filename
from nemo_curator.utils.import_utils import gpu_only_import
from nemo_curator.utils.text_utils import get_words

cudf = gpu_only_import("cudf")


class TaskDecontamination(BaseModule):
    def __init__(  # noqa: PLR0913
        self,
        tasks: DownstreamTask | Iterable[DownstreamTask],
        text_field: str = "text",
        max_ngram_size: int = 13,
        max_matches: int = 10,
        min_document_length: int = 200,
        remove_char_each_side: int = 200,
        max_splits: int = 10,
        removed_dir: str | None = None,
    ) -> None:
        """
        Removes segments of downstream evaluation tasks from a dataset
        Args:
            max_ngram_size: The maximum amount of task grams that are considered at once for contamination.
            max_matches: If an ngram is found more than max_matches times, it is considered too common and will not be removed from the dataset.
            min_document_length: When a document is split, if a split falls below this character length it is discarded.
            remove_char_each_side: The number of characters to remove on either side of the matching ngram
            max_splits: The maximum number of times a document may be split before being entirely discarded.
            removed_dir: If not None, the documents split too many times will be written to this directory using the filename in the dataset.
        """
        super().__init__(input_backend="pandas")
        if isinstance(tasks, DownstreamTask):
            tasks = [tasks]
        self.tasks = tasks
        self.text_field = text_field
        self.max_ngram_size = max_ngram_size
        self.max_matches = max_matches
        self.min_document_length = min_document_length
        self.remove_char_each_side = remove_char_each_side
        self.max_splits = max_splits
        self.removed_dir = removed_dir

    def call(self, dataset: DocumentDataset) -> DocumentDataset:
        # Convert the dataframe to delayed objects for complex operations
        original_meta = dataset.df.dtypes.to_dict()
        delayed_dataset = dataset.df.to_delayed()

        # Perform task decontamintation
        task_ngrams = self.prepare_task_ngram_count()
        found_result = self._find_matching_ngrams(task_ngrams, delayed_dataset)
        matched_ngrams, ngram_freq = (
            found_result["matched-ngrams"],
            found_result["ngrams-freq"],
        )
        delayed_removed_dataset = self._remove_matching_ngrams(matched_ngrams, ngram_freq, delayed_dataset)

        # Restore the dataset to its original format
        return DocumentDataset(dataset_df=dd.from_delayed(delayed_removed_dataset, meta=original_meta))

    @staticmethod
    def _merge_task_ngrams(first: dict[str, int], second: dict[str, int]) -> dict[str, int]:
        first.update(second)
        return first

    def prepare_task_ngram_count(self) -> dict:
        """
        Computes a dictionary of all ngrams in each task as keys and each value set to 0.
        """
        delayed_ngrams = [delayed(task.generate_ngrams)() for task in self.tasks]

        return delayed(reduce)(TaskDecontamination._merge_task_ngrams, delayed_ngrams)

    @staticmethod
    def _compute_ngram_freq_sorted(task_ngrams: dict[str, int]) -> list[tuple[int, int]]:
        ngrams_freq = defaultdict(int)
        for ngram_key in task_ngrams:
            ngram_words, _ = get_words(ngram_key)
            length = len(ngram_words)
            ngrams_freq[length] += 1

        return sorted(ngrams_freq.items(), key=lambda item: item[0])

    def find_matching_ngrams(self, task_ngrams: dict, dataset: DocumentDataset) -> dict:
        delayed_dataset = dataset.df.to_delayed()

        return self._find_matching_ngrams(task_ngrams, delayed_dataset)

    def _find_matching_ngrams(self, task_ngrams: dict, delayed_dataset: list[dd.DataFrame]) -> dict:
        task_ngrams_frequency_sorted = delayed(self._compute_ngram_freq_sorted)(task_ngrams)
        delayed_counts = [
            delayed(self._find_ngrams_partition)(partition, task_ngrams, task_ngrams_frequency_sorted)
            for partition in delayed_dataset
        ]
        combined_counts = delayed(reduce)(self._merge_counts, delayed_counts)

        return delayed(self._format_matching_ngrams_result)(combined_counts, task_ngrams_frequency_sorted)

    def _find_ngrams_partition(
        self,
        dataset_partition: Union[pd.DataFrame, "cudf.DataFrame"],
        task_ngrams: dict[str, int],
        ngrams_freq_sorted: list[tuple[int, int]],
    ) -> dict[str, int]:
        partition_count = defaultdict(int)
        for document in dataset_partition[self.text_field]:
            doc_result = self._find_ngrams(document, task_ngrams, ngrams_freq_sorted)
            partition_count = TaskDecontamination._merge_counts(partition_count, doc_result)

        return partition_count

    @staticmethod
    def _merge_counts(first: dict[str, int], second: dict[str, int]) -> dict[str, int]:
        for ngram, count in second.items():
            first[ngram] = first.get(ngram, 0) + count

        return first

    @staticmethod
    def _format_matching_ngrams_result(
        matched_ngrams: dict[str, int], ngram_freq: list[tuple[int, int]]
    ) -> dict[str, Any]:
        return {
            "matched-ngrams": matched_ngrams,
            "ngrams-freq": ngram_freq,
        }

    def _find_ngrams(  # noqa: C901
        self, document: str, task_ngrams: dict[str, int], ngrams_freq_sorted: list[tuple[int, int]]
    ) -> dict[str, int]:
        """
        Searches for matching n-grams in a document
        """
        text_buf = [document]

        local_ngram = defaultdict(int)
        while len(text_buf) > 0:
            # get the first one from the buffer
            text = text_buf.pop(0)
            words, positions = get_words(text)

            ngram_free = True
            # First, loop over all n-grams in document
            for i in range(len(words) - self.max_ngram_size + 1):
                # Check if we found a matching n-gram
                check_ngram_free = TaskDecontamination._check_text(
                    words[i : i + self.max_ngram_size],
                    task_ngrams,
                    text,
                    positions[i],
                    text_buf,
                    local_ngram,
                )

                # If we found a match, break
                # the remainder of the text is appended to text_buf
                # for futher processing
                if not check_ngram_free:
                    ngram_free = False
                    break

                # Continue searching for the remaining dominant n-grams
                for ngram_len, _ in ngrams_freq_sorted:
                    # Check if we found a matching n-gram
                    check_ngram_free = TaskDecontamination._check_text(
                        words[i : i + ngram_len],
                        task_ngrams,
                        text,
                        positions[i],
                        text_buf,
                        local_ngram,
                    )

                    # Again, if we find match, break
                    # the remainder of the text is appended to text_buf
                    # for futher processing
                    if not check_ngram_free:
                        ngram_free = False
                        break

                # Additional break to break out of both loops
                if not ngram_free:
                    break

            # If did not find a match for the max_ngram_size
            # check the ending n-gram
            if ngram_free and len(words) - self.max_ngram_size > 0:
                # get the last words of the lax max ngram
                last_seq_words = words[len(words) - self.max_ngram_size : len(words)]
                last_seq_start_position = len(words) - self.max_ngram_size

                # check all n-grams lower than max ngram-len
                for _pos, (ngram_len, _) in enumerate(ngrams_freq_sorted):
                    # ignore the max ngram as has been considered already
                    if ngram_len == self.max_ngram_size:
                        continue

                    # find each ngram of ngram_len in max n-grams and check
                    for i in range(len(last_seq_words) - ngram_len + 1):
                        # Check for matching n-grams
                        check_ngram_free = TaskDecontamination._check_text(
                            last_seq_words[i : i + ngram_len],
                            task_ngrams,
                            text,
                            positions[last_seq_start_position + i],
                            text_buf,
                            local_ngram,
                        )

                        # If we find a match, break
                        if not check_ngram_free:
                            ngram_free = False
                            break

                    # Break from both loops
                    if not ngram_free:
                        break

        return local_ngram

    @staticmethod
    def _check_text(  # noqa: PLR0913
        words: list[str],
        task_ngrams: dict[str, int],
        text: str,
        start_position: int,
        text_buf: list[str],
        local_ngram: dict[str, int],
    ) -> bool:
        seq = " ".join(words)
        if seq in task_ngrams:
            print(f" [matched]: {seq}", flush=True)
            # If this flag is set, we just look for matching n-grams
            # we don't remove any matching n-grams
            # Count the matched n-gram and consider it later
            local_ngram[seq] += 1
            if (start_position + len(seq) + 1) < len(text):
                text_buf.append(text[start_position + len(seq) + 1 : len(text)])
            return False

        return True

    def remove_matching_ngrams(
        self, matched_ngrams: dict, ngram_freq: list[tuple], dataset: DocumentDataset
    ) -> DocumentDataset:
        original_meta = dataset.df.dtypes.to_dict()
        delayed_dataset = dataset.df.to_delayed()
        delayed_removed_dataset = self._remove_matching_ngrams(matched_ngrams, ngram_freq, delayed_dataset)

        return DocumentDataset(dataset_df=dd.from_delayed(delayed_removed_dataset, meta=original_meta))

    def _remove_matching_ngrams(
        self, matched_ngrams: dict, ngram_freq: list[tuple], delayed_dataset: list[dd.DataFrame]
    ) -> list[dd.DataFrame]:
        threshhold_ngrams = delayed(self._threshold_ngram_count)(matched_ngrams)

        return [
            delayed(self._remove_ngrams_partition)(partition, threshhold_ngrams, ngram_freq)
            for partition in delayed_dataset
        ]

    def _threshold_ngram_count(self, matched_ngrams: dict) -> set:
        filtered_ngrams = set()
        for ngram, count in matched_ngrams.items():
            if count <= self.max_matches:
                filtered_ngrams.add(ngram)

        return filtered_ngrams

    def _remove_ngrams_partition(
        self,
        partition: Union[pd.DataFrame, "cudf.DataFrame"],
        task_ngrams: dict[str, int],
        ngrams_freq_sorted: list[tuple[int, int]],
    ) -> Union[pd.DataFrame, "cudf.DataFrame"]:
        text_type = partition[self.text_field].dtype

        document_fn = partial(
            self._remove_ngrams,
            task_ngrams=task_ngrams,
            ngrams_freq_sorted=ngrams_freq_sorted,
        )
        split_text = partition[self.text_field].apply(document_fn)
        num_splits = split_text.apply(len)

        valid_documents_mask = (num_splits >= 1) & (num_splits <= self.max_splits)

        if self.removed_dir:
            removed_docs = partition[~valid_documents_mask]
            single_partition_write_with_filename(removed_docs, self.removed_dir)

        partition[self.text_field] = split_text
        filtered_partition = partition[valid_documents_mask]
        exploded_partition = filtered_partition.explode(self.text_field, ignore_index=True)
        # After exploding, the string datatype can become an "object" type
        exploded_partition[self.text_field] = exploded_partition[self.text_field].astype(text_type)

        return exploded_partition

    def _remove_ngrams(  # noqa: C901, PLR0912
        self, document: str, task_ngrams: dict[str, int], ngrams_freq_sorted: list[tuple[int, int]]
    ) -> list[str]:
        """
        Searches for matching n-grams in a document
        """
        text_buf = [document]

        text_buf_ngram_free = []
        while len(text_buf) > 0:
            # get the first one from the buffer
            text = text_buf.pop(0)
            words, positions = get_words(text)

            ngram_free = True
            # First, loop over all n-grams in document
            for i in range(len(words) - self.max_ngram_size + 1):
                # Check if we found a matching n-gram
                check_ngram_free = self._clean_text(
                    words[i : i + self.max_ngram_size],
                    task_ngrams,
                    text,
                    positions[i],
                    text_buf,
                    text_buf_ngram_free,
                )

                # If we found a match, break
                # the remainder of the text is appended to text_buf
                # for futher processing
                if not check_ngram_free:
                    ngram_free = False
                    break

                # Continue searching for the remaining dominant n-grams
                for ngram_len, _ in ngrams_freq_sorted:
                    # Check if we found a matching n-gram
                    check_ngram_free = self._clean_text(
                        words[i : i + ngram_len],
                        task_ngrams,
                        text,
                        positions[i],
                        text_buf,
                        text_buf_ngram_free,
                    )

                    # Again, if we find match, break
                    # the remainder of the text is appended to text_buf
                    # for futher processing
                    if not check_ngram_free:
                        ngram_free = False
                        break

                # Additional break to break out of both loops
                if not ngram_free:
                    break

            # If did not find a match for the max_ngram_size
            # check the ending n-gram
            if ngram_free and len(words) - self.max_ngram_size > 0:
                # get the last words of the lax max ngram
                last_seq_words = words[len(words) - self.max_ngram_size : len(words)]
                last_seq_start_position = len(words) - self.max_ngram_size

                # check all n-grams lower than max ngram-len
                for _pos, (ngram_len, _) in enumerate(ngrams_freq_sorted):
                    # ignore the max ngram as has been considered already
                    if ngram_len == self.max_ngram_size:
                        continue

                    # find each ngram of ngram_len in max n-grams and check
                    for i in range(len(last_seq_words) - ngram_len + 1):
                        # Check for matching n-grams
                        check_ngram_free = self._clean_text(
                            last_seq_words[i : i + ngram_len],
                            task_ngrams,
                            text,
                            positions[last_seq_start_position + i],
                            text_buf,
                            text_buf_ngram_free,
                        )

                        # If we find a match, break
                        if not check_ngram_free:
                            ngram_free = False
                            break

                    # Break from both loops
                    if not ngram_free:
                        break

            # texts are ngram free
            if ngram_free:
                text_buf_ngram_free.append(text)

        return text_buf_ngram_free

    def _clean_text(  # noqa: PLR0913
        self,
        words: list[str],
        matched_ngrams: dict[str, int],
        text: str,
        start_position: int,
        text_buf: list[str],
        text_buf_ngram_free: list[str],
        nosplit_remove: bool = False,
    ) -> bool:
        seq = " ".join(words)
        if seq in matched_ngrams:
            print(f" [matched]: {seq}", flush=True)

            # for NMT data we want to completely remove the sample
            # which has a match
            if nosplit_remove:
                return False

            # split the text
            text_first, text_second = TaskDecontamination._split_text(
                text,
                start_position,
                self.remove_char_each_side,
                seq,
            )

            # Free up the first part of matching n-grams
            if len(text_first) > self.min_document_length:
                text_buf_ngram_free.append(text_first)

            # The second part of the text is added to the output buffer
            # and will be processed later
            if len(text_second) > self.min_document_length:
                text_buf.append(text_second)

            # Is not free of matching ngrams
            return False

        # Free of matching n-grams
        return True

    @staticmethod
    def _split_text(text: str, start_pos: int, remove_char_each_side: int, seq: str) -> tuple[str, str]:
        # first part of the text
        punctuations = ".!?"
        pos = start_pos - remove_char_each_side
        text_first = ""
        while pos > 0 and text[pos] not in punctuations:
            pos -= 1
        if pos > 0:
            text_first = text[0 : pos + 1]

        # add length of seq and remove_char_each_side
        pos = start_pos + len(seq) + remove_char_each_side

        # last part of the text
        text_second = ""
        while pos < len(text) and text[pos] not in punctuations:
            pos += 1
        if pos + 1 < len(text):
            text_second = text[pos + 1 : len(text)]

        return text_first, text_second
