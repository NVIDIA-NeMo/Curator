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
from dataclasses import dataclass
from functools import partial, reduce

import pandas as pd
from loguru import logger

from ray_curator.backends.base import NodeInfo, WorkerMetadata
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.utils.text_utils import get_words
from ray_curator.tasks import DocumentBatch, Task


@dataclass
class NGramFrequencyTask(Task[dict[str, int]]):
    data: dict[str, int]

    def get_matched_ngrams(self) -> dict[str, int]:
        return self.data["matched-ngrams"]

    def get_ngram_freq(self) -> list[tuple[int, int]]:
        return self.data["ngrams-freq"]

    @property
    def num_items(self) -> int:
        return len(self.data["matched-ngrams"])

    def validate(self) -> bool:
        return "matched-ngrams" in self.data and "ngrams-freq" in self.data


class EvalSetNGramFrequencyStage(ProcessingStage[DocumentBatch, NGramFrequencyTask]):
    def __init__(  # noqa: PLR0913
        self,
        eval_set_ngrams: dict[str, int],
        eval_set_ngrams_frequency_sorted: list[tuple[int, int]],
        text_field: str = "text",
        max_ngram_size: int = 13,
        min_document_length: int = 200,
        remove_char_each_side: int = 200,
        max_splits: int = 10,
        removed_dir: str | None = None,
    ) -> None:
        """
        Removes segments of downstream evaluation tasks from a dataset
        Args:
            eval_sets: The evaluation sets to use for decontamination.
            text_field: The field in the dataset that contains the text to be decontaminated.
            max_ngram_size: The maximum amount of task grams that are considered at once for contamination.
            min_document_length: When a document is split, if a split falls below this character length it is discarded.
            remove_char_each_side: The number of characters to remove on either side of the matching ngram
            max_splits: The maximum number of times a document may be split before being entirely discarded.
            removed_dir: If not None, the documents split too many times will be written to this directory using the filename in the dataset.
        """
        self.eval_set_ngrams = eval_set_ngrams
        self.eval_set_ngrams_frequency_sorted = eval_set_ngrams_frequency_sorted

        self.text_field = text_field
        self.max_ngram_size = max_ngram_size
        self.min_document_length = min_document_length
        self.remove_char_each_side = remove_char_each_side
        self.max_splits = max_splits
        self.removed_dir = removed_dir

    @staticmethod
    def _merge_eval_set_ngrams(first: dict[str, int], second: dict[str, int]) -> dict[str, int]:
        first.update(second)
        return first

    def process(self, task: DocumentBatch) -> NGramFrequencyTask:
        logger.info(f"Processing task {task.task_id} with {task.num_items} items")
        # This is the only part that needs to be done on this
        found_result = self._find_ngrams_task(task, self.eval_set_ngrams, self.eval_set_ngrams_frequency_sorted)
        return NGramFrequencyTask(
            data={
                "matched-ngrams": found_result,
                # TODO: Remove this
                "ngrams-freq": self.eval_set_ngrams_frequency_sorted,
            },
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )

    @staticmethod
    def _compute_ngram_freq_sorted(task_ngrams: dict[str, int]) -> list[tuple[int, int]]:
        ngrams_freq = defaultdict(int)
        for ngram_key in task_ngrams:
            ngram_words, _ = get_words(ngram_key)
            length = len(ngram_words)
            ngrams_freq[length] += 1

        return sorted(ngrams_freq.items(), key=lambda item: item[0])

    def _find_ngrams_task(
        self,
        df: pd.DataFrame,
        task_ngrams: dict[str, int],
        ngrams_freq_sorted: list[tuple[int, int]],
    ) -> dict[str, int]:
        df = df.to_pandas()
        count = defaultdict(int)
        for document in df[self.text_field]:
            doc_result = self._find_ngrams(document, task_ngrams, ngrams_freq_sorted)
            count = self._merge_counts(count, doc_result)
        return count

    @staticmethod
    def _merge_counts(first: dict[str, int], second: dict[str, int]) -> dict[str, int]:
        for ngram, count in second.items():
            first[ngram] = first.get(ngram, 0) + count

        return first

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
                check_ngram_free = EvalSetNGramFrequencyStage._check_text(
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
                    check_ngram_free = EvalSetNGramFrequencyStage._check_text(
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
                        check_ngram_free = EvalSetNGramFrequencyStage._check_text(
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
            logger.debug(f" [matched]: {seq}")
            # If this flag is set, we just look for matching n-grams
            # we don't remove any matching n-grams
            # Count the matched n-gram and consider it later
            local_ngram[seq] += 1
            if (start_position + len(seq) + 1) < len(text):
                text_buf.append(text[start_position + len(seq) + 1 : len(text)])
            return False

        return True


class EvalSetNGramRemovalStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    def __init__(
        self,
        matched_ngrams: dict[str, int] | list[dict[str, int]],
        ngram_freq: list[tuple[int, int]],
        text_field: str = "text",
        max_ngram_size: int = 13,
        max_matches: int = 10,
        min_document_length: int = 200,
        remove_char_each_side: int = 200,
        max_splits: int = 10,
        removed_dir: str | None = None,
    ):
        self.matched_ngrams: dict[str, int] | list[dict[str, int]] = matched_ngrams
        self.ngram_freq: list[tuple[int, int]] = ngram_freq
        self.text_field = text_field
        self.max_ngram_size = max_ngram_size
        self.max_matches = max_matches
        self.min_document_length = min_document_length
        self.remove_char_each_side = remove_char_each_side
        self.max_splits = max_splits
        self.removed_dir = removed_dir

    def setup(self, _1: NodeInfo | None = None, _2: WorkerMetadata | None = None) -> None:
        if isinstance(self.matched_ngrams, list):
            self.matched_ngrams = reduce(EvalSetNGramFrequencyStage._merge_counts, self.matched_ngrams)

        self.filtered_ngrams = self._threshold_ngram_count(self.matched_ngrams)

    def process(self, task: DocumentBatch) -> DocumentBatch:
        df = self._remove_ngrams_partition(task.to_pandas(), self.filtered_ngrams, self.ngram_freq)
        return DocumentBatch(
            data=df,
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )

    def _threshold_ngram_count(self, matched_ngrams: dict) -> set:
        filtered_ngrams = set()
        for ngram, count in matched_ngrams.items():
            if count <= self.max_matches:
                filtered_ngrams.add(ngram)

        return filtered_ngrams

    def _remove_ngrams_partition(
        self,
        df: pd.DataFrame,
        task_ngrams: dict[str, int],
        ngrams_freq_sorted: list[tuple[int, int]],
    ) -> DocumentBatch:
        text_type = df[self.text_field].dtype

        document_fn = partial(
            self._remove_ngrams,
            task_ngrams=task_ngrams,
            ngrams_freq_sorted=ngrams_freq_sorted,
        )
        split_text = df[self.text_field].apply(document_fn)
        num_splits = split_text.apply(len)

        valid_documents_mask = (num_splits >= 1) & (num_splits <= self.max_splits)

        if self.removed_dir:
            removed_docs = df[~valid_documents_mask]
            # TODO: Add write
            logger.warning(f"No write available. Removing {len(removed_docs)} documents due to too many splits")
            # single_partition_write_with_filename(removed_docs, self.removed_dir)

        df[self.text_field] = split_text
        filtered_df = df[valid_documents_mask]
        exploded_df = filtered_df.explode(self.text_field, ignore_index=True)
        # After exploding, the string datatype can become an "object" type
        exploded_df[self.text_field] = exploded_df[self.text_field].astype(text_type)

        return exploded_df

    def _remove_ngrams(
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

    def _clean_text(
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
            logger.debug(f" [matched]: {seq}")

            # for NMT data we want to completely remove the sample
            # which has a match
            if nosplit_remove:
                return False

            # split the text
            text_first, text_second = self._split_text(
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
