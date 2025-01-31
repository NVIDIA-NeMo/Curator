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

from nemo_curator import (
    ExactDuplicates,
    FuzzyDuplicates,
    FuzzyDuplicatesConfig,
    Modify,
    ScoreFilter,
    SemDedup,
    SemDedupConfig,
    Sequential,
)
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import (
    DocumentFilter,
    RepeatedParagraphsFilter,
    RepeatingTopNGramsFilter,
    UrlsFilter,
    WordCountFilter,
)
from nemo_curator.filters.code import (
    NumberOfLinesOfCodeFilter,
    PythonCommentToCodeFilter,
)
from nemo_curator.modifiers import DocumentModifier
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.modifiers.unicode_reformatter import UnicodeReformatter
from nemo_curator.utils.file_utils import expand_outdir_and_mkdir


class QuotationUnifier(DocumentModifier):
    """
    A simple modifier that unifies the quotation marks in the documents.
    """

    def modify_document(self, text: str) -> str:
        """
        Modifies the given text by replacing left and right single quotes with normal single quotes,
        and replacing left and right double quotes with normal double quotes.

        Args:
            text (str): The text to be modified.

        Returns:
            str: The modified text.
        """
        text = text.replace("‘", "'").replace("’", "'")
        text = text.replace("“", '"').replace("”", '"')
        return text


def clean_and_unify(dataset: DocumentDataset) -> DocumentDataset:
    """
    Cleans and unifies the given dataset using a set of predefined cleaners.

    Args:
        dataset (DocumentDataset): The dataset to be cleaned and unified.

    Returns:
        DocumentDataset: The cleaned and unified dataset.
    """
    cleaners = Sequential(
        [
            # Unify all the quotation marks
            Modify(QuotationUnifier(), text_field="text"),
            # Unify all unicode
            Modify(UnicodeReformatter(), text_field="text"),
        ]
    )
    return cleaners(dataset)


def filter_text(dataset: DocumentDataset) -> DocumentDataset:
    """
    Filters the given dataset based on various criteria.
    Refer to the full list of all filters here:
    https://github.com/NVIDIA/NeMo-Curator/blob/main/config/heuristic_filter_en.yaml
    https://github.com/NVIDIA/NeMo-Curator/blob/main/tutorials/peft-curation/main.py

    Args:
        dataset (DocumentDataset): The dataset to be filtered.

    Returns:
        DocumentDataset: The filtered dataset.
    """
    filters = Sequential(
        [
            # If a document contains a number of words not
            # within a specified range then discard
            ScoreFilter(
                WordCountFilter(min_words=50, max_words=100000),
                text_field="text",
                score_field="word_count",
                score_type=int,
            ),
            # If the document shrinks by > x% in terms of number of characters after
            # removing the top n-grams then discard. Source: Gopher (Rae et al., 2021)
            ScoreFilter(
                RepeatingTopNGramsFilter(n=2, max_repeating_ngram_ratio=0.2),
                text_field="text",
                score_type=float,
            ),
            ScoreFilter(
                RepeatingTopNGramsFilter(n=3, max_repeating_ngram_ratio=0.18),
                text_field="text",
                score_type=float,
            ),
            ScoreFilter(
                RepeatingTopNGramsFilter(n=4, max_repeating_ngram_ratio=0.16),
                text_field="text",
                score_type=float,
            ),
            ScoreFilter(
                RepeatedParagraphsFilter(max_repeated_paragraphs_ratio=0.7),
                text_field="text",
                score_type=float,
            ),
            # If more than 20% of the document is comprised of URLs then discard
            ScoreFilter(
                UrlsFilter(max_url_to_text_ratio=0.2),
                text_field="text",
                score_type=float,
            ),
        ]
    )
    filtered_dataset = filters(dataset)
    return filtered_dataset


def filter_code(dataset: DocumentDataset) -> DocumentDataset:
    """
    Filters the given dataset based on various criteria.
    Refer to the full list of all filters here:
    https://github.com/NVIDIA/NeMo-Curator/blob/main/config/heuristic_filter_en.yaml
    https://github.com/NVIDIA/NeMo-Curator/blob/main/tutorials/peft-curation/main.py

    Args:
        dataset (DocumentDataset): The dataset to be filtered.

    Returns:
        DocumentDataset: The filtered dataset.
    """
    filters = Sequential(
        [
            # If a document contains a number of words not
            # within a specified range then discard
            ScoreFilter(
                WordCountFilter(min_words=5, max_words=100000),
                text_field="text",
                score_field="word_count",
                score_type=int,
            ),
        ]
    )
    filtered_dataset = filters(dataset)
    return filtered_dataset


def filter_code_dataset(dataset: DocumentDataset) -> DocumentDataset:
    """
    Filters the given code dataset based on various code-specific criteria.
    Refer to the full list of all filters here:
    https://github.com/NVIDIA/NeMo-Curator/blob/main/config/heuristic_filter_code.yaml
    https://github.com/NVIDIA/NeMo-Curator/blob/main/nemo_curator/filters/code.py

    Args:
        dataset (DocumentDataset): The dataset to be filtered.

    Returns:
        DocumentDataset: The filtered dataset.
    """
    filters = Sequential(
        [
            # if the comment to code ratio is not within the specified range,
            # discard the document
            ScoreFilter(
                PythonCommentToCodeFilter(
                    min_comment_to_code_ratio=0.001, max_comment_to_code_ratio=0.80
                ),
                text_field="text",
                score_type=float,
            ),
            # if the number of lines of code is not within the specified range,
            # discard the document
            ScoreFilter(
                NumberOfLinesOfCodeFilter(min_lines=5, max_lines=20000),
                text_field="text",
                score_type=int,
            ),
        ]
    )
    filtered_dataset = filters(dataset)
    return filtered_dataset


def redact_pii(dataset: DocumentDataset) -> DocumentDataset:
    """
    Redacts personally identifiable information (PII) from extracted comment lines in a given dataset.

    Args:
        dataset (DocumentDataset): The dataset containing documents with PII.

    Returns:
        DocumentDataset: The redacted dataset with PII replaced by a generic value.
    """
    DEFAULT_MAX_DOC_SIZE = 3000000
    redactor = Modify(
        PiiModifier(
            supported_entities=[
                "PERSON",
                "EMAIL_ADDRESS",
            ],
            anonymize_action="replace",
            device="gpu",
        ),
        text_field="extracted_comment",
    )
    return redactor(dataset)


def redact_code(dataset: DocumentDataset) -> DocumentDataset:
    """
    Redact PII on comment lines on a  given code dataset.

    Args:
        dataset (DocumentDataset): The dataset to be redacted.

    Returns:
        DocumentDataset: The redacted dataset.
    """

    # functions to extract comment lines from each row in a dataframe
    def func(row):
        return row["text"][row["text"].find("/*") : row["text"].find("*/") + 2]

    def func2(row):
        comment = row["text"][row["text"].find("/*") : row["text"].find("*/") + 2]
        return row["text"].replace(comment, str(row["extracted_comment"]))

    dataset.df["extracted_comment"] = dataset.df.apply(func, axis=1, meta=(None, str))
    redacted_dataset = redact_pii(dataset)
    redacted_dataset.df["text"] = redacted_dataset.df.apply(
        func2, axis=1, meta=(None, str)
    )
    redacted_dataset.df = redacted_dataset.df.drop(["extracted_comment"], axis=1)

    return redacted_dataset


def exact_dedupe(dataset: DocumentDataset) -> DocumentDataset:
    """
    Remove exact duplicates from the given DocumentDataset.

    Args:
        dataset (DocumentDataset): The dataset containing documents.

    Returns:
        DocumentDataset: The deduplicated dataset.
    """
    deduplicator = ExactDuplicates(id_field="id", text_field="text", hash_method="md5")
    # Find the duplicates
    duplicates = deduplicator(dataset)
    docs_to_remove = duplicates.df.map_partitions(
        lambda x: x[x._hashes.duplicated(keep="first")]
    )
    # Remove the duplicates using their IDs.
    duplicate_ids = list(docs_to_remove.compute().id)
    dataset_df = dataset.df
    deduped = dataset_df[~dataset_df.id.isin(duplicate_ids)]
    return DocumentDataset(deduped)


def fuzzy_dedupe(dataset: DocumentDataset, cache: str) -> DocumentDataset:
    """
    Removes near-duplicate documents and code lines

    Args:
        dataset (DocumentDataset): The dataset containing documents.
        type (str): Document type to process.

    Returns:
        DocumentDataset: The deduplicated dataset.
    """
    fuzzy_dedup_config = FuzzyDuplicatesConfig(
        cache_dir=cache,
        id_field="id",
        text_field="text",
        seed=42,
        char_ngrams=20,
        num_buckets=20,
        hashes_per_bucket=13,
        use_64_bit_hash=False,
        buckets_per_shuffle=5,
        false_positive_check=False,
        num_anchors=2,
        jaccard_threshold=0.8,
    )
    fuzzy_dup = FuzzyDuplicates(config=fuzzy_dedup_config)
    duplicates = fuzzy_dup(dataset)

    docs_to_remove = duplicates.df.map_partitions(
        lambda x: x[x.group.duplicated(keep="first")]
    )

    # When there are few duplicates we can compute the results to a list and use `isin`.
    duplicate_ids = docs_to_remove.compute().id.to_arrow().to_pylist()
    dataset_df = dataset.df
    deduped = dataset_df[~dataset_df.id.isin(duplicate_ids)]
    return DocumentDataset(deduped)


def semantic_dedupe(
    dataset: DocumentDataset, sem_dedupe_config_yaml_path: str, cache_dir: str
):
    """
    Perform semantic deduplication on the given dataset.

    Args:
        dataset (DocumentDataset): The dataset containing documents.
        type (str): Document type to process.

    Returns:
        The deduplicated DocumentDataset.
    """
    partition_lengths = dataset.df.map_partitions(len).compute()
    non_empty_partitions = [
        i for i, length in enumerate(partition_lengths) if length > 0
    ]
    dataset.df = dataset.df.partitions[non_empty_partitions]

    semdedup_config = SemDedupConfig.from_yaml(sem_dedupe_config_yaml_path)
    expand_outdir_and_mkdir(semdedup_config.cache_dir)
    semdup = SemDedup(config=semdedup_config, id_column_type="str")
    duplicates = semdup(dataset)
    return duplicates


class TextLineCountFilter(DocumentFilter):
    """
    Discard text files based on number of lines.
    """

    def __init__(self, min_lines: int = 10):
        super().__init__()
        self._min_lines = min_lines

    def score_document(self, text: str) -> bool:
        words = text.split()
        if words[0] == "text" and int(words[2]) < self._min_lines:
            return False
        else:
            return True

    def keep_document(self, score) -> bool:
        return score


class CodeLineCountFilter(DocumentFilter):
    """
    Discard code files based on number of lines.
    """

    def __init__(self, min_lines: int = 10, max_lines: int = 20000):
        super().__init__()
        self._min_lines = min_lines
        self._max_lines = max_lines

    def score_document(self, text: str) -> bool:
        words = text.split()
        if words[0] == "code" and (
            int(words[2]) < self._min_lines or int(words[2]) > self._max_lines
        ):
            return False
        else:
            return True

    def keep_document(self, score) -> bool:
        return score


def rm_dir(cache_dir):
    if os.path.isdir(cache_dir):
        os.system(f"rm -rf {cache_dir}")
