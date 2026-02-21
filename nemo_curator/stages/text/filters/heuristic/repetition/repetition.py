# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_curator.stages.text.filters.doc_filter import DocumentFilter
from nemo_curator.stages.text.utils.text_utils import (
    get_ngrams,
    get_paragraphs,
    get_sentences,
    get_word_splitter,
)


class RepeatedLinesFilter(DocumentFilter):
    """
    If the document shrinks by > 30% in terms of number of lines after
    removing duplicate lines, then discard.
    Source: Gopher (Rae et al., 2021)
    """

    def __init__(self, max_repeated_line_fraction: float = 0.7):
        super().__init__()
        self._cutoff = max_repeated_line_fraction
        self._name = "repeated_lines"

    def score_document(self, text: str) -> float:
        sentences = self._sentences
        if sentences is None:
            sentences = get_sentences(text)
        return len(set(sentences)) / len(sentences)

    def keep_document(self, score: float) -> bool:
        return score >= self._cutoff


class RepeatedParagraphsFilter(DocumentFilter):
    """
    If the document shrinks by > 30% in terms of number of lines after
    removing duplicate paragraphs, then discard.
    Source: Gopher (Rae et al., 2021)
    """

    def __init__(self, max_repeated_paragraphs_ratio: float = 0.7):
        super().__init__()
        self._max_repeated_paragraphs_ratio = max_repeated_paragraphs_ratio
        self._name = "repeated_paragraphs"

    def score_document(self, text: str) -> float:
        paragraphs = self._paragraphs
        if paragraphs is None:
            paragraphs = get_paragraphs(text)
        return len(set(paragraphs)) / len(paragraphs)

    def keep_document(self, score: float) -> bool:
        return score >= self._max_repeated_paragraphs_ratio


class RepeatedLinesByCharFilter(DocumentFilter):
    """
    If the document shrinks by > 20% in terms of number of lines
    after removing duplicate lines, then discard.
    Source: Gopher (Rae et al., 2021)
    """

    def __init__(self, max_repeated_lines_char_ratio: float = 0.8):
        super().__init__()
        self._cutoff = max_repeated_lines_char_ratio
        self._name = "repeated_lines_char"

    def score_document(self, text: str) -> float:
        sentences = self._sentences
        if sentences is None:
            sentences = get_sentences(text)

        return len("".join(set(sentences))) / len("".join(sentences))

    def keep_document(self, score: float) -> bool:
        return score >= self._cutoff


class RepeatedParagraphsByCharFilter(DocumentFilter):
    """
    If the document shrinks by > 10% in terms of number of lines after
    removing duplicate paragraphs, then discard.
    Source: Gopher (Rae et al., 2021)
    """

    def __init__(self, max_repeated_paragraphs_char_ratio: float = 0.8):
        super().__init__()
        self._cutoff = max_repeated_paragraphs_char_ratio
        self._name = "repeated_paragraphs_char"

    def score_document(self, text: str) -> float:
        paragraphs = self._paragraphs
        if paragraphs is None:
            paragraphs = get_paragraphs(text)

        return len("".join(set(paragraphs))) / len("".join(paragraphs))

    def keep_document(self, score: float) -> bool:
        return score >= self._cutoff


class RepeatingTopNGramsFilter(DocumentFilter):
    """
    If the document shrinks by > x% in terms of number of characters after
    removing the top n-grams, then discard.
    Source: Gopher (Rae et al., 2021)

    For Chinese and Japanese text, we use external libraries to split the text
    because these languages are not separated by spaces. For all other languages,
    such as English, we assume words are separated by spaces.
    """

    def __init__(self, n: int = 2, max_repeating_ngram_ratio: float = 0.2, lang: str = "en"):
        super().__init__()
        self._n = n
        self._cutoff = max_repeating_ngram_ratio
        self._max_ratio = 1.0
        self._word_splitter = get_word_splitter(lang)
        self._name = f"repeating_top_{n}grams"

    def score_document(self, text: str) -> float:
        ngrams = self._ngrams
        if ngrams is None:
            split_text = self._word_splitter(text.strip())
            if len(split_text) < self._n:
                return self._max_ratio
            ngrams = get_ngrams(split_text, self._n)
        unique_ngrams = set(ngrams)
        # Find the most frequent ngram in the zipped ngram list
        counts = {ngram: {"freq": 0, "num_chars": sum(len(word) for word in ngram)} for ngram in unique_ngrams}
        for ngram in ngrams:
            counts[ngram]["freq"] += 1
        most_frqnt_ngram = " ".join(max(counts, key=lambda x: counts[x]["freq"]))
        # Find the number of characters the most frequent ngram
        # contributes to the document
        nchar = len(text)
        len_diff = nchar - len(text.replace(most_frqnt_ngram, ""))
        # Remove if the document is empty
        return len_diff / nchar if nchar > 0 else 1.0

    def keep_document(self, score: float) -> bool:
        return score <= self._cutoff


class RepeatingDuplicateNGramsFilter(DocumentFilter):
    """
    If the document shrinks by > x% in terms of number of characters
    after removing all duplicate n-grams, then discard.
    Source: Gopher (Rae et al., 2021)

    For Chinese and Japanese text, we use external libraries to split the text
    because these languages are not separated by spaces. For all other languages,
    such as English, we assume words are separated by spaces.
    """

    def __init__(self, n: int = 2, max_repeating_duplicate_ngram_ratio: float = 0.2, lang: str = "en"):
        super().__init__()
        self._n = n
        self._cutoff = max_repeating_duplicate_ngram_ratio
        self._max_ratio = 1.0
        self._word_splitter = get_word_splitter(lang)
        self._name = f"repeating_dup_{n}gram"

    def score_document(self, text: str) -> float:
        ngrams = self._ngrams
        if ngrams is None:
            split_text = self._word_splitter(text.strip())
            if len(split_text) < self._n:
                return self._max_ratio
            ngrams = get_ngrams(split_text, self._n)

        counts = {}
        duplicated_nchar = 0
        overlapping_ngrams = 0
        for ngram in ngrams:
            counts[ngram] = counts.get(ngram, 0) + 1
            if counts[ngram] > 1:
                # Count the number of characters in this ngram that haven't been counted already
                duplicated_ngrams = sum(len(gram) for gram in ngram[overlapping_ngrams:])
                # Count the spaces between the ngrams
                nspaces = min(self._n - overlapping_ngrams, self._n - 1)
                duplicated_nchar += duplicated_ngrams + nspaces
                overlapping_ngrams = self._n
            overlapping_ngrams = max(overlapping_ngrams - 1, 0)

        nchar = len(text)
        # Remove if the document is empty
        return duplicated_nchar / nchar if nchar > 0 else 1.0

    def keep_document(self, score: float) -> bool:
        return score <= self._cutoff
