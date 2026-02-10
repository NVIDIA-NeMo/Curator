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

from typing import Literal

from nemo_curator.stages.text.filters.doc_filter import DocumentFilter
from nemo_curator.stages.text.utils.constants import (
    bullet_list,
    common_english_words,
    ellipsis_marks,
    end_marks,
    policy_substrings,
    regex_alpha,
    regex_alphanum,
    regex_digit,
    regex_hash,
    regex_paren,
    regex_url,
    white_space_list,
)
from nemo_curator.stages.text.utils.text_utils import (
    get_paragraphs,
    get_sentences,
    get_word_splitter,
)


class NonAlphaNumericFilter(DocumentFilter):
    """
    If more than 25% of the document is non-alphanumeric, then discard.
    Intended to be applied only to English text.
    Source: Adapted from Gopher (Rae et al., 2021)
    """

    def __init__(self, max_non_alpha_numeric_to_text_ratio: float = 0.25):
        super().__init__()
        self._cutoff = max_non_alpha_numeric_to_text_ratio
        self._name = "alpha_numeric"

    def score_document(self, text: str) -> float:
        nchar = len(text)
        # Remove the document if it is empty
        return (nchar - len(regex_alphanum.findall(text))) / nchar if nchar > 0 else 1.0

    def keep_document(self, score: float) -> bool:
        return score <= self._cutoff


class SymbolsToWordsFilter(DocumentFilter):
    """
    Remove any document with a symbol-to-word ratio greater than
    0.1 for either the hash symbol or the elipsis.
    Source: Gopher (Rae et al., 2021)

    For Chinese and Japanese text, we use external libraries to split the text
    because these languages are not separated by spaces. For all other languages,
    such as English, we assume words are separated by spaces.
    """

    def __init__(self, max_symbol_to_word_ratio: float = 0.1, lang: str = "en"):
        super().__init__()
        self._cutoff = max_symbol_to_word_ratio
        self._word_splitter = get_word_splitter(lang)
        self._name = "symbol_to_word"

    def score_document(self, text: str) -> float:
        num_symbol_words = 0
        words = self._word_splitter(text.strip())
        for w in words:
            word = w.strip()
            # Checks if the word is an elipsis or consists mostly of symbols.
            symbol_ratio = len(regex_hash.findall(word)) / len(word)
            if word in ellipsis_marks or symbol_ratio > 0.5:  # noqa: PLR2004
                num_symbol_words += 1
        return num_symbol_words / len(words)

    def keep_document(self, score: float) -> bool:
        return score <= self._cutoff


class NumbersFilter(DocumentFilter):
    """
    If more than 15% of the document contains numbers, then discard.
    """

    def __init__(self, max_number_to_text_ratio: float = 0.15):
        super().__init__()
        self._cutoff = max_number_to_text_ratio
        self._name = "numbers_ratio"

    def score_document(self, text: str) -> float:
        nchar = len(text)
        # Remove if the document is empty
        return len(regex_digit.findall(text)) / nchar if nchar > 0 else 1.0

    def keep_document(self, score: float) -> bool:
        return score <= self._cutoff


class UrlsFilter(DocumentFilter):
    """
    If more than 20% of the document is comprised of URLs, then discard.
    """

    def __init__(self, max_url_to_text_ratio: float = 0.2):
        super().__init__()
        self._cutoff = max_url_to_text_ratio
        self._name = "urls_ratio"

    def score_document(self, text: str) -> float:
        all_urls = regex_url.findall(text)
        url_chars = sum([len(url) for url in all_urls])
        nchar = len(text)
        # Remove if the document is empty
        return url_chars / nchar if nchar > 0 else 1.0

    def keep_document(self, score: float) -> bool:
        return score <= self._cutoff


class BulletsFilter(DocumentFilter):
    """
    If more than 90% of the lines start with a bullet, then discard.
    Source: Gopher (Rae et al., 2021)
    """

    def __init__(self, max_bullet_lines_ratio: float = 0.9):
        super().__init__()
        self._cutoff = max_bullet_lines_ratio
        self._sentences = None
        self._name = "bullet_ratio"

    def score_document(self, text: str) -> float:
        # Get sentences
        sentences = self._sentences
        if sentences is None:
            sentences = get_sentences(text)
        num_bullet_lines = 0
        for sentence in sentences:
            for bullet in bullet_list:
                if sentence.strip().startswith(bullet):
                    num_bullet_lines += 1
                    break
        return num_bullet_lines / len(sentences)

    def keep_document(self, score: float) -> bool:
        return score <= self._cutoff


class WhiteSpaceFilter(DocumentFilter):
    """
    If the document contains a significant number
    of white space characters, then discard.
    """

    def __init__(self, max_white_space_ratio: float = 0.25):
        super().__init__()
        self._cutoff = max_white_space_ratio
        self._name = "white_space"

    def score_document(self, text: str) -> float:
        # Do not strip the document since we want to
        # include leading and trailing whitepsaces as well.
        nchar = len(text)
        # Remove if the document is empty
        return len([x for x in text if x in white_space_list]) / nchar if nchar > 0 else 1.0

    def keep_document(self, score: float) -> bool:
        return score <= self._cutoff


class ParenthesesFilter(DocumentFilter):
    """
    If more than 10% of the sentence is in parentheses, then discard.
    """

    def __init__(self, max_parentheses_ratio: float = 0.1):
        super().__init__()
        self._max_parentheses_ratio = max_parentheses_ratio
        self._name = "parentheses_ratio"

    def score_document(self, text: str) -> float:
        nchar = len(text)
        # Remove if the document is empty
        return len(regex_paren.findall(text)) / nchar if nchar > 0 else 1.0

    def keep_document(self, score: float) -> bool:
        return score <= self._max_parentheses_ratio


class LongWordFilter(DocumentFilter):
    """
    If the document contains a word longer than 1000 characters, then discard.
    NOTE: This seems to be catching things like minified `.js` files
    that don't have spaces anywhere.
    Source: C4 (Google)

    For Chinese and Japanese text, we use external libraries to split the text
    because these languages are not separated by spaces. For all other languages,
    such as English, we assume words are separated by spaces.
    """

    def __init__(self, max_word_length: int = 1000, lang: str = "en"):
        super().__init__()
        self._max_word_length = max_word_length
        self._word_splitter = get_word_splitter(lang)
        self._name = "max_word_length"

    def score_document(self, text: str) -> float:
        return max(len(w) for w in self._word_splitter(text.strip()))

    def keep_document(self, score: float) -> bool:
        return score <= self._max_word_length


class WordCountFilter(DocumentFilter):
    """
    If a document contains a number of words not
    within a specified range, then discard.

    For Chinese and Japanese text, we use external libraries to split the text
    because these languages are not separated by spaces. For all other languages,
    such as English, we assume words are separated by spaces.
    """

    def __init__(self, min_words: int = 50, max_words: int = 100000, lang: str = "en"):
        super().__init__()
        self._min_words = min_words
        self._max_words = max_words
        self._word_splitter = get_word_splitter(lang)
        self._name = "word_count"

    def score_document(self, text: str) -> float:
        return len(self._word_splitter(text.strip()))

    def keep_document(self, score: float) -> bool:
        return self._min_words <= score <= self._max_words


class BoilerPlateStringFilter(DocumentFilter):
    """
    If more than 40% of paragraphs contain boilerplate strings, then discard.
    This includes things like "terms of use", "privacy policy", etc.
    Source: Adapted significantly from Google C4 processing.
    """

    def __init__(
        self,
        remove_if_at_top_or_bottom: bool = True,
        max_boilerplate_string_ratio: float = 0.4,
    ):
        super().__init__()
        self._remove_if_at_top_or_bottom = remove_if_at_top_or_bottom
        self._max_boilerplate_string_ratio = max_boilerplate_string_ratio
        self._boilerplate_paragraph_indices = []
        self._max_ratio = 1.0
        self._name = "boilerplate_string_ratio"

    def score_document(self, text: str) -> float:
        # Initialize variables
        boilerplate_paragraph_count = 0

        # Get the paragraphs
        paragraphs = get_paragraphs(text)

        # Check each paragraph
        for _idx, each_paragraph in enumerate(paragraphs):
            paragraph = each_paragraph.strip().lower()
            if "lorem ipsum" in paragraph:
                return self._max_ratio
            if any(p in paragraph for p in policy_substrings):
                boilerplate_paragraph_count += 1

        return boilerplate_paragraph_count / len(paragraphs)

    def keep_document(self, score: float) -> bool:
        return score <= self._max_boilerplate_string_ratio


class MeanWordLengthFilter(DocumentFilter):
    """
    If the mean word length is not in a specified range, then discard.

    For Chinese and Japanese text, we use external libraries to split the text
    because these languages are not separated by spaces. For all other languages,
    such as English, we assume words are separated by spaces.
    """

    def __init__(
        self,
        min_mean_word_length: int = 3,
        max_mean_word_length: int = 10,
        lang: str = "en",
    ):
        super().__init__()
        self._min_cutoff = min_mean_word_length
        self._max_cutoff = max_mean_word_length
        self._word_splitter = get_word_splitter(lang)
        self._name = "mean_word_length"

    def score_document(self, text: str) -> float:
        word_lens = [len(w) for w in self._word_splitter(text.strip()) if len(w) > 0]
        return sum(word_lens) / len(word_lens)

    def keep_document(self, score: float) -> bool:
        return self._min_cutoff <= score <= self._max_cutoff


class PunctuationFilter(DocumentFilter):
    """
    If more than 85% of the sentences do not end with a
    punctuation mark, then discard.
    Source: Google C4 processing
    """

    def __init__(self, max_num_sentences_without_endmark_ratio: float = 0.85):
        super().__init__()
        self._cutoff = max_num_sentences_without_endmark_ratio
        self._name = "punctuation"

    def score_document(self, text: str) -> float:
        sentences = self._sentences
        if sentences is None:
            sentences = get_sentences(text)
        num_sentence_without_endmarks = len([s for s in sentences if not s.strip().endswith(end_marks)])
        return num_sentence_without_endmarks / len(sentences)

    def keep_document(self, score: float) -> bool:
        return score <= self._cutoff


class EllipsisFilter(DocumentFilter):
    """
    If more than 30% of the sentences end with an elipsis, then discard.
    Source: Google C4 processing
    """

    def __init__(self, max_num_lines_ending_with_ellipsis_ratio: float = 0.3):
        super().__init__()
        self._cutoff = max_num_lines_ending_with_ellipsis_ratio
        self._name = "ellipsis"

    def score_document(self, text: str) -> float:
        sentences = self._sentences
        if sentences is None:
            sentences = get_sentences(text)
        num_lines_ending_with_ellipsis = 0
        for sentence in sentences:
            for ellipsis in ellipsis_marks:
                if sentence.strip().lower().endswith(ellipsis):
                    num_lines_ending_with_ellipsis += 1
                    break
        return num_lines_ending_with_ellipsis / len(sentences)

    def keep_document(self, score: float) -> bool:
        return score <= self._cutoff


class CommonEnglishWordsFilter(DocumentFilter):
    """
    If the sentence contains at least 2 common English words, then keep it.
    NOTE: We purposefully check for the lowercase versions of those common words
    to remove documents with over-capitalization.

    For Chinese and Japanese text, we use external libraries to split the text
    because these languages are not separated by spaces. For all other languages,
    such as English, we assume words are separated by spaces.
    """

    def __init__(self, min_num_common_words: int = 2, stop_at_false: bool = True):
        super().__init__()
        self._cutoff = min_num_common_words
        self._stop_at_false = stop_at_false
        self._word_splitter = get_word_splitter("en")
        self._name = "common_english_words"

    def score_document(self, text: str) -> int:
        common_word_counter = 0
        for word in self._word_splitter(text.strip()):
            if word in common_english_words:
                common_word_counter += 1
            if self._stop_at_false and common_word_counter >= self._cutoff:
                return common_word_counter

        return common_word_counter

    def keep_document(self, score: int) -> bool:
        return score >= self._cutoff


class WordsWithoutAlphabetsFilter(DocumentFilter):
    """
    80% of words in a document must contain at least one alphabetic character.
    Source: Gopher (Rae et al., 2021)

    For Chinese and Japanese text, we use external libraries to split the text
    because these languages are not separated by spaces. For all other languages,
    such as English, we assume words are separated by spaces.
    """

    def __init__(self, min_words_with_alphabets: float = 0.8, lang: str = "en"):
        super().__init__()
        self._cutoff = min_words_with_alphabets
        self._word_splitter = get_word_splitter(lang)
        self._name = "words_without_alphabets"

    def score_document(self, text: str) -> float:
        num_english_alpha = 0
        words = self._word_splitter(text.strip())
        for word in words:
            if regex_alpha.search(word):
                num_english_alpha += 1

        return num_english_alpha / len(words)

    def keep_document(self, score: float) -> bool:
        return score >= self._cutoff


class PornographicUrlsFilter(DocumentFilter):
    """
    Check if any of the URLs within the document point to pornography.
    """

    def __init__(self):
        super().__init__()

    def score_document(self, text: str) -> int:
        all_urls = regex_url.findall(text)
        for url in all_urls:
            if "porn" in url:
                return 1

        return 0

    def keep_document(self, score: int) -> bool:
        return score != 1


class SubstringFilter(DocumentFilter):
    """
    Keeps documents that contain a substring in a given position.
    Gives a score of 1 if the substring is found in the given position, otherwise 0.
    """

    def __init__(self, substring: str, position: Literal["prefix", "suffix", "any"]):
        """
        Args:
            substring (str): The substring to check for.
            position (Literal["prefix", "suffix", "any"]): The position of the substring.
        """
        super().__init__()
        self._substring = substring
        if position not in ["prefix", "suffix", "any"]:
            msg = f"Invalid position: {position}. Must be one of: prefix, suffix, any."
            raise ValueError(msg)
        self._position = position

    def score_document(self, text: str) -> int:
        if self._position == "prefix":
            return int(text.startswith(self._substring))
        elif self._position == "suffix":
            return int(text.endswith(self._substring))
        elif self._position == "any":
            return int(self._substring in text)
        else:
            msg = f"Invalid position: {self._position}. Must be one of: prefix, suffix, any."
            raise ValueError(msg)

    def keep_document(self, score: int) -> bool:
        return score == 1
