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

import regex

from nemo_curator.filters.doc_filter import DocumentFilter, import_filter
from nemo_curator.utils.constants import (
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
from nemo_curator.utils.text_utils import (
    get_ngrams,
    get_paragraphs,
    get_sentences,
    get_word_splitter,
    is_paragraph_indices_in_top_or_bottom_only,
)


class NonAlphaNumericFilter(DocumentFilter):
    """
    If more than 25% of the document is non-alphanumeric then discard
    Intended to be applied only too english text
    Source: Adapted from Gopher (Rae et al., 2021)
    """

    def __init__(self, max_non_alpha_numeric_to_text_ratio=0.25):
        super().__init__()
        self._cutoff = max_non_alpha_numeric_to_text_ratio
        self._name = "alpha_numeric"

    def score_document(self, text):
        nchar = len(text)
        if nchar > 0:
            score = (nchar - len(regex_alphanum.findall(text))) / nchar
        else:
            # Remove the document if it is empty
            score = 1.0
        return score

    def keep_document(self, score):
        return score <= self._cutoff


class SymbolsToWordsFilter(DocumentFilter):
    """
    Remove any document with symbol-to-word ratio greater than
    0.1 for either the hash symbol or the elipsis
    Source: Gopher (Rae et al., 2021)
    """

    def __init__(self, max_symbol_to_word_ratio=0.1, lang="en"):
        super().__init__()
        self._cutoff = max_symbol_to_word_ratio
        self._word_splitter = get_word_splitter(lang)
        self._name = "symbol_to_word"

    def score_document(self, text):
        num_symbol_words = 0
        words = self._word_splitter(text.strip())
        for word in words:
            word = word.strip()
            # Checks if the word is an elipsis or consists mostly of symbols.
            symbol_ratio = len(regex_hash.findall(word)) / len(word)
            if word in ellipsis_marks or symbol_ratio > 0.5:
                num_symbol_words += 1
        return num_symbol_words / len(words)

    def keep_document(self, score):
        return score <= self._cutoff


class NumbersFilter(DocumentFilter):
    """
    If more than 15% of the document contains numbers then discard
    """

    def __init__(self, max_number_to_text_ratio=0.15):
        super().__init__()
        self._cutoff = max_number_to_text_ratio
        self._name = "numbers_ratio"

    def score_document(self, text):
        nchar = len(text)
        if nchar > 0:
            score = len(regex_digit.findall(text)) / nchar
        else:
            # Remove if the document is empty
            score = 1.0
        return score

    def keep_document(self, score):
        return score <= self._cutoff


class UrlsFilter(DocumentFilter):
    """
    If more than 20% of the document is comprised of URLs then discard
    """

    def __init__(self, max_url_to_text_ratio=0.2):
        super().__init__()
        self._cutoff = max_url_to_text_ratio
        self._name = "urls_ratio"

    def score_document(self, text):
        all_urls = regex_url.findall(text)
        url_chars = sum([len(url) for url in all_urls])
        nchar = len(text)
        if nchar > 0:
            score = url_chars / nchar
        else:
            # Remove if the document is empty
            score = 1.0
        return score

    def keep_document(self, score):
        return score <= self._cutoff


class BulletsFilter(DocumentFilter):
    """
    If more than 90% of the lines start with a bullet then discard
    Source: Gopher (Rae et al., 2021)
    """

    def __init__(self, max_bullet_lines_ratio=0.9):
        super().__init__()
        self._cutoff = max_bullet_lines_ratio
        self._sentences = None
        self._name = "bullet_ratio"

    def score_document(self, text):
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

    def keep_document(self, score):
        return score <= self._cutoff


class WhiteSpaceFilter(DocumentFilter):
    """
    If the document contains a significant number
    of white space characters then discard
    """

    def __init__(self, max_white_space_ratio=0.25):
        super().__init__()
        self._cutoff = max_white_space_ratio
        self._name = "white_space"

    def score_document(self, text):
        # Do not strip the document since we want to
        # include leading and trailing whitepsaces as well.
        nchar = len(text)
        if nchar > 0:
            score = len([x for x in text if x in white_space_list]) / nchar
        else:
            # Remove if the document is empty
            score = 1.0
        return score

    def keep_document(self, score):
        return score <= self._cutoff


class ParenthesesFilter(DocumentFilter):
    """
    If more than 10% of the sentence is in parentheses then discard
    """

    def __init__(self, max_parentheses_ratio=0.1):
        super().__init__()
        self._max_parentheses_ratio = max_parentheses_ratio
        self._name = "parentheses_ratio"

    def score_document(self, text):
        nchar = len(text)
        if nchar > 0:
            score = len(regex_paren.findall(text)) / nchar
        else:
            # Remove if the document is empty
            score = 1.0
        return score

    def keep_document(self, score):
        return score <= self._max_parentheses_ratio


class LongWordFilter(DocumentFilter):
    """
    If the document contains a word longer than 1000 characters then discard
    NOTE: This seems to be catching things like minified `.js` files
    that don't have spaces anywhere.
    Source: C4 (Google)
    """

    def __init__(self, max_word_length=1000, lang="en"):
        super().__init__()
        self._max_word_length = max_word_length
        self._word_splitter = get_word_splitter(lang)
        self._name = "max_word_length"

    def score_document(self, text):
        return max(len(w) for w in self._word_splitter(text.strip()))

    def keep_document(self, score):
        return score <= self._max_word_length


class WordCountFilter(DocumentFilter):
    """
    If a document contains a number of words not
    within a specified range then discard
    """

    def __init__(self, min_words=50, max_words=100000, lang="en"):
        super().__init__()
        self._min_words = min_words
        self._max_words = max_words
        self._word_splitter = get_word_splitter(lang)
        self._name = "word_count"

    def score_document(self, text):
        return len(self._word_splitter(text.strip()))

    def keep_document(self, score):
        return self._min_words <= score <= self._max_words


class BoilerPlateStringFilter(DocumentFilter):
    """
    If more than 40% of paragraphs contain boilerplate strings then discard.
    This includes things like "terms of use", "privacy policy", etc.
    Source: Adapted significantly from Google C4 processing.
    """

    def __init__(
        self,
        remove_if_at_top_or_bottom=True,
        max_boilerplate_string_ratio=0.4,
    ):
        super().__init__()
        self._remove_if_at_top_or_bottom = remove_if_at_top_or_bottom
        self._max_boilerplate_string_ratio = max_boilerplate_string_ratio
        self._boilerplate_paragraph_indices = []
        self._max_ratio = 1.0
        self._name = "boilerplate_string_ratio"

    def score_document(self, text):
        # Initialize variables
        boilerplate_paragraph_count = 0

        # Get the paragraphs
        paragraphs = get_paragraphs(text)

        # Check each paragraph
        for idx, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip().lower()
            if "lorem ipsum" in paragraph:
                return self._max_ratio
            if any(p in paragraph for p in policy_substrings):
                boilerplate_paragraph_count += 1

        return boilerplate_paragraph_count / len(paragraphs)

    def keep_document(self, score):
        return score <= self._max_boilerplate_string_ratio


class MeanWordLengthFilter(DocumentFilter):
    """
    If the mean word length is not in a specified range then discard
    """

    def __init__(
        self,
        min_mean_word_length=3,
        max_mean_word_length=10,
        lang="en",
    ):
        super().__init__()
        self._min_cutoff = min_mean_word_length
        self._max_cutoff = max_mean_word_length
        self._word_splitter = get_word_splitter(lang)
        self._name = "mean_word_length"

    def score_document(self, text):
        word_lens = [len(w) for w in self._word_splitter(text.strip()) if len(w) > 0]
        return sum(word_lens) / len(word_lens)

    def keep_document(self, score):
        return self._min_cutoff <= score <= self._max_cutoff


class RepeatedLinesFilter(DocumentFilter):
    """
    If the document shrinks by > 30% in terms of number of lines after
    removing duplicate lines then discard
    Source: Gopher (Rae et al., 2021)
    """

    def __init__(self, max_repeated_line_fraction=0.7):
        super().__init__()
        self._cutoff = max_repeated_line_fraction
        self._name = "repeated_lines"

    def score_document(self, text):
        sentences = self._sentences
        if sentences is None:
            sentences = get_sentences(text)
        return len(set(sentences)) / len(sentences)

    def keep_document(self, score):
        return score >= self._cutoff


class RepeatedParagraphsFilter(DocumentFilter):
    """
    If the document shrinks by > 30% in terms of number of lines after
    removing duplicate paragraphs then discard.
    Source: Gopher (Rae et al., 2021)
    """

    def __init__(self, max_repeated_paragraphs_ratio=0.7):
        super().__init__()
        self._max_repeated_paragraphs_ratio = max_repeated_paragraphs_ratio
        self._name = "repeated_paragraphs"

    def score_document(self, text):
        paragraphs = self._paragraphs
        if paragraphs is None:
            paragraphs = get_paragraphs(text)
        return len(set(paragraphs)) / len(paragraphs)

    def keep_document(self, score):
        return score >= self._max_repeated_paragraphs_ratio


class RepeatedLinesByCharFilter(DocumentFilter):
    """
    If the document shrinks by > 20% in terms of number of lines
    after removing duplicate lines then discard
    Source: Gopher (Rae et al., 2021)
    """

    def __init__(self, max_repeated_lines_char_ratio=0.8):
        super().__init__()
        self._cutoff = max_repeated_lines_char_ratio
        self._name = "repeated_lines_char"

    def score_document(self, text):
        sentences = self._sentences
        if sentences is None:
            sentences = get_sentences(text)

        return len("".join(set(sentences))) / len("".join(sentences))

    def keep_document(self, score):
        return score >= self._cutoff


class RepeatedParagraphsByCharFilter(DocumentFilter):
    """
    If the document shrinks by > 10% in terms of number of lines after
    removing duplicate paragraphs then discard.
    Source: Gopher (Rae et al., 2021)
    """

    def __init__(self, max_repeated_paragraphs_char_ratio=0.8):
        super().__init__()
        self._cutoff = max_repeated_paragraphs_char_ratio
        self._name = "repeated_paragraphs_char"

    def score_document(self, text):
        paragraphs = self._paragraphs
        if paragraphs is None:
            paragraphs = get_paragraphs(text)

        return len("".join(set(paragraphs))) / len("".join(paragraphs))

    def keep_document(self, score):
        return score >= self._cutoff


class RepeatingTopNGramsFilter(DocumentFilter):
    """
    If the document shrinks by > x% in terms of number of characters after
    removing the top n-grams then discard.
    Source: Gopher (Rae et al., 2021)
    """

    def __init__(self, n=2, max_repeating_ngram_ratio=0.2, lang="en"):
        super().__init__()
        self._n = n
        self._cutoff = max_repeating_ngram_ratio
        self._max_ratio = 1.0
        self._word_splitter = get_word_splitter(lang)
        self._name = f"repeating_top_{n}grams"

    def score_document(self, text):
        ngrams = self._ngrams
        if ngrams is None:
            split_text = self._word_splitter(text.strip())
            if len(split_text) < self._n:
                return self._max_ratio
            ngrams = get_ngrams(split_text, self._n)
        unique_ngrams = set(ngrams)
        # Find the most frequent ngram in the zipped ngram list
        counts = {
            ngram: {"freq": 0, "num_chars": sum(len(word) for word in ngram)}
            for ngram in unique_ngrams
        }
        for ngram in ngrams:
            counts[ngram]["freq"] += 1
        most_frqnt_ngram = " ".join(max(counts, key=lambda x: counts[x]["freq"]))
        # Find the number of characters the most frequent ngram
        # contributes to the document
        nchar = len(text)
        len_diff = nchar - len(text.replace(most_frqnt_ngram, ""))
        if nchar > 0:
            score = len_diff / nchar
        else:
            # Remove if the document is empty
            score = 1.0
        return score

    def keep_document(self, score):
        return score <= self._cutoff


class RepeatingDuplicateNGramsFilter(DocumentFilter):
    """
    If the document shrinks by > x% in terms of number of characters
    after removing all duplicate n-grams then discard.
    Source: Gopher (Rae et al., 2021)
    """

    def __init__(self, n=2, max_repeating_duplicate_ngram_ratio=0.2, lang="en"):
        super().__init__()
        self._n = n
        self._cutoff = max_repeating_duplicate_ngram_ratio
        self._max_ratio = 1.0
        self._word_splitter = get_word_splitter(lang)
        self._name = f"repeating_dup_{n}gram"

    def score_document(self, text):
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
                duplicated_ngrams = sum(
                    len(gram) for gram in ngram[overlapping_ngrams:]
                )
                # Count the spaces between the ngrams
                nspaces = min(self._n - overlapping_ngrams, self._n - 1)
                duplicated_nchar += duplicated_ngrams + nspaces
                overlapping_ngrams = self._n
            overlapping_ngrams = max(overlapping_ngrams - 1, 0)

        nchar = len(text)
        if nchar > 0:
            score = duplicated_nchar / nchar
        else:
            # Remove if the document is empty
            score = 1.0
        return score

    def keep_document(self, score):
        return score <= self._cutoff


class PunctuationFilter(DocumentFilter):
    """
    If more than 85% of the sentences do not end with a
    punctuation mark then discard.
    Source: Google C4 processing
    """

    def __init__(self, max_num_sentences_without_endmark_ratio=0.85):
        super().__init__()
        self._cutoff = max_num_sentences_without_endmark_ratio
        self._name = "punctuation"

    def score_document(self, text):
        sentences = self._sentences
        if sentences is None:
            sentences = get_sentences(text)
        num_sentence_without_endmarks = len(
            [s for s in sentences if not s.strip().endswith(end_marks)]
        )
        return num_sentence_without_endmarks / len(sentences)

    def keep_document(self, score):
        return score <= self._cutoff


class EllipsisFilter(DocumentFilter):
    """
    If more than 30% of the sentences end with an elipsis then discard.
    Source: Google C4 processing
    """

    def __init__(self, max_num_lines_ending_with_ellipsis_ratio=0.3):
        super().__init__()
        self._cutoff = max_num_lines_ending_with_ellipsis_ratio
        self._name = "ellipsis"

    def score_document(self, text):
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

    def keep_document(self, score):
        return score <= self._cutoff


class CommonEnglishWordsFilter(DocumentFilter):
    """
    If the sentence contains at least 2 common english words, keep
    NOTE: we purposefully check for the lowercase versions of those common words
    to remove documents with over-capitalization.
    """

    def __init__(self, min_num_common_words=2, stop_at_false=True):
        super().__init__()
        self._cutoff = min_num_common_words
        self._stop_at_false = stop_at_false
        self._word_splitter = get_word_splitter("en")
        self._name = "common_english_words"

    def score_document(self, text):
        common_word_counter = 0
        for word in self._word_splitter(text.strip()):
            if word in common_english_words:
                common_word_counter += 1
            if self._stop_at_false and common_word_counter >= self._cutoff:
                return common_word_counter

        return common_word_counter

    def keep_document(self, score):
        return score >= self._cutoff


class WordsWithoutAlphabetsFilter(DocumentFilter):
    """
    80% of words in a document must contain at least one alphabetic character
    Source: Gopher (Rae et al., 2021)
    """

    def __init__(self, min_words_with_alphabets=0.8, lang="en"):
        super().__init__()
        self._cutoff = min_words_with_alphabets
        self._word_splitter = get_word_splitter(lang)
        self._name = "words_without_alphabets"

    def score_document(self, text):
        num_english_alpha = 0
        words = self._word_splitter(text.strip())
        for word in words:
            if regex_alpha.search(word):
                num_english_alpha += 1

        return num_english_alpha / len(words)

    def keep_document(self, score):
        return score >= self._cutoff


class PornographicUrlsFilter(DocumentFilter):
    """
    Check if any of the urls within the document point to porn
    """

    def __init__(self):
        super().__init__()

    def score_document(self, text):
        all_urls = regex_url.findall(text)
        for url in all_urls:
            if "porn" in url:
                return 1

        return 0

    def keep_document(self, score):
        return score != 1
