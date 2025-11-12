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

import os.path
import tarfile
from typing import Literal

import huggingface_hub
import requests
from platformdirs import user_cache_dir
from transformers import AutoTokenizer

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
    get_ngrams,
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


class TokenCountFilter(DocumentFilter):
    """
    If the document contains more or less than a specified number of tokens, then discard.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer | None = None,
        hf_model_name: str | None = None,
        hf_token: str | None = None,
        min_tokens: int = 0,
        max_tokens: int = float("inf"),
    ):
        """
        Args:
            tokenizer (AutoTokenizer | None): The pre-loaded tokenizer to use to count the tokens.
                If None, the tokenizer will be initialized from the hf_model_name.
            hf_model_name (str | None): The name of the Hugging Face model to use to count the tokens.
                If None, the pre-loaded tokenizer must be provided via the tokenizer argument.
            hf_token (str | None): The token to use to access the Hugging Face model, if needed.
            min_tokens (int): The minimum number of tokens the document must contain.
                Set to 0 to disable the minimum token count filter.
            max_tokens (int): The maximum number of tokens the document can contain.
                Set to infinity to disable the maximum token count filter.
        """
        super().__init__()

        if tokenizer is None and hf_model_name is None:
            msg = "Either tokenizer or hf_model_name must be provided"
            raise ValueError(msg)
        if tokenizer is not None and hf_model_name is not None:
            msg = "Either tokenizer or hf_model_name must be provided, not both"
            raise ValueError(msg)

        self._token_count_filter_tokenizer = tokenizer
        self._hf_model_name = hf_model_name
        self._hf_token = hf_token
        self._min_tokens = min_tokens
        self._max_tokens = max_tokens
        self._name = "token_count"

    def model_check_or_download(self) -> None:
        if self._hf_model_name is not None:
            # Use snapshot_download to download all files without loading the model into memory.
            huggingface_hub.snapshot_download(
                repo_id=self._hf_model_name,
                token=self._hf_token,
                local_files_only=False,  # Download if not cached
                resume_download=True,  # Resume interrupted downloads
            )

    def load_tokenizer(self) -> None:
        if self._hf_model_name is not None:
            self._token_count_filter_tokenizer = AutoTokenizer.from_pretrained(
                self._hf_model_name, local_files_only=True
            )

    def score_document(self, text: str) -> int:
        tokens = self._token_count_filter_tokenizer.encode(text)
        return len(tokens)

    def keep_document(self, score: int) -> bool:
        return self._min_tokens <= score <= self._max_tokens


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


class HistogramFilter(DocumentFilter):
    """Histogram filter used by the NLLB paper (https://arxiv.org/pdf/2207.04672). See p30 for details.

    The high-level idea of histogram filter can be described as a cheap version of language ID.
    Basically, it checks what ratio of characters in the data instance are included in the character historgrams collected from trusted data in the corresponding language.
    If the ratio is too low, then there is a good chance that there is a language ID mismatch and the data instance should be discarded.

    Written with reference to the original fairseq implementation at:
    https://github.com/facebookresearch/fairseq/blob/main/examples/m2m_100/process_data/clean_histogram.py.
    """

    def __init__(
        self,
        lang: str | None = "en",
        threshold: float | None = 0.8,
        cache_dir: str | None = "",
        threshold_char: str | None = "]",
    ):
        """Args:
        lang (str, optional): Expected language of the segment. This will decide which histogram will be loaded. Defaults to "en".
        threshold (float, optional): Threshold for ratio of characters in the histogram. Defaults to 0.8.
        cache_dir (str, optional): Cache dir download histogram files. Defaults to "".
        threshold_char (str, optional): Formatter character of the histogram files. You should not change this unless you rebuilt your own histogram. Defaults to "]".
        """
        super().__init__()
        self._lang = lang
        self._threshold = threshold
        self._cache_dir = cache_dir if cache_dir else user_cache_dir()
        self._threshold_char = threshold_char
        self._name = "histogram"

        if not os.path.isdir(os.path.join(self._cache_dir, "histograms")):
            self._download_histograms()

        self._read_hist()

    def _download_histograms(self) -> None:
        """Download and process histograms from default repo.

        Raises:
            requests.exceptions.RequestException: If download fails.
        """

        # Send a GET request to the URL
        response = requests.get("https://dl.fbaipublicfiles.com/m2m_100/histograms.tar.gz")  # noqa: S113

        # Check if the request was successful
        if response.status_code != 200:  # noqa: PLR2004
            msg = f"Failed to download histogram file. Status code: {response.status_code}"
            raise requests.exceptions.RequestException(msg)

        # Open a file to write the content
        os.makedirs(self._cache_dir, exist_ok=True)
        download_dest_path = os.path.join(self._cache_dir, "histograms.tar.gz")
        with open(download_dest_path, "wb") as file:
            file.write(response.content)

        extract_path = os.path.join(self._cache_dir, "histograms")
        with tarfile.open(download_dest_path, "r:gz") as tar:
            # Extract all the contents into the specified directory
            tar.extractall(path=extract_path)  # noqa: S202

    def _read_hist(self) -> None:
        """Load histogram files."""

        self._histogram = []
        with open(
            os.path.join(
                self._cache_dir,
                "histograms",
                "checkpoint",
                "edunov",
                "cc60_multilingual",
                "clean_hists",
                self._lang,
            )
        ) as f:
            for line in f:
                c = line[0]
                if c == self._threshold_char:
                    break
                self._histogram.append(c)
        self._histogram = set(self._histogram)

    def score_document(self, text: str) -> float:
        """Compute histogram token ratio of a text data instance according to the loaded histogram.

        Args:
            text (str): Text data instance.

        Returns:
            float: Ratio of tokens included in the histogram.
        """
        cnt = len([c for c in text.strip() if c in self._histogram])
        return 1 if cnt / len(text) > self._threshold else 0

    def keep_document(self, score: float) -> bool:
        return score == 1
