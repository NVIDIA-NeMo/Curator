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

import csv
import warnings

import sentencepiece
from bs4 import BeautifulSoup
from comment_parser import comment_parser

from nemo_curator.stages.text.filters.doc_filter import DocumentFilter
from nemo_curator.stages.text.utils.constants import regex_alpha, regex_alphanum
from nemo_curator.stages.text.utils.text_utils import get_comments_and_docstring


class PythonCommentToCodeFilter(DocumentFilter):
    def __init__(
        self,
        min_comment_to_code_ratio: float = 0.01,
        max_comment_to_code_ratio: float = 0.85,
    ):
        self._min_threshold = min_comment_to_code_ratio
        self._max_threshold = max_comment_to_code_ratio

        self._name = "python_comment_ratio"

    def score_document(self, source: str) -> float:
        docstrings, comments = get_comments_and_docstring(source)
        if docstrings is None or comments is None:
            return 0
        # No need to be super precise about the way this formatted,
        # we just need to count the number of characters.
        return (len(comments) + len(docstrings)) / len(source)

    def keep_document(self, score: float) -> bool:
        return self._min_threshold <= score <= self._max_threshold


class GeneralCommentToCodeFilter(DocumentFilter):
    def __init__(
        self,
        language: str,
        min_comment_to_code_ratio: float = 0.01,
        max_comment_to_code_ratio: float = 0.85,
    ):
        """
        Does not include the comment characters (// or /**/) towards the length of the comment.
        Args:
          language: Mime string of language
        """
        self._lang = language
        self._min_threshold = min_comment_to_code_ratio
        self._max_threshold = max_comment_to_code_ratio

        self._name = "comment_ratio"

    def score_document(self, source: str) -> float:
        try:
            comments = comment_parser.extract_comments_from_str(
                source,
                mime=self._lang,
            )
            comments = " ".join([x.text() for x in comments])
        except Exception:  # noqa: BLE001
            warnings.warn("tokenization error, no comment is extracted", stacklevel=2)
            return 9999
        if comments is None:
            return 0
        return len(comments) / len(source)

    def keep_document(self, score: float) -> bool:
        return self._min_threshold <= score <= self._max_threshold


class NumberOfLinesOfCodeFilter(DocumentFilter):
    def __init__(self, min_lines: int = 10, max_lines: int = 20000):
        self._min_lines = min_lines
        self._max_lines = max_lines

        self._name = "num_lines"

    def score_document(self, source: str) -> int:
        return source.count("\n") + 1

    def keep_document(self, score: int) -> bool:
        return self._min_lines <= score <= self._max_lines


class TokenizerFertilityFilter(DocumentFilter):
    def __init__(self, path_to_tokenizer: str | None = None, min_char_to_token_ratio: float = 2.5):
        if path_to_tokenizer is None:
            msg = "Must provide a valid path to a SentencePiece tokenizer"
            raise ValueError(msg)
        self._tokenizer = sentencepiece.SentencePieceProcessor()
        self._tokenizer.Load(path_to_tokenizer)
        self._threshold = min_char_to_token_ratio

        self._name = "tokenizer_fertility"

    def score_document(self, source: str) -> float:
        tokens = self._tokenizer.encode_as_pieces(source)
        num_chars = len(source)
        num_tokens = len(tokens)
        if num_tokens == 0:
            return -1
        return num_chars / num_tokens

    def keep_document(self, score: float) -> bool:
        return score >= self._threshold


class XMLHeaderFilter(DocumentFilter):
    """
    This filter tries to identify files that have incorrect file extensions.
    In many cases, these end up being XML files and we try to identify them
    based on the header.
    (Source: Starcoder https://arxiv.org/abs/2305.06161)
    """

    def __init__(self, char_prefix_search_length: int = 100):
        self._char_prefix_search_length = char_prefix_search_length

        self._name = "xml_header"

    def score_document(self, source: str) -> float:
        source_prefix = source[: self._char_prefix_search_length]
        if "<?xml version=" in source_prefix:
            return 1
        else:
            return 0

    def keep_document(self, score: float) -> bool:
        return score != 1


class AlphaFilter(DocumentFilter):
    """
    This filter tries to identify files that have large tensors, or tables stored
    as raw text within code files.
    (Source: Starcoder https://arxiv.org/abs/2305.06161)
    """

    def __init__(self, min_alpha_ratio: float = 0.25):
        self._min_alpha_ratio = min_alpha_ratio

        self._name = "alpha_filter"

    def score_document(self, source: str) -> float:
        return len(regex_alpha.findall(source)) / len(source)

    def keep_document(self, score: float) -> bool:
        return score >= self._min_alpha_ratio


class HTMLBoilerplateFilter(DocumentFilter):
    """
    This filter tries to identify HTML that is largely boilerplate.
    """

    def __init__(self, min_lang_content_ratio: float = 0.2, min_lang_content_num_chars: int = 100):
        self._min_lang_content_ratio = min_lang_content_ratio
        self._min_lang_content_num_chars = min_lang_content_num_chars

        self._name = "html_boilerplate"

    def score_document(self, source: str) -> float | None:
        try:
            soup = BeautifulSoup(source, features="html.parser")
        except (TypeError, UnboundLocalError):
            return None

        # kill all script and style elements
        for script in soup(["script", "style"]):
            script.extract()  # rip it out

        # get text
        text = soup.get_text()
        ratio = len(text) / len(source)

        if len(text) < self._min_lang_content_num_chars:
            return 0

        return ratio

    def keep_document(self, score: float) -> bool:
        return score >= self._min_lang_content_ratio


class PerExtensionFilter(DocumentFilter):
    """
    This filter that has specific conditions depending on the file extension.
    """

    def __init__(self, lang: str, extension: str, metadata_file: str = "code_meta.csv"):
        self._metadata_file = metadata_file
        self._lang = lang
        self._extension = extension
        self._ext_to_filter = self._load_filter_csv(metadata_file, lang)
        self._name = "per_extension_filter"

    def _load_filter_csv(self, path: str, language: str | None = None) -> dict:
        """Load csv file that specifies the filter to apply for each (lang, extension)."""
        # (Lang, extension) -> filter_args
        ext_to_filter = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                # Only take the rows corresponding to the language if specified
                if language is None or row["language"] == language:
                    ext_to_filter[(row["language"], row["extension"])] = self._get_filter_params(row)
        if len(ext_to_filter) <= 0:
            msg = f"Did not find filtering params corresponding to language: `{language}` in: {path}"
            raise ValueError(msg)

        return ext_to_filter

    def _get_filter_params(self, row: dict) -> tuple[bool, int | None, float | None, float | None, float | None]:
        """Extract filter parameters from csv row"""
        include = row["Include"] == "1"
        try:
            line_max = int(row["Long_line_threshold"])
        except ValueError:
            line_max = None
        line_mean = 100 if line_max else None
        try:
            alphanum_frac = float(row["Alphanum_threshold"])
        except ValueError:
            alphanum_frac = None
        try:
            alphabetic_frac = float(row["Alpha filter"])
        except ValueError:
            alphabetic_frac = None
        return include, line_max, line_mean, alphanum_frac, alphabetic_frac

    def _language_format_from_dataset(self, lang: str) -> str:
        """Convert: Language field in dataset -> language field in csv file that defines the filters."""
        if lang == "C#":
            return "c-sharp"
        if lang == "F#":
            return "f-sharp"
        return lang.lower().replace(" ", "-")

    def _line_statistics(self, source: str) -> tuple[int, float]:
        lengths = [len(x) for x in source.split("\n")]
        max_length = max(lengths)
        mean_length = (len(source) + 1) / len(lengths) - 1

        return max_length, mean_length

    def _alphanum_fraction(self, source: str) -> float:
        return len(regex_alphanum.findall(source)) / len(source)

    def score_document(self, source: str) -> float:
        """Filter files based on line length and % alphanumeric characters.
        The filtering parameters depend on the file extension, given by `ext_to_filter`
        """
        # Get the filter-params we want to use
        # extension `None` is an empty string in the csv
        try:
            (include, line_max, line_mean, alphanum_frac, alphabetic_frac) = self._ext_to_filter[
                (
                    self._language_format_from_dataset(self._lang),
                    self._extension if self._extension is not None else "",
                )
            ]
        except KeyError as e:
            # Some extensions are not in the csv. This happens for dockerfiles.
            # Exclude these files
            print(str(e) + f":{self._extension} not in ext_to_filter")
            include = False

        if not include:
            return 0

        max_length, mean_length = self._line_statistics(source)

        if (
            (line_max and max_length > line_max)
            or (line_mean and mean_length > line_mean)
            or (alphanum_frac and self._alphanum_fraction(source) < alphanum_frac)
            or (alphabetic_frac and sum(map(str.isalpha, source)) < alphabetic_frac * len(source))
        ):
            return 0

        return 1

    def keep_document(self, score: float | None) -> bool:
        return not (score is None or score == 0)
