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

from .bitext_filter import BitextFilter
from .classifier_filter import (
    FastTextLangId,
    FastTextQualityFilter,
    QualityEstimationFilter,
)
from .code import (
    AlphaFilter,
    GeneralCommentToCodeFilter,
    HTMLBoilerplateFilter,
    NumberOfLinesOfCodeFilter,
    PerExtensionFilter,
    PythonCommentToCodeFilter,
    TokenizerFertilityFilter,
    XMLHeaderFilter,
)
from .doc_filter import DocumentFilter, import_filter
from .heuristic_filter import (
    BoilerPlateStringFilter,
    BulletsFilter,
    CommonEnglishWordsFilter,
    EllipsisFilter,
    HistogramFilter,
    LengthRatioFilter,
    LongWordFilter,
    MeanWordLengthFilter,
    NonAlphaNumericFilter,
    NumbersFilter,
    ParenthesesFilter,
    PornographicUrlsFilter,
    PunctuationFilter,
    RepeatedLinesByCharFilter,
    RepeatedLinesFilter,
    RepeatedParagraphsByCharFilter,
    RepeatedParagraphsFilter,
    RepeatingDuplicateNGramsFilter,
    RepeatingTopNGramsFilter,
    SubstringFilter,
    SymbolsToWordsFilter,
    TokenCountFilter,
    UrlsFilter,
    WhiteSpaceFilter,
    WordCountFilter,
    WordsWithoutAlphabetsFilter,
)
from .synthetic import AnswerabilityFilter, EasinessFilter

__all__ = [
    "AlphaFilter",
    "AnswerabilityFilter",
    "BitextFilter",
    "BoilerPlateStringFilter",
    "BulletsFilter",
    "CommonEnglishWordsFilter",
    "DocumentFilter",
    "EasinessFilter",
    "EllipsisFilter",
    "FastTextLangId",
    "FastTextQualityFilter",
    "GeneralCommentToCodeFilter",
    "HTMLBoilerplateFilter",
    "HistogramFilter",
    "LengthRatioFilter",
    "LongWordFilter",
    "MeanWordLengthFilter",
    "NonAlphaNumericFilter",
    "NumberOfLinesOfCodeFilter",
    "NumbersFilter",
    "ParenthesesFilter",
    "PerExtensionFilter",
    "PornographicUrlsFilter",
    "PunctuationFilter",
    "PythonCommentToCodeFilter",
    "QualityEstimationFilter",
    "RepeatedLinesByCharFilter",
    "RepeatedLinesFilter",
    "RepeatedParagraphsByCharFilter",
    "RepeatedParagraphsFilter",
    "RepeatingDuplicateNGramsFilter",
    "RepeatingTopNGramsFilter",
    "SubstringFilter",
    "SymbolsToWordsFilter",
    "TokenCountFilter",
    "TokenizerFertilityFilter",
    "UrlsFilter",
    "WhiteSpaceFilter",
    "WordCountFilter",
    "WordsWithoutAlphabetsFilter",
    "XMLHeaderFilter",
    "import_filter",
]
