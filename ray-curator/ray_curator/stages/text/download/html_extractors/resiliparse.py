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

from loguru import logger
from resiliparse.extract.html2text import extract_plain_text as resiliparse_extract_plain_text

from .base import HTMLExtractorAlgorithm


class ResiliparseExtractor(HTMLExtractorAlgorithm):
    def __init__(
        self,
        required_stopword_density: float = 0.32,
        main_content: bool = True,
        alt_texts: bool = False,
    ):
        """
        Initialize the Resiliparse text extraction algorithm with specified parameters.

        The Resiliparse algorithm extracts structural or semantic information from noisy raw web data for further processing,
        such as (main) content extraction / boilerplate removal, schema extraction, general web data cleansing, and more.

        It is implemented via the `extract_plain_text` function in the `resiliparse.extract.html2text` module.
        Resiliparse HTML2Text is a very fast and rule-based plain text extractor for HTML pages which uses the Resiliparse DOM parser.
        The `extract_plain_text` function extracts all visible text nodes inside the HTML document's <body>.
        Only <script>, <style> and a few other (generally) invisible elements are skipped and very basic ASCII formatting is applied.

        Please refer to the Resiliparse documentation for more details: https://resiliparse.chatnoir.eu/en/latest/man/extract/html2text.html

        NeMo Curator has added a stopword density filter to the Resiliparse extraction process, which requires that a paragraph contains a certain proportion of stopwords.

        Args:
            required_stopword_density: Proportion of stopwords required preserve an extracted paragraph.
                Studies on stopword lists and their distribution in various text corpora often
                suggest that around 30-40% of a typical English text consists of stopwords.
            main_content: Whether to apply simple heuristics for extracting only "main-content" elements.
            alt_texts: Whether to preserve alternative text descriptions (e.g., for images).

        """
        self.required_stopword_density = required_stopword_density
        self.main_content = main_content
        self.alt_texts = alt_texts

    def extract_text(self, html: str, stop_words: frozenset[str], language: str) -> list[str] | None:
        text = resiliparse_extract_plain_text(html, main_content=self.main_content, alt_texts=self.alt_texts)

        paragraphs = list(filter(None, text.split("\n")))

        if language in self.NON_SPACED_LANGUAGES:
            logger.warning("stopword_density is ignored for non-space-separated languages.")
            result = paragraphs
        else:
            result = []

            for paragraph in paragraphs:
                words = paragraph.split()
                length = len(words)

                if length == 0:
                    continue

                stopwords = [word for word in words if word in stop_words]
                stopword_density = len(stopwords) / length

                if stopword_density >= self.required_stopword_density:
                    result.append(paragraph)

        return result
