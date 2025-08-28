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

from typing import ClassVar

import justext
import lxml
from loguru import logger

from .base import HTMLExtractorAlgorithm


class JusTextExtractor(HTMLExtractorAlgorithm):
    # Class-level set to track which languages we've already logged warnings for
    _logged_languages: ClassVar[set[str]] = set()

    def __init__(  # noqa: PLR0913
        self,
        length_low: int = 70,
        length_high: int = 200,
        stopwords_low: float = 0.30,
        stopwords_high: float = 0.32,
        max_link_density: float = 0.2,
        max_heading_distance: int = 200,
        no_headings: bool = False,
        is_boilerplate: bool | None = None,
    ):
        """
        Initialize the jusText text extraction algorithm with specified parameters.

        jusText is a tool for removing boilerplate content, such as navigation links, headers, and footers from HTML pages.
        It is designed to preserve mainly text containing full sentences and it is therefore well suited for creating linguistic resources such as Web corpora.
        The key idea is that long blocks can often be classified with high confidence, while shorter blocks require context-based adjustments.

        Here is an overview of the jusText algorithm:
            • Segmentation: The document is split into textual blocks based on HTML tags that typically define separate sections (e.g., <div>, <p>, <table>).
            • Preprocessing: Contents of <header>, <style>, and <script> tags are removed.
                Certain elements (e.g., <select>, copyright symbols) are immediately classified as boilerplate.
            • Context-Free Classification: Each block is classified as:
                - Bad (boilerplate) if it has high link density.
                - Short if it is too small to be classified reliably.
                - Near-Good if it has a moderate density of stopwords.
                - Good (main content) if it is long and contains many stopwords.
            • Context-Sensitive Classification: Blocks that were classified as short or near-good are reclassified based on surrounding blocks.
                The assumption is that main content clusters together, as does boilerplate.
            • Headings Processing: Header elements (e.g., <h1>, <h2>) are treated separately to ensure useful headings are preserved.
                Short headers near good content may be reclassified as near-good or good.

        Please refer to the jusText documentation for more details: https://corpus.tools/wiki/Justext/Algorithm

        Args:
            length_low: Minimum length of text to be considered for extraction.
            length_high: Maximum length of text to be considered for extraction.
            stopwords_low: Minimum proportion of stopwords in the text to be considered for extraction.
            stopwords_high: Maximum proportion of stopwords in the text to be considered for extraction.
            max_link_density: Maximum allowed link density in the text.
            max_heading_distance: Maximum distance from a heading to consider text for extraction.
            no_headings: If True, text extraction will ignore headings.
            is_boilerplate: If True, text extraction will ignore boilerplate content.
                Default is True for space-separated languages and False for non-space-separated languages
                (Thai, Chinese, Japanese, and Korean).
            logger: Optional logger instance for logging messages.

        """
        self.length_low = length_low
        self.length_high = length_high
        self.stopwords_low = stopwords_low
        self.stopwords_high = stopwords_high
        self.max_link_density = max_link_density
        self.max_heading_distance = max_heading_distance
        self.no_headings = no_headings
        self.is_boilerplate = is_boilerplate

    def extract_text(self, html: str, stop_words: frozenset[str], language: str) -> list[str] | None:
        # Segment the HTML into paragraphs
        try:
            # Form the DOM tree
            dom = justext.core.html_to_dom(html)
            cleaned_dom = justext.core.preprocessor(dom)
            # Get the paragraphs from the DOM
            handler = justext.core.ParagraphMaker()
            lxml.sax.saxify(cleaned_dom, handler)
        except (lxml.etree.ParserError, ValueError, Exception):
            # Return nothing when we cannot segment the document
            logger.info("Could not segment paragaphs in the document")
            return None

        paragraphs = handler.paragraphs

        # Context free classification
        justext.core.classify_paragraphs(
            paragraphs,
            stop_words,
            self.length_low,
            self.length_high,
            self.stopwords_low,
            self.stopwords_high,
            self.max_link_density,
            self.no_headings,
        )

        # Copy the context free class to the class_style
        # This handles the headings as described in the
        # documentation
        for paragraph in paragraphs:
            paragraph.class_type = paragraph.cf_class

        # Context sensitive classification
        justext.core.revise_paragraph_classification(
            paragraphs,
            self.max_heading_distance,
        )

        if self.is_boilerplate is None:
            if language in self.NON_SPACED_LANGUAGES:
                if language not in self._logged_languages:
                    logger.warning(f"Disabling is_boilerplate check for jusText extraction for language: {language}")
                    self._logged_languages.add(language)
                is_boilerplate = False
            else:
                is_boilerplate = True

        else:
            is_boilerplate = self.is_boilerplate

        if is_boilerplate:
            return [p.text for p in paragraphs if not p.is_boilerplate]

        else:
            return [p.text for p in paragraphs]
