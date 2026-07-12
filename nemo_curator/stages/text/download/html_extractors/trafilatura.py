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

from copy import deepcopy

from loguru import logger
from trafilatura import extract as extract_with_trafilatura
from trafilatura.settings import DEFAULT_CONFIG as TRAFILATURA_DEFAULT_CONFIG

from .base import HTMLExtractorAlgorithm


class TrafilaturaExtractor(HTMLExtractorAlgorithm):
    def __init__(  # noqa: PLR0913
        self,
        required_stopword_density: float = 0.32,
        min_extracted_size: int = 250,
        min_extracted_comm_size: int = 1,
        min_output_size: int = 1,
        min_output_comm_size: int = 1,
        max_tree_size: int | None = None,
        min_duplcheck_size: int = 100,
        max_repetitions: int = 2,
        **extract_kwargs,
    ):
        """
        Initialize the Trafilatura text extraction algorithm with specified parameters.

        The Trafilatura extraction process combines readability-lxml and jusText as fallbacks to ensure robustness.
        Trafilatura's own algorithm follows a cascade of rule-based filters and content heuristics:
            • Content Delimitation: Uses XPath expressions to exclude unwanted HTML elements (e.g., navigation bars) and focus on relevant content (e.g., article body).
                Extracted HTML nodes are analyzed for relevance based on element type, text length, and link density.
            • Fallback Mechanism: If extraction seems faulty, alternative algorithms are run as backups.
                These use heuristics like line length, text-to-markup ratio, and HTML depth to improve extraction.
                Outputs are compared, prioritizing longer extractions with fewer impurities.
            • Baseline Extraction: If all else fails, it searches for text elements that might have been missed, discarding irrelevant content.

        The system balances precision and recall, extracting main text, comments, and metadata (title, site name, author, date, categories, tags).

        Please refer to the Trafilatura documentation for more details:
            https://trafilatura.readthedocs.io/en/latest/ and https://aclanthology.org/2021.acl-demo.15/

        NeMo Curator has added a stopword density filter to the Trafilatura extraction process, which requires that a paragraph contains a certain proportion of stopwords.

        Args:
            required_stopword_density: Proportion of stopwords required preserve an extracted paragraph.
                Studies on stopword lists and their distribution in various text corpora often
                suggest that around 30-40% of a typical English text consists of stopwords.
            min_extracted_size: Acceptable size in characters (used to trigger fallbacks).
                Defaults to 250. See Trafilatura documentation: https://trafilatura.readthedocs.io/en/latest/settings.html.
            min_extracted_comm_size: Works the same as min_output_comm_size for comment extraction.
                Defaults to 1. See Trafilatura documentation: https://trafilatura.readthedocs.io/en/latest/settings.html.
            min_output_size: Absolute acceptable minimum for main text output.
                Defaults to 1. See Trafilatura documentation: https://trafilatura.readthedocs.io/en/latest/settings.html.
            min_output_comm_size: Works the same as min_output_comm_size for comment extraction.
                Defaults to 1. See Trafilatura documentation: https://trafilatura.readthedocs.io/en/latest/settings.html.
            max_tree_size: Used to discard documents with too many elements. Defaults to None.
            min_duplcheck_size: Minimum size in characters to run deduplication on.
                Defaults to 100. See Trafilatura documentation: https://trafilatura.readthedocs.io/en/latest/settings.html.
            max_repetitions: Maximum number of duplicates allowed.
                Defaults to 2. See Trafilatura documentation: https://trafilatura.readthedocs.io/en/latest/settings.html.
            extract_kwargs: Additional keyword arguments for the Trafilatura extract function.
                See API documentation https://trafilatura.readthedocs.io/en/latest/corefunctions.html#extract
                for list of possible parameters.
                All arguments are set to their default values, except for deduplicate (bool) which is set to True.

        """
        self.required_stopword_density = required_stopword_density
        self.min_extracted_size = min_extracted_size
        self.min_extracted_comm_size = min_extracted_comm_size
        self.min_output_size = min_output_size
        self.min_output_comm_size = min_output_comm_size
        self.max_tree_size = max_tree_size
        self.min_duplcheck_size = min_duplcheck_size
        self.max_repetitions = max_repetitions
        self.extract_kwargs = extract_kwargs

    def extract_text(self, html: str, stop_words: frozenset[str], language: str) -> list[str] | None:
        trafilatura_config = deepcopy(TRAFILATURA_DEFAULT_CONFIG)
        trafilatura_config["DEFAULT"]["MIN_EXTRACTED_SIZE"] = str(self.min_extracted_size)
        trafilatura_config["DEFAULT"]["MIN_EXTRACTED_COMM_SIZE"] = str(self.min_extracted_comm_size)
        trafilatura_config["DEFAULT"]["MIN_OUTPUT_SIZE"] = str(self.min_output_size)
        trafilatura_config["DEFAULT"]["MIN_OUTPUT_COMM_SIZE"] = str(self.min_output_comm_size)
        if self.max_tree_size:
            trafilatura_config["DEFAULT"]["MAX_TREE_SIZE"] = str(self.max_tree_size)
        trafilatura_config["DEFAULT"]["MIN_DUPLCHECK_SIZE"] = str(self.min_duplcheck_size)
        trafilatura_config["DEFAULT"]["MAX_REPETITIONS"] = str(self.max_repetitions)

        # Recommended to set deduplicate=True
        self.extract_kwargs.setdefault("deduplicate", True)

        text = extract_with_trafilatura(html, config=trafilatura_config, **self.extract_kwargs)

        if text is not None:
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

        else:
            return None

        if len(result) == 0:
            return None
        return result
