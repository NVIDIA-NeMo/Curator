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
from .kenlm_utility import KenlmModel

class PerplexityFilter(DocumentFilter):
    """
    Filters documents based on KenLM n-gram language model perplexity.

    Documents whose perplexity falls outside [min_perplexity, max_perplexity]
    are discarded. Lower perplexity indicates text that more closely resembles
    the training corpus of the language model (typically high-quality web text).

    """

    def __init__(
        self,
        model_path: str,
        min_perplexity: float,
        max_perplexity: float,
        lang: str = "en",
        lower_case: bool = False,
        remove_accents: bool = False,
        normalize_numbers: bool = True,
        punctuation: int = 1,
    ) -> None:
        """
        Args:
            model_path (str): Path to a directory containing the KenLM binary model
                ({lang}.arpa.bin) and SentencePiece model ({lang}.sp.model).
            min_perplexity (float): Minimum perplexity threshold (inclusive). Documents
                with perplexity below this value are discarded.
            max_perplexity (float): Maximum perplexity threshold (inclusive). Documents
                with perplexity above this value are discarded.
            lang (str): Language code used to locate the model files. Defaults to "en".
            lower_case (bool): Whether to lowercase text before scoring. Defaults to False.
            remove_accents (bool): Whether to strip accent characters before scoring. Defaults to False.
            normalize_numbers (bool): Whether to replace digits with 0 before scoring. Defaults to True.
            punctuation (int): Punctuation handling mode — 1 replaces Unicode punctuation
                with ASCII equivalents, 2 removes it entirely, 0 disables. Defaults to 1.
        """
        if min_perplexity < 0 or max_perplexity < 0:
            msg = "Perplexity thresholds must be non-negative"
            raise ValueError(msg)
        if min_perplexity > max_perplexity:
            msg = "min_perplexity must be less than or equal to max_perplexity"
            raise ValueError(msg)
        if punctuation not in (0, 1, 2):
            msg = "punctuation must be one of 0 (disabled), 1 (replace), or 2 (remove)"
            raise ValueError(msg)

        super().__init__()
        self.model_path=model_path
        self.language=lang
        self.lower_case=lower_case
        self.remove_accents=remove_accents
        self.normalize_numbers=normalize_numbers
        self.punctuation=punctuation
        self.min_perplexity = min_perplexity
        self.max_perplexity = max_perplexity

    def load_model(self) -> None:
        self._kenlm_model = KenlmModel(
            model_path=self.model_path,
            language=self.language,
            lower_case=self.lower_case,
            remove_accents=self.remove_accents,
            normalize_numbers=self.normalize_numbers,
            punctuation=self.punctuation
        )
    def score_document(self, text: str) -> float:
        """
        Compute the KenLM perplexity score for the given text.

        Args:
            text (str): The document text to score.

        Returns:
            float: The perplexity score. Lower values indicate text more similar
                to the language model's training corpus.
        """
        return self._kenlm_model.get_perplexity(text, normalize=True)

    def keep_document(self, score: float) -> bool:
        """
        Determine whether to keep a document based on its perplexity score.

        Args:
            score (float): The perplexity score returned by score_document().

        Returns:
            bool: True if the score is within [min_perplexity, max_perplexity], False otherwise.
        """
        return self.min_perplexity <= score <= self.max_perplexity