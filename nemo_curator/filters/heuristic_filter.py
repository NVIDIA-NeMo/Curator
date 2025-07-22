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

from nemo_curator.filters.bitext_filter import BitextFilter
from nemo_curator.utils.text_utils import get_word_splitter


class LengthRatioFilter(BitextFilter):
    """(Bitext filter) Length ratio filter for bitext, similar to the one implemented in Moses toolkit (`https://github.com/moses-smt/mosesdecoder/blob/master/scripts/training/clean-corpus-n.perl`).

    If the ratio between source and target tokens is not within a specified range then discard. Either direction (src/tgt, tgt/src) is considered.
    """

    def __init__(self, max_ratio: float = 3.0, src_lang: str = "en", tgt_lang: str = "en", **kwargs):
        """Args:
        max_ratio (float, optional): Maximum allowed length ratio between either direction of the bitext. Defaults to 3.0.
        src_lang (str, optional): Language of the source data (needed for tokenization). Defaults to "en".
        tgt_lang (str, optional): Language of the target data (needed for tokenization). Defaults to "en".
        """

        super().__init__(**kwargs)
        self._max_ratio = float(max_ratio)
        self._src_word_splitter = get_word_splitter(src_lang)
        self._tgt_word_splitter = get_word_splitter(tgt_lang)
        self._name = "length_ratio"

    def score_bitext(self, src: str, tgt: str) -> float:
        """Tokenize the source and target sentences and compute length ratio.

        Args:
            src (str): Source document string.
            tgt (str): Target document string.

        Returns:
            float: The maximum ratio among the two translation directions of the bitext.
        """
        src_len = len(self._src_word_splitter(src.strip()))
        tgt_len = len(self._tgt_word_splitter(tgt.strip()))
        return max(src_len / tgt_len, tgt_len / src_len)

    def keep_bitext(self, score: float) -> bool:
        """Decides whether a single document should be retained according to the computed length ratio."""
        return score < self._max_ratio
