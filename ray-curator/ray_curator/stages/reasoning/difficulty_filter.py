# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the specific language for the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

import pandas as pd

from ray_curator.stages.filters.doc_filter import DocumentFilter
from ray_curator.stages.modules.score_filter import ScoreFilter


class ReasoningLengthDifficultyFilter(DocumentFilter):
    def __init__(self, min_length: int):
        self._min_length = min_length
        self._name = "reasoning_length_difficulty_filter"

    def score_document(self, text: str) -> int:
        word_count = len(text.split())
        return 1 if word_count > self._min_length else 0

    def keep_document(self, score: int) -> bool:
        return score == 1


class LLMBasedDifficultyFilterFunction(DocumentFilter):
    def __init__(self, llm_correctness_fields: list[str]):
        self._name = "llm_based_difficulty_filter"
        self.llm_correctness_fields = llm_correctness_fields

    def score_document(self, sample: dict) -> int:
        return 1 - (1 if all(sample[item] == "Yes" for item in self.llm_correctness_fields) else 0)

    def keep_document(self, score: int) -> bool:
        return score == 1


@dataclass
class LLMBasedDifficultyFilter(ScoreFilter):
    """
    ...

    Args:
        llm_correctness_fields (list[str]): The fields that will be used to determine the difficulty of the document.

    """

    llm_correctness_fields: list[str] = None
    _name: str = "LLMBasedDifficultyFilter"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], self.llm_correctness_fields

    def compute_filter_mask(self, df: pd.DataFrame) -> pd.Series:
        """Compute the bool mask to filter the dataset.

        Args:
            df (pd.DataFrame): The dataset to compute filter mask on.

        Returns:
            Series: A mask corresponding to each data instance indicating whether it will be retained.

        """

        scores = df[self.llm_correctness_fields].apply(self.filter_obj.score_document, axis=1)

        if self.score_field is not None:
            df[self.score_field] = scores

        bool_mask = scores.apply(self.filter_obj.keep_document)

        if self.invert:
            bool_mask = ~bool_mask

        return bool_mask
