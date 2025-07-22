from dataclasses import dataclass

import pandas as pd

from ray_curator.stages.filters.doc_filter import DocumentFilter
from ray_curator.stages.modules.score_filter import ScoreFilter


class ReasoningLengthDifficultyFilter(DocumentFilter):
    def __init__(self, min_length: int):
        self._min_length = min_length
        self._name = "reasoning_length_difficulty_filter"

    def score_document(self, text: str) -> float:
        word_count = len(text.split())
        return 1.0 if word_count > self._min_length else 0.0

    def keep_document(self, score: float) -> bool:
        return score == 1.0


class LLMBasedDifficultyFilterFunction(DocumentFilter):
    def __init__(self, llm_correctness_fields: list[str]):
        self._name = "llm_based_difficulty_filter"
        self.llm_correctness_fields = llm_correctness_fields

    def score_document(self, sample: dict) -> float:
        return 1.0 - (1.0 if all(sample[item] == "Yes" for item in self.llm_correctness_fields) else 0.0)

    def keep_document(self, score: float) -> bool:
        return score == 1.0


@dataclass
class LLMBasedDifficultyFilter(ScoreFilter):
    """
    ...

    Args:
        llm_correctness_fields (list[str]): The fields that will be used to determine the difficulty of the document.

    """

    llm_correctness_fields: list[str] = None

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
