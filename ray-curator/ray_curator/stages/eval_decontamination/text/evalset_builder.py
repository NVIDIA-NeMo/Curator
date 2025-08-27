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

from abc import ABC, abstractmethod
from collections import defaultdict
from functools import reduce

import ray
from loguru import logger

from ray_curator.stages.utils.text_utils import get_words

from .evalset_base import EvaluationSetBase


class BaseEvalSetBuilder(ABC):
    """
    Base class for building evaluation sets and computing merged ngrams.

    This class provides the interface and common functionality for constructing
    evaluation sets and computing the merged ngram data needed by decontamination stages.
    """

    def __init__(self, eval_set_specs: list[tuple[type[EvaluationSetBase], tuple, dict]]):
        """
        Initialize the builder with evaluation set specifications.

        Args:
            eval_set_specs: List of tuples, each containing:
                - eval_set_class: The EvaluationSetBase subclass to instantiate
                - args: Positional arguments for the class constructor
                - kwargs: Keyword arguments for the class constructor
        """
        self.eval_set_specs = eval_set_specs

    @classmethod
    def from_classes_and_args(
        cls,
        eval_set_classes: list[type[EvaluationSetBase]],
        args_list: list[tuple] | None = None,
        kwargs_list: list[dict] | None = None,
    ) -> "BaseEvalSetBuilder":
        """
        Convenience constructor that takes separate lists of classes, args, and kwargs.

        Args:
            eval_set_classes: List of EvaluationSetBase subclasses
            args_list: List of positional argument tuples (one per class)
            kwargs_list: List of keyword argument dicts (one per class)

        Returns:
            BaseEvalSetBuilder instance
        """
        if args_list is None:
            args_list = [() for _ in eval_set_classes]
        if kwargs_list is None:
            kwargs_list = [{} for _ in eval_set_classes]

        if len(eval_set_classes) != len(args_list) or len(eval_set_classes) != len(kwargs_list):
            msg = "Number of classes, args, and kwargs must match"
            raise ValueError(msg)

        eval_set_specs = [
            (cls, args, kwargs) for cls, args, kwargs in zip(eval_set_classes, args_list, kwargs_list, strict=True)
        ]

        return cls(eval_set_specs)

    @abstractmethod
    def _construct_eval_sets(self) -> list[EvaluationSetBase]:
        """Construct the evaluation sets. Implementation varies by subclass."""

    @staticmethod
    def _merge_eval_set_ngrams(first: dict[str, int], second: dict[str, int]) -> dict[str, int]:
        """Merge two ngram dictionaries by updating the first with the second."""
        first.update(second)
        return first

    @staticmethod
    def _compute_ngram_freq_sorted(task_ngrams: dict[str, int]) -> list[tuple[int, int]]:
        """
        Compute frequency distribution of ngrams by length.

        Args:
            task_ngrams: Dictionary of ngrams to their counts

        Returns:
            List of (length, count) tuples sorted by length
        """
        ngrams_freq = defaultdict(int)
        for ngram_key in task_ngrams:
            ngram_words, _ = get_words(ngram_key)
            length = len(ngram_words)
            ngrams_freq[length] += 1

        return sorted(ngrams_freq.items(), key=lambda item: item[0])

    def build(self) -> tuple[dict[str, int], list[tuple[int, int]]]:
        """
        Build evaluation sets and compute merged ngram data.

        Returns:
            EvalSetBuildResult containing eval_sets, merged ngrams, and frequency data
        """
        logger.info(f"Building {len(self.eval_set_specs)} evaluation sets")

        # Construct evaluation sets
        eval_sets = self._construct_eval_sets()

        # Merge ngrams from all evaluation sets
        logger.info("Merging ngrams from all evaluation sets")
        eval_set_ngrams = reduce(self._merge_eval_set_ngrams, [eval_set.ngrams for eval_set in eval_sets])

        # Compute frequency distribution
        logger.info("Computing ngram frequency distribution")
        eval_set_ngrams_frequency_sorted = self._compute_ngram_freq_sorted(eval_set_ngrams)

        logger.info(f"Successfully built evaluation sets with {len(eval_set_ngrams)} total ngrams")

        return (eval_set_ngrams, eval_set_ngrams_frequency_sorted)


class SequentialEvalSetBuilder(BaseEvalSetBuilder):
    """
    Sequential implementation of evaluation set builder.

    Constructs evaluation sets one by one in sequence. This is the fallback
    implementation that doesn't require Ray.
    """

    def _construct_eval_sets(self) -> list[EvaluationSetBase]:
        """Construct evaluation sets sequentially."""
        eval_sets = []
        for eval_set_class, args, kwargs in self.eval_set_specs:
            logger.info(f"Constructing {eval_set_class.__name__}")
            eval_set = eval_set_class(*args, **kwargs)
            eval_sets.append(eval_set)
        return eval_sets


@ray.remote
def _construct_eval_set(eval_set_class: type[EvaluationSetBase], *args, **kwargs) -> EvaluationSetBase:
    """Ray remote function to construct an evaluation set."""
    return eval_set_class(*args, **kwargs)


class ParallelEvalSetBuilder(BaseEvalSetBuilder):
    """
    Parallel Ray-based implementation of evaluation set builder.

    Constructs evaluation sets in parallel using Ray for improved performance
    when dealing with multiple large evaluation sets.
    """

    def _construct_eval_sets(self) -> list[EvaluationSetBase]:
        """Construct evaluation sets in parallel using Ray."""
        logger.info("Submitting evaluation set construction tasks to Ray")

        # Submit all construction tasks to Ray
        ray_futures = []
        for eval_set_class, args, kwargs in self.eval_set_specs:
            logger.info(f"Submitting construction task for {eval_set_class.__name__}")
            future = _construct_eval_set.remote(eval_set_class, *args, **kwargs)
            ray_futures.append(future)

        # Wait for all tasks to complete and collect results
        logger.info("Waiting for all evaluation set construction tasks to complete")
        return ray.get(ray_futures)
