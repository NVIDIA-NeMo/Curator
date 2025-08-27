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


class DocumentFilter(ABC):
    """
    An abstract base class for text-based document filters.

    This class serves as a template for creating specific document filters
    in the library. Subclasses should implement the abstract methods to
    define custom filtering behavior.
    """

    def __init__(self):
        super().__init__()
        self._name = self.__class__.__name__
        self._sentences = None
        self._paragraphs = None
        self._ngrams = None

    @abstractmethod
    def score_document(self, text: str) -> float | list[int | float]:
        """
        Calculate a score for the given document text.

        This method should be implemented by subclasses to define how
        a document's text is evaluated and scored.

        Args:
            text (str): The text content of the document to be scored.

        Returns:
            Any: A score or set of scores representing the document's
            relevance or quality. The type and structure of the
            return value should be consistent for each subclass.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        msg = "score_document method must be implemented by subclasses"
        raise NotImplementedError(msg)

    @abstractmethod
    def keep_document(self, scores: float | list[int | float]) -> bool:
        """
        Determine whether to keep a document based on its scores.

        This method should be implemented by subclasses to define the
        criteria for keeping or discarding a document based on the
        scores calculated by score_document().

        Args:
            scores (float | list[int | float]): The score or set of scores returned by score_document().
                          The type should match what is returned by score_document().

        Returns:
            bool: True if the document should be kept, False otherwise.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        msg = "keep_document method must be implemented by subclasses"
        raise NotImplementedError(msg)

    @property
    def name(self) -> str:
        return self._name

    @property
    def sentences(self) -> list:
        return self._sentences

    @sentences.setter
    def sentences(self, sentences: list) -> None:
        self._sentences = sentences

    @property
    def paragraphs(self) -> list:
        return self._paragraphs

    @paragraphs.setter
    def paragraphs(self, paragraphs: list) -> None:
        self._paragraphs = paragraphs

    @property
    def ngrams(self) -> dict:
        return self._ngrams

    @ngrams.setter
    def ngrams(self, ngrams: dict) -> None:
        self._ngrams = ngrams
