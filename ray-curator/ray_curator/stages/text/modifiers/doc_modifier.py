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


class DocumentModifier(ABC):
    """
    Abstract base class for text-based document modifiers.

    Subclasses must implement `modify_document` to transform input value(s)
    and return the modified value. This supports both single-input and
    multi-input usage:
    - Single input: `modify_document(value)`
    - Multiple inputs: `modify_document(**values)` where each input field is
      expanded as a keyword argument (e.g., `modify_document(column_1=..., column_2=...)`).
    """

    def __init__(self) -> None:
        super().__init__()
        self._name = self.__class__.__name__
        self._sentences = None
        self._paragraphs = None
        self._ngrams = None

    @abstractmethod
    def modify_document(self, *args: object, **kwargs: object) -> object:
        """Transform the provided value(s) and return the result."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self._name
