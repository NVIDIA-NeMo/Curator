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

from abc import ABC, abstractmethod
from typing import Any


class DocumentExtractor(ABC):
    """Abstract base class for document extractors.

    Takes a record dict and returns processed record dict or None to skip.
    Can transform any fields in the input dict.
    """

    @abstractmethod
    def extract(self, record: dict[str, str]) -> dict[str, Any] | None:
        """Extract/transform a record dict into final record dict."""
        ...

    @abstractmethod
    def input_columns(self) -> list[str]:
        """Define input columns - produces DocumentBatch with records."""
        ...

    @abstractmethod
    def output_columns(self) -> list[str]:
        """Define output columns - produces DocumentBatch with records."""
        ...
