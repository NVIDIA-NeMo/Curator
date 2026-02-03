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
from collections.abc import Iterator
from typing import Any


class DocumentIterator(ABC):
    """Abstract base class for document iterators.

    Always yields dict[str, str] records. For raw content that needs extraction,
    the iterator can put it in any field (e.g., "raw_content", "html", "content", etc.)
    """

    @abstractmethod
    def iterate(self, file_path: str) -> Iterator[dict[str, Any]]:
        """Iterate over records in a file, yielding dict records."""
        ...

    @abstractmethod
    def output_columns(self) -> list[str]:
        """Define output columns - produces DocumentBatch with records."""
        ...
