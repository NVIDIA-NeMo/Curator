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

"""Embedding result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EmbeddingBundle:
    pair_ids: list[str]
    modalities: list[str]
    records: list[dict[str, Any]]
    experts: list[str]
    embeddings: dict[str, list[list[float]]] = field(default_factory=dict)

    def raw_embeddings(self, expert: str) -> list[list[float]]:
        return self.embeddings[expert][0::2]

    def annotation_embeddings(self, expert: str) -> list[list[float]]:
        return self.embeddings[expert][1::2]

    @property
    def embedding_dim(self) -> int:
        if not self.experts:
            return 0
        first = self.embeddings[self.experts[0]]
        return len(first[0]) if first else 0
