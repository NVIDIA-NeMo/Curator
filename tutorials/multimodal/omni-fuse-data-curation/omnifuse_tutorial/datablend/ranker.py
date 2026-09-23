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

"""Query-based datablend ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omnifuse_tutorial.config.models import DatablendConfig
from omnifuse_tutorial.data.io import cosine_similarity
from omnifuse_tutorial.eee.backends import EEEBackend
from omnifuse_tutorial.projection.trainer import ProjectionResult


@dataclass
class DatablendRanker:
    config: DatablendConfig
    backend: EEEBackend

    def rank(self, records: list[dict[str, Any]], projection: ProjectionResult) -> list[dict[str, Any]]:
        query_embedding = self.backend.embed_query(self.config.query, expert="text-based")
        rows: list[dict[str, Any]] = []
        for index, (record, projected) in enumerate(zip(records, projection.projected_raw)):
            score = cosine_similarity(projected, query_embedding)
            row = {
                "rank": 0,
                "score": score,
                "pair_id": record["pair_id"],
                "pool": record["pool"],
                "modality": record["modality"],
                "raw_path": record["raw_path"],
                "annotation": record.get("sns_annotation") or record.get("annotation"),
                "original_annotation": record.get("annotation"),
                "source_index": index,
            }
            if self.config.include_metadata:
                row["metadata"] = record.get("metadata", {})
            rows.append(row)

        rows.sort(key=lambda item: item["score"], reverse=True)
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        return rows

    def select_top(self, ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.config.top_k is not None:
            return ranked[: self.config.top_k]
        if self.config.blend_fraction is not None:
            count = max(1, int(len(ranked) * self.config.blend_fraction))
            return ranked[:count]
        return ranked
