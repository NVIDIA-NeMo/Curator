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

"""Typed checkpoint decisions and shared policy constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_CHECKPOINT_REF = "ManifestCheckpointStage"


_CHECKPOINT_PATH_PARAM = "output_path"


_ANALYSIS_PATH = "/__audio_agent_checkpoint_probe__.jsonl"


_OPS = frozenset({"lt", "le", "eq", "ne", "ge", "gt"})


_DECLINED = "declined"


# Statuses meaning "the user has not yet said yes or no to this checkpoint". A derived path
# removes the question of WHERE, never the question of WHETHER, so a candidate the core
# located on its own still holds the gate open until it is accepted or declined.
_UNDECIDED = frozenset({"needs_output_path", "needs_decision"})


@dataclass(frozen=True)
class DecisionPair:
    """One card-declared annotation producer and its configured downstream gate."""

    producer_index: int
    selector_index: int
    producer_stage: str
    selector_stage: str
    decision_kind: str
    score_key: str | None
    score_keys: tuple[str, ...]
    operator: str
    target_value: Any
    conditions: tuple[dict[str, Any], ...]
    value_type: str
    decision: dict[str, Any]
    scope: str = "task"
    items_key: str | None = None
