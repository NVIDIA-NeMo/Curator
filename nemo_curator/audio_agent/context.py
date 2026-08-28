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

"""Context Assembler — package Knowledge Index results + facts into PlanningContext.

Deterministic assembly only. Selection (which categories/stages) is the host
router's job; this packages whatever the host has drilled into (``selected_stages``)
together with the L0 category tree, matched blueprints/recipes/patterns, the
role-graph slice, and the profiler/env facts. It never dumps source.
"""

from __future__ import annotations

from typing import Any

from nemo_curator.audio_agent.contracts import PlanningContext
from nemo_curator.audio_agent.env_health import env_report
from nemo_curator.audio_agent.index import get_index
from nemo_curator.audio_agent.profiler import probe_env, profile_data


def assemble(  # noqa: PLR0913 - compact context surface stays keyword-addressable
    goal: dict[str, Any] | None = None,
    *,
    data: str | None = None,
    selected_stages: list[str] | None = None,
    roles: list[str] | None = None,
    include_env: bool = True,
    planning_preference: dict[str, Any] | None = None,
) -> PlanningContext:
    """Build a compact PlanningContext for the host router/planner."""
    from nemo_curator.audio_agent.recipe import parse_planning_preference

    goal = dict(goal or {})
    preference = parse_planning_preference(planning_preference)
    idx = get_index()

    selected = idx.full_cards(selected_stages) if selected_stages else []
    matched_blueprints = idx.match_blueprints(goal)
    matched_recipes = idx.match_recipes(goal)

    presets = _collect_presets(idx, selected_stages or [], matched_blueprints)
    data_profile = profile_data(data).to_dict() if data else None
    env_obj = probe_env() if include_env else None
    env_profile = env_obj.to_dict() if env_obj is not None else None
    env_health = env_report(env_obj).to_dict() if env_obj is not None else None

    return PlanningContext(
        goal=goal,
        category_tree=idx.category_tree(),
        selected_stages=selected,
        presets=presets,
        matched_blueprints=matched_blueprints,
        matched_recipes=matched_recipes,
        patterns=idx.patterns(),
        role_graph_slice=idx.role_neighborhood(roles),
        data_profile=data_profile,
        env_profile=env_profile,
        env_health=env_health,
        planning_preference=preference,
    )


def _collect_presets(idx: Any, stages: list[str], blueprints: list[dict[str, Any]]) -> dict[str, Any]:  # noqa: ANN401
    presets: dict[str, Any] = {}
    for name in stages:
        card = idx.card(name)
        if card and isinstance(card.get("presets"), dict):
            presets[name] = card["presets"]
    for bp in blueprints:
        if isinstance(bp.get("presets"), dict):
            presets.setdefault("_blueprint_" + str(bp.get("blueprint_id", "bp")), bp["presets"])
    return presets
