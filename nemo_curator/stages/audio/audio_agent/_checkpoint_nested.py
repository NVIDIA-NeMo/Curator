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

"""Nested collection selector matching."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nemo_curator.stages.audio.audio_agent import _checkpoint_types

if TYPE_CHECKING:
    from nemo_curator.stages.audio.audio_agent.recipe import (
        Recipe,
    )


def _nested_pair_for_producer(  # noqa: C901, PLR0911, PLR0912, PLR0913, PLR0915
    recipe: Recipe,
    built: list[Any],
    producer_index: int,
    producer_stage: str,
    producer: Any,  # noqa: ANN401 - heterogeneous configured AgentReady stage
    decision: dict[str, Any],
) -> tuple[_checkpoint_types.DecisionPair | None, str]:
    """Resolve an exact one-level nested-list selector for segment annotation."""
    from nemo_curator.stages.audio.audio_agent import _checkpoint_compound, _checkpoint_decisions

    kind = str(decision.get("kind") or "scalar")
    selector = decision.get("selector") or {}
    if not isinstance(selector, dict):
        return None, "decision.selector is not a mapping"
    selector_stage = str(selector.get("stage_id") or "")
    conditions_param = str(selector.get("conditions_param") or "conditions")
    required_operator = str(selector.get("required_operator") or "")
    value_type = str(decision.get("value_type") or "")

    expected_conditions: list[dict[str, Any]] = []
    scalar_score_key: str | None = None
    if kind == "scalar":
        score_key_param = str(decision.get("score_key_param") or "")
        scalar_score_key = getattr(producer, score_key_param, None) if score_key_param else None
        if not isinstance(scalar_score_key, str) or not scalar_score_key:
            return None, f"configured producer has no score key through {score_key_param!r}"
        threshold_param = str(decision.get("threshold_param") or "")
        threshold = getattr(producer, threshold_param, None) if threshold_param else None
        value_reason = _checkpoint_decisions._decision_value_reason(
            threshold,
            operator=required_operator,
            value_type=value_type,
        )
        if value_reason:
            return None, f"{threshold_param}: {value_reason}"
        expected_conditions.append(
            {
                "input_value_key": scalar_score_key,
                "target_value": threshold,
                "operator": required_operator,
            }
        )
    elif kind == "compound":
        dimensions = decision.get("dimensions")
        if not isinstance(dimensions, list) or not dimensions:
            return None, "compound decision has no declared dimensions"
        for dimension in dimensions:
            if not isinstance(dimension, dict):
                return None, "compound decision dimension is not a mapping"
            threshold_param = str(dimension.get("threshold_param") or "")
            threshold = getattr(producer, threshold_param, None)
            if threshold is None:
                continue
            score_key_param = str(dimension.get("score_key_param") or "")
            score_key = getattr(producer, score_key_param, None)
            if not isinstance(score_key, str) or not score_key:
                return None, f"configured producer has no score key through {score_key_param!r}"
            value_reason = _checkpoint_decisions._decision_value_reason(
                threshold,
                operator=required_operator,
                value_type=value_type,
            )
            if value_reason:
                return None, f"{threshold_param}: {value_reason}"
            expected_conditions.append(
                {
                    "input_value_key": score_key,
                    "target_value": threshold,
                    "operator": required_operator,
                }
            )
        if not expected_conditions:
            return (
                None,
                "compound segment separation requires at least one enabled threshold so "
                "unscorable-child dropping remains mechanically expressible",
            )
    else:
        return None, f"decision kind {kind!r} is not mechanically supported"

    expected_map, expected_reason = _checkpoint_compound._condition_map(expected_conditions)
    if expected_reason:
        return None, expected_reason
    score_keys = tuple(condition["input_value_key"] for condition in expected_conditions)
    source_param = str(selector.get("items_key_source_param") or "")
    items_key = getattr(producer, source_param, None) if source_param else None
    if not isinstance(items_key, str) or not items_key:
        return None, f"configured producer has no nested list key through {source_param!r}"

    for index in range(producer_index + 1, len(recipe.stages)):
        ref = recipe.stages[index]
        if ref.ref != selector_stage:
            continue
        configured = built[index]
        policy_reason = _checkpoint_decisions._selector_policy_reason(
            ref,
            configured,
            selector,
            decision,
            producer=producer,
            operator=None,
        )
        if policy_reason:
            return None, policy_reason
        configured_conditions = getattr(configured, "normalized_conditions", None)
        if not isinstance(configured_conditions, tuple):
            return None, f"{selector_stage}.{conditions_param} has no canonical condition surface"
        configured_map, configured_reason = _checkpoint_compound._condition_map(list(configured_conditions))
        if configured_reason:
            return None, configured_reason
        if configured_map != expected_map:
            return (
                None,
                "nested selector conditions must exactly match every enabled producer "
                "threshold, configured score key, and required operator",
            )
        lineage_reason = _checkpoint_decisions._score_lineage_reason(
            recipe,
            built,
            producer_index=producer_index,
            selector_index=index,
            score_keys=score_keys,
        )
        if lineage_reason:
            return None, lineage_reason
        return (
            _checkpoint_types.DecisionPair(
                producer_index=producer_index,
                selector_index=index,
                producer_stage=producer_stage,
                selector_stage=selector_stage,
                decision_kind=kind,
                score_key=scalar_score_key,
                score_keys=score_keys,
                operator=required_operator if kind == "scalar" else "and",
                target_value=(
                    expected_conditions[0]["target_value"]
                    if kind == "scalar"
                    else {condition["input_value_key"]: condition["target_value"] for condition in expected_conditions}
                ),
                conditions=tuple(expected_conditions),
                value_type=value_type,
                decision=dict(decision),
                scope="segments",
                items_key=items_key,
            ),
            "",
        )
    return None, f"no exact nested {selector_stage} below the producer"
