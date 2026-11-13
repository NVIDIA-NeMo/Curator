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

"""Compound selector conditions and checkpoint evidence."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from typing import Any

from nemo_curator.stages.audio.audio_agent import _checkpoint_types
from nemo_curator.stages.audio.audio_agent.recipe import (
    Recipe,
    StageRef,
)


def _compound_pair_for_producer(  # noqa: C901, PLR0911, PLR0912, PLR0913
    recipe: Recipe,
    built: list[Any],
    producer_index: int,
    producer_stage: str,
    producer: Any,  # noqa: ANN401 - heterogeneous configured AgentReady stage
    decision: dict[str, Any],
) -> tuple[_checkpoint_types.DecisionPair | None, str]:
    """Resolve one exact all-enabled-dimensions AND selector."""
    from nemo_curator.stages.audio.audio_agent import _checkpoint_decisions

    selector = decision.get("selector") or {}
    if not isinstance(selector, dict):
        return None, "decision.selector is not a mapping"
    selector_stage = str(selector.get("stage_id") or "")
    conditions_param = str(selector.get("conditions_param") or "conditions")
    required_operator = str(selector.get("required_operator") or "")
    value_type = str(decision.get("value_type") or "")
    dimensions = decision.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        return None, "compound decision has no declared dimensions"

    expected_conditions: list[dict[str, Any]] = []
    for dimension in dimensions:
        if not isinstance(dimension, dict):
            return None, "compound decision dimension is not a mapping"
        threshold_param = str(dimension.get("threshold_param") or "")
        score_key_param = str(dimension.get("score_key_param") or "")
        threshold = getattr(producer, threshold_param, None)
        if threshold is None:
            continue
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
            "compound separation requires at least one enabled threshold so unscorable-row "
            "dropping remains mechanically expressible",
        )

    expected_map, expected_reason = _condition_map(expected_conditions)
    if expected_reason:
        return None, expected_reason
    score_keys = tuple(condition["input_value_key"] for condition in expected_conditions)
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
        configured_map, configured_reason = _condition_map(list(configured_conditions))
        if configured_reason:
            return None, configured_reason
        if configured_map != expected_map:
            return (
                None,
                "compound selector conditions must exactly match every enabled producer "
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
                decision_kind="compound",
                score_key=None,
                score_keys=score_keys,
                operator="and",
                target_value={
                    condition["input_value_key"]: condition["target_value"] for condition in expected_conditions
                },
                conditions=tuple(expected_conditions),
                value_type=value_type,
                decision=dict(decision),
                scope="task",
            ),
            "",
        )
    return None, f"no exact {selector_stage} below the producer"


def _condition_map(
    conditions: list[dict[str, Any]],
) -> tuple[dict[str, tuple[str, Any]], str]:
    """Canonicalize one-condition-per-key AND semantics."""
    result: dict[str, tuple[str, Any]] = {}
    for condition in conditions:
        key = condition.get("input_value_key")
        operator = condition.get("operator")
        if not isinstance(key, str) or not key:
            return {}, "compound selector contains an invalid input_value_key"
        if key in result:
            return {}, f"compound selector contains duplicate conditions for {key!r}"
        result[key] = (str(operator), condition.get("target_value"))
    return result, ""


def _normalize_decision_conditions(  # noqa: C901, PLR0911, PLR0912 - fail closed per condition invariant
    recipe: Recipe,
    pair: _checkpoint_types.DecisionPair,
    raw: Any,  # noqa: ANN401 - untrusted Python/CLI/MCP boundary
) -> tuple[tuple[dict[str, Any], ...], str]:
    """Validate and canonicalize a complete compound-selector replacement."""
    from nemo_curator.stages.audio.audio_agent import _checkpoint_decisions

    declared, declared_reason = _declared_compound_score_keys(recipe, pair)
    if declared_reason:
        return (), declared_reason
    selector = pair.decision.get("selector") or {}
    required_operator = str(selector.get("required_operator") or "")
    if required_operator != "ge":
        return (), ("decision_conditions requires a card-declared compound decision with exact ge operators")

    entries: list[Mapping[str, Any]] = []
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if isinstance(value, Mapping):
                unknown = set(value) - {"target_value", "operator"}
                if unknown:
                    return (), (f"decision_conditions[{key!r}] has unknown field(s) {sorted(unknown, key=repr)!r}")
                if "target_value" not in value:
                    return (), f"decision_conditions[{key!r}] must define target_value"
                entries.append(
                    {
                        "input_value_key": key,
                        "target_value": value.get("target_value"),
                        "operator": value.get("operator", required_operator),
                    }
                )
            else:
                entries.append(
                    {
                        "input_value_key": key,
                        "target_value": value,
                        "operator": required_operator,
                    }
                )
    elif isinstance(raw, list):
        for index, condition in enumerate(raw):
            if not isinstance(condition, Mapping):
                return (), f"decision_conditions[{index}] must be a mapping"
            required = {"input_value_key", "target_value", "operator"}
            missing = required - set(condition)
            unknown = set(condition) - required
            if missing or unknown:
                detail = []
                if missing:
                    detail.append(f"missing {sorted(missing)!r}")
                if unknown:
                    detail.append(f"unknown {sorted(unknown, key=repr)!r}")
                return (), f"decision_conditions[{index}] has " + " and ".join(detail)
            entries.append(condition)
    else:
        return (), "decision_conditions must be a non-empty JSON list or mapping"
    if not entries:
        return (), (
            "decision_conditions must contain at least one enabled dimension; "
            "an empty selector cannot preserve unscorable-row drop semantics"
        )

    by_key: dict[str, dict[str, Any]] = {}
    for index, condition in enumerate(entries):
        key = condition.get("input_value_key")
        if not isinstance(key, str) or not key:
            return (), f"decision_conditions[{index}].input_value_key must be a non-empty string"
        if key in by_key:
            return (), f"decision_conditions contains duplicate conditions for {key!r}"
        if key not in declared:
            return (), (
                f"decision_conditions key {key!r} is not a configured score key "
                f"of the card-declared compound decision; expected a subset of {list(declared)!r}"
            )
        operator = condition.get("operator")
        if operator != required_operator:
            return (), (
                f"decision_conditions[{index}].operator must be exactly {required_operator!r}, got {operator!r}"
            )
        target = condition.get("target_value")
        value_reason = _checkpoint_decisions._decision_value_reason(
            target,
            operator=required_operator,
            value_type=pair.value_type,
        )
        if value_reason:
            return (), f"decision_conditions[{index}].target_value: {value_reason}"
        by_key[key] = {
            "input_value_key": key,
            "target_value": target,
            "operator": required_operator,
        }
    return tuple(by_key[key] for key in declared if key in by_key), ""


def _declared_compound_score_keys(  # noqa: C901, PLR0911 - fail closed per declaration invariant
    recipe: Recipe,
    pair: _checkpoint_types.DecisionPair,
) -> tuple[dict[str, dict[str, Any]], str]:
    """Configured score keys for every card-declared compound dimension."""
    from nemo_curator.stages.audio import agent as foundation
    from nemo_curator.stages.audio.audio_agent.recipe import build_stages

    built, _issues = build_stages(recipe)
    if not built or pair.producer_index >= len(built):
        return {}, "compound annotation producer could not be constructed"
    producer = built[pair.producer_index]
    dimensions = pair.decision.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        return {}, "compound decision has no declared dimensions"

    declared: dict[str, dict[str, Any]] = {}
    for dimension in dimensions:
        if not isinstance(dimension, dict):
            return {}, "compound decision dimension is not a mapping"
        score_key_param = str(dimension.get("score_key_param") or "")
        score_key = getattr(producer, score_key_param, None)
        if not isinstance(score_key, str) or not score_key:
            return {}, f"configured producer has no score key through {score_key_param!r}"
        if score_key in declared:
            return {}, f"compound decision has ambiguous duplicate score key {score_key!r}"
        declared[score_key] = dict(dimension)

    contract = foundation.build_contract(producer)
    guaranteed = set(contract.writes.segment_data_keys) if pair.scope == "segments" else set(contract.writes.data_keys)
    conditional_groups: list[set[str]] = []
    for conditional in contract.conditional_writes:
        if conditional.value_origin != "stage_generated":
            continue
        scoped = (
            set(conditional.writes.segment_data_keys)
            if pair.scope == "segments"
            else set(conditional.writes.data_keys)
        )
        if scoped:
            conditional_groups.append(scoped)
    declared_keys = set(declared)
    if not any(declared_keys <= guaranteed | conditional for conditional in conditional_groups) and not (
        declared_keys <= guaranteed
    ):
        available = guaranteed | set().union(*conditional_groups)
        missing = declared_keys - available
        detail = (
            f"missing {sorted(missing)!r}"
            if missing
            else "the scores are split across separate conditional write groups"
        )
        return {}, (
            "annotation producer contract does not atomically write every declared compound score "
            f"at {pair.scope} scope: {detail}"
        )
    return declared, ""


def _with_decision_conditions(
    recipe: Recipe,
    pair: _checkpoint_types.DecisionPair,
    conditions: tuple[dict[str, Any], ...],
) -> Recipe:
    """Replace only the declared compound selector's complete condition set."""
    from nemo_curator.stages.audio.audio_agent import _checkpoint_candidates

    selector = pair.decision["selector"]
    conditions_param = str(selector.get("conditions_param") or "conditions")
    stages = [StageRef(ref=s.ref, params=dict(s.params)) for s in recipe.stages]
    stages[pair.selector_index].params[conditions_param] = [dict(condition) for condition in conditions]
    candidate = _checkpoint_candidates._copy_recipe(recipe, stages)
    candidate.checkpoint_decision = None
    return candidate.freeze()


def _refresh_compound_pair(
    recipe: Recipe,
    original: _checkpoint_types.DecisionPair,
    conditions: tuple[dict[str, Any], ...],
) -> tuple[_checkpoint_types.DecisionPair | None, str]:
    """Re-prove the selector policy and score lineage after compound feedback."""
    from nemo_curator.stages.audio.audio_agent import _checkpoint_decisions
    from nemo_curator.stages.audio.audio_agent.recipe import build_stages

    built, _issues = build_stages(recipe)
    if not built:
        return None, "changing decision_conditions made the recipe unconstructible"
    producer = built[original.producer_index]
    configured = built[original.selector_index]
    selector = original.decision.get("selector") or {}
    configured_conditions = getattr(configured, "normalized_conditions", None)
    if not isinstance(configured_conditions, tuple):
        return None, "compound selector has no canonical condition surface"
    configured_map, configured_reason = _condition_map(list(configured_conditions))
    expected_map, expected_reason = _condition_map(list(conditions))
    if configured_reason or expected_reason or configured_map != expected_map:
        return None, (
            configured_reason
            or expected_reason
            or "compound selector did not retain the complete canonical condition replacement"
        )
    policy_reason = _checkpoint_decisions._selector_policy_reason(
        recipe.stages[original.selector_index],
        configured,
        selector,
        original.decision,
        producer=producer,
        operator=None,
    )
    if policy_reason:
        return None, policy_reason
    score_keys = tuple(condition["input_value_key"] for condition in conditions)
    lineage_reason = _checkpoint_decisions._score_lineage_reason(
        recipe,
        built,
        producer_index=original.producer_index,
        selector_index=original.selector_index,
        score_keys=score_keys,
    )
    if lineage_reason:
        return None, lineage_reason
    return (
        _checkpoint_types.DecisionPair(
            producer_index=original.producer_index,
            selector_index=original.selector_index,
            producer_stage=original.producer_stage,
            selector_stage=original.selector_stage,
            decision_kind="compound",
            score_key=None,
            score_keys=score_keys,
            operator="and",
            target_value={condition["input_value_key"]: condition["target_value"] for condition in conditions},
            conditions=conditions,
            value_type=original.value_type,
            decision=dict(original.decision),
            scope=original.scope,
            items_key=original.items_key,
        ),
        "",
    )


def _checkpoint_condition_evidence_reason(  # noqa: C901, PLR0911, PLR0912 - fail closed per artifact invariant
    recipe: Recipe,
    *,
    checkpoint_index: int,
    pair: _checkpoint_types.DecisionPair,
) -> str:
    """Prove an existing annotation checkpoint carries the requested score keys."""
    output_path = recipe.stages[checkpoint_index].params.get(_checkpoint_types._CHECKPOINT_PATH_PARAM)
    if not isinstance(output_path, str) or not output_path:
        return "configured checkpoint has no path for compound score evidence"
    expanded = os.path.realpath(os.path.expanduser(output_path))
    marker = f"{expanded}._COMPLETE"
    if not os.path.exists(expanded) and not os.path.exists(marker):
        # First-run configured recipe: the producer contract proves what the
        # not-yet-created annotation checkpoint will contain.
        return ""
    if not os.path.isfile(expanded) or not os.path.isfile(marker):
        return "compound feedback requires a complete local JSONL annotation checkpoint"

    requested = set(pair.score_keys)
    saw_scored_item = False

    def _item_reason(item: Any, location: str) -> str:  # noqa: ANN401
        nonlocal saw_scored_item
        if not isinstance(item, Mapping):
            return f"{location} is not a mapping"
        present = requested & set(item)
        if present and present != requested:
            return (
                f"{location} contains only part of the requested compound score set: "
                f"found {sorted(present)!r}, expected {sorted(requested)!r}"
            )
        if present:
            for key in requested:
                value = item[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    return f"{location}.{key} is not a finite numeric annotation score"
            saw_scored_item = True
        return ""

    try:
        with open(expanded, encoding="utf-8") as checkpoint:
            for line_number, line in enumerate(checkpoint, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if pair.scope == "segments":
                    if not isinstance(row, Mapping):
                        return f"checkpoint row {line_number} is not a mapping"
                    items = row.get(pair.items_key)
                    if not isinstance(items, list):
                        return f"checkpoint row {line_number} does not contain nested list {pair.items_key!r}"
                    for item_index, item in enumerate(items):
                        reason = _item_reason(
                            item,
                            f"checkpoint row {line_number} child {item_index}",
                        )
                        if reason:
                            return reason
                else:
                    reason = _item_reason(row, f"checkpoint row {line_number}")
                    if reason:
                        return reason
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return f"compound feedback checkpoint could not be read as complete JSONL: {type(exc).__name__}: {exc}"
    if not saw_scored_item:
        return (
            "compound feedback checkpoint contains no row or nested child with "
            f"the complete requested score set {sorted(requested)!r}"
        )
    return ""
