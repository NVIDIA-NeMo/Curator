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

"""Card-declared producer and selector matching."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from nemo_curator.stages.audio.audio_agent import _checkpoint_types

if TYPE_CHECKING:
    from nemo_curator.stages.audio.audio_agent.recipe import (
        Recipe,
        StageRef,
    )


def _decision_pairs(
    recipe: Recipe, *, decision_stage: str | None
) -> tuple[list[_checkpoint_types.DecisionPair], list[dict[str, Any]]]:
    from nemo_curator.stages.audio.audio_agent.index import get_index
    from nemo_curator.stages.audio.audio_agent.recipe import build_stages

    built, build_issues = build_stages(recipe)
    if not built or len(built) != len(recipe.stages):
        return [], [{"reason": "recipe does not build", "issues": list(build_issues)}]
    pairs: list[_checkpoint_types.DecisionPair] = []
    rejected: list[dict[str, Any]] = []
    for producer_index, (stage_ref, stage) in enumerate(zip(recipe.stages, built, strict=True)):
        if decision_stage and stage_ref.ref != decision_stage:
            continue
        card = get_index().card(stage_ref.ref) or {}
        decision = card.get("decision")
        if not isinstance(decision, dict) or decision.get("separable_from_producer") is not True:
            continue
        declarations = _decision_declarations(decision)
        producer_mode = getattr(stage, "mode", None)
        mode_bound = [
            declaration
            for declaration in declarations
            if isinstance(declaration.get("producer_constraints"), dict)
            and "mode" in declaration["producer_constraints"]
        ]
        if producer_mode == "auto" and mode_bound:
            rejected.append(
                {
                    "producer_stage": stage_ref.ref,
                    "producer_index": producer_index,
                    "reason": (
                        "producer mode='auto' is data-dependent; reusable decision separation "
                        "requires explicit mode='task' or mode='segments'"
                    ),
                }
            )
            continue
        matching_mode = [
            declaration
            for declaration in mode_bound
            if declaration["producer_constraints"].get("mode") == producer_mode
        ]
        candidates = matching_mode or declarations
        reasons: list[str] = []
        for declaration in candidates:
            pair, reason = _pair_for_producer(
                recipe,
                built,
                producer_index,
                stage_ref.ref,
                stage,
                declaration,
            )
            if pair is not None:
                pairs.append(pair)
                break
            if reason and reason not in reasons:
                reasons.append(reason)
        else:
            rejected.append(
                {
                    "producer_stage": stage_ref.ref,
                    "producer_index": producer_index,
                    "reason": "; ".join(reasons) or "no matching selector",
                }
            )
    return pairs, rejected


def _decision_declarations(decision: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the backward-compatible primary decision and its full variants."""
    primary = {key: value for key, value in decision.items() if key != "variants"}
    variants = decision.get("variants")
    if not isinstance(variants, list):
        return [primary]
    return [primary, *(dict(variant) for variant in variants if isinstance(variant, dict))]


def _pair_for_producer(  # noqa: C901, PLR0911, PLR0912, PLR0913 - one refusal per broken proof
    recipe: Recipe,
    built: list[Any],
    producer_index: int,
    producer_stage: str,
    producer: Any,  # noqa: ANN401 - heterogeneous configured AgentReady stage
    decision: dict[str, Any],
) -> tuple[_checkpoint_types.DecisionPair | None, str]:
    from nemo_curator.stages.audio.audio_agent import _checkpoint_compound, _checkpoint_nested

    kind = str(decision.get("kind") or "scalar")
    constraints = decision.get("producer_constraints") or {}
    if not isinstance(constraints, dict):
        return None, "decision.producer_constraints is not a mapping"
    for param, expected in constraints.items():
        actual = getattr(producer, str(param), None)
        if actual != expected:
            return (
                None,
                f"producer requires {param}={expected!r} for exact separation, got {actual!r}",
            )
    if decision.get("scope") == "segments":
        return _checkpoint_nested._nested_pair_for_producer(
            recipe,
            built,
            producer_index,
            producer_stage,
            producer,
            decision,
        )
    if kind == "compound":
        return _checkpoint_compound._compound_pair_for_producer(
            recipe,
            built,
            producer_index,
            producer_stage,
            producer,
            decision,
        )
    if kind != "scalar":
        return None, f"decision kind {kind!r} is not mechanically supported"

    selector = decision.get("selector") or {}
    if not isinstance(selector, dict):
        return None, "decision.selector is not a mapping"
    selector_stage = str(selector.get("stage_id") or selector.get("stage") or "")
    score_key_param = str(decision.get("score_key_param") or "")
    score_key = getattr(producer, score_key_param, None) if score_key_param else None
    if not isinstance(score_key, str) or not score_key:
        return None, f"configured producer has no score key through {score_key_param!r}"
    key_param = str(selector.get("key_param") or "input_value_key")
    value_param = str(selector.get("value_param") or "target_value")
    operator_param = str(selector.get("operator_param") or "operator")
    allowed = {str(op) for op in (selector.get("allowed_operators") or [])} or set(_checkpoint_types._OPS)
    value_type = str(decision.get("value_type") or "")
    for index in range(producer_index + 1, len(recipe.stages)):
        ref = recipe.stages[index]
        if ref.ref != selector_stage:
            continue
        configured = built[index]
        if getattr(configured, key_param, ref.params.get(key_param)) != score_key:
            continue
        lineage_reason = _score_lineage_reason(
            recipe,
            built,
            producer_index=producer_index,
            selector_index=index,
            score_keys=(score_key,),
        )
        if lineage_reason:
            return None, lineage_reason
        operator = str(ref.params.get(operator_param, selector.get("default_operator") or "eq"))
        if operator not in allowed:
            return None, f"selector operator {operator!r} is outside declared operators {sorted(allowed)}"
        policy_reason = _selector_policy_reason(
            ref,
            configured,
            selector,
            decision,
            producer=producer,
            operator=operator,
        )
        if policy_reason:
            return None, policy_reason
        target = ref.params.get(value_param, getattr(configured, value_param, None))
        value_reason = _decision_value_reason(
            target,
            operator=operator,
            value_type=value_type,
        )
        if value_reason:
            return None, value_reason
        return (
            _checkpoint_types.DecisionPair(
                producer_index=producer_index,
                selector_index=index,
                producer_stage=producer_stage,
                selector_stage=selector_stage,
                decision_kind="scalar",
                score_key=score_key,
                score_keys=(score_key,),
                operator=operator,
                target_value=target,
                conditions=(),
                value_type=value_type,
                decision=dict(decision),
                scope="task",
            ),
            "",
        )
    return None, f"no {selector_stage} below the producer reads {score_key!r}"


def _selector_policy_reason(  # noqa: PLR0913 - producer binding is required only for nested scope
    ref: StageRef,
    configured: Any,  # noqa: ANN401 - heterogeneous configured selector
    selector: dict[str, Any],
    decision: dict[str, Any],
    *,
    producer: Any,  # noqa: ANN401 - heterogeneous configured producer
    operator: str | None,
) -> str:
    """Prove exact operator and missing-score behavior on a configured selector."""
    required_operator = selector.get("required_operator")
    if operator is not None and required_operator is not None and operator != required_operator:
        return f"selector operator must be exactly {required_operator!r}, got {operator!r}"
    condition_logic_param = selector.get("condition_logic_param")
    required_condition_logic = selector.get("required_condition_logic")
    if condition_logic_param is not None or required_condition_logic is not None:
        logic_param = str(condition_logic_param or "condition_logic")
        actual_condition_logic = getattr(
            configured,
            logic_param,
            ref.params.get(logic_param),
        )
        if actual_condition_logic != required_condition_logic:
            return (
                f"selector {logic_param} must be exactly {required_condition_logic!r} "
                f"for exact native-filter equivalence, got {actual_condition_logic!r}"
            )
    missing_param = str(selector.get("missing_policy_param") or "missing_value_policy")
    expected_missing = selector.get(
        "required_missing_policy",
        "drop" if decision.get("missing_score_policy") == "selector_drop" else "error",
    )
    actual_missing = getattr(
        configured,
        missing_param,
        ref.params.get(missing_param),
    )
    if actual_missing != expected_missing:
        return f"selector {missing_param} must be exactly {expected_missing!r}, got {actual_missing!r}"
    if decision.get("scope") == "segments":
        items_param = str(selector.get("items_key_param") or "items_key")
        source_param = str(selector.get("items_key_source_param") or "segments_key")
        expected_items_key = getattr(producer, source_param, None)
        actual_items_key = getattr(
            configured,
            items_param,
            ref.params.get(items_param),
        )
        if actual_items_key != expected_items_key:
            return (
                f"selector {items_param} must exactly match producer {source_param} "
                f"{expected_items_key!r}, got {actual_items_key!r}"
            )
        empty_param = str(selector.get("empty_policy_param") or "drop_parent_if_empty")
        expected_empty = selector.get("required_empty_policy")
        actual_empty = getattr(
            configured,
            empty_param,
            ref.params.get(empty_param),
        )
        if actual_empty is not expected_empty:
            return f"selector {empty_param} must be exactly {expected_empty!r}, got {actual_empty!r}"
    return ""


def _score_lineage_reason(
    recipe: Recipe,
    built: list[Any],
    *,
    producer_index: int,
    selector_index: int,
    score_keys: tuple[str, ...],
) -> str:
    """Prove the selector still reads the exact producer value.

    Cross only independent, key-preserving annotations. Row filters are safe
    only after a checkpoint already preserves the unfiltered score population.
    Unknown effects, score replacement and changes of entity remain barriers.
    """
    intervening = list(range(producer_index + 1, selector_index))
    if not intervening:
        return ""
    from nemo_curator.stages.audio._agent._agent_registry import build_contract

    protected = set(score_keys)
    producer_contract = build_contract(built[producer_index])
    nested_scores = bool(producer_contract.writes.segment_data_keys) or any(
        write.writes.segment_data_keys for write in producer_contract.conditional_writes
    )
    if producer_contract.writes.segment_data_keys:
        protected.update(producer_contract.writes.data_keys)
    checkpoint_seen = False
    for index in intervening:
        contract = build_contract(built[index])
        writes = set(contract.writes.data_keys) | set(contract.writes.segment_data_keys)
        for conditional in contract.conditional_writes:
            writes.update(conditional.writes.data_keys)
            writes.update(conditional.writes.segment_data_keys)
        affected = writes | set(contract.removes_keys) | set(contract.invalidates_keys)
        prior_writes = getattr(contract, "invalidates_prior_writes", None)
        affected |= set(getattr(prior_writes, "data_keys", ())) | set(getattr(prior_writes, "segment_data_keys", ()))
        is_checkpoint = recipe.stages[index].ref == _checkpoint_types._CHECKPOINT_REF
        safe_shape = contract.cardinality == "1:1" or (
            checkpoint_seen and not is_checkpoint and contract.cardinality == "filter"
        )
        safe_effects = is_checkpoint or (
            contract.gates.per_row_independent is True
            and not contract.gates.writes_to_disk
            and not contract.gates.lifecycle_side_effects
        )
        if (
            contract.contract_resolution != "configured"
            or not contract.wrappable
            or not safe_shape
            or not safe_effects
            or not contract.preserves_upstream_keys
            or (nested_scores and not getattr(contract, "preserves_upstream_segment_keys", True))
            or protected & affected
        ):
            return (
                f"exact score lineage across {recipe.stages[index].ref} at index {index} "
                f"is not proven for {sorted(protected)!r}; requires independent key preservation "
                "and a checkpoint before any intervening row filter"
            )
        checkpoint_seen |= is_checkpoint
    return ""


def _decision_value_reason(  # noqa: PLR0911 - one refusal per scalar contract
    value: Any,  # noqa: ANN401 - untrusted JSON boundary
    *,
    operator: str,
    value_type: str,
) -> str:
    if not isinstance(value, (type(None), bool, int, float, str)):
        return "selector target_value must be a JSON scalar (null, boolean, number, or string)"
    if isinstance(value, float) and not math.isfinite(value):
        return "selector target_value must be a finite JSON number"
    if value_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "selector target_value must be a finite JSON number for this numeric score"
        return ""
    if value_type == "string":
        if not isinstance(value, str):
            return "selector target_value must be a JSON string for this string score"
        return ""
    if value_type == "boolean":
        if not isinstance(value, bool):
            return "selector target_value must be a JSON boolean for this boolean score"
        if operator not in {"eq", "ne"}:
            return "boolean selector targets support only eq or ne"
        return ""
    return f"decision value_type {value_type!r} is not mechanically supported"
