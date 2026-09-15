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

"""Deterministic candidate recipes for feedback-tunable, same-dataset pipelines.

The host decides whether a reusable boundary is worth offering.  This module decides the
mechanical half: a producer/gate pair must be explicitly declared by a capability card, the
configured keys and operator must match that declaration, and a dedicated metadata checkpoint
must survive the same validation and disk-boundary simulation as a real continuation.

No native filter is rewritten here. Supported shapes are deliberately card-declared:
an annotate-only producer followed by an exact scalar, compound, or one-level nested-list
selector. Inserting the pass-through checkpoint changes no row verdict in the first run and
leaves the decision parameter wholly below the persisted producer identity.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from nemo_curator.audio_agent.recipe import (
    REUSABLE_CHECKPOINT_PROVENANCE,
    Recipe,
    StageRef,
)

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


def plan(  # noqa: C901, PLR0912, PLR0913, PLR0915 - explicit mechanical refusal branches
    recipe: Recipe,
    *,
    output_path: str | None = None,
    dataset_key: str = "",
    accept: bool = False,
    decision_stage: str | None = None,
    decision_value: Any = None,  # noqa: ANN401 - card-declared scalar/categorical value
    decision_conditions: Any = None,  # noqa: ANN401 - complete card-declared compound surface
    retention_sec: int = 0,
    owner: str = "user",
) -> dict[str, Any]:
    """Return the unchanged baseline and mechanically proven checkpoint candidates.

    ``decision_value`` changes only a scalar selector target.
    ``decision_conditions`` replaces the complete compound selector condition
    set. In both cases the producer and checkpoint stay byte-identical, which
    is the property a feedback rerun needs.

    With ``dataset_key`` a candidate is materialized at its managed location rather than
    asking the caller for one; ``accept`` is the user's yes to that candidate. Without a
    dataset key the step key cannot be computed, so the caller is asked for a path exactly
    as before. WHETHER to spend a checkpoint stays a user decision either way -- only the
    question of where to put it goes away.
    """
    rec = recipe.freeze()
    baseline = {
        "id": "baseline",
        "config_hash": rec.config_hash,
        "recipe": rec.to_dict(),
        "effect": "run the authored pipeline without adding a reuse checkpoint",
    }
    pairs, rejected = _decision_pairs(rec, decision_stage=decision_stage)
    candidates: list[dict[str, Any]] = []
    if decision_value is not None and decision_conditions is not None:
        return {
            "status": "no_candidate",
            "scope": "same_dataset_only",
            "baseline": baseline,
            "candidates": [],
            "rejected": [
                {
                    "producer_stage": decision_stage,
                    "reason": (
                        "decision_value and decision_conditions are mutually exclusive; "
                        "use the scalar surface or the complete compound surface"
                    ),
                }
            ],
            "checkpoint_decision_required": False,
            "requires_authoritative_smoke": True,
        }
    for pair in pairs:
        if decision_value is not None:
            if pair.decision_kind != "scalar":
                rejected.append(
                    _rejected(
                        pair,
                        "compound decisions cannot be tuned with scalar decision_value; "
                        "submit an explicit complete conditions surface instead",
                    )
                )
                continue
            value_reason = _decision_value_reason(
                decision_value,
                operator=pair.operator,
                value_type=pair.value_type,
            )
            if value_reason:
                rejected.append(_rejected(pair, value_reason))
                continue
        if decision_conditions is not None:
            if pair.decision_kind != "compound":
                rejected.append(
                    _rejected(
                        pair,
                        "decision_conditions is supported only for a card-declared compound decision",
                    )
                )
                continue
            normalized, conditions_reason = _normalize_decision_conditions(
                rec,
                pair,
                decision_conditions,
            )
            if conditions_reason:
                rejected.append(_rejected(pair, conditions_reason))
                continue
            selected = _with_decision_conditions(rec, pair, normalized)
            refreshed, refresh_reason = _refresh_compound_pair(
                selected,
                pair,
                normalized,
            )
            if refreshed is None:
                rejected.append(_rejected(pair, refresh_reason))
                continue
        else:
            selected = _with_decision_value(rec, pair, decision_value) if decision_value is not None else rec
            refreshed = _refresh_pair(selected, pair)
            if refreshed is None:
                rejected.append(_rejected(pair, "changing the decision made its declared selector unresolvable"))
                continue
        existing = _checkpoint_between(selected, refreshed.producer_index, refreshed.selector_index)
        if existing is not None:
            configured = _with_checkpoint_provenance(selected, existing)
            reason = _configured_checkpoint_path_reason(configured, existing)
            if reason:
                rejected.append(_rejected(refreshed, reason))
                continue
            if decision_conditions is not None:
                evidence_reason = _checkpoint_condition_evidence_reason(
                    configured,
                    checkpoint_index=existing,
                    pair=refreshed,
                )
                if evidence_reason:
                    rejected.append(_rejected(refreshed, evidence_reason))
                    continue
            baseline_without_checkpoint = _without_stage(configured, existing)
            legal, why = _validate_candidate(
                baseline_without_checkpoint,
                configured,
                checkpoint_index=existing,
            )
            if not legal:
                rejected.append(_rejected(refreshed, why))
                continue
            candidates.append(
                _covered_candidate(
                    configured,
                    refreshed,
                    checkpoint_index=existing,
                    decision_changed=(decision_value is not None or decision_conditions is not None),
                )
            )
            continue
        chosen_path, path_source = output_path, "explicit"
        if not chosen_path:
            probe = _insert_checkpoint(
                selected,
                index=refreshed.selector_index,
                output_path=_ANALYSIS_PATH,
                retention_sec=retention_sec,
                owner=owner,
            )
            legal, why = _validate_candidate(
                selected,
                probe,
                checkpoint_index=refreshed.selector_index,
            )
            if not legal:
                rejected.append(_rejected(refreshed, why))
                continue
            path_source = "derived"
            chosen_path = _derived_path(
                probe,
                checkpoint_index=refreshed.selector_index,
                dataset_key=dataset_key,
            )
            if not chosen_path:
                expensive_prefix = _expensive_prefix(selected, refreshed.selector_index)
                candidates.append(
                    {
                        "id": f"checkpoint-after-{refreshed.producer_index}",
                        "status": "needs_output_path",
                        "producer_stage": refreshed.producer_stage,
                        "selector_stage": refreshed.selector_stage,
                        **_decision_identity(refreshed),
                        "checkpoint_index": refreshed.selector_index,
                        "expensive_prefix": expensive_prefix,
                        "recommended": bool(expensive_prefix),
                        "cost_evidence": _static_cost_evidence(selected, refreshed.selector_index),
                        "trust": _prefix_trust(probe, refreshed.selector_index + 1),
                        "residency": {
                            "format": "complete serializable task.data JSONL",
                            "waveform_persisted": False,
                            "suffix_survives_metadata_boundary": True,
                        },
                        "next": "call plan_checkpoint again with output_path to materialize and validate this candidate",
                    }
                )
                continue
        reason = _path_reason(selected, chosen_path, derived=path_source == "derived")
        if reason:
            rejected.append(_rejected(refreshed, reason))
            continue
        candidate = _insert_checkpoint(
            selected,
            index=refreshed.selector_index,
            output_path=chosen_path,
            retention_sec=retention_sec,
            owner=owner,
        )
        legal, why = _validate_candidate(selected, candidate, checkpoint_index=refreshed.selector_index)
        if not legal:
            rejected.append(_rejected(refreshed, why))
            continue
        candidates.append(
            _candidate(
                candidate,
                refreshed,
                output_path=chosen_path,
                path_source=path_source,
                # An explicit path IS the acceptance; a derived one still needs the user's yes.
                accepted=path_source == "explicit" or accept,
                decision_changed=(decision_value is not None or decision_conditions is not None),
                prior_target=(
                    [dict(condition) for condition in pair.conditions]
                    if decision_conditions is not None
                    else pair.target_value
                ),
            )
        )
    decision_required = any(
        candidate.get("status") in _UNDECIDED and candidate.get("recommended") is True for candidate in candidates
    )
    return {
        "status": "candidates" if candidates else "no_candidate",
        "scope": "same_dataset_only",
        "baseline": baseline,
        "candidates": candidates,
        "rejected": rejected,
        "checkpoint_decision_required": decision_required,
        "requires_authoritative_smoke": True,
        "host_directive": (
            "Choose only among the returned recipes. The core proved composition and resume "
            "legality; the host decides whether the user's likely feedback justifies one metadata "
            "file. Never mutate action, residency, score keys, selectors, or output paths by hand. "
            "A recommended option must be materialized or explicitly declined before smoke. "
            "Any selected recipe needs a new validate, semantic critique, smoke token, and exact-hash approval."
        ),
    }


def recommended_candidate_ids(result: dict[str, Any]) -> list[str]:
    """Stable IDs of pre-smoke options whose prefix contains meaningful work."""
    return sorted(
        str(candidate["id"])
        for candidate in result.get("candidates", [])
        if candidate.get("status") in _UNDECIDED and candidate.get("recommended") is True and candidate.get("id")
    )


def with_declined_checkpoint(recipe: Recipe, candidate_ids: list[str]) -> Recipe:
    """Bind an explicit baseline choice to this exact recipe and option set."""
    rec = _copy_recipe(
        recipe.freeze(), [StageRef(ref=stage.ref, params=dict(stage.params)) for stage in recipe.stages]
    ).freeze()
    rec.checkpoint_decision = {
        "status": _DECLINED,
        "recipe_config_hash": rec.config_hash,
        "candidate_ids": sorted(candidate_ids),
        "planner": REUSABLE_CHECKPOINT_PROVENANCE,
    }
    return rec


def checkpoint_decision_requirement(recipe: Recipe) -> dict[str, Any] | None:
    """Return the unresolved recommended choice that must precede smoke/run."""
    rec = recipe.freeze()
    result = plan(rec)
    candidate_ids = recommended_candidate_ids(result)
    if not candidate_ids:
        return None
    decision = rec.checkpoint_decision
    if (
        isinstance(decision, dict)
        and decision.get("status") == _DECLINED
        and decision.get("recipe_config_hash") == rec.config_hash
        and decision.get("planner") == REUSABLE_CHECKPOINT_PROVENANCE
        and sorted(str(value) for value in decision.get("candidate_ids", [])) == candidate_ids
    ):
        return None
    options = [
        {
            key: candidate.get(key)
            for key in (
                "id",
                "producer_stage",
                "selector_stage",
                "score_key",
                "score_keys",
                "scope",
                "items_key",
                "conditions",
                "expensive_prefix",
                "next",
            )
        }
        for candidate in result.get("candidates", [])
        if str(candidate.get("id") or "") in candidate_ids
    ]
    return {
        "reason_code": "checkpoint_decision_required",
        "reason": (
            "a recommended reusable metadata checkpoint must be materialized or "
            "explicitly declined before authoritative smoke or full execution"
        ),
        "config_hash": rec.config_hash,
        "candidates": options,
        "next": (
            "call plan_checkpoint with choice='checkpoint' to accept (the location is "
            "derived; output_path only overrides it), or with choice='baseline' after the "
            "user declines; use only the returned recipe"
        ),
    }


def _decision_pairs(recipe: Recipe, *, decision_stage: str | None) -> tuple[list[DecisionPair], list[dict[str, Any]]]:
    from nemo_curator.audio_agent.index import get_index
    from nemo_curator.audio_agent.recipe import build_stages

    built, build_issues = build_stages(recipe)
    if not built or len(built) != len(recipe.stages):
        return [], [{"reason": "recipe does not build", "issues": list(build_issues)}]
    pairs: list[DecisionPair] = []
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
) -> tuple[DecisionPair | None, str]:
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
        return _nested_pair_for_producer(
            recipe,
            built,
            producer_index,
            producer_stage,
            producer,
            decision,
        )
    if kind == "compound":
        return _compound_pair_for_producer(
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
    allowed = {str(op) for op in (selector.get("allowed_operators") or [])} or set(_OPS)
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
            DecisionPair(
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


def _compound_pair_for_producer(  # noqa: C901, PLR0911, PLR0912, PLR0913
    recipe: Recipe,
    built: list[Any],
    producer_index: int,
    producer_stage: str,
    producer: Any,  # noqa: ANN401 - heterogeneous configured AgentReady stage
    decision: dict[str, Any],
) -> tuple[DecisionPair | None, str]:
    """Resolve one exact all-enabled-dimensions AND selector."""
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
        value_reason = _decision_value_reason(
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
        policy_reason = _selector_policy_reason(
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
        lineage_reason = _score_lineage_reason(
            recipe,
            built,
            producer_index=producer_index,
            selector_index=index,
            score_keys=score_keys,
        )
        if lineage_reason:
            return None, lineage_reason
        return (
            DecisionPair(
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


def _nested_pair_for_producer(  # noqa: C901, PLR0911, PLR0912, PLR0913, PLR0915
    recipe: Recipe,
    built: list[Any],
    producer_index: int,
    producer_stage: str,
    producer: Any,  # noqa: ANN401 - heterogeneous configured AgentReady stage
    decision: dict[str, Any],
) -> tuple[DecisionPair | None, str]:
    """Resolve an exact one-level nested-list selector for segment annotation."""
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
        value_reason = _decision_value_reason(
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
            value_reason = _decision_value_reason(
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

    expected_map, expected_reason = _condition_map(expected_conditions)
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
        policy_reason = _selector_policy_reason(
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
                "nested selector conditions must exactly match every enabled producer "
                "threshold, configured score key, and required operator",
            )
        lineage_reason = _score_lineage_reason(
            recipe,
            built,
            producer_index=producer_index,
            selector_index=index,
            score_keys=score_keys,
        )
        if lineage_reason:
            return None, lineage_reason
        return (
            DecisionPair(
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

    Adjacency is the primary contract. The only currently supported intervening
    stage is this feature's checkpoint, whose configured contract is explicitly
    1:1, preserves upstream keys, and does not write the score key.
    """
    intervening = list(range(producer_index + 1, selector_index))
    if not intervening:
        return ""
    if any(recipe.stages[index].ref != _CHECKPOINT_REF for index in intervening):
        names = [recipe.stages[index].ref for index in intervening]
        return (
            "the declared selector is not adjacent to its producer and exact score "
            f"lineage across {names!r} is not proven"
        )

    from nemo_curator.stages.audio._agent._agent_registry import build_contract

    for index in intervening:
        contract = build_contract(built[index])
        writes = set(contract.writes.data_keys) | set(contract.writes.segment_data_keys)
        for conditional in contract.conditional_writes:
            writes.update(conditional.writes.data_keys)
            writes.update(conditional.writes.segment_data_keys)
        if contract.cardinality != "1:1" or not contract.preserves_upstream_keys or set(score_keys) & writes:
            return (
                f"{_CHECKPOINT_REF} at index {index} does not mechanically prove "
                f"1:1 preservation of {list(score_keys)!r}"
            )
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


def _normalize_decision_conditions(  # noqa: C901, PLR0911, PLR0912 - fail closed per condition invariant
    recipe: Recipe,
    pair: DecisionPair,
    raw: Any,  # noqa: ANN401 - untrusted Python/CLI/MCP boundary
) -> tuple[tuple[dict[str, Any], ...], str]:
    """Validate and canonicalize a complete compound-selector replacement."""
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
        value_reason = _decision_value_reason(
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


def _declared_compound_score_keys(  # noqa: PLR0911 - fail closed per declaration invariant
    recipe: Recipe,
    pair: DecisionPair,
) -> tuple[dict[str, dict[str, Any]], str]:
    """Configured score keys for every card-declared compound dimension."""
    from nemo_curator.audio_agent.recipe import build_stages
    from nemo_curator.stages.audio import agent as foundation

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
    available = set(contract.writes.segment_data_keys) if pair.scope == "segments" else set(contract.writes.data_keys)
    missing = set(declared) - available
    if missing:
        return {}, (
            "annotation producer contract does not write every declared compound score "
            f"at {pair.scope} scope: missing {sorted(missing)!r}"
        )
    return declared, ""


def _with_decision_conditions(
    recipe: Recipe,
    pair: DecisionPair,
    conditions: tuple[dict[str, Any], ...],
) -> Recipe:
    """Replace only the declared compound selector's complete condition set."""
    selector = pair.decision["selector"]
    conditions_param = str(selector.get("conditions_param") or "conditions")
    stages = [StageRef(ref=s.ref, params=dict(s.params)) for s in recipe.stages]
    stages[pair.selector_index].params[conditions_param] = [dict(condition) for condition in conditions]
    candidate = _copy_recipe(recipe, stages)
    candidate.checkpoint_decision = None
    return candidate.freeze()


def _refresh_compound_pair(
    recipe: Recipe,
    original: DecisionPair,
    conditions: tuple[dict[str, Any], ...],
) -> tuple[DecisionPair | None, str]:
    """Re-prove the selector policy and score lineage after compound feedback."""
    from nemo_curator.audio_agent.recipe import build_stages

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
    policy_reason = _selector_policy_reason(
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
    lineage_reason = _score_lineage_reason(
        recipe,
        built,
        producer_index=original.producer_index,
        selector_index=original.selector_index,
        score_keys=score_keys,
    )
    if lineage_reason:
        return None, lineage_reason
    return (
        DecisionPair(
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
    pair: DecisionPair,
) -> str:
    """Prove an existing annotation checkpoint carries the requested score keys."""
    output_path = recipe.stages[checkpoint_index].params.get(_CHECKPOINT_PATH_PARAM)
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


def _with_decision_value(recipe: Recipe, pair: DecisionPair, value: Any) -> Recipe:  # noqa: ANN401
    selector = pair.decision["selector"]
    stages = [StageRef(ref=s.ref, params=dict(s.params)) for s in recipe.stages]
    if pair.scope == "segments":
        conditions_param = str(selector.get("conditions_param") or "conditions")
        stages[pair.selector_index].params[conditions_param] = [
            {
                **condition,
                "target_value": (
                    value if condition["input_value_key"] == pair.score_key else condition["target_value"]
                ),
            }
            for condition in pair.conditions
        ]
    else:
        value_param = str(selector.get("value_param") or "target_value")
        stages[pair.selector_index].params[value_param] = value
    candidate = _copy_recipe(recipe, stages)
    candidate.checkpoint_decision = None
    return candidate.freeze()


def _refresh_pair(recipe: Recipe, original: DecisionPair) -> DecisionPair | None:
    pairs, _ = _decision_pairs(recipe, decision_stage=original.producer_stage)
    return next((p for p in pairs if p.producer_index == original.producer_index), None)


def _checkpoint_between(recipe: Recipe, producer_index: int, selector_index: int) -> int | None:
    return next(
        (index for index in range(producer_index + 1, selector_index) if recipe.stages[index].ref == _CHECKPOINT_REF),
        None,
    )


def _insert_checkpoint(
    recipe: Recipe,
    *,
    index: int,
    output_path: str,
    retention_sec: int,
    owner: str,
) -> Recipe:
    stages = [StageRef(ref=s.ref, params=dict(s.params)) for s in recipe.stages]
    stages.insert(
        index,
        StageRef(
            ref=_CHECKPOINT_REF,
            params={
                _CHECKPOINT_PATH_PARAM: output_path,
                "retention_sec": int(retention_sec),
                "owner": str(owner),
                "planning_provenance": REUSABLE_CHECKPOINT_PROVENANCE,
            },
        ),
    )
    candidate = _copy_recipe(recipe, stages)
    candidate.checkpoint_decision = None
    return candidate.freeze()


def _with_checkpoint_provenance(recipe: Recipe, checkpoint_index: int) -> Recipe:
    stages = [StageRef(ref=s.ref, params=dict(s.params)) for s in recipe.stages]
    stages[checkpoint_index].params["planning_provenance"] = REUSABLE_CHECKPOINT_PROVENANCE
    candidate = _copy_recipe(recipe, stages)
    candidate.checkpoint_decision = None
    return candidate.freeze()


def _without_stage(recipe: Recipe, index: int) -> Recipe:
    stages = [
        StageRef(ref=stage.ref, params=dict(stage.params))
        for stage_index, stage in enumerate(recipe.stages)
        if stage_index != index
    ]
    return _copy_recipe(recipe, stages).freeze()


def _copy_recipe(recipe: Recipe, stages: list[StageRef]) -> Recipe:
    return Recipe(
        stages=stages,
        inputs=dict(recipe.inputs),
        preset=recipe.preset,
        acceptance_criteria=list(recipe.acceptance_criteria),
        rationale=recipe.rationale,
        name=recipe.name,
        machine_plan=dict(recipe.machine_plan) if recipe.machine_plan else None,
        data_derived=dict(recipe.data_derived) if recipe.data_derived else None,
        config_strategy=list(recipe.config_strategy) if recipe.config_strategy else None,
        knowledge_version=recipe.knowledge_version,
        parent_run_id=recipe.parent_run_id,
        checkpoint_decision=(
            dict(recipe.checkpoint_decision) if isinstance(recipe.checkpoint_decision, dict) else None
        ),
        planning_preference=(
            dict(recipe.planning_preference) if isinstance(recipe.planning_preference, dict) else None
        ),
    )


def _validate_candidate(baseline: Recipe, candidate: Recipe, *, checkpoint_index: int) -> tuple[bool, str]:
    from nemo_curator.audio_agent.checkpoint import _error_codes
    from nemo_curator.audio_agent.continuation import _resume_breaks_on_disk_boundary

    before = _error_codes(baseline)
    after = _error_codes(candidate)
    if before is None or after is None:
        return False, "candidate could not be built"
    added = after - before
    if "tensor_into_sink" in added:
        return False, "a live waveform reaches the checkpoint; this feature never materializes audio implicitly"
    if added:
        return False, "checkpoint candidate adds validation errors: " + ", ".join(sorted(added.elements()))
    broken = _resume_breaks_on_disk_boundary(candidate, checkpoint_index + 1)
    if broken:
        return False, f"the suffix cannot resume from metadata: {broken}"
    return True, ""


def _derived_path(probe: Recipe, *, checkpoint_index: int, dataset_key: str) -> str:
    """The managed checkpoint location for this step, or ``""`` when it cannot be derived.

    Safe to compute from ``probe`` -- which carries a placeholder ``output_path`` -- because
    ``output_path`` is an ``OUTPUT_LOCATION_PARAM`` and so is stripped from the step key.
    The placeholder cannot perturb the key it is being used to compute.

    Returns ``""`` rather than raising: failing to derive a path must fall back to asking
    the caller for one, never break planning.
    """
    if not dataset_key:
        return ""
    try:
        from nemo_curator.audio_agent.artifacts import plan_steps
        from nemo_curator.audio_agent.run_store import checkpoint_path

        steps = plan_steps(probe, dataset_key)
        if not 0 <= checkpoint_index < len(steps):
            return ""
        return checkpoint_path(steps[checkpoint_index].step_key) or ""
    except Exception:  # noqa: BLE001 - an underivable path degrades to asking, never to a crash
        return ""


def _path_reason(recipe: Recipe, output_path: str, *, derived: bool = False) -> str:
    parsed = urlsplit(output_path)
    if parsed.scheme:
        return "dedicated checkpoint reuse requires a plain local path, not a URI"
    expanded = os.path.realpath(os.path.expanduser(output_path))
    for stage in recipe.stages:
        for key, value in stage.params.items():
            if "output" not in key and not key.endswith(("_dir", "_path")):
                continue
            if not isinstance(value, str) or not value:
                continue
            other = urlsplit(value)
            if other.scheme:
                continue
            if os.path.realpath(os.path.expanduser(value)) == expanded:
                return f"checkpoint path collides with {stage.ref}.{key}"
    return "" if derived else _occupied_reason(expanded)


def _occupied_reason(expanded: str) -> str:
    """Why a user-named path is already taken, or ``""``.

    Asked only of a path the user named. At a derived one an existing file is this very
    step's own prior output -- a live artifact reuse-scan is about to resume from, or an
    orphan whose bytes the same step key would reproduce. Refusing there would make the
    cache unusable the second time it is consulted.
    """
    if os.path.exists(expanded):
        return "checkpoint path already exists; choose a new versioned path instead of replacing prior work"
    if os.path.exists(f"{expanded}._COMPLETE"):
        return "a stale completion marker exists at this path; choose a new versioned checkpoint path"
    return ""


def _configured_checkpoint_path_reason(  # noqa: C901 - one refusal per path invariant
    recipe: Recipe,
    checkpoint_index: int,
) -> str:
    stage = recipe.stages[checkpoint_index]
    output_path = stage.params.get(_CHECKPOINT_PATH_PARAM)
    if not isinstance(output_path, str) or not output_path:
        return "configured checkpoint has no non-empty output_path"
    parsed = urlsplit(output_path)
    if parsed.scheme:
        return "configured checkpoint requires a plain local path, not a URI"
    expanded = os.path.realpath(os.path.expanduser(output_path))
    for index, other_stage in enumerate(recipe.stages):
        for key, value in other_stage.params.items():
            if index == checkpoint_index and key == _CHECKPOINT_PATH_PARAM:
                continue
            if "output" not in key and not key.endswith(("_dir", "_path")):
                continue
            if not isinstance(value, str) or not value:
                continue
            other = urlsplit(value)
            if other.scheme:
                continue
            if os.path.realpath(os.path.expanduser(value)) == expanded:
                return f"checkpoint path collides with {other_stage.ref}.{key}"
    output_exists = os.path.exists(expanded)
    marker_exists = os.path.exists(f"{expanded}._COMPLETE")
    if marker_exists and not output_exists:
        return "a stale completion marker exists without its configured checkpoint"
    if output_exists and not marker_exists:
        return "configured checkpoint path exists without a completion marker; refusing a partial or unproven artifact"
    return ""


def _decision_identity(pair: DecisionPair) -> dict[str, Any]:
    """Unambiguous scalar or complete compound candidate identity."""
    nested = (
        {
            "scope": "segments",
            "items_key": pair.items_key,
            "conditions": [dict(condition) for condition in pair.conditions],
        }
        if pair.scope == "segments"
        else {"scope": "task"}
    )
    if pair.decision_kind == "compound":
        return {
            "decision_kind": "compound",
            "score_keys": list(pair.score_keys),
            "operator": "and",
            "conditions": [dict(condition) for condition in pair.conditions],
            **nested,
        }
    return {
        "decision_kind": "scalar",
        "score_key": pair.score_key,
        "operator": pair.operator,
        "target_value": pair.target_value,
        **nested,
    }


def _candidate(  # noqa: PLR0913 - one materialized option gathers its whole description
    recipe: Recipe,
    pair: DecisionPair,
    *,
    output_path: str,
    decision_changed: bool,
    prior_target: Any,  # noqa: ANN401
    path_source: str = "explicit",
    accepted: bool = True,
) -> dict[str, Any]:
    trust = _prefix_trust(recipe, pair.selector_index + 1)
    expensive = _expensive_prefix(recipe, pair.selector_index)
    checkpoint_stage = recipe.stages[pair.selector_index]
    return {
        "id": f"checkpoint-after-{pair.producer_index}",
        "status": "ready" if accepted else "needs_decision",
        "producer_stage": pair.producer_stage,
        "producer_index": pair.producer_index,
        "selector_stage": pair.selector_stage,
        "selector_index": pair.selector_index + 1,
        **_decision_identity(pair),
        "decision_contract": {
            "kind": pair.decision_kind,
            "scope": pair.decision.get("scope"),
            "monotonic_direction": pair.decision.get("monotonic_direction"),
            "missing_score_policy": pair.decision.get("missing_score_policy"),
            "atomic": pair.decision.get("atomic"),
            "producer_identity": "threshold_free",
        },
        "checkpoint": {
            "stage": _CHECKPOINT_REF,
            "index": pair.selector_index,
            "output_path": output_path,
            "path_source": path_source,
            "retention_sec": checkpoint_stage.params.get("retention_sec"),
            "owner": checkpoint_stage.params.get("owner"),
            "planning_provenance": checkpoint_stage.params.get("planning_provenance"),
        },
        "expensive_prefix": expensive,
        "recommended": bool(expensive),
        "cost_evidence": _static_cost_evidence(recipe, pair.selector_index),
        "cardinality": {
            "first_run": "unchanged; the checkpoint is pass-through",
            "feedback_run": "resume before the existing selector, so discarded rows remain available",
        },
        "residency": {
            "format": "complete serializable task.data JSONL",
            "waveform_persisted": False,
            "suffix_survives_metadata_boundary": True,
        },
        "trust": trust,
        "diff": {
            "inserted": [{"index": pair.selector_index, "stage": _CHECKPOINT_REF}],
            "changed": (
                [
                    {
                        "stage": pair.selector_stage,
                        "param": str(
                            (pair.decision.get("selector") or {}).get(
                                "conditions_param" if pair.decision_kind == "compound" else "value_param"
                            )
                            or ("conditions" if pair.decision_kind == "compound" else "target_value")
                        ),
                        "from": prior_target,
                        "to": (
                            [dict(condition) for condition in pair.conditions]
                            if pair.decision_kind == "compound"
                            else pair.target_value
                        ),
                    }
                ]
                if decision_changed
                else []
            ),
        },
        "config_hash": recipe.config_hash,
        "planning_provenance": _planning_provenance(recipe.config_hash),
        "execution_requirements": _execution_requirements(recipe.config_hash),
        "recipe": recipe.to_dict(),
        "effect": (
            "first run writes one metadata checkpoint; a same-dataset change to this selector "
            "can resume after the unchanged producer prefix"
        ),
        "next": (
            "validate this recipe, then smoke and run it"
            if accepted
            else "ask the user whether to spend this checkpoint, then call plan_checkpoint "
            "again with choice='checkpoint' to accept the location above, or "
            "choice='baseline' to decline; do not ask them for a path"
        ),
    }


def _covered_candidate(
    recipe: Recipe,
    pair: DecisionPair,
    *,
    checkpoint_index: int,
    decision_changed: bool,
) -> dict[str, Any]:
    stage = recipe.stages[checkpoint_index]
    return {
        "id": f"existing-checkpoint-{checkpoint_index}",
        "status": "configured",
        "producer_stage": pair.producer_stage,
        "selector_stage": pair.selector_stage,
        **_decision_identity(pair),
        "decision_contract": {
            "kind": pair.decision_kind,
            "scope": pair.decision.get("scope"),
            "monotonic_direction": pair.decision.get("monotonic_direction"),
            "missing_score_policy": pair.decision.get("missing_score_policy"),
            "atomic": pair.decision.get("atomic"),
            "producer_identity": "threshold_free",
        },
        "checkpoint": {
            "stage": stage.ref,
            "index": checkpoint_index,
            "output_path": stage.params.get(_CHECKPOINT_PATH_PARAM),
            "planning_provenance": stage.params.get("planning_provenance"),
        },
        "residency": {
            "format": "complete serializable task.data JSONL",
            "waveform_persisted": False,
            "suffix_survives_metadata_boundary": True,
        },
        "trust": _prefix_trust(recipe, checkpoint_index + 1),
        "decision_changed": decision_changed,
        "config_hash": recipe.config_hash,
        "planning_provenance": _planning_provenance(recipe.config_hash),
        "execution_requirements": _execution_requirements(recipe.config_hash),
        "recipe": recipe.to_dict(),
        "requires_reuse_scan": True,
        "effect": (
            "the recipe already has the right boundary; reuse-scan must still verify that a "
            "complete matching artifact exists before running only the selector and suffix"
        ),
    }


def _planning_provenance(config_hash: str | None) -> dict[str, Any]:
    return {
        "planner": REUSABLE_CHECKPOINT_PROVENANCE,
        "recipe_config_hash": config_hash,
        "scope": "checkpoint_planned_recipe",
    }


def _execution_requirements(config_hash: str | None) -> dict[str, Any]:
    return {
        "validate": {
            "required": True,
            "recipe_config_hash": config_hash,
        },
        "semantic_review": {
            "required": True,
            "recipe_config_hash": config_hash,
            "required_response": {
                "mechanically_runnable": True,
                "recipe_config_hash": config_hash,
                "intent_status": "pass",
            },
            "enforcement": (
                "host response contract; the core has no semantic-review token because intent critique is host-owned"
            ),
        },
        "smoke": {
            "required": True,
            "recipe_config_hash": config_hash,
            "token_must_match_exact_hash": True,
        },
        "approval": {
            "required": True,
            "confirm_with_exact_hash": config_hash,
            "bare_true_allowed": False,
        },
    }


def _prefix_trust(recipe: Recipe, prefix: int) -> dict[str, Any]:
    from nemo_curator.audio_agent.artifacts import stage_trust

    deterministic = True
    ttl = 0
    low_trust: list[str] = []
    for stage in recipe.stages[:prefix]:
        stage_det, stage_ttl = stage_trust(stage.ref)
        deterministic = deterministic and stage_det
        if not stage_det:
            low_trust.append(stage.ref)
        if stage_ttl:
            ttl = min(ttl, stage_ttl) if ttl else stage_ttl
    return {
        "deterministic": deterministic,
        "ttl_sec": ttl,
        "low_trust_stages": low_trust,
        "requires_explicit_reuse_approval": not deterministic,
    }


def _expensive_prefix(recipe: Recipe, prefix: int) -> list[str]:
    from nemo_curator.audio_agent.artifacts import stage_is_costly

    return [stage.ref for stage in recipe.stages[:prefix] if stage_is_costly(stage.ref)]


def _static_cost_evidence(recipe: Recipe, prefix: int) -> list[dict[str, Any]]:
    from nemo_curator.audio_agent.index import get_index

    evidence: list[dict[str, Any]] = []
    for stage in recipe.stages[:prefix]:
        card = get_index().card(stage.ref) or {}
        resource = card.get("resource") or {}
        evidence.append(
            {
                "stage": stage.ref,
                "bound": resource.get("bound"),
                "model_id": card.get("model_id"),
                "throughput_hint": resource.get("throughput_hint"),
                "measured": False,
            }
        )
    return evidence


def _rejected(pair: DecisionPair, reason: str) -> dict[str, Any]:
    return {
        "producer_stage": pair.producer_stage,
        "producer_index": pair.producer_index,
        "selector_stage": pair.selector_stage,
        "selector_index": pair.selector_index,
        "reason": reason,
    }
