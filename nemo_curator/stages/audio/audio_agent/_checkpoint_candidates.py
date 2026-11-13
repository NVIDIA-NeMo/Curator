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

"""Checkpoint recipe transformations, paths, and candidate evidence."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit

from nemo_curator.stages.audio.audio_agent import _checkpoint_types
from nemo_curator.stages.audio.audio_agent.recipe import (
    REUSABLE_CHECKPOINT_PROVENANCE,
    Recipe,
    StageRef,
)


def _with_decision_value(recipe: Recipe, pair: _checkpoint_types.DecisionPair, value: Any) -> Recipe:  # noqa: ANN401
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


def _refresh_pair(recipe: Recipe, original: _checkpoint_types.DecisionPair) -> _checkpoint_types.DecisionPair | None:
    from nemo_curator.stages.audio.audio_agent import _checkpoint_decisions

    pairs, _ = _checkpoint_decisions._decision_pairs(recipe, decision_stage=original.producer_stage)
    return next((p for p in pairs if p.producer_index == original.producer_index), None)


def _checkpoint_between(recipe: Recipe, producer_index: int, selector_index: int) -> int | None:
    return next(
        (
            index
            for index in range(producer_index + 1, selector_index)
            if recipe.stages[index].ref == _checkpoint_types._CHECKPOINT_REF
        ),
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
            ref=_checkpoint_types._CHECKPOINT_REF,
            params={
                _checkpoint_types._CHECKPOINT_PATH_PARAM: output_path,
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
    from nemo_curator.stages.audio.audio_agent.checkpoint import _error_codes
    from nemo_curator.stages.audio.audio_agent.continuation import _resume_breaks_on_disk_boundary

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
        from nemo_curator.stages.audio.audio_agent.artifacts import plan_steps
        from nemo_curator.stages.audio.audio_agent.run_store import checkpoint_path

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
    output_path = stage.params.get(_checkpoint_types._CHECKPOINT_PATH_PARAM)
    if not isinstance(output_path, str) or not output_path:
        return "configured checkpoint has no non-empty output_path"
    parsed = urlsplit(output_path)
    if parsed.scheme:
        return "configured checkpoint requires a plain local path, not a URI"
    expanded = os.path.realpath(os.path.expanduser(output_path))
    for index, other_stage in enumerate(recipe.stages):
        for key, value in other_stage.params.items():
            if index == checkpoint_index and key == _checkpoint_types._CHECKPOINT_PATH_PARAM:
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


def _decision_identity(pair: _checkpoint_types.DecisionPair) -> dict[str, Any]:
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
    pair: _checkpoint_types.DecisionPair,
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
            "stage": _checkpoint_types._CHECKPOINT_REF,
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
            "inserted": [{"index": pair.selector_index, "stage": _checkpoint_types._CHECKPOINT_REF}],
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
    pair: _checkpoint_types.DecisionPair,
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
            "output_path": stage.params.get(_checkpoint_types._CHECKPOINT_PATH_PARAM),
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
    from nemo_curator.stages.audio.audio_agent.artifacts import stage_trust

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
    from nemo_curator.stages.audio.audio_agent.artifacts import stage_is_costly

    return [stage.ref for stage in recipe.stages[:prefix] if stage_is_costly(stage.ref)]


def _static_cost_evidence(recipe: Recipe, prefix: int) -> list[dict[str, Any]]:
    from nemo_curator.stages.audio.audio_agent.index import get_index

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


def _rejected(pair: _checkpoint_types.DecisionPair, reason: str) -> dict[str, Any]:
    return {
        "producer_stage": pair.producer_stage,
        "producer_index": pair.producer_index,
        "selector_stage": pair.selector_stage,
        "selector_index": pair.selector_index,
        "reason": reason,
    }
