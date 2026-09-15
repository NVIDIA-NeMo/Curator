# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fail-closed, non-blocking advice for soft curation planning preferences."""

from __future__ import annotations

from typing import Any

from nemo_curator.audio_agent.recipe import Recipe, StageRef

_QUALITY_STAGES = frozenset({"UTMOSFilterStage", "SIGMOSFilterStage"})
_CHECKPOINT_REF = "ManifestCheckpointStage"


def build_planning_advisories(  # noqa: C901 - each branch removes an unproven advisory
    recipe: Recipe,
    configured_stages: list[Any],
    *,
    initial_keys: set[str],
    data_profile: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return refine-later suggestions only when an exact alternative is proven.

    The proof reuses the capability-card decision declaration and the reusable
    checkpoint planner. Any unsupported scope, selector, tensor boundary, or
    uncertain file/waveform equivalence produces no advice.
    """
    preference = recipe.planning_preference or {}
    if preference.get("curation_mode") != "refine_later":
        return []

    from nemo_curator.audio_agent import reusable_pipeline
    from nemo_curator.audio_agent.index import get_index

    exact_pairs, _ = reusable_pipeline._decision_pairs(
        recipe,
        decision_stage=None,
    )
    exact_producers = {pair.producer_index for pair in exact_pairs}
    advisories: list[dict[str, Any]] = []

    for stage_index, (stage_ref, stage) in enumerate(zip(recipe.stages, configured_stages, strict=True)):
        if stage_ref.ref not in _QUALITY_STAGES or stage_index in exact_producers:
            continue
        card = get_index().card(stage_ref.ref) or {}
        decision = card.get("decision")
        if not isinstance(decision, dict) or decision.get("separable_from_producer") is not True:
            continue

        scope = _explicit_scope(
            stage,
            stage_index=stage_index,
            configured_stages=configured_stages,
            initial_keys=initial_keys,
            data_profile=data_profile,
        )
        if scope is None:
            continue

        file_equivalent = _file_input_is_equivalent(
            stage,
            stage_index=stage_index,
            configured_stages=configured_stages,
            initial_keys=initial_keys,
            data_profile=data_profile,
        )
        alternative = (
            _exact_alternative(
                recipe,
                stage_index=stage_index,
                stage=stage,
                decision=decision,
                card=card,
                scope=scope,
                force_file=True,
            )
            if file_equivalent
            else None
        )
        proof = _checkpoint_proof(alternative, stage_index) if alternative is not None else None
        residency_reason = ""
        if proof is None:
            alternative = _file_backed_live_waveform_alternative(
                recipe,
                stage_index=stage_index,
                stage=stage,
                decision=decision,
                card=card,
                scope=scope,
            )
            if alternative is None:
                continue
            proof = _checkpoint_proof(alternative, stage_index)
            if proof is None:
                continue
            residency_reason = "live_waveform_residency"

        pair, checkpoint = proof
        reasons = []
        if getattr(stage, "action", None) == "filter":
            reasons.append("native_filter")
        if getattr(stage, "mode", None) == "auto":
            reasons.append("data_dependent_auto")
        if residency_reason:
            reasons.append(residency_reason)
        if not reasons:
            continue

        selector = alternative.stages[pair.selector_index]
        advisories.append(
            {
                "code": "refine_later_reusable_decision_available",
                "stage_index": stage_index,
                "stage": stage_ref.ref,
                "reasons": reasons,
                "message": (
                    f"{stage_ref.ref} at recipe index {stage_index} is valid as configured. "
                    "For easier later threshold refinement, its card declares an exact "
                    f"{scope}-scope annotation/selector alternative with one optional "
                    "metadata checkpoint."
                ),
                "suggested_shape": {
                    "ordering": "annotate -> metadata checkpoint -> exact selector",
                    "producer": {
                        "ref": stage_ref.ref,
                        "params": {
                            "action": "annotate",
                            "mode": scope,
                            "input_residency": "file",
                        },
                    },
                    "checkpoint": {
                        "ref": _CHECKPOINT_REF,
                        "position": "immediately after producer",
                        "metadata_only": True,
                        "optional": True,
                    },
                    "selector": selector.to_dict(),
                    "scope": scope,
                    "file_backed_boundary": True,
                },
                "checkpoint_evidence": {
                    "candidate_id": checkpoint.get("id"),
                    "recommended": True,
                    "first_run_recipe_unchanged": False,
                },
                "guidance": (
                    "Do not rewrite automatically or force a checkpoint. Prefer this shape "
                    "only if future tuning is worth one metadata file; otherwise keep the "
                    "current valid recipe and briefly explain the deviation."
                ),
            }
        )
    advisories.extend(_row_independence_advisories(recipe, configured_stages))
    return advisories


def _row_independence_advisories(
    recipe: Recipe,
    configured_stages: list[Any],
) -> list[dict[str, Any]]:
    """Warn when a PARAM, not the stage itself, forfeits incremental reuse.

    A delta merges manifest rows, so it can only resume from a step inside the run of stages
    whose per-row work is independent. A stage that declares ``per_row_independent=False``
    ends that run, and anything persisted below it -- usually the terminal manifest, the one
    artifact a merge can actually rewrite -- becomes unreachable. Adding one file to the
    corpus then costs a full recompute.

    Some stages are inherently cross-row and there is nothing to say. This advises only
    where the gate is CONDITIONAL on a param the author of the recipe chose: SplitASRAlignJoin
    declares ``per_row_independent = self.output_dir is None``, so pointing its chunks at a
    shared directory -- a tidy, entirely reasonable-looking choice -- silently trades away
    every future delta. That is precisely the trade ``refine_later`` exists to surface.

    Which param is found by re-deriving the gate with each configured param dropped, never
    by naming one here: the next stage to make its independence conditional gets the same
    advice without touching this file.
    """
    from nemo_curator.audio_agent import artifacts as art_mod

    advisories: list[dict[str, Any]] = []
    for index, (stage_ref, stage) in enumerate(zip(recipe.stages, configured_stages, strict=True)):
        if _independence(stage) is not False:
            continue
        # Nothing persists below it, so the truncation costs no reuse that existed.
        if not any(art_mod.output_uri(later)[0] for later in recipe.stages[index + 1 :]):
            continue
        culprits = _params_costing_independence(recipe, index)
        if not culprits:
            continue  # inherent to the stage; the user has no choice to be told about
        advisories.append(
            {
                "code": "refine_later_row_independence_forfeited",
                "stage_index": index,
                "stage": stage_ref.ref,
                "reasons": ["param_conditional_row_independence"],
                "params_responsible": culprits,
                "message": (
                    f"{stage_ref.ref} at recipe index {index} is valid as configured, but "
                    f"{', '.join(culprits)} makes its per-row work interdependent. A later "
                    "delta can only resume from a step above it, so adding one file to this "
                    "corpus would recompute the whole pipeline instead of just the new file."
                ),
                "suggested_shape": {
                    "drop_params": culprits,
                    "effect": "restores per-row independence, so a one-file addition can delta",
                },
                "guidance": (
                    "Do not drop these automatically -- they were set for a reason, and for "
                    "an output directory that reason is usually keeping intermediate files "
                    "out of the user's source folder. Put the trade to the user: a tidy "
                    "intermediate location now, or incremental reuse when the corpus grows."
                ),
            }
        )
    return advisories


def _independence(stage: Any) -> bool | None:  # noqa: ANN401 - a configured stage instance
    """The stage's configured ``per_row_independent``, or ``None`` when undeclared/unreadable."""
    from nemo_curator.stages.audio import agent as foundation

    try:
        return foundation.build_contract(stage).gates.per_row_independent
    except Exception:  # noqa: BLE001 - advice must never break validation
        return None


def _params_costing_independence(recipe: Recipe, index: int) -> list[str]:
    """Configured params of ``recipe.stages[index]`` that each, alone, cause ``False``."""
    from nemo_curator.audio_agent.recipe import build_stages

    responsible: list[str] = []
    for name in sorted(recipe.stages[index].params):
        stages = [StageRef(ref=s.ref, params=dict(s.params)) for s in recipe.stages]
        stages[index].params.pop(name, None)
        try:
            built, _ = build_stages(_copy_recipe(recipe, stages))
            if built and len(built) == len(stages) and _independence(built[index]) is not False:
                responsible.append(name)
        except Exception:  # noqa: BLE001, S112 - an unbuildable variant proves nothing
            continue
    return responsible


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


def _explicit_scope(  # noqa: PLR0911 - fail closed at each missing proof
    stage: Any,  # noqa: ANN401 - configured heterogeneous stage
    *,
    stage_index: int,
    configured_stages: list[Any],
    initial_keys: set[str],
    data_profile: dict[str, Any] | None,
) -> str | None:
    mode = getattr(stage, "mode", None)
    if mode in {"task", "segments"}:
        return str(mode)
    if mode != "auto":
        return None

    segments_key = getattr(stage, "segments_key", None)
    if not isinstance(segments_key, str) or not segments_key:
        return None
    produced_unconditionally, _produced_possibly = _keys_before(
        configured_stages,
        stage_index=stage_index,
        initial_keys=set(),
    )
    _unconditional, possible = _keys_before(
        configured_stages,
        stage_index=stage_index,
        initial_keys=initial_keys,
    )
    if segments_key in produced_unconditionally:
        return "segments"
    if segments_key in possible:
        return None
    if _profile_completely_defines_input_keys(data_profile):
        return "task"
    return None


def _keys_before(
    configured_stages: list[Any],
    *,
    stage_index: int,
    initial_keys: set[str],
) -> tuple[set[str], set[str]]:
    """Unconditional and possible task-data keys before one configured stage."""
    from nemo_curator.stages.audio import agent as foundation

    unconditional = set(initial_keys)
    possible = set(initial_keys)
    for prior in configured_stages[:stage_index]:
        contract = foundation.build_contract(prior)
        removed = set(contract.removes_keys)
        unconditional -= removed
        possible -= removed
        writes = set(contract.writes.data_keys) | set(contract.writes.segment_data_keys)
        unconditional |= writes
        possible |= writes
        for conditional in contract.conditional_writes:
            possible |= set(conditional.writes.data_keys)
            possible |= set(conditional.writes.segment_data_keys)
    return unconditional, possible


def _profile_completely_defines_input_keys(
    data_profile: dict[str, Any] | None,
) -> bool:
    if not data_profile or data_profile.get("source_errors"):
        return False
    return data_profile.get("kind") in {"manifest", "folder"}


def _file_input_is_equivalent(
    stage: Any,  # noqa: ANN401
    *,
    stage_index: int,
    configured_stages: list[Any],
    initial_keys: set[str],
    data_profile: dict[str, Any] | None,
) -> bool:
    residency = getattr(stage, "input_residency", None)
    if residency == "file":
        return True
    if residency != "auto" or not _profile_completely_defines_input_keys(data_profile):
        return False
    waveform_key = getattr(stage, "waveform_key", None)
    if not isinstance(waveform_key, str) or not waveform_key:
        return False
    _unconditional, possible = _keys_before(
        configured_stages,
        stage_index=stage_index,
        initial_keys=initial_keys,
    )
    return waveform_key not in possible


def _declarations(decision: dict[str, Any]) -> list[dict[str, Any]]:
    primary = {key: value for key, value in decision.items() if key != "variants"}
    variants = decision.get("variants")
    extras = variants if isinstance(variants, list) else []
    return [
        primary,
        *(dict(item) for item in extras if isinstance(item, dict)),
    ]


def _declaration_for_scope(
    decision: dict[str, Any],
    scope: str,
) -> dict[str, Any] | None:
    for declaration in _declarations(decision):
        constraints = declaration.get("producer_constraints")
        if (
            declaration.get("separable_from_producer") is True
            and declaration.get("scope", "task") == scope
            and isinstance(constraints, dict)
            and constraints.get("action") == "annotate"
            and constraints.get("mode") == scope
        ):
            return declaration
    return None


def _exact_alternative(  # noqa: PLR0913 - proof inputs stay explicit
    recipe: Recipe,
    *,
    stage_index: int,
    stage: Any,  # noqa: ANN401
    decision: dict[str, Any],
    card: dict[str, Any],
    scope: str,
    force_file: bool,
) -> Recipe | None:
    declaration = _declaration_for_scope(decision, scope)
    if declaration is None:
        return None
    stages = [StageRef(ref=item.ref, params=dict(item.params)) for item in recipe.stages]
    producer = stages[stage_index]
    producer.params["mode"] = scope
    if force_file:
        producer.params["input_residency"] = "file"

    action = getattr(stage, "action", None)
    if action == "filter":
        producer.params["action"] = "annotate"
        selector = _selector_for(stage, declaration, card, scope=scope)
        if selector is None:
            return None
        stages.insert(stage_index + 1, selector)
    elif action == "annotate" and getattr(stage, "mode", None) == "auto":
        pass
    else:
        return None
    return _copy_recipe(recipe, stages).freeze()


def _selector_for(  # noqa: C901, PLR0911, PLR0912 - fail closed per card field
    stage: Any,  # noqa: ANN401
    declaration: dict[str, Any],
    card: dict[str, Any],
    *,
    scope: str,
) -> StageRef | None:
    selector = declaration.get("selector")
    if not isinstance(selector, dict):
        return None
    selector_ref = selector.get("stage_id") or selector.get("stage")
    if not isinstance(selector_ref, str) or not selector_ref:
        return None
    required_operator = selector.get("required_operator")
    if not isinstance(required_operator, str) or not required_operator:
        return None

    kind = declaration.get("kind", "scalar")
    conditions: list[dict[str, Any]] = []
    if kind == "scalar":
        score_param = declaration.get("score_key_param")
        threshold_param = declaration.get("threshold_param")
        if not threshold_param:
            metrics = card.get("metrics")
            metric_entries = (
                [entry for entry in metrics.values() if isinstance(entry, dict)] if isinstance(metrics, dict) else []
            )
            threshold_params = {
                entry.get("threshold_param") for entry in metric_entries if entry.get("threshold_param")
            }
            if len(threshold_params) != 1:
                return None
            threshold_param = next(iter(threshold_params))
        score_key = getattr(stage, str(score_param), None)
        threshold = getattr(stage, str(threshold_param), None)
        if (
            not isinstance(score_key, str)
            or not score_key
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
        ):
            return None
        conditions.append(
            {
                "input_value_key": score_key,
                "target_value": threshold,
                "operator": required_operator,
            }
        )
    elif kind == "compound":
        dimensions = declaration.get("dimensions")
        if not isinstance(dimensions, list):
            return None
        for dimension in dimensions:
            if not isinstance(dimension, dict):
                return None
            threshold = getattr(stage, str(dimension.get("threshold_param") or ""), None)
            if threshold is None:
                continue
            score_key = getattr(stage, str(dimension.get("score_key_param") or ""), None)
            if (
                not isinstance(score_key, str)
                or not score_key
                or isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
            ):
                return None
            conditions.append(
                {
                    "input_value_key": score_key,
                    "target_value": threshold,
                    "operator": required_operator,
                }
            )
        if not conditions:
            return None
    else:
        return None

    params: dict[str, Any]
    if kind == "scalar" and scope == "task":
        params = {
            str(selector.get("key_param") or "input_value_key"): conditions[0]["input_value_key"],
            str(selector.get("value_param") or "target_value"): conditions[0]["target_value"],
            str(selector.get("operator_param") or "operator"): required_operator,
        }
    else:
        params = {
            str(selector.get("conditions_param") or "conditions"): conditions,
        }
        params[str(selector.get("condition_logic_param") or "condition_logic")] = selector.get(
            "required_condition_logic", "and"
        )
    params[str(selector.get("missing_policy_param") or "missing_value_policy")] = selector.get(
        "required_missing_policy", "drop"
    )
    if scope == "segments":
        source_param = str(selector.get("items_key_source_param") or "")
        items_key = getattr(stage, source_param, None)
        if not isinstance(items_key, str) or not items_key:
            return None
        params[str(selector.get("items_key_param") or "items_key")] = items_key
        params[str(selector.get("empty_policy_param") or "drop_parent_if_empty")] = selector.get(
            "required_empty_policy"
        )
    return StageRef(ref=selector_ref, params=params)


def _checkpoint_proof(
    alternative: Recipe,
    producer_index: int,
) -> tuple[Any, dict[str, Any]] | None:
    from nemo_curator.audio_agent import reusable_pipeline

    pairs, _ = reusable_pipeline._decision_pairs(
        alternative,
        decision_stage=alternative.stages[producer_index].ref,
    )
    pair = next(
        (item for item in pairs if item.producer_index == producer_index),
        None,
    )
    if pair is None:
        return None
    result = reusable_pipeline.plan(alternative)
    checkpoint = next(
        (
            candidate
            for candidate in result.get("candidates", [])
            if candidate.get("producer_stage") == pair.producer_stage
            and candidate.get("selector_stage") == pair.selector_stage
            and candidate.get("status") in {"needs_output_path", "configured"}
            and candidate.get("recommended") is True
        ),
        None,
    )
    return (pair, checkpoint) if checkpoint is not None else None


def _file_backed_live_waveform_alternative(  # noqa: PLR0913 - proof inputs stay explicit
    recipe: Recipe,
    *,
    stage_index: int,
    stage: Any,  # noqa: ANN401
    decision: dict[str, Any],
    card: dict[str, Any],
    scope: str,
) -> Recipe | None:
    """Retry only when an existing persisted transform can drop its live copy."""
    if scope != "task":
        return None
    stages = [StageRef(ref=item.ref, params=dict(item.params)) for item in recipe.stages]
    producer_params = stages[stage_index].params
    if producer_params.get("input_residency", getattr(stage, "input_residency", None)) not in {
        "file",
        "waveform",
        "auto",
    }:
        return None
    persisted_index = next(
        (
            index
            for index in range(stage_index - 1, -1, -1)
            if stages[index].params.get("keep_waveform_in_task") is True
            and stages[index].params.get("write_to_disk") is True
            and stages[index].params.get("update_audio_filepath") is True
        ),
        None,
    )
    if persisted_index is None:
        return None
    stages[persisted_index].params["keep_waveform_in_task"] = False
    base = _copy_recipe(recipe, stages)
    return _exact_alternative(
        base,
        stage_index=stage_index,
        stage=stage,
        decision=decision,
        card=card,
        scope=scope,
        force_file=True,
    )
