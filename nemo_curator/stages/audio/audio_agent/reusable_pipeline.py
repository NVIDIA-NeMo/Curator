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
leaves the decision parameter wholly below the persisted producer identity."""

from __future__ import annotations

from typing import Any

from nemo_curator.stages.audio.audio_agent import (
    _checkpoint_candidates,
    _checkpoint_compound,
    _checkpoint_decisions,
    _checkpoint_types,
)
from nemo_curator.stages.audio.audio_agent._checkpoint_types import DecisionPair
from nemo_curator.stages.audio.audio_agent.recipe import (
    REUSABLE_CHECKPOINT_PROVENANCE,
    Recipe,
    StageRef,
)


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
    pairs, rejected = _checkpoint_decisions._decision_pairs(rec, decision_stage=decision_stage)
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
                    _checkpoint_candidates._rejected(
                        pair,
                        "compound decisions cannot be tuned with scalar decision_value; "
                        "submit an explicit complete conditions surface instead",
                    )
                )
                continue
            value_reason = _checkpoint_decisions._decision_value_reason(
                decision_value,
                operator=pair.operator,
                value_type=pair.value_type,
            )
            if value_reason:
                rejected.append(_checkpoint_candidates._rejected(pair, value_reason))
                continue
        if decision_conditions is not None:
            if pair.decision_kind != "compound":
                rejected.append(
                    _checkpoint_candidates._rejected(
                        pair,
                        "decision_conditions is supported only for a card-declared compound decision",
                    )
                )
                continue
            normalized, conditions_reason = _checkpoint_compound._normalize_decision_conditions(
                rec,
                pair,
                decision_conditions,
            )
            if conditions_reason:
                rejected.append(_checkpoint_candidates._rejected(pair, conditions_reason))
                continue
            selected = _checkpoint_compound._with_decision_conditions(rec, pair, normalized)
            refreshed, refresh_reason = _checkpoint_compound._refresh_compound_pair(
                selected,
                pair,
                normalized,
            )
            if refreshed is None:
                rejected.append(_checkpoint_candidates._rejected(pair, refresh_reason))
                continue
        else:
            selected = (
                _checkpoint_candidates._with_decision_value(rec, pair, decision_value)
                if decision_value is not None
                else rec
            )
            refreshed = _checkpoint_candidates._refresh_pair(selected, pair)
            if refreshed is None:
                rejected.append(
                    _checkpoint_candidates._rejected(
                        pair, "changing the decision made its declared selector unresolvable"
                    )
                )
                continue
        existing = _checkpoint_candidates._checkpoint_between(
            selected, refreshed.producer_index, refreshed.selector_index
        )
        if existing is not None:
            configured = _checkpoint_candidates._with_checkpoint_provenance(selected, existing)
            reason = _checkpoint_candidates._configured_checkpoint_path_reason(configured, existing)
            if reason:
                rejected.append(_checkpoint_candidates._rejected(refreshed, reason))
                continue
            if decision_conditions is not None:
                evidence_reason = _checkpoint_compound._checkpoint_condition_evidence_reason(
                    configured,
                    checkpoint_index=existing,
                    pair=refreshed,
                )
                if evidence_reason:
                    rejected.append(_checkpoint_candidates._rejected(refreshed, evidence_reason))
                    continue
            baseline_without_checkpoint = _checkpoint_candidates._without_stage(configured, existing)
            legal, why = _checkpoint_candidates._validate_candidate(
                baseline_without_checkpoint,
                configured,
                checkpoint_index=existing,
            )
            if not legal:
                rejected.append(_checkpoint_candidates._rejected(refreshed, why))
                continue
            candidates.append(
                _checkpoint_candidates._covered_candidate(
                    configured,
                    refreshed,
                    checkpoint_index=existing,
                    decision_changed=(decision_value is not None or decision_conditions is not None),
                )
            )
            continue
        chosen_path, path_source = output_path, "explicit"
        if not chosen_path:
            probe = _checkpoint_candidates._insert_checkpoint(
                selected,
                index=refreshed.selector_index,
                output_path=_checkpoint_types._ANALYSIS_PATH,
                retention_sec=retention_sec,
                owner=owner,
            )
            legal, why = _checkpoint_candidates._validate_candidate(
                selected,
                probe,
                checkpoint_index=refreshed.selector_index,
            )
            if not legal:
                rejected.append(_checkpoint_candidates._rejected(refreshed, why))
                continue
            path_source = "derived"
            chosen_path = _checkpoint_candidates._derived_path(
                probe,
                checkpoint_index=refreshed.selector_index,
                dataset_key=dataset_key,
            )
            if not chosen_path:
                expensive_prefix = _checkpoint_candidates._expensive_prefix(selected, refreshed.selector_index)
                candidates.append(
                    {
                        "id": f"checkpoint-after-{refreshed.producer_index}",
                        "status": "needs_output_path",
                        "producer_stage": refreshed.producer_stage,
                        "selector_stage": refreshed.selector_stage,
                        **_checkpoint_candidates._decision_identity(refreshed),
                        "checkpoint_index": refreshed.selector_index,
                        "expensive_prefix": expensive_prefix,
                        "recommended": bool(expensive_prefix),
                        "cost_evidence": _checkpoint_candidates._static_cost_evidence(
                            selected, refreshed.selector_index
                        ),
                        "trust": _checkpoint_candidates._prefix_trust(probe, refreshed.selector_index + 1),
                        "residency": {
                            "format": "complete serializable task.data JSONL",
                            "waveform_persisted": False,
                            "suffix_survives_metadata_boundary": True,
                        },
                        "next": "call plan_checkpoint again with output_path to materialize and validate this candidate",
                    }
                )
                continue
        reason = _checkpoint_candidates._path_reason(selected, chosen_path, derived=path_source == "derived")
        if reason:
            rejected.append(_checkpoint_candidates._rejected(refreshed, reason))
            continue
        candidate = _checkpoint_candidates._insert_checkpoint(
            selected,
            index=refreshed.selector_index,
            output_path=chosen_path,
            retention_sec=retention_sec,
            owner=owner,
        )
        legal, why = _checkpoint_candidates._validate_candidate(
            selected, candidate, checkpoint_index=refreshed.selector_index
        )
        if not legal:
            rejected.append(_checkpoint_candidates._rejected(refreshed, why))
            continue
        candidates.append(
            _checkpoint_candidates._candidate(
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
        candidate.get("status") in _checkpoint_types._UNDECIDED and candidate.get("recommended") is True
        for candidate in candidates
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
        if candidate.get("status") in _checkpoint_types._UNDECIDED
        and candidate.get("recommended") is True
        and candidate.get("id")
    )


def with_declined_checkpoint(recipe: Recipe, candidate_ids: list[str]) -> Recipe:
    """Bind an explicit baseline choice to this exact recipe and option set."""
    rec = _checkpoint_candidates._copy_recipe(
        recipe.freeze(), [StageRef(ref=stage.ref, params=dict(stage.params)) for stage in recipe.stages]
    ).freeze()
    rec.checkpoint_decision = {
        "status": _checkpoint_types._DECLINED,
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
        and decision.get("status") == _checkpoint_types._DECLINED
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


__all__ = [
    "DecisionPair",
    "checkpoint_decision_requirement",
    "plan",
    "recommended_candidate_ids",
    "with_declined_checkpoint",
]
