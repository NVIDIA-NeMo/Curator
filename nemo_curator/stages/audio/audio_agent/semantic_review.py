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

"""Build evidence for a host-LLM semantic review of a configured recipe.

This module deliberately does not judge user intent or recipe correctness.  It
uses configured, dynamic stage contracts to co-locate exact field lineage,
cardinality changes, and the relevant capability-card prose.  The host LLM is
the reviewer; this packet is only its read-only evidence."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from nemo_curator.stages.audio.audio_agent import _semantic_lineage, _semantic_stages, _semantic_values

_RESPONSE_SECTIONS: tuple[str, ...] = (
    "stage_reviews",
    "field_reviews",
    "behavior_checks",
    "transform_checks",
    "model_checks",
    "assumptions_or_questions",
)


def semantic_response_contract() -> dict[str, Any]:
    """Canonical host response shape for every semantic-review packet.

    ``mechanically_runnable`` is copied from the containing deterministic
    Verdict.  This module deliberately cannot fill it in: semantic evidence is
    also useful to SDK callers that construct it without a Verdict.
    """
    return {
        "schema_version": 2,
        "mechanically_runnable": {
            "type": "boolean",
            "source": "containing_verdict.runnable",
            "required": True,
        },
        "recipe_config_hash": {
            "type": "string",
            "source": "semantic_review.recipe.config_hash",
            "required": True,
            "rule": "Copy exactly; a different recipe requires another validation and critique.",
        },
        "intent_status": ["pass", "revise", "ask"],
        "sections": list(_RESPONSE_SECTIONS),
        "rule": (
            "Only mechanically_runnable=true, an exact copied recipe_config_hash, "
            "and intent_status=pass may proceed to smoke; revise, ask, or a recipe "
            "hash change requires another validate and semantic review."
        ),
    }


def build_semantic_review(  # noqa: C901, PLR0912, PLR0915
    stages: Sequence[Any],
    *,
    initial_keys: Sequence[str] | set[str] | None = None,
    recipe: Any = None,  # noqa: ANN401 - accepts Recipe or its JSON-safe mapping
    data_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe, advisory-only semantic-review evidence packet.

    Args:
        stages: Already-configured stage instances, in execution order.
        initial_keys: Exact top-level keys proven to exist on the input tasks.
        recipe: Optional Recipe/mapping, used only for authored refs and identity.
        data_profile: Optional profiler facts, reduced to a redacted semantic summary.

    The function never executes a stage, interprets user intent, changes a recipe,
    or returns a runnable/refusal verdict.
    """
    from nemo_curator.stages.audio import agent as foundation

    configured_stages = list(stages)
    (
        execution_groups,
        recipe_stage_views,
        composite_views,
        contract_issues,
        missing_card_semantics,
    ) = _semantic_stages._expand_configured_stages(
        configured_stages,
        recipe=recipe,
        foundation=foundation,
    )
    stage_packets: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    cardinality_seams: list[dict[str, Any]] = []
    unresolved_lineage: list[dict[str, Any]] = []
    semantic_evidence_gaps: list[dict[str, Any]] = []

    # Active writers reflect contract-declared key preservation/removal. Complete
    # history remains available to explain why a formerly-produced key is no
    # longer an active lineage source.
    active_writers: dict[tuple[str, str], dict[str, Any]] = {}
    writer_history: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key in sorted({str(key) for key in (initial_keys or ())}):
        active_writers[("task", key)] = {
            "kind": "initial_input",
            "key": key,
            "scope": "task",
            "stage_index": None,
            "basis": "declared_initial_key",
        }
    declared_initial_writers = copy.deepcopy(active_writers)

    for group in execution_groups:
        # ``initial_keys`` describe configured source output, never its EmptyTask
        # input. Strip only those declared keys (not real upstream producers) so they
        # don't appear as inputs to a source's own inner leaves -- a second source must
        # not wipe the keys an earlier source already produced.
        if group["is_source"]:
            for slot in declared_initial_writers:
                # Match on what currently OCCUPIES the slot, not on the slot's name. The
                # intent stated above is to strip declared keys and spare real upstream
                # producers, but a real producer writing the same key name lands in the very
                # same slot -- so a second source silently deleted the lineage of a key an
                # earlier stage had genuinely produced, and every downstream read of it then
                # looked like it came from nowhere.
                if (active_writers.get(slot) or {}).get("basis") == "declared_initial_key":
                    active_writers.pop(slot, None)

        for leaf in group["leaves"]:
            stage = leaf["stage"]
            stage_index = int(leaf["stage_index"])
            stage_ref = str(leaf["stage_ref"])
            provenance = dict(leaf["provenance"])
            material = _semantic_stages._semantic_material(stage_ref)
            gap = _semantic_stages._card_semantic_gap(
                material,
                stage_index=stage_index,
                stage=stage_ref,
                evidence_view="execution_leaf",
                provenance=provenance,
            )
            if gap is not None:
                missing_card_semantics.append(gap)
            try:
                contract = foundation.build_contract(stage)
            except Exception as exc:  # noqa: BLE001 - evidence remains JSON-safe and partial
                contract_issues.append(
                    {
                        "stage_index": stage_index,
                        "stage": stage_ref,
                        "provenance": provenance,
                        "code": "contract_unavailable",
                        "message": _semantic_stages._safe_error(exc),
                    }
                )
                stage_packets.append(
                    {
                        "stage_index": stage_index,
                        "stage": stage_ref,
                        "provenance": provenance,
                        "contract_status": "unavailable",
                        "cardinality": "unknown",
                        "iteration_key": None,
                        "preserves_upstream_keys": None,
                        "wrappable": None,
                        "accepts_task_type": None,
                        "produces_task_type": None,
                        "configured_params": _semantic_values._best_effort_instance_params(
                            stage,
                            source=("composite_effective" if provenance["execution_path"] else "configured_instance"),
                        ),
                        "reads": [],
                        "writes": [],
                        "removes_keys": [],
                        "invalidates_keys": [],
                        "invalidates_prior_writes": {"task": [], "segment": []},
                        "semantic_material": material,
                    }
                )
                # Without a contract we cannot prove this leaf preserved earlier keys.
                active_writers.clear()
                continue

            is_composite_leaf = bool(provenance["execution_path"])
            authored = (
                {}
                if is_composite_leaf
                else _semantic_values._recipe_stage_params(
                    recipe,
                    int(provenance["recipe_stage_index"]),
                )
            )
            implicit_source = (
                "composite_effective"
                if is_composite_leaf
                else ("configured_instance" if recipe is None else "default")
            )
            key_params = _semantic_stages._configured_key_params(stage, contract)
            configured_params = _semantic_values._configured_params(
                stage,
                contract,
                authored,
                implicit_source=implicit_source,
            )
            reads = _semantic_lineage._contract_reads(contract, key_params)
            writes = _semantic_lineage._contract_writes(contract, key_params)
            prior_writes = getattr(contract, "invalidates_prior_writes", None)
            stage_info = {
                "stage_index": stage_index,
                "stage": stage_ref,
                "provenance": provenance,
                "contract_status": "configured",
                "cardinality": str(getattr(contract, "cardinality", "1:1")),
                "iteration_key": getattr(contract, "iteration_key", None),
                "preserves_upstream_keys": bool(getattr(contract, "preserves_upstream_keys", True)),
                "preserves_upstream_segment_keys": bool(getattr(contract, "preserves_upstream_segment_keys", True)),
                "wrappable": bool(getattr(contract, "wrappable", True)),
                "accepts_task_type": getattr(contract, "accepts_task_type", None),
                "produces_task_type": getattr(contract, "produces_task_type", None),
                "configured_params": configured_params,
                "reads": reads,
                "writes": writes,
                "removes_keys": [str(key) for key in (getattr(contract, "removes_keys", ()) or ())],
                "invalidates_keys": [str(key) for key in (getattr(contract, "invalidates_keys", ()) or ())],
                "invalidates_prior_writes": {
                    "task": list(getattr(prior_writes, "data_keys", ())),
                    "segment": list(getattr(prior_writes, "segment_data_keys", ())),
                },
                "semantic_material": material,
            }
            stage_packets.append(stage_info)

            consumer = _semantic_lineage._stage_endpoint(stage_info)
            for read in reads:
                slot = (read["scope"], read["key"])
                latest = copy.deepcopy(active_writers.get(slot))
                if latest is None and read.get("requirement") == "optional":
                    continue
                if latest is None:
                    latest = {
                        "kind": "unresolved",
                        "key": read["key"],
                        "scope": read["scope"],
                        "stage_index": None,
                        "basis": "no_active_contract_writer_or_declared_initial_key",
                    }
                history = copy.deepcopy(writer_history.get(slot, []))
                earlier = history[:-1] if latest.get("kind") == "stage" and history else history
                edge = {
                    "consumer": copy.deepcopy(consumer),
                    "read": copy.deepcopy(read),
                    "latest_upstream_producer": latest,
                    "earlier_contract_writers": earlier,
                    "crossed_cardinality_seams": _semantic_lineage._crossed_seams(
                        cardinality_seams,
                        latest,
                        stage_index,
                    ),
                }
                if latest["kind"] == "initial_input":
                    edge["semantic_provenance"] = {
                        "status": "unresolved_source_schema",
                        "reason": ("initial key presence does not establish meaning, unit, entity, or granularity"),
                    }
                    semantic_evidence_gaps.append(
                        {
                            "code": "initial_key_semantics_unresolved",
                            "key": read["key"],
                            "scope": read["scope"],
                            "consumer": copy.deepcopy(consumer),
                            "source": copy.deepcopy(latest),
                            "message": (
                                "the profiler/initial contract proves this key exists, "
                                "but no source-schema semantic provenance was supplied"
                            ),
                        }
                    )
                lineage.append(edge)
                if latest["kind"] == "unresolved":
                    unresolved_lineage.append(copy.deepcopy(edge))

            active_before_stage = copy.deepcopy(active_writers)
            for scope, keys in stage_info["invalidates_prior_writes"].items():
                for key in keys:
                    active_writers.pop((scope, key), None)
                    active_before_stage.pop((scope, key), None)
            if not stage_info["preserves_upstream_keys"]:
                # ``preserves_upstream_keys`` describes task.data reconstruction.
                # Task metadata is a separate channel and remains live unless a
                # future contract explicitly declares metadata removal.
                active_writers = {slot: writer for slot, writer in active_writers.items() if slot[0] == "metadata"}
            elif not stage_info["preserves_upstream_segment_keys"]:
                active_writers = {slot: writer for slot, writer in active_writers.items() if slot[0] != "segment"}
            for key in stage_info["removes_keys"]:
                active_writers.pop(("task", key), None)
            for key in stage_info["invalidates_keys"]:
                active_writers.pop(("task", key), None)
            for write in writes:
                slot = (write["scope"], write["key"])
                conditions = list(write.get("conditions", []))
                origins = {str(condition.get("value_origin", "stage_generated")) for condition in conditions}
                prior = copy.deepcopy(active_before_stage.get(slot))
                passthrough_conditions = [
                    copy.deepcopy(condition)
                    for condition in conditions
                    if condition.get("value_origin") == "upstream_same_key"
                ]

                # An unchanged pass-through is not a new producer. Preserve the
                # exact prior producer across a non-preserving rebuild and add
                # traversal evidence for the critic.
                if origins == {"upstream_same_key"}:
                    if prior is not None:
                        preserved = copy.deepcopy(prior)
                        preserved.setdefault("conditional_passthroughs", []).append(
                            {
                                "through": _semantic_lineage._stage_endpoint(stage_info),
                                "conditions": passthrough_conditions,
                                "basis": "configured_conditional_write",
                            }
                        )
                        active_writers[slot] = preserved
                    continue

                writer = _semantic_lineage._writer_record(stage_info, write)
                if passthrough_conditions:
                    writer["alternative_upstream_same_key"] = {
                        "conditions": passthrough_conditions,
                        "producer": prior
                        or {
                            "kind": "unresolved",
                            "key": write["key"],
                            "scope": write["scope"],
                            "stage_index": None,
                            "basis": "no_active_same_key_writer",
                        },
                    }
                same_key_relationships = [
                    copy.deepcopy(condition)
                    for condition in conditions
                    if condition.get("value_origin")
                    in {
                        "augments_upstream_same_key",
                        "transforms_upstream_same_key",
                    }
                ]
                if same_key_relationships:
                    writer["same_key_upstream"] = {
                        "relationships": same_key_relationships,
                        "producer": prior
                        or {
                            "kind": "unresolved",
                            "key": write["key"],
                            "scope": write["scope"],
                            "stage_index": None,
                            "basis": "no_active_same_key_writer",
                        },
                    }
                if (
                    write.get("certainty") == "conditional"
                    and prior is not None
                    and (
                        write["scope"] == "metadata"
                        or (
                            stage_info["preserves_upstream_keys"]
                            and (write["scope"] != "segment" or stage_info["preserves_upstream_segment_keys"])
                        )
                    )
                ):
                    writer["when_condition_not_met"] = {
                        "outcome": "upstream_same_key_value_remains",
                        "producer": prior,
                    }
                active_writers[slot] = writer
                writer_history.setdefault(slot, []).append(writer)

            seam = _semantic_lineage._seam_record(stage_info)
            if seam is not None:
                cardinality_seams.append(seam)

            # A non-composite leaf that still declares itself opaque cannot be
            # inspected recursively. Keep the lineage barrier explicit.
            if not stage_info["wrappable"]:
                contract_issues.append(
                    {
                        "stage_index": stage_index,
                        "stage": stage_ref,
                        "provenance": provenance,
                        "code": "opaque_execution_leaf",
                        "message": (
                            "an execution leaf declares wrappable=false and exposes no further configured lineage"
                        ),
                    }
                )
                active_writers.clear()

        if group["is_source"]:
            # Preserve concrete leaf writers, then add only profiled source keys
            # the execution contracts did not explicitly declare.
            for slot, origin in declared_initial_writers.items():
                if slot in active_writers:
                    continue
                marked = copy.deepcopy(origin)
                marked["basis"] = "declared_post_source_initial_key"
                marked["visibility_uncertainty"] = {
                    "recipe_stage_index": group["recipe_stage_index"],
                    "recipe_stage_ref": group["recipe_stage_ref"],
                    "reason": "profiled_source_key_without_declared_semantic_schema",
                }
                active_writers[slot] = marked

    recipe_stage_count: int | None = None
    entries = getattr(recipe, "stages", None)
    if entries is None and isinstance(recipe, Mapping):
        entries = recipe.get("stages")
    if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
        recipe_stage_count = len(entries)
    if recipe_stage_count is not None and recipe_stage_count != len(configured_stages):
        contract_issues.append(
            {
                "code": "recipe_stage_count_mismatch",
                "message": (
                    f"recipe has {recipe_stage_count} stage refs but "
                    f"{len(configured_stages)} configured stages were supplied"
                ),
            }
        )

    recipe_identity = _semantic_values._recipe_identity(recipe, len(configured_stages))
    recipe_identity["execution_leaf_count"] = len(stage_packets)
    if configured_stages and not recipe_identity.get("config_hash"):
        contract_issues.append(
            {
                "code": "recipe_identity_unavailable",
                "message": (
                    "no canonical recipe config_hash is available; the host cannot "
                    "bind its semantic critique to this configured candidate"
                ),
            }
        )

    checklist = _semantic_lineage._checklist(
        stages=stage_packets,
        recipe_stages=recipe_stage_views,
        composites=composite_views,
        lineage=lineage,
        seams=cardinality_seams,
        unresolved=unresolved_lineage,
        missing_cards=missing_card_semantics,
        semantic_gaps=semantic_evidence_gaps,
        contract_issues=contract_issues,
    )
    incomplete = bool(contract_issues or missing_card_semantics or semantic_evidence_gaps)
    return {
        "status": "partial" if incomplete else "complete",
        "review_required": bool(configured_stages),
        "advisory_only": True,
        "intent_interpretation_performed": False,
        "required_response": semantic_response_contract(),
        "recipe": recipe_identity,
        "data_profile": _semantic_values._data_profile_summary(data_profile),
        "recipe_stages": recipe_stage_views,
        "composites": composite_views,
        "stages": stage_packets,
        "lineage": lineage,
        "cardinality_seams": cardinality_seams,
        "unresolved_lineage": unresolved_lineage,
        "semantic_evidence_gaps": semantic_evidence_gaps,
        "missing_card_semantics": missing_card_semantics,
        "contract_issues": contract_issues,
        "checklist": checklist,
    }
