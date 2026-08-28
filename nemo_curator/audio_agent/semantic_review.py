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

"""Build evidence for a host-LLM semantic review of a configured recipe.

This module deliberately does not judge user intent or recipe correctness.  It
uses configured, dynamic stage contracts to co-locate exact field lineage,
cardinality changes, and the relevant capability-card prose.  The host LLM is
the reviewer; this packet is only its read-only evidence.
"""

# ruff: noqa: B023 - the `visit` closure captures per-stage loop vars but is invoked
# synchronously in the same iteration (see the immediate `visit(configured_stage, ...)` call),
# never stored, so the late binding B023 warns about cannot occur here.

from __future__ import annotations

import contextlib
import copy
import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

from nemo_curator.audio_agent.index import get_index

_SEMANTIC_CARD_FIELDS: tuple[str, ...] = (
    "summary",
    "tags",
    "model_id",
    "model_version",
    "domain",
    "constraints",
    "resource",
    "use_cases",
    "composition",
    "params_of_note",
    "param_dependencies",
    "notes",
    "caveats",
    "comparison",
    "metrics",
    "semantic_facts",
    "verified",
    "provenance",
)

_MAX_EXECUTION_LEAVES = 512

_CARDINALITY_SEAM_KINDS: dict[str, str] = {
    "1:N fan-out": "fan_out",
    "N:1": "aggregation",
    "1:1 nested-list": "nested_collection",
    "filter": "filter",
}

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


def _json_safe(value: Any) -> Any:  # noqa: ANN401
    """Recursively preserve JSON values and stringify an unexpected leaf."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item) for item in sorted(value, key=repr)]
    return str(value)


def _recipe_stage_ref(recipe: Any, index: int, stage: Any) -> str:  # noqa: ANN401
    """Resolve the authored recipe ref without requiring a concrete Recipe type."""
    entries: Any = getattr(recipe, "stages", None)
    if entries is None and isinstance(recipe, Mapping):
        entries = recipe.get("stages")
    if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)) and index < len(entries):
        entry = entries[index]
        ref = getattr(entry, "ref", None)
        if ref is None and isinstance(entry, Mapping):
            ref = entry.get("ref")
        if ref:
            return str(ref)
    return type(stage).__name__


def _recipe_stage_params(recipe: Any, index: int) -> dict[str, Any]:  # noqa: ANN401
    """Return the authored params for one recipe entry, if available."""
    entries: Any = getattr(recipe, "stages", None)
    if entries is None and isinstance(recipe, Mapping):
        entries = recipe.get("stages")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)) or index >= len(entries):
        return {}
    entry = entries[index]
    params = getattr(entry, "params", None)
    if params is None and isinstance(entry, Mapping):
        params = entry.get("params")
    return dict(params) if isinstance(params, Mapping) else {}


def _jsonable(value: Any) -> Any:  # noqa: ANN401
    """Bound configured values to JSON-safe data without executing anything."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        with contextlib.suppress(Exception):
            return _jsonable(value.to_dict())
    return f"<{type(value).__name__}>"


def _configured_params(
    stage: Any,  # noqa: ANN401
    contract: Any,  # noqa: ANN401
    authored: dict[str, Any],
    *,
    implicit_source: str = "default",
) -> dict[str, dict[str, Any]]:
    """Effective parameter evidence, preserving whether a value was explicit."""
    from nemo_curator.audio_agent import _safety

    configured: dict[str, dict[str, Any]] = {}
    missing = object()
    specs = {str(getattr(param, "name", "")): param for param in (getattr(contract, "params", ()) or ())}
    for name, spec in specs.items():
        if not name:
            continue
        if name in authored:
            value = authored[name]
            source = "recipe"
        else:
            try:
                value = getattr(stage, name)
            except Exception:  # noqa: BLE001 - advisory inspection stays best-effort
                value = missing
            if callable(value):
                callable_name = getattr(value, "__name__", None)
                choices = getattr(spec, "choices", None) or ()
                value = (
                    callable_name
                    if isinstance(callable_name, str)
                    and (callable_name in choices or str(getattr(spec, "type", "")) == "str")
                    else missing
                )
            if value is missing:
                value = getattr(spec, "default", None)
            source = implicit_source
        configured[name] = {"value": _jsonable(value), "source": source}
    # Framework knobs may not appear in the stage constructor contract but still
    # affect the authored plan. Include them as explicit evidence.
    for name, value in authored.items():
        configured.setdefault(str(name), {"value": _jsonable(value), "source": "recipe"})
    return _safety.redact(configured)


def _best_effort_instance_params(
    stage: Any,  # noqa: ANN401
    *,
    source: str,
) -> dict[str, dict[str, Any]]:
    """Configured values for a leaf whose AgentReady contract is unavailable."""
    from nemo_curator.audio_agent import _safety

    names: list[str] = []
    if dataclasses.is_dataclass(stage) and not isinstance(stage, type):
        names = [field.name for field in dataclasses.fields(stage) if field.init and not field.name.startswith("_")]
    else:
        with contextlib.suppress(TypeError, ValueError):
            import inspect

            names = [
                name
                for name, param in inspect.signature(type(stage).__init__).parameters.items()
                if name != "self"
                and not name.startswith("_")
                and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
            ]
    excluded = {"name", "resources", "batch_size", "runtime_env", "num_workers"}
    configured: dict[str, dict[str, Any]] = {}
    for name in names:
        if name in excluded:
            continue
        with contextlib.suppress(Exception):
            value = getattr(stage, name)
            if not callable(value):
                configured[name] = {"value": _jsonable(value), "source": source}
    return _safety.redact(configured)


def _recipe_value(recipe: Any, name: str, default: Any = None) -> Any:  # noqa: ANN401
    value = getattr(recipe, name, default)
    if isinstance(recipe, Mapping):
        value = recipe.get(name, value)
    return value


def _recipe_identity(recipe: Any, stage_count: int) -> dict[str, Any]:  # noqa: ANN401
    from nemo_curator.audio_agent import _safety

    if recipe is None:
        return {"stage_count": stage_count}
    authored_config_hash = _recipe_value(recipe, "config_hash")
    config_hash: str | None = None
    hash_source: str | None = None
    compute_hash = getattr(recipe, "compute_hash", None)
    if callable(compute_hash):
        with contextlib.suppress(Exception):
            config_hash = str(compute_hash())
            hash_source = "computed_canonical_recipe"
    elif isinstance(recipe, Mapping):
        # Public callers may pass the same JSON-safe recipe mapping accepted by
        # validate. Compute its portable canonical hash without mutating it.
        with contextlib.suppress(Exception):
            from nemo_curator.audio_agent.recipe import Recipe

            config_hash = Recipe.from_dict(dict(recipe)).compute_hash()
            hash_source = "computed_canonical_recipe"
    if config_hash is None and authored_config_hash:
        # This fallback is useful to non-Recipe SDK callers, but is explicitly
        # labeled unverified rather than silently treating a supplied value as
        # the canonical identity.
        config_hash = str(authored_config_hash)
        hash_source = "provided_unverified"
    recipe_id = _recipe_value(recipe, "recipe_id")
    out: dict[str, Any] = {"stage_count": stage_count}
    if recipe_id:
        out["recipe_id"] = str(recipe_id)
    if config_hash:
        out["config_hash"] = config_hash
        out["config_hash_source"] = hash_source
    if authored_config_hash and config_hash and str(authored_config_hash) != config_hash:
        out["authored_config_hash_mismatch"] = True
    for name in ("preset", "rationale", "acceptance_criteria", "config_strategy"):
        value = _recipe_value(recipe, name)
        if value not in (None, "", [], {}):
            out[name] = _jsonable(value)
    return _safety.redact(out)


def _data_profile_summary(data_profile: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return only bounded, critique-relevant profile facts (never examples/content)."""
    if not isinstance(data_profile, Mapping):
        return None
    from nemo_curator.audio_agent import _safety

    fields = (
        "kind",
        "num_files",
        "sample_rates",
        "channels",
        "total_duration_sec",
        "mean_duration_sec",
        "codecs",
        "has_transcripts",
        "manifest_keys",
        "unreadable",
        "source_errors",
        "notes",
        "fingerprint_tier",
    )
    summary = {name: _jsonable(data_profile[name]) for name in fields if name in data_profile}
    return _safety.redact(summary)


def _safe_error(exc: BaseException) -> str:
    from nemo_curator.audio_agent import _safety

    return _safety.redact_secret_text(f"{type(exc).__name__}: {exc}")


def _semantic_material(stage_ref: str) -> dict[str, Any]:
    """Copy card prose verbatim so callers cannot mutate the cached index."""
    card = get_index().card(stage_ref)
    if not isinstance(card, dict):
        return {
            "available": False,
            "stage_id": stage_ref,
            "semantic_facts_status": "card_absent",
        }
    semantic_facts = card.get("semantic_facts")
    has_semantic_facts = (
        bool(semantic_facts.strip())
        if isinstance(semantic_facts, str)
        else isinstance(semantic_facts, Mapping) and bool(semantic_facts)
    )
    material: dict[str, Any] = {
        "available": True,
        "stage_id": str(card.get("stage_id") or stage_ref),
        "category": card.get("category"),
        "semantic_facts_status": ("present" if has_semantic_facts else "semantic_facts_absent"),
    }
    for field in _SEMANTIC_CARD_FIELDS:
        if field in card:
            material[field] = _json_safe(copy.deepcopy(card[field]))
    return material


def _card_semantic_gap(
    material: Mapping[str, Any],
    *,
    stage_index: int | None,
    stage: str,
    evidence_view: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any] | None:
    status = str(material.get("semantic_facts_status") or "")
    if status == "present":
        return None
    reason = "card_absent" if not material.get("available") else "semantic_facts_absent"
    gap: dict[str, Any] = {
        "stage_index": stage_index,
        "stage": stage,
        "reason": reason,
        "evidence_view": evidence_view,
        "provenance": _json_safe(dict(provenance)),
    }
    if reason == "card_absent":
        gap["message"] = "no capability card is available for this configured stage"
    else:
        gap["message"] = "a capability card exists, but it does not declare semantic_facts"
    return gap


def _expand_configured_stages(  # noqa: C901, PLR0915
    configured_stages: Sequence[Any],
    *,
    recipe: Any,  # noqa: ANN401
    foundation: Any,  # noqa: ANN401
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Resolve the exact execution leaves without executing stage processing.

    Expansion intentionally mirrors the pipeline/planner boundary:
    ``CompositeStage.decompose_and_apply_with`` is used so composite ``with_``
    overrides are reflected in the evidence, and expansion is SINGLE-LEVEL because
    that is the only shape the executor runs.  An exception, empty decomposition,
    nested composite, or leaf-budget overflow produces no fabricated leaf for that
    branch and leaves the packet partial.
    """
    from nemo_curator.audio_agent import _safety
    from nemo_curator.stages.base import CompositeStage, ProcessingStage

    execution_groups: list[dict[str, Any]] = []
    recipe_stage_views: list[dict[str, Any]] = []
    composite_views: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    card_gaps: list[dict[str, Any]] = []
    execution_leaf_count = 0
    leaf_limit_reported = False

    def issue(  # noqa: PLR0913
        *,
        code: str,
        message: str,
        recipe_stage_index: int,
        recipe_stage_ref: str,
        execution_path: Sequence[int],
        stage: str,
    ) -> None:
        issues.append(
            {
                "code": code,
                "message": _safety.redact_secret_text(message),
                "recipe_stage_index": recipe_stage_index,
                "recipe_stage_ref": recipe_stage_ref,
                "execution_path": list(execution_path),
                "stage": stage,
            }
        )

    for recipe_stage_index, configured_stage in enumerate(configured_stages):
        recipe_stage_ref = _recipe_stage_ref(
            recipe,
            recipe_stage_index,
            configured_stage,
        )
        authored_params = _recipe_stage_params(recipe, recipe_stage_index)
        is_composite = isinstance(configured_stage, CompositeStage)
        group: dict[str, Any] = {
            "recipe_stage_index": recipe_stage_index,
            "recipe_stage_ref": recipe_stage_ref,
            "is_composite": is_composite,
            "is_source": False,
            "leaves": [],
        }
        recipe_view: dict[str, Any] = {
            "recipe_stage_index": recipe_stage_index,
            "recipe_stage_ref": recipe_stage_ref,
            "kind": "composite" if is_composite else "execution_leaf",
            "authored_params": _safety.redact(_jsonable(authored_params)),
            "execution_leaf_indices": [],
        }

        # The outer configured contract is useful even though a composite never
        # executes: it identifies source boundaries and its effective params.
        outer_contract: Any = None
        try:
            outer_contract = foundation.build_contract(configured_stage)
            group["is_source"] = getattr(outer_contract, "accepts_task_type", None) == "EmptyTask"
        except Exception as exc:  # noqa: BLE001 - evidence assembly is fail-closed
            if is_composite:
                issue(
                    code="composite_contract_unavailable",
                    message=_safe_error(exc),
                    recipe_stage_index=recipe_stage_index,
                    recipe_stage_ref=recipe_stage_ref,
                    execution_path=(),
                    stage=recipe_stage_ref,
                )

        def visit(  # noqa: C901, PLR0912
            stage: Any,  # noqa: ANN401
            *,
            path: tuple[int, ...],
        ) -> list[int]:
            nonlocal execution_leaf_count, leaf_limit_reported
            stage_name = type(stage).__name__
            provenance = {
                "recipe_stage_index": recipe_stage_index,
                "recipe_stage_ref": recipe_stage_ref,
                "execution_path": list(path),
            }
            if isinstance(stage, CompositeStage):
                composite_ref = recipe_stage_ref if not path else stage_name
                material = _semantic_material(composite_ref)
                gap = _card_semantic_gap(
                    material,
                    stage_index=None,
                    stage=composite_ref,
                    evidence_view="authored_composite" if not path else "nested_composite",
                    provenance=provenance,
                )
                if gap is not None:
                    card_gaps.append(gap)
                view: dict[str, Any] = {
                    **provenance,
                    "composite": composite_ref,
                    "authored": not path,
                    "authored_params": (_safety.redact(_jsonable(authored_params)) if not path else {}),
                    "configured_params": {},
                    "semantic_material": material,
                    "expansion_status": "partial",
                    "execution_leaf_indices": [],
                }
                try:
                    composite_contract = (
                        outer_contract if not path and outer_contract is not None else foundation.build_contract(stage)
                    )
                    view["configured_params"] = _configured_params(
                        stage,
                        composite_contract,
                        authored_params if not path else {},
                        implicit_source=(
                            "default"
                            if not path and recipe is not None
                            else ("configured_instance" if not path else "composite_effective")
                        ),
                    )
                except Exception as exc:  # noqa: BLE001 - expansion may still be inspectable
                    issue(
                        code="composite_contract_unavailable",
                        message=_safe_error(exc),
                        recipe_stage_index=recipe_stage_index,
                        recipe_stage_ref=recipe_stage_ref,
                        execution_path=path,
                        stage=composite_ref,
                    )
                composite_views.append(view)

                # The executor refuses a composite below the top level: ``_decompose_stages``
                # expands once and raises on a child that decomposes further. Reviewing it
                # would describe a plan that cannot run, so report the same limit the backend
                # enforces -- which also makes a depth bound and cycle check unreachable.
                if path:
                    issue(
                        code="nested_composite_unsupported",
                        message=(
                            "composite decomposition returned another composite; the "
                            "executor does not support nested composition"
                        ),
                        recipe_stage_index=recipe_stage_index,
                        recipe_stage_ref=recipe_stage_ref,
                        execution_path=path,
                        stage=composite_ref,
                    )
                    view["expansion_error"] = "nested_composite"
                    return []
                expansion_issue_count = len(issues)
                try:
                    raw_children = stage.decompose_and_apply_with()
                    children: list[Any] = []
                    for child_index, child in enumerate(raw_children or ()):
                        if child_index >= _MAX_EXECUTION_LEAVES:
                            issue(
                                code="composite_child_bound_exceeded",
                                message=(
                                    "one composite produced more children than the "
                                    f"{_MAX_EXECUTION_LEAVES}-leaf review bound"
                                ),
                                recipe_stage_index=recipe_stage_index,
                                recipe_stage_ref=recipe_stage_ref,
                                execution_path=path,
                                stage=composite_ref,
                            )
                            view["expansion_error"] = "child_bound"
                            break
                        children.append(child)
                except Exception as exc:  # noqa: BLE001 - no guessed leaf on failure
                    issue(
                        code="composite_decomposition_failed",
                        message=_safe_error(exc),
                        recipe_stage_index=recipe_stage_index,
                        recipe_stage_ref=recipe_stage_ref,
                        execution_path=path,
                        stage=composite_ref,
                    )
                    view["expansion_error"] = "decomposition_failed"
                    return []
                if not children:
                    issue(
                        code="composite_decomposition_empty",
                        message="composite decomposition produced no execution stages",
                        recipe_stage_index=recipe_stage_index,
                        recipe_stage_ref=recipe_stage_ref,
                        execution_path=path,
                        stage=composite_ref,
                    )
                    view["expansion_error"] = "empty"
                    return []

                child_leaf_indices: list[int] = []
                for child_index, child in enumerate(children):
                    if not isinstance(child, ProcessingStage):
                        issue(
                            code="invalid_composite_child",
                            message=(
                                f"composite decomposition returned {type(child).__name__}, not a ProcessingStage"
                            ),
                            recipe_stage_index=recipe_stage_index,
                            recipe_stage_ref=recipe_stage_ref,
                            execution_path=(*path, child_index),
                            stage=composite_ref,
                        )
                        continue
                    child_leaf_indices.extend(visit(child, path=(*path, child_index)))
                view["child_count"] = len(children)
                view["execution_leaf_indices"] = child_leaf_indices
                if "expansion_error" not in view and len(issues) == expansion_issue_count:
                    view["expansion_status"] = "complete"
                return child_leaf_indices

            if execution_leaf_count >= _MAX_EXECUTION_LEAVES:
                if not leaf_limit_reported:
                    issue(
                        code="execution_leaf_bound_exceeded",
                        message=(
                            f"configured recipe expands beyond the {_MAX_EXECUTION_LEAVES}-leaf semantic-review bound"
                        ),
                        recipe_stage_index=recipe_stage_index,
                        recipe_stage_ref=recipe_stage_ref,
                        execution_path=path,
                        stage=stage_name,
                    )
                    leaf_limit_reported = True
                return []
            leaf_index = execution_leaf_count
            execution_leaf_count += 1
            group["leaves"].append(
                {
                    "stage": stage,
                    "stage_ref": stage_name if path else recipe_stage_ref,
                    "stage_index": leaf_index,
                    "provenance": {
                        **provenance,
                        "execution_leaf_index": leaf_index,
                    },
                }
            )
            return [leaf_index]

        leaf_indices = visit(configured_stage, path=())
        recipe_view["execution_leaf_indices"] = leaf_indices
        if is_composite:
            recipe_view["composite_view_index"] = next(
                (
                    index
                    for index, view in enumerate(composite_views)
                    if view["recipe_stage_index"] == recipe_stage_index and not view["execution_path"]
                ),
                None,
            )
        execution_groups.append(group)
        recipe_stage_views.append(recipe_view)

    return (
        execution_groups,
        recipe_stage_views,
        composite_views,
        issues,
        card_gaps,
    )


def _configured_key_params(stage: Any, contract: Any) -> dict[str, list[str]]:  # noqa: ANN401
    """Map configured literal key values to the ``*_key`` params that selected them."""
    bindings: dict[str, list[str]] = {}
    for param in getattr(contract, "params", ()) or ():
        name = str(getattr(param, "name", ""))
        if not name.endswith("_key"):
            continue
        with contextlib.suppress(Exception):
            value = getattr(stage, name)
            if isinstance(value, str) and value:
                bindings.setdefault(value, []).append(name)
    return {key: sorted(set(names)) for key, names in bindings.items()}


def _field_entry(  # noqa: PLR0913
    key: str,
    *,
    scope: str,
    contract: Any,  # noqa: ANN401
    key_params: dict[str, list[str]],
    requirement: str | None = None,
    alternative_group: int | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "key": str(key),
        "scope": scope,
        "role": str((getattr(contract, "key_roles", {}) or {}).get(key, "unknown")),
        "configured_by": list(key_params.get(key, [])),
    }
    if requirement is not None:
        entry["requirement"] = requirement
    if alternative_group is not None:
        entry["alternative_group"] = alternative_group
    return entry


def _contract_reads(contract: Any, key_params: dict[str, list[str]]) -> list[dict[str, Any]]:  # noqa: ANN401
    reads: list[dict[str, Any]] = []
    primary = getattr(contract, "reads", None)
    if primary is not None:
        reads.extend(
            _field_entry(key, scope="task", contract=contract, key_params=key_params, requirement="required")
            for key in getattr(primary, "data_keys", ()) or ()
        )
        reads.extend(
            _field_entry(key, scope="segment", contract=contract, key_params=key_params, requirement="required")
            for key in getattr(primary, "segment_data_keys", ()) or ()
        )
    for option_index, option in enumerate(getattr(contract, "reads_one_of", ()) or ()):
        reads.extend(
            _field_entry(
                key,
                scope="task",
                contract=contract,
                key_params=key_params,
                requirement="one_of",
                alternative_group=option_index,
            )
            for key in getattr(option, "data_keys", ()) or ()
        )
        reads.extend(
            _field_entry(
                key,
                scope="segment",
                contract=contract,
                key_params=key_params,
                requirement="one_of",
                alternative_group=option_index,
            )
            for key in getattr(option, "segment_data_keys", ()) or ()
        )
    reads.extend(
        _field_entry(
            key,
            scope="metadata",
            contract=contract,
            key_params=key_params,
            requirement="required",
        )
        for key in getattr(contract, "metadata_reads", ()) or ()
    )
    return reads


def _contract_writes(contract: Any, key_params: dict[str, list[str]]) -> list[dict[str, Any]]:  # noqa: ANN401, C901
    """Return configured writes with generic runtime-presence provenance.

    ``StageContract.writes`` remains the legacy mechanical declaration.
    ``conditional_writes`` overlays those slots (or adds advisory-only
    pass-through slots) without interpreting a stage name or user intent.
    """
    writes_by_slot: dict[tuple[str, str], dict[str, Any]] = {}

    def add_legacy(key: str, scope: str) -> None:
        slot = (scope, str(key))
        writes_by_slot.setdefault(
            slot,
            {
                **_field_entry(key, scope=scope, contract=contract, key_params=key_params),
                "certainty": "definite",
                "conditions": [],
                "legacy_mechanical_write": True,
            },
        )

    spec = getattr(contract, "writes", None)
    if spec is not None:
        for key in getattr(spec, "data_keys", ()) or ():
            add_legacy(key, "task")
        for key in getattr(spec, "segment_data_keys", ()) or ():
            add_legacy(key, "segment")
    for key in getattr(contract, "metadata_writes", ()) or ():
        add_legacy(key, "metadata")

    for conditional in getattr(contract, "conditional_writes", ()) or ():
        condition = {
            "condition": str(getattr(conditional, "condition", "")),
            "value_origin": str(getattr(conditional, "value_origin", "stage_generated")),
        }
        conditional_spec = getattr(conditional, "writes", None)
        if conditional_spec is None:
            continue
        for scope, keys in (
            ("task", getattr(conditional_spec, "data_keys", ()) or ()),
            (
                "segment",
                getattr(conditional_spec, "segment_data_keys", ()) or (),
            ),
            ("metadata", getattr(conditional, "metadata_writes", ()) or ()),
        ):
            for key in keys:
                slot = (scope, str(key))
                entry = writes_by_slot.setdefault(
                    slot,
                    {
                        **_field_entry(
                            key,
                            scope=scope,
                            contract=contract,
                            key_params=key_params,
                        ),
                        "certainty": "conditional",
                        "conditions": [],
                        "legacy_mechanical_write": False,
                    },
                )
                entry["certainty"] = "conditional"
                if condition not in entry["conditions"]:
                    entry["conditions"].append(copy.deepcopy(condition))

    return list(writes_by_slot.values())


def _stage_endpoint(stage_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "stage",
        "stage_index": stage_info["stage_index"],
        "stage": stage_info["stage"],
        "provenance": copy.deepcopy(stage_info["provenance"]),
        "semantic_material": copy.deepcopy(stage_info["semantic_material"]),
    }


def _writer_record(stage_info: dict[str, Any], write: dict[str, Any]) -> dict[str, Any]:
    return {
        **_stage_endpoint(stage_info),
        "write": copy.deepcopy(write),
        "basis": "configured_dynamic_contract",
    }


def _seam_record(stage_info: dict[str, Any]) -> dict[str, Any] | None:
    cardinality = stage_info["cardinality"]
    kind = _CARDINALITY_SEAM_KINDS.get(cardinality)
    if kind is None:
        return None
    return {
        "stage_index": stage_info["stage_index"],
        "stage": stage_info["stage"],
        "provenance": copy.deepcopy(stage_info["provenance"]),
        "kind": kind,
        "cardinality": cardinality,
        "iteration_key": stage_info["iteration_key"],
        "preserves_upstream_keys": stage_info["preserves_upstream_keys"],
        "semantic_material": copy.deepcopy(stage_info["semantic_material"]),
    }


def _crossed_seams(
    seams: list[dict[str, Any]],
    producer: dict[str, Any],
    consumer_index: int,
) -> list[dict[str, Any]]:
    producer_index = producer.get("stage_index")
    start = int(producer_index) if isinstance(producer_index, int) else 0
    return [copy.deepcopy(seam) for seam in seams if start <= int(seam["stage_index"]) < consumer_index]


def _checklist(  # noqa: PLR0913
    *,
    stages: list[dict[str, Any]],
    recipe_stages: list[dict[str, Any]],
    composites: list[dict[str, Any]],
    lineage: list[dict[str, Any]],
    seams: list[dict[str, Any]],
    unresolved: list[dict[str, Any]],
    missing_cards: list[dict[str, Any]],
    semantic_gaps: list[dict[str, Any]],
    contract_issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generic host-review prompts; none asserts that a semantic mismatch exists."""
    caveat_evidence = sum(
        1
        for stage in stages
        if any(
            field in stage["semantic_material"] for field in ("model_id", "domain", "metrics", "comparison", "caveats")
        )
    )
    conditional_write_count = sum(
        1 for stage in stages for write in stage["writes"] if write.get("certainty") == "conditional"
    )
    return [
        {
            "id": "per_stage_justification",
            "required": bool(stages),
            "evidence_count": len(stages),
            "instruction": (
                "For every concrete execution leaf, state which user goal, constraint, or "
                "acceptance criterion justifies its presence; flag any leaf with no intent-level purpose."
            ),
        },
        {
            "id": "authored_recipe_stage_justification",
            "required": bool(recipe_stages),
            "evidence_count": len(recipe_stages),
            "instruction": (
                "Justify every authored recipe-stage choice. For a composite, review both "
                "its outer card/configuration and every concrete execution leaf it expands into."
            ),
        },
        {
            "id": "composite_expansion_coverage",
            "required": bool(composites),
            "evidence_count": len(composites),
            "instruction": (
                "Verify that every composite expansion is complete and that enabled inner "
                "filters, transforms, models, and side effects all match the request."
            ),
        },
        {
            "id": "transform_output_consumption",
            "required": bool(stages),
            "evidence_count": sum(len(stage["writes"]) for stage in stages),
            "instruction": (
                "For each transform or annotation, verify that its configured output is consumed "
                "downstream or is itself an intended terminal output, and that no result is silently discarded."
            ),
        },
        {
            "id": "conditional_write_presence",
            "required": bool(conditional_write_count),
            "evidence_count": conditional_write_count,
            "instruction": (
                "For every conditional write, check which profiled rows take the declared "
                "runtime branch and do not treat a possible field as universally present. "
                "For same-key pass-through, retain the original producer's meaning."
            ),
        },
        {
            "id": "model_domain_metric_caveats",
            "required": bool(caveat_evidence),
            "evidence_count": caveat_evidence,
            "instruction": (
                "Compare model, domain, metric, comparison, and caveat prose with the request; "
                "state what each selected metric or model does and does not establish."
            ),
        },
        {
            "id": "literal_field_meaning",
            "required": bool(lineage),
            "evidence_count": len(lineage),
            "instruction": (
                "For each exact-key consumer, compare the producer and consumer card prose "
                "and decide whether the field's meaning matches the user's intended operation."
            ),
        },
        {
            "id": "cardinality_scope",
            "required": bool(seams),
            "evidence_count": len(seams),
            "instruction": (
                "Review field scope and granularity across every fan-out, aggregation, "
                "nested-collection, or filter seam before confirmation."
            ),
        },
        {
            "id": "unresolved_lineage",
            "required": bool(unresolved),
            "evidence_count": len(unresolved),
            "instruction": (
                "Treat unresolved exact-key origins as missing evidence; inspect the source schema "
                "or upstream contract instead of inventing field semantics."
            ),
        },
        {
            "id": "missing_card_semantics",
            "required": bool(missing_cards),
            "evidence_count": len(missing_cards),
            "instruction": (
                "Distinguish a missing card from a present card without semantic_facts; "
                "in either case do not infer undocumented meaning from a parameter name alone."
            ),
        },
        {
            "id": "semantic_provenance_gaps",
            "required": bool(semantic_gaps),
            "evidence_count": len(semantic_gaps),
            "instruction": (
                "A source key's presence is known but its meaning/unit/entity is not. "
                "Inspect the source schema or ask instead of treating presence as semantic provenance."
            ),
        },
        {
            "id": "contract_visibility",
            "required": bool(contract_issues),
            "evidence_count": len(contract_issues),
            "instruction": (
                "Dynamic contract evidence is incomplete for these stages; keep the semantic review partial."
            ),
        },
    ]


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
    ) = _expand_configured_stages(
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
            material = _semantic_material(stage_ref)
            gap = _card_semantic_gap(
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
                        "message": _safe_error(exc),
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
                        "configured_params": _best_effort_instance_params(
                            stage,
                            source=("composite_effective" if provenance["execution_path"] else "configured_instance"),
                        ),
                        "reads": [],
                        "writes": [],
                        "removes_keys": [],
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
                else _recipe_stage_params(
                    recipe,
                    int(provenance["recipe_stage_index"]),
                )
            )
            implicit_source = (
                "composite_effective"
                if is_composite_leaf
                else ("configured_instance" if recipe is None else "default")
            )
            key_params = _configured_key_params(stage, contract)
            configured_params = _configured_params(
                stage,
                contract,
                authored,
                implicit_source=implicit_source,
            )
            reads = _contract_reads(contract, key_params)
            writes = _contract_writes(contract, key_params)
            stage_info = {
                "stage_index": stage_index,
                "stage": stage_ref,
                "provenance": provenance,
                "contract_status": "configured",
                "cardinality": str(getattr(contract, "cardinality", "1:1")),
                "iteration_key": getattr(contract, "iteration_key", None),
                "preserves_upstream_keys": bool(getattr(contract, "preserves_upstream_keys", True)),
                "wrappable": bool(getattr(contract, "wrappable", True)),
                "accepts_task_type": getattr(contract, "accepts_task_type", None),
                "produces_task_type": getattr(contract, "produces_task_type", None),
                "configured_params": configured_params,
                "reads": reads,
                "writes": writes,
                "removes_keys": [str(key) for key in (getattr(contract, "removes_keys", ()) or ())],
                "semantic_material": material,
            }
            stage_packets.append(stage_info)

            consumer = _stage_endpoint(stage_info)
            for read in reads:
                slot = (read["scope"], read["key"])
                latest = copy.deepcopy(active_writers.get(slot))
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
                    "crossed_cardinality_seams": _crossed_seams(
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
            if not stage_info["preserves_upstream_keys"]:
                # ``preserves_upstream_keys`` describes task.data reconstruction.
                # Task metadata is a separate channel and remains live unless a
                # future contract explicitly declares metadata removal.
                active_writers = {slot: writer for slot, writer in active_writers.items() if slot[0] == "metadata"}
            for key in stage_info["removes_keys"]:
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
                                "through": _stage_endpoint(stage_info),
                                "conditions": passthrough_conditions,
                                "basis": "configured_conditional_write",
                            }
                        )
                        active_writers[slot] = preserved
                    continue

                writer = _writer_record(stage_info, write)
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
                    and (write["scope"] == "metadata" or stage_info["preserves_upstream_keys"])
                ):
                    writer["when_condition_not_met"] = {
                        "outcome": "upstream_same_key_value_remains",
                        "producer": prior,
                    }
                active_writers[slot] = writer
                writer_history.setdefault(slot, []).append(writer)

            seam = _seam_record(stage_info)
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

    recipe_identity = _recipe_identity(recipe, len(configured_stages))
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

    checklist = _checklist(
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
        "data_profile": _data_profile_summary(data_profile),
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
