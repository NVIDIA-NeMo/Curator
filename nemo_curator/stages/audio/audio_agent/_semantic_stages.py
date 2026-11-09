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

"""Configured composite expansion and semantic capability evidence."""

from __future__ import annotations

import contextlib
import copy
from collections.abc import Mapping, Sequence
from typing import Any

from nemo_curator.stages.audio.audio_agent import _semantic_values
from nemo_curator.stages.audio.audio_agent.index import get_index

# ruff: noqa: B023 - stage visitors execute synchronously within the current iteration.

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


def _safe_error(exc: BaseException) -> str:
    from nemo_curator.stages.audio.audio_agent import _safety

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
            material[field] = _semantic_values._json_safe(copy.deepcopy(card[field]))
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
        "provenance": _semantic_values._json_safe(dict(provenance)),
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
    from nemo_curator.stages.audio.audio_agent import _safety
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
        recipe_stage_ref = _semantic_values._recipe_stage_ref(
            recipe,
            recipe_stage_index,
            configured_stage,
        )
        authored_params = _semantic_values._recipe_stage_params(recipe, recipe_stage_index)
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
            "authored_params": _safety.redact(_semantic_values._jsonable(authored_params)),
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
                    "authored_params": (
                        _safety.redact(_semantic_values._jsonable(authored_params)) if not path else {}
                    ),
                    "configured_params": {},
                    "semantic_material": material,
                    "expansion_status": "partial",
                    "execution_leaf_indices": [],
                }
                try:
                    composite_contract = (
                        outer_contract if not path and outer_contract is not None else foundation.build_contract(stage)
                    )
                    view["configured_params"] = _semantic_values._configured_params(
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
