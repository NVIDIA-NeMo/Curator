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

"""Field lineage, cardinality seams, and semantic review checklists."""

from __future__ import annotations

import copy
from typing import Any

_CARDINALITY_SEAM_KINDS: dict[str, str] = {
    "1:N fan-out": "fan_out",
    "N:1": "aggregation",
    "1:1 nested-list": "nested_collection",
    "filter": "filter",
}


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
    optional = getattr(contract, "optional_reads", None)
    if optional is not None:
        reads.extend(
            _field_entry(key, scope="task", contract=contract, key_params=key_params, requirement="optional")
            for key in getattr(optional, "data_keys", ()) or ()
        )
        reads.extend(
            _field_entry(key, scope="segment", contract=contract, key_params=key_params, requirement="optional")
            for key in getattr(optional, "segment_data_keys", ()) or ()
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
