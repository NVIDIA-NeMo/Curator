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

"""Card <-> stage conformance gate: keeps capability cards honest (2.1).

Mechanical facts in a card are checked against the real stage so a card cannot
drift (the exact ``resample`` / ``audio_to_document`` failure class):

* ``stage_id`` must resolve to a registered agent-ready stage.
* ``params_of_note`` / ``presets`` keys that name a stage parameter must actually
  exist on the constructor. (``constraints`` are model facts, e.g.
  ``supported_sample_rates`` / ``max_speakers`` -- NOT constructor params -- so they
  are intentionally not checked against the signature.)
* ``resource`` uses only known keys with numeric values where numeric is expected.
* ``model_version`` is NOT required, even on a model stage: nothing compares it between runs,
  so demanding it only produced pin-shaped strings that were not pins. Say how far to trust it
  in ``verified`` instead.
* a ``metrics`` block, if present, must use a valid ``scale.direction``, a real
  ``threshold_param``, and a ``[lo, hi]`` ``valid_range``.
* a ``decision`` block, if present, must describe one of the explicitly supported
  task or one-level segment score producers and its real downstream selector
  parameters. The score key's declared fallback is checked against the constructor
  default and the key parameter must control a real stage write.
* ``semantic_facts``, if present, is shape-checked as advisory prose.  The gate
  never interprets scope or turns a fact into a module-specific pipeline rule.
* a capability ``tag`` must reflect the stage's DEFAULT behavior: a tag that maps to an
  unconditional boolean contract gate (``writes_disk``/``needs_ffmpeg``) is checked against
  a default-constructed instance's gate. An opt-in capability belongs in a param (knob),
  not a tag. (``needs_gpu`` and ``needs_internet_first_run`` are intentionally NOT checked:
  their gates are conditional -- ``resources.gpus > 0`` and ``model_path/auto_download`` --
  so they aren't a clean tag<->gate equality.)

Measured facts (actual VRAM) are a GPU-CI hook, not checked here; best-guess facts
(``use_cases`` / ``domain``) are shape-checked only. CPU-runnable, no models load.
Run: ``python -m nemo_curator.audio_agent.card_conformance``.
"""

from __future__ import annotations

import json
from typing import Any

_KNOWN_RESOURCE_KEYS = frozenset(
    {"cpus", "gpu_mem_gb", "host_mem_gb", "gpu_optional", "bound", "throughput_hint", "disk_expansion"}
)
_NUMERIC_RESOURCE_KEYS = frozenset({"cpus", "gpu_mem_gb", "host_mem_gb", "disk_expansion"})
_KNOWN_BOUND = frozenset({"cpu", "gpu", "io"})
_VERIFIED_TIERS = frozenset({"mechanical", "measured", "best_guess"})
_DIRECTIONS = frozenset({"higher_better", "lower_better"})
_SEMANTIC_PROSE_FIELDS = frozenset({"meaning", "unit", "provenance", "scope", "propagation"})
_DECISION_PRODUCERS = frozenset(
    {
        "GetAudioDurationStage",
        "GetPairwiseWerStage",
        "SIGMOSFilterStage",
        "UTMOSFilterStage",
    }
)
_DECISION_KINDS = {
    "GetAudioDurationStage": "scalar",
    "GetPairwiseWerStage": "scalar",
    "SIGMOSFilterStage": "compound",
    "UTMOSFilterStage": "scalar",
}
_DECISION_VALUE_TYPES = frozenset({"number", "string", "boolean"})
_DECISION_FIELDS = frozenset(
    {
        "kind",
        "separable_from_producer",
        "score_key_param",
        "score_key_default",
        "threshold_param",
        "dimensions",
        "value_type",
        "scope",
        "selector",
        "producer_constraints",
        "missing_score_policy",
        "monotonic_direction",
        "atomic",
        "variants",
    }
)
_DECISION_COMMON_REQUIRED_FIELDS = frozenset(
    {
        "separable_from_producer",
        "value_type",
        "scope",
        "selector",
        "missing_score_policy",
        "atomic",
    }
)
_DECISION_SCALAR_REQUIRED_FIELDS = frozenset({"score_key_param", "score_key_default"})
_DECISION_COMPOUND_REQUIRED_FIELDS = frozenset({"dimensions"})
_DECISION_SCALAR_SELECTOR_FIELDS = frozenset(
    {"stage_id", "key_param", "value_param", "operator_param", "allowed_operators"}
)
_DECISION_COMPOUND_SELECTOR_FIELDS = frozenset(
    {
        "stage_id",
        "conditions_param",
        "condition_logic_param",
        "required_condition_logic",
        "missing_policy_param",
        "required_missing_policy",
        "required_operator",
    }
)
_DECISION_SEGMENT_SELECTOR_FIELDS = frozenset(
    {
        "stage_id",
        "conditions_param",
        "condition_logic_param",
        "required_condition_logic",
        "missing_policy_param",
        "required_missing_policy",
        "required_operator",
        "items_key_param",
        "items_key_source_param",
        "empty_policy_param",
        "required_empty_policy",
    }
)
_DECISION_OPTIONAL_SELECTOR_FIELDS = frozenset(
    {"missing_policy_param", "required_missing_policy", "required_operator"}
)
_DECISION_DIMENSION_FIELDS = frozenset({"score_key_param", "score_key_default", "threshold_param"})
_DECISION_OPERATORS = frozenset({"lt", "le", "eq", "ne", "ge", "gt"})
_DECISION_SELECTOR_PARAMS = {
    "key_param": "input_value_key",
    "value_param": "target_value",
    "operator_param": "operator",
}
_DECISION_COMPOUND_SELECTOR_PARAMS = {
    "conditions_param": "conditions",
    "condition_logic_param": "condition_logic",
    "missing_policy_param": "missing_value_policy",
}
_DECISION_SEGMENT_SELECTOR_PARAMS = {
    **_DECISION_COMPOUND_SELECTOR_PARAMS,
    "items_key_param": "items_key",
    "empty_policy_param": "drop_parent_if_empty",
}
_DECISION_MISSING_POLICIES = frozenset({"selector_drop", "selector_error"})


def _stage_param_specs(stage_id: str) -> dict[str, Any] | None:
    """Constructor params keyed by name, or None if the stage doesn't resolve."""
    from nemo_curator.audio_agent._resolve import resolve_stage_class
    from nemo_curator.stages.audio._agent._agent_registry import stage_params

    try:
        cls = resolve_stage_class(stage_id)
    except Exception:  # noqa: BLE001 - unknown/unimportable stage
        return None
    try:
        return {p.name: p for p in stage_params(cls)}
    except Exception:  # noqa: BLE001
        return {}


def _stage_param_names(stage_id: str) -> set[str] | None:
    """Constructor param names of the stage, or None if it doesn't resolve."""
    specs = _stage_param_specs(stage_id)
    return None if specs is None else set(specs)


def _decision_score_key_is_written(stage_id: str, score_key_param: str) -> bool:
    """Whether changing the declared key param changes a mechanically declared write."""
    from nemo_curator.audio_agent._resolve import resolve_stage_class
    from nemo_curator.stages.audio import agent as foundation

    marker = "__decision_score_key__"
    try:
        stage = resolve_stage_class(stage_id)(**{score_key_param: marker})
        contract = foundation.build_contract(stage)
    except Exception:  # noqa: BLE001 - malformed key params fail conformance
        return False
    written = set(contract.writes.data_keys)
    for conditional in contract.conditional_writes:
        written.update(conditional.writes.data_keys)
    return marker in written


def _decision_score_binding_violations(
    stage_id: str,
    binding: dict[str, Any],
    *,
    prefix: str,
    producer_specs: dict[str, Any],
) -> list[str]:
    """Validate one configurable producer score key and its declared default."""
    violations: list[str] = []
    score_key_param = binding.get("score_key_param")
    if not isinstance(score_key_param, str) or not score_key_param:
        violations.append(f"{prefix}.score_key_param must be a non-empty string")
        return violations
    if score_key_param not in producer_specs:
        violations.append(f"{prefix}.score_key_param {score_key_param!r} is not a constructor param of the producer")
        return violations

    spec = producer_specs[score_key_param]
    score_key_default = binding.get("score_key_default")
    if spec.required:
        violations.append(f"{prefix}.score_key_param {score_key_param!r} must have a constructor default")
    elif score_key_default != spec.default:
        violations.append(
            f"{prefix}.score_key_default {score_key_default!r} does not match "
            f"{score_key_param!r}'s constructor default {spec.default!r}"
        )
    if not isinstance(score_key_default, str) or not score_key_default:
        violations.append(f"{prefix}.score_key_default must be a non-empty string")
    if not _decision_score_key_is_written(stage_id, score_key_param):
        violations.append(f"{prefix}.score_key_param {score_key_param!r} does not control a declared producer write")
    return violations


def _decision_constraints_violations(
    stage_id: str,
    raw: dict[str, Any],
    *,
    prefix: str,
    producer_specs: dict[str, Any],
) -> list[str]:
    """Require card constraints to exactly mirror producer-declared safe settings."""
    from nemo_curator.audio_agent._resolve import resolve_stage_class

    try:
        stage_class = resolve_stage_class(stage_id)
        by_scope = getattr(stage_class, "SEPARABLE_DECISION_CONSTRAINTS_BY_SCOPE", {})
        if by_scope:
            scope = raw.get("scope")
            if scope not in by_scope:
                return [
                    f"{prefix}.scope {scope!r} has no producer-declared safe settings (supported: {sorted(by_scope)})"
                ]
            expected = dict(by_scope[scope])
        else:
            expected = dict(
                getattr(
                    stage_class,
                    "SEPARABLE_DECISION_CONSTRAINTS",
                    {},
                )
            )
    except Exception:  # noqa: BLE001
        expected = {}
    constraints = raw.get("producer_constraints")
    if not expected and constraints is None:
        return []
    if not isinstance(constraints, dict):
        return [f"{prefix}.producer_constraints must be a mapping"]
    violations = (
        [
            f"{prefix}.producer_constraints {constraints!r} must exactly match "
            f"the producer-declared safe settings {expected!r}"
        ]
        if constraints != expected
        else []
    )
    for param in constraints:
        if param not in producer_specs:
            violations.append(f"{prefix}.producer_constraints names {param!r}, which is not a constructor param")
    return violations


def _decision_dimension_violations(
    stage_id: str,
    raw: dict[str, Any],
    *,
    prefix: str,
    producer_specs: dict[str, Any],
) -> list[str]:
    """Validate every dimension of an atomic compound decision."""
    from nemo_curator.audio_agent._resolve import resolve_stage_class

    dimensions = raw.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        return [f"{prefix}.dimensions must be a non-empty list"]
    violations: list[str] = []
    actual_pairs: list[tuple[str, str]] = []
    for index, dimension in enumerate(dimensions):
        dimension_prefix = f"{prefix}.dimensions[{index}]"
        if not isinstance(dimension, dict):
            violations.append(f"{dimension_prefix} must be a mapping")
            continue
        violations.extend(
            f"{dimension_prefix} has unknown key {key!r}"
            for key in sorted(dimension, key=str)
            if key not in _DECISION_DIMENSION_FIELDS
        )
        violations.extend(
            f"{dimension_prefix} is missing required field {key!r}"
            for key in sorted(_DECISION_DIMENSION_FIELDS)
            if key not in dimension
        )
        violations.extend(
            _decision_score_binding_violations(
                stage_id,
                dimension,
                prefix=dimension_prefix,
                producer_specs=producer_specs,
            )
        )
        threshold_param = dimension.get("threshold_param")
        if not isinstance(threshold_param, str) or not threshold_param:
            violations.append(f"{dimension_prefix}.threshold_param must be a non-empty string")
        elif threshold_param not in producer_specs:
            violations.append(
                f"{dimension_prefix}.threshold_param {threshold_param!r} is not a constructor param of the producer"
            )
        score_key_param = dimension.get("score_key_param")
        if isinstance(threshold_param, str) and isinstance(score_key_param, str):
            actual_pairs.append((threshold_param, score_key_param))

    try:
        expected_pairs = list(
            getattr(
                resolve_stage_class(stage_id),
                "SEPARABLE_DECISION_DIMENSIONS",
                (),
            )
        )
    except Exception:  # noqa: BLE001
        expected_pairs = []
    if actual_pairs != expected_pairs:
        violations.append(
            f"{prefix}.dimensions must exactly cover the producer-declared threshold/key dimensions {expected_pairs!r}"
        )
    return violations


def _decision_selector_violations(  # noqa: C901, PLR0912, PLR0915 - exact refusal per selector invariant
    stage_id: str,
    raw: dict[str, Any],
    *,
    prefix: str,
    kind: str,
    producer_specs: dict[str, Any],
) -> list[str]:
    """Validate the exact scalar or compound selector surface."""
    selector = raw.get("selector")
    if not isinstance(selector, dict):
        return [f"{prefix}.selector must be a mapping"]

    if raw.get("scope") == "segments":
        required_fields = _DECISION_SEGMENT_SELECTOR_FIELDS
        allowed_fields = required_fields
        expected_stage = "PreserveByValueConditionsStage"
        expected_params = _DECISION_SEGMENT_SELECTOR_PARAMS
    elif kind == "compound":
        required_fields = _DECISION_COMPOUND_SELECTOR_FIELDS
        allowed_fields = required_fields
        expected_stage = "PreserveByValueConditionsStage"
        expected_params = _DECISION_COMPOUND_SELECTOR_PARAMS
    else:
        required_fields = _DECISION_SCALAR_SELECTOR_FIELDS
        allowed_fields = required_fields | _DECISION_OPTIONAL_SELECTOR_FIELDS
        expected_stage = "PreserveByValueStage"
        expected_params = _DECISION_SELECTOR_PARAMS

    violations = [
        f"{prefix}.selector has unknown key {key!r} (allowed: {sorted(allowed_fields)})"
        for key in sorted(selector, key=str)
        if key not in allowed_fields
    ]
    violations.extend(
        f"{prefix}.selector is missing required field {key!r}"
        for key in sorted(required_fields)
        if key not in selector
    )

    selector_stage = selector.get("stage_id")
    selector_specs = _stage_param_specs(selector_stage) if isinstance(selector_stage, str) and selector_stage else None
    if selector_specs is None:
        violations.append(f"{prefix}.selector.stage_id {selector_stage!r} is not a registered stage")
        return violations
    if selector_stage != expected_stage:
        violations.append(f"{prefix}.selector.stage_id must be {expected_stage!r} for a {kind} decision")
        return violations

    for field, expected_param in expected_params.items():
        value = selector.get(field)
        if not isinstance(value, str) or not value:
            violations.append(f"{prefix}.selector.{field} must be a non-empty string")
        elif value not in selector_specs:
            violations.append(f"{prefix}.selector.{field} {value!r} is not a constructor param of {selector_stage}")
        elif value != expected_param:
            violations.append(f"{prefix}.selector.{field} must be {expected_param!r} for {selector_stage}")

    if kind == "scalar" and raw.get("scope") != "segments":
        operators = selector.get("allowed_operators")
        if (
            not isinstance(operators, list)
            or not operators
            or any(not isinstance(operator, str) for operator in operators)
        ):
            violations.append(f"{prefix}.selector.allowed_operators must be a non-empty list of operators")
        else:
            unknown_operators = sorted(set(operators) - _DECISION_OPERATORS)
            if unknown_operators:
                violations.append(
                    f"{prefix}.selector.allowed_operators contains unsupported operators "
                    f"{unknown_operators!r} (allowed: {sorted(_DECISION_OPERATORS)})"
                )
            if len(operators) != len(set(operators)):
                violations.append(f"{prefix}.selector.allowed_operators must not contain duplicates")

    required_operator = selector.get("required_operator")
    if required_operator is not None:
        if required_operator not in _DECISION_OPERATORS:
            violations.append(f"{prefix}.selector.required_operator must be one of {sorted(_DECISION_OPERATORS)}")
        operators = selector.get("allowed_operators")
        if isinstance(operators, list) and required_operator not in operators:
            violations.append(f"{prefix}.selector.required_operator must be present in allowed_operators")

    if expected_stage == "PreserveByValueConditionsStage":
        required_logic = selector.get("required_condition_logic")
        if required_logic != "and":
            violations.append(
                f"{prefix}.selector.required_condition_logic must be 'and' for exact native-filter equivalence"
            )

    if raw.get("scope") == "segments":
        source_param = selector.get("items_key_source_param")
        if not isinstance(source_param, str) or not source_param:
            violations.append(f"{prefix}.selector.items_key_source_param must be a non-empty string")
        elif source_param not in producer_specs:
            violations.append(
                f"{prefix}.selector.items_key_source_param {source_param!r} is not a constructor param of {stage_id}"
            )
        if selector.get("required_empty_policy") is not True:
            violations.append(
                f"{prefix}.selector.required_empty_policy must be true for native segment-filter equivalence"
            )

    missing_policy_param = selector.get("missing_policy_param", "missing_value_policy")
    expected_missing = "drop" if raw.get("missing_score_policy") == "selector_drop" else "error"
    missing_spec = selector_specs.get(str(missing_policy_param))
    if missing_spec is None:
        violations.append(
            f"{prefix}.selector.missing_policy_param {missing_policy_param!r} "
            f"is not a constructor param of {selector_stage}"
        )
    else:
        effective_missing = selector.get("required_missing_policy", missing_spec.default)
        if effective_missing != expected_missing:
            violations.append(
                f"{prefix}.selector requires missing policy {expected_missing!r} "
                f"for {raw.get('missing_score_policy')!r}"
            )
    return violations


def _decision_violations(  # noqa: C901, PLR0912, PLR0915 - one branch per card invariant
    stage_id: str,
    raw: Any,  # noqa: ANN401 - untrusted card boundary
) -> list[str]:
    """Validate the narrow producer/selector split consumed by strategy code."""
    prefix = f"{stage_id}: decision"
    if not isinstance(raw, dict):
        return [f"{prefix} must be a mapping"]

    kind = raw.get("kind", "scalar")
    required_fields = set(_DECISION_COMMON_REQUIRED_FIELDS)
    required_fields.update(
        _DECISION_COMPOUND_REQUIRED_FIELDS if kind == "compound" else _DECISION_SCALAR_REQUIRED_FIELDS
    )
    violations = [
        f"{prefix} has unknown key {key!r} (allowed: {sorted(_DECISION_FIELDS)})"
        for key in sorted(raw, key=str)
        if key not in _DECISION_FIELDS
    ]
    violations.extend(
        f"{prefix} is missing required field {key!r}" for key in sorted(required_fields) if key not in raw
    )

    if stage_id not in _DECISION_PRODUCERS:
        violations.append(f"{prefix} is only supported for {sorted(_DECISION_PRODUCERS)} in this phase")
    expected_kind = _DECISION_KINDS.get(stage_id)
    if kind not in {"scalar", "compound"}:
        violations.append(f"{prefix}.kind must be 'scalar' or 'compound'")
    elif expected_kind is not None and kind != expected_kind:
        violations.append(f"{prefix}.kind must be {expected_kind!r} for {stage_id}")
    if raw.get("separable_from_producer") is not True:
        violations.append(f"{prefix}.separable_from_producer must be true")

    producer_specs = _stage_param_specs(stage_id) or {}
    violations.extend(
        _decision_constraints_violations(
            stage_id,
            raw,
            prefix=prefix,
            producer_specs=producer_specs,
        )
    )
    if kind == "compound":
        violations.extend(
            _decision_dimension_violations(
                stage_id,
                raw,
                prefix=prefix,
                producer_specs=producer_specs,
            )
        )
    else:
        violations.extend(
            _decision_score_binding_violations(
                stage_id,
                raw,
                prefix=prefix,
                producer_specs=producer_specs,
            )
        )
        if raw.get("scope") == "segments":
            threshold_param = raw.get("threshold_param")
            if not isinstance(threshold_param, str) or not threshold_param:
                violations.append(f"{prefix}.threshold_param must be a non-empty string for segment scope")
            elif threshold_param not in producer_specs:
                violations.append(
                    f"{prefix}.threshold_param {threshold_param!r} is not a constructor param of the producer"
                )

    if raw.get("scope") not in {"task", "segments"}:
        violations.append(f"{prefix}.scope must be 'task' or 'segments'")
    if raw.get("value_type") not in _DECISION_VALUE_TYPES:
        violations.append(f"{prefix}.value_type must be one of {sorted(_DECISION_VALUE_TYPES)}")
    if raw.get("missing_score_policy") not in _DECISION_MISSING_POLICIES:
        violations.append(f"{prefix}.missing_score_policy must be one of {sorted(_DECISION_MISSING_POLICIES)}")
    if raw.get("atomic") is not True:
        violations.append(f"{prefix}.atomic must be true for this single-decision phase")
    if "monotonic_direction" in raw and raw.get("monotonic_direction") not in _DIRECTIONS:
        violations.append(f"{prefix}.monotonic_direction must be one of {sorted(_DIRECTIONS)} when present")
    violations.extend(
        _decision_selector_violations(
            stage_id,
            raw,
            prefix=prefix,
            kind=str(kind),
            producer_specs=producer_specs,
        )
    )

    variants = raw.get("variants")
    if variants is not None:
        if not isinstance(variants, list) or not variants:
            violations.append(f"{prefix}.variants must be a non-empty list")
        else:
            seen_scopes = {raw.get("scope")}
            for index, variant in enumerate(variants):
                variant_prefix = f"{prefix}.variants[{index}]"
                if not isinstance(variant, dict):
                    violations.append(f"{variant_prefix} must be a mapping")
                    continue
                if "variants" in variant:
                    violations.append(f"{variant_prefix} must not contain nested variants")
                    continue
                scope = variant.get("scope")
                if scope in seen_scopes:
                    violations.append(f"{variant_prefix}.scope {scope!r} duplicates another decision scope")
                seen_scopes.add(scope)
                violations.extend(_decision_violations(stage_id, variant))
    return violations


def _semantic_fact_violations(stage_id: str, raw: Any) -> list[str]:  # noqa: ANN401, C901
    """Validate only the JSON/YAML shape of optional semantic reasoning prose.

    A compact string and a richer mapping are both accepted.  Meaning, scope,
    and propagation remain free text: conformance can ensure the packet is
    readable, but only a reviewer can judge whether it matches user intent.
    """
    if raw is None:
        return []
    if not isinstance(raw, dict):
        return [f"{stage_id}: semantic_facts must be a mapping"]
    violations: list[str] = []
    for anchor, fact in raw.items():
        if not isinstance(anchor, str) or not anchor.strip():
            violations.append(f"{stage_id}: semantic_facts keys must be non-empty strings")
            continue
        prefix = f"{stage_id}: semantic_facts[{anchor!r}]"
        if isinstance(fact, str):
            if not fact.strip():
                violations.append(f"{prefix} must not be empty")
            continue
        if not isinstance(fact, dict):
            violations.append(f"{prefix} must be prose or a mapping")
            continue
        for field in _SEMANTIC_PROSE_FIELDS:
            if field in fact and (not isinstance(fact[field], str) or not fact[field].strip()):
                violations.append(f"{prefix}.{field} must be a non-empty string")
        counterexamples = fact.get("counterexamples")
        if counterexamples is not None and (
            not isinstance(counterexamples, list)
            or not counterexamples
            or any(not isinstance(item, str) or not item.strip() for item in counterexamples)
        ):
            violations.append(f"{prefix}.counterexamples must be a non-empty list of non-empty strings")
    return violations


# Capability tag -> the contract gate it must mirror, checked against a DEFAULT-constructed
# instance. Only unconditional booleans qualify: ``needs_gpu`` is excluded because its gate is
# ``resources.gpus > 0``, true for gpu_optional stages that rightly omit the tag, and
# ``needs_internet_first_run`` because its gate depends on a knob. The remaining tags
# (produces_score, is_filter, fanout, sink, batch_only, needs_hf_token) are role facts, not gates.
_TAG_GATES: dict[str, str] = {
    "writes_disk": "writes_to_disk",
    "needs_ffmpeg": "requires_ffmpeg",
}


def _effective_default_gates(stage_id: str) -> dict[str, bool] | None:
    """DEFAULT-constructed gate values for the tag-checked attrs, or ``None`` if the stage
    can't be built cheaply (required args / build error).

    A composite hides its own gates (``wrappable=False``), so its *effective* gate is the OR
    of its decomposed inner stages' gates -- e.g. SplitASRAlignJoin writes to disk because its
    inner SplitLongAudioStage does, even though the composite's own contract declares nothing.
    Best-effort: never raises.
    """
    from nemo_curator.audio_agent._resolve import resolve_stage_class
    from nemo_curator.stages.audio import agent as foundation
    from nemo_curator.stages.base import CompositeStage

    attrs = set(_TAG_GATES.values())
    try:
        inst = resolve_stage_class(stage_id)()
        contract = foundation.build_contract(inst)
        if not contract.wrappable and isinstance(inst, CompositeStage):
            inner = [foundation.build_contract(s).gates for s in inst.decompose()]
            return {a: any(bool(getattr(g, a, False)) for g in inner) for a in attrs}
        return {a: bool(getattr(contract.gates, a, False)) for a in attrs}
    except Exception:  # noqa: BLE001 - not default-buildable -> can't verify (skip, not a failure)
        return None


def _tag_gate_violations(stage_id: str, card: dict[str, Any]) -> list[str]:
    """Tags that claim a capability the stage does NOT do by default.

    Forward-only (tag present -> default gate must be True): a stage may legitimately have a
    gate on by default without the tag (e.g. gpu_optional), so we do not flag the reverse.
    Composites are judged by the OR of their inner stages' gates; stages that aren't
    default-buildable are skipped (can't verify), never failed.
    """
    checkable = {t for t in (card.get("tags") or []) if t in _TAG_GATES}
    if not checkable:
        return []
    gates = _effective_default_gates(stage_id)
    if gates is None:
        return []
    out: list[str] = []
    for tag in sorted(checkable):
        attr = _TAG_GATES[tag]
        if not gates.get(attr, False):
            out.append(
                f"{stage_id}: card tag {tag!r} but the stage's DEFAULT contract gate {attr}=False "
                f"-- a tag states DEFAULT behavior; make this an opt-in param (knob) instead of a tag, "
                f"or fix the gate"
            )
    return out


def _filter_tag_violations(stage_id: str, card: dict[str, Any]) -> list[str]:
    """A contract declaring ``cardinality="filter"`` must carry the ``is_filter`` tag.

    The contract is the stricter statement and the one a stage author is most likely to write
    alone, having just made the stage drop rows. Without the tag nothing assembling a recipe
    knows it can: a stage that silently discards most of a corpus reads as a pass-through
    exactly where the decision to include it is made.

    Only this direction. The tag is the broader planner-facing notion -- ``OverlapFilterStage``
    and ``ALMDataOverlapStage`` filter WITHIN a row, shrinking a segment list while every row
    survives -- so a tag without ``cardinality="filter"`` is a correct pairing, not a drift.
    Checking the converse would demand those stages declare a row cardinality they do not have.
    """
    if "is_filter" in (card.get("tags") or []):
        return []
    from nemo_curator.audio_agent._resolve import resolve_stage_class
    from nemo_curator.stages.audio import agent as foundation

    try:
        contract = foundation.build_contract(resolve_stage_class(stage_id)())
    except Exception:  # noqa: BLE001 - not default-buildable -> can't verify (skip, not a failure)
        return []
    if getattr(contract, "cardinality", None) != "filter":
        return []
    return [
        f"{stage_id}: the stage's DEFAULT contract declares cardinality='filter' but the card "
        f"has no 'is_filter' tag, so nothing planning a recipe knows this stage can drop rows"
    ]


# `provenance` is required because a card is read as current: without card_version and
# last_validated there is nothing to say WHEN its best_guess facts were last checked against the
# code, and a stale guess is indistinguishable from a fresh one. 47 of 49 cards carried it by
# convention; the two that did not were simply the newest, which is how an unenforced convention
# always fails.
_REQUIRED_FIELDS = ("category", "summary", "verified", "provenance")

# Every top-level key a card may carry, closed because the failure it prevents is silent: an
# unread key is absent from the host packet while the card still passes conformance. Two shipped
# cards wrote ``gotchas`` and ``relationships`` where readers expect ``counterexamples`` and
# ``comparison``, so their disambiguation prose reached nobody. Adding a key here is the
# deliberate half of adding a reader for it.
_KNOWN_CARD_FIELDS = frozenset(
    {
        "stage_id",
        "category",
        "summary",
        "tags",
        "model_id",
        "model_version",
        "domain",
        "constraints",
        "resource",
        "use_cases",
        "composition",
        "verified",
        "params_of_note",
        "provenance",
        "notes",
        "param_dependencies",
        "comparison",
        "semantic_facts",
        "conflicts_with",
        "presets",
        "caveats",
        "metrics",
        "versions",
        "deterministic",
        "decision",
    }
)


def _composition_violations(stage_id: str, card: dict[str, Any]) -> list[str]:
    """Check the stages a card recommends chaining with actually exist and don't contradict.

    ``composition`` is the part of a card an agent acts on most directly -- it is read as "these
    are the stages to put either side of this one" -- so a wrong name here is worse than a
    missing one. ``ALMDataBuilderStage`` recommended ``PrepareModuleSegmentsStage`` upstream for
    two card versions while that pairing raised ``TypeError`` on the first window, because the
    one writes ``metrics.bandwidth`` as a per-word list and the other compares it to an int.
    Nothing caught it: the recommendation was prose pointing at a name.

    Full edge simulation was considered and rejected -- building two default-constructed stages
    and validating them flags every pair needing params or a seed, and a gate that cries wolf
    gets ignored. These are the checks that cannot false-positive: a name must resolve, and a
    stage cannot be recommended and forbidden at once.
    """
    comp = card.get("composition")
    if not isinstance(comp, dict):
        return []
    v: list[str] = []
    typical: set[str] = set()
    for field in ("typical_upstream", "typical_downstream"):
        names = comp.get(field) or []
        if not isinstance(names, list):
            v.append(f"{stage_id}: composition.{field} must be a list of stage ids")
            continue
        if field == "typical_upstream":
            typical |= set(names)
        v += [
            f"{stage_id}: composition.{field} names {name!r}, which is not a registered stage"
            for name in names
            if _stage_param_names(str(name)) is None
        ]
    return v + _incompatible_violations(stage_id, comp, typical)


def _incompatible_violations(stage_id: str, comp: dict[str, Any], typical: set[str]) -> list[str]:
    """Violations in the optional ``incompatible_upstream`` map."""
    incompatible = comp.get("incompatible_upstream") or {}
    if not isinstance(incompatible, dict):
        return [f"{stage_id}: composition.incompatible_upstream must be a mapping of stage id -> reason"]
    v: list[str] = []
    for name, reason in incompatible.items():
        if _stage_param_names(str(name)) is None:
            v.append(f"{stage_id}: composition.incompatible_upstream names {name!r}, which is not a registered stage")
        if not str(reason or "").strip():
            v.append(f"{stage_id}: composition.incompatible_upstream[{name!r}] must say WHY, not just name the stage")
        if name in typical:
            v.append(f"{stage_id}: composition lists {name!r} as both typical_upstream and incompatible_upstream")
    return v


def _composite_legibility(stage_id: str) -> list[str]:
    """A composite must reveal its stages, or declare its own I/O; silence is not an option.

    A ``CompositeStage`` whose ``describe()`` returns a bare ``StageContract(wrappable=False)``
    tells a reader it has no reads and no writes, which is not the same as having unknown ones.
    Validation opens composites now, so this holds the door open: a new composite that neither
    decomposes at plan time nor states its own contract reintroduces the blind spot that let a
    ``segments``/``diar_segments`` mismatch survive validation and fail after two model
    downloads and a GPU diarization pass.
    """
    from nemo_curator.audio_agent._resolve import resolve_stage_class
    from nemo_curator.stages.audio._agent._composite import expand_composites

    try:
        cls = resolve_stage_class(stage_id)
        from nemo_curator.stages.base import CompositeStage

        if not (isinstance(cls, type) and issubclass(cls, CompositeStage)):
            return []
        instance = cls()
    except Exception:  # noqa: BLE001 - a composite needing constructor args is exercised by its own tests
        return []
    if expand_composites([instance]).fully_resolved:
        return []
    contract = getattr(instance, "describe", lambda: None)()
    if contract is not None and (contract.reads.data_keys or contract.writes.data_keys):
        return []
    return [
        (
            f"{stage_id}: composite neither decomposes at plan time nor declares its own "
            f"reads/writes, so nothing downstream of it can be validated"
        ),
    ]


def check_card(  # noqa: C901, PLR0912, PLR0915 - one branch per card section
    stage_id: str,
    card: Any,  # noqa: ANN401 - untrusted card boundary
) -> list[str]:
    """Return a list of mechanical conformance violations for one card (empty = ok)."""
    if not isinstance(card, dict):
        return [f"{stage_id}: card is not a mapping"]
    v: list[str] = []

    params = _stage_param_names(stage_id)
    if params is None:
        return [f"{stage_id}: stage_id does not resolve to a registered agent-ready stage"]

    # required structural fields (a card must state its category, a summary, and how
    # honest each fact is — the verified tiers).
    for f in _REQUIRED_FIELDS:
        if not card.get(f):
            v.append(f"{stage_id}: missing required field {f!r}")

    v += [
        f"{stage_id}: unknown top-level field {f!r}; nothing reads it, so its content reaches "
        f"nobody (allowed: {sorted(_KNOWN_CARD_FIELDS)})"
        for f in sorted(card)
        if f not in _KNOWN_CARD_FIELDS
    ]

    # params_of_note keys must be real constructor params (the drift catch).
    for k in card.get("params_of_note") or {}:
        if k not in params:
            v.append(f"{stage_id}: params_of_note lists {k!r} which is not a constructor param of the stage")

    # preset values are param bundles the agent applies as-is: every key must be a real
    # param, else applying the preset would raise bad_params (the asr batch_size drift).
    presets = card.get("presets") or {}
    if isinstance(presets, dict):
        for pname, pvals in presets.items():
            if isinstance(pvals, dict):
                for k in pvals:
                    if k not in params:
                        v.append(f"{stage_id}: preset {pname!r} sets {k!r} which is not a constructor param")

    # resource block: known keys + numeric where expected + valid bound.
    res = card.get("resource") or {}
    if isinstance(res, dict):
        for k, val in res.items():
            if k not in _KNOWN_RESOURCE_KEYS:
                v.append(f"{stage_id}: resource has unknown key {k!r} (allowed: {sorted(_KNOWN_RESOURCE_KEYS)})")
            elif k in _NUMERIC_RESOURCE_KEYS and val is not None and not isinstance(val, (int, float)):
                v.append(f"{stage_id}: resource.{k} must be a number or null, got {val!r}")
            elif k == "bound" and val is not None and val not in _KNOWN_BOUND:
                v.append(f"{stage_id}: resource.bound must be one of {sorted(_KNOWN_BOUND)} or null, got {val!r}")

    v.extend(_composition_violations(stage_id, card))
    v.extend(_composite_legibility(stage_id))
    if "decision" in card:
        v.extend(_decision_violations(stage_id, card["decision"]))

    # Model stages no longer have to pin a ``model_version``. Nothing ever compared it between
    # runs -- reuse identity uses the recipe's own value plus the source digest -- so the rule
    # demanded a value nobody checks, and where upstream ships no revision the only way to
    # satisfy it was to invent something pin-shaped. ``verified.model_version: best_guess``
    # already carries that caveat honestly.

    # metrics block (1A.2): the deterministic source of absolute targets. Validate its
    # shape so the config-strategy resolver can trust it (drift-proof anchors/presets).
    metrics = card.get("metrics") or {}
    if isinstance(metrics, dict):
        for mkey, mblock in metrics.items():
            if not isinstance(mblock, dict):
                v.append(f"{stage_id}: metrics[{mkey!r}] must be a mapping")
                continue
            scale = mblock.get("scale")
            if scale is not None and (not isinstance(scale, dict) or scale.get("direction") not in _DIRECTIONS):
                v.append(f"{stage_id}: metrics[{mkey!r}].scale needs direction in {sorted(_DIRECTIONS)} (+ min/max)")
            tp = mblock.get("threshold_param")
            if tp and tp not in params:
                v.append(f"{stage_id}: metrics[{mkey!r}].threshold_param {tp!r} is not a constructor param")
            for pname, pvals in (mblock.get("presets") or {}).items():
                if isinstance(pvals, dict):
                    for k in pvals:
                        if k not in params:
                            v.append(f"{stage_id}: metrics[{mkey!r}] preset {pname!r} sets non-param {k!r}")
            vr = mblock.get("valid_range")
            if vr is not None and not (isinstance(vr, list) and len(vr) == 2):  # noqa: PLR2004 - [lo, hi]
                v.append(f"{stage_id}: metrics[{mkey!r}].valid_range must be [lo, hi]")

    # A stage advertising a score must say what the score MEANS numerically. The checks above
    # validate a metrics block only IF one is present, so a scorer with no block at all passed
    # silently -- which is how BandwidthEstimationStage shipped `produces_score` with no scale,
    # no range and no direction. Nothing downstream could then derive a comparison operator, and
    # the generic filter the resolver emits for an annotate-only scorer is exactly where an
    # inverted filter comes from. Deliberately requires a BLOCK, not a direction: a categorical
    # metric (BandFilterStage's band_prediction) legitimately has no numeric scale.
    if "produces_score" in (card.get("tags") or []) and not metrics:
        v.append(
            f"{stage_id}: card tag 'produces_score' but no metrics block -- declare the score's "
            f"scale/direction (or valid values, if categorical) so a downstream filter cannot be inverted"
        )

    # Semantic facts are retrieval material for the host critic.  Validate
    # shape only; do not encode field meaning or scope into deterministic rules.
    v.extend(_semantic_fact_violations(stage_id, card.get("semantic_facts")))

    # versions block (optional, model-backed stages): {model_id: "when-to-use"} for
    # checkpoints verified interchangeable via model_name/model_path (same output structure,
    # no module code change). Keep it honest + drift-proof: a version an agent can *select*
    # (a preset that sets model_name/model_path) must be documented here.
    versions = card.get("versions")
    if versions is not None:
        if not isinstance(versions, dict) or not all(
            isinstance(mid, str) and isinstance(desc, str) for mid, desc in versions.items()
        ):
            v.append(f"{stage_id}: versions must be a mapping of {{model_id: 'when-to-use string'}}")
        else:
            if not card.get("model_id"):
                v.append(f"{stage_id}: versions is set but model_id is null (versions document model checkpoints)")
            if isinstance(presets, dict):
                for pname, pvals in presets.items():
                    if not isinstance(pvals, dict):
                        continue
                    for mk in ("model_name", "model_path"):
                        if mk in pvals and pvals[mk] not in versions:
                            v.append(
                                f"{stage_id}: preset {pname!r} selects {mk}={pvals[mk]!r} which is not documented in versions"
                            )

    # verified tiers, when present, must use the known vocabulary.
    verified = card.get("verified")
    if not isinstance(verified, dict):
        v.append(f"{stage_id}: verified must be a mapping of fact names to evidence tiers")
    else:
        for fact, tier in verified.items():
            if tier not in _VERIFIED_TIERS:
                v.append(f"{stage_id}: verified[{fact!r}]={tier!r} not in {sorted(_VERIFIED_TIERS)}")
        if card.get("semantic_facts") is not None and "semantic_facts" not in verified:
            v.append(f"{stage_id}: semantic_facts must declare its evidence tier in verified.semantic_facts")
        if "decision" in card and verified.get("decision") != "mechanical":
            v.append(
                f"{stage_id}: decision must declare mechanically checked evidence as verified.decision: mechanical"
            )
        if "decision" not in card and "decision" in verified:
            v.append(f"{stage_id}: verified.decision is set but the card has no decision block")

    # tag <-> default-gate consistency (M5b): a capability tag must reflect DEFAULT behavior.
    v.extend(_tag_gate_violations(stage_id, card))
    v.extend(_filter_tag_violations(stage_id, card))

    return v


def audit(index: Any = None) -> dict[str, Any]:  # noqa: ANN401
    """Audit all cards: mechanical violations + coverage (uncarded stages, orphan cards)."""
    from nemo_curator.audio_agent.index import get_index

    idx = index or get_index()
    cards = idx.all_cards()
    violations = {sid: vs for sid, vs in ((sid, check_card(sid, c)) for sid, c in cards.items()) if vs}
    carded = set(cards)
    all_stages = set(idx.stage_names())
    # Which stages the tag/gate check could not judge at all. It compares a card's tags against a
    # DEFAULT-CONSTRUCTED contract, so a stage with required constructor args yields None from
    # _effective_default_gates and is skipped -- correctly, since guessing is worse, but silently.
    # That blind spot is not academic: SnippetExtractionStage carried a `1:N fan-out` contract
    # with no `fanout` tag across card versions and nothing could see it, because the stage needs
    # constructor args. Report the set so a partial check does not read as full coverage.
    gate_unverified = sorted(s for s in (carded & all_stages) if _effective_default_gates(s) is None)
    return {
        "violations": violations,
        "orphan_cards": sorted(carded - all_stages),  # cards for a stage that no longer exists
        "uncarded_stages": sorted(all_stages - carded),  # stages still missing a card (coverage gap)
        "gate_unverified": gate_unverified,  # tags present but unjudgeable: not default-constructible
        "carded_count": len(carded),
        "stage_count": len(all_stages),
    }


def _blueprint_violations(blueprint_id: str, blueprint: Any) -> list[str]:  # noqa: ANN401
    """Check a blueprint's stage refs resolve and its presets name real parameters.

    Blueprints are shown to the planner as worked examples, and its presets are read as
    "these are the knobs for this pipeline" -- so a preset naming a parameter that does not
    exist is a confident instruction to do something impossible. Every shipped preset was
    wrong this way (``utmos_mos_threshold`` for what the stage calls ``mos_threshold``,
    ``wer_threshold`` for ``target_value``), and nothing caught it: the conformance gate
    covered cards only, while the blueprints declared ``validated: true``.

    A preset parameter is accepted when SOME stage the blueprint lists accepts it -- the
    presets are pipeline-level, so they are not attributable to one stage.
    """
    if not isinstance(blueprint, dict):
        return [f"{blueprint_id}: blueprint is not a mapping"]
    v: list[str] = []
    refs = [str(s.get("ref")) for s in (blueprint.get("stages") or []) if isinstance(s, dict) and s.get("ref")]
    accepted: set[str] = set()
    for ref in refs:
        params = _stage_param_names(ref)
        if params is None:
            v.append(f"{blueprint_id}: stages names {ref!r}, which is not a registered stage")
            continue
        accepted |= params
    presets = blueprint.get("presets") or {}
    if not isinstance(presets, dict):
        return [*v, f"{blueprint_id}: presets must be a mapping of name -> {{param: value}}"]
    for name, values in presets.items():
        if not isinstance(values, dict):
            v.append(f"{blueprint_id}: preset {name!r} must be a mapping of {{param: value}}")
            continue
        v += [
            f"{blueprint_id}: preset {name!r} sets {param!r}, which no stage in this blueprint accepts"
            for param in values
            if refs and param not in accepted
        ]
    return v


def audit_blueprints(index: Any = None) -> dict[str, Any]:  # noqa: ANN401
    """Mechanical violations across every blueprint (same contract as :func:`audit`)."""
    from nemo_curator.audio_agent.index import get_index

    idx = index or get_index()
    violations: dict[str, list[str]] = {}
    for blueprint in idx.blueprints():
        blueprint_id = str((blueprint or {}).get("blueprint_id") or "<unnamed>")
        found = _blueprint_violations(blueprint_id, blueprint)
        if found:
            violations[blueprint_id] = found
    return {"violations": violations, "blueprint_count": len(idx.blueprints())}


def main(argv: list[str] | None = None) -> int:
    """Print the audit as JSON; exit non-zero if any card has a mechanical violation."""
    import argparse

    ap = argparse.ArgumentParser(description="Audio-agent capability-card conformance gate")
    ap.add_argument(
        "--allow-uncarded",
        action="store_true",
        help="do not fail on coverage gaps (default: an uncarded stage fails the gate)",
    )
    args = ap.parse_args(argv)

    result = audit()
    # Blueprints are planner-facing worked examples, so a preset naming a parameter that
    # does not exist misleads exactly like a drifted card; gate them the same way.
    blueprints = audit_blueprints()
    result["blueprint_violations"] = blueprints["violations"]
    result["blueprint_count"] = blueprints["blueprint_count"]
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    # Drift, orphan cards, and coverage gaps are all hard failures. An uncarded stage used to be
    # reported only, which meant a stage could ship plannable with no card semantics behind it --
    # `discover` lists it, the planner may pick it, and the host critic gets no meaning, scope or
    # counterexample to reason from. --allow-uncarded is the deliberate escape hatch for a
    # work-in-progress stage; the flag existed for this and was simply never wired up.
    ok = not result["violations"] and not result["orphan_cards"] and not blueprints["violations"]
    if result["uncarded_stages"] and not args.allow_uncarded:
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
