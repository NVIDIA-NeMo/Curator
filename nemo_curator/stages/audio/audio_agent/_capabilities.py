# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small card declarations for existing source and artifact protocols.

These declarations name constructor parameters, never executable code. Configured
stage contracts remain authoritative for writes and their sandboxable locations.
"""

from __future__ import annotations

import math
from typing import Any

_SOURCE_FIELDS = frozenset(
    {
        "kind",
        "path_param",
        "audio_filepath_key_param",
        "include_files_param",
        "extensions_param",
        "recursive_param",
        "limit_param",
    }
)
_OUTPUT_FIELDS = frozenset({"kind", "primary", "relocatable", "per_item"})
_KINDS = frozenset({"manifest", "audio_dir", "rttm_dir", "text_dir", "archive", "unknown"})


def io_violations(  # noqa: C901, PLR0912 - independent mechanical schema checks
    ref: str, card: dict[str, Any], params: set[str]
) -> list[str]:
    """Check parameter bindings without constructing a model or doing I/O."""
    if "io" not in card:
        return []
    from nemo_curator.stages.audio.audio_agent._safety import is_secret_key

    block = card["io"]
    prefix = f"{ref}: io"
    if not isinstance(block, dict) or set(block) - {"source", "outputs", "dependency_params"}:
        return [f"{prefix} must contain only source, outputs, and dependency_params"]
    issues: list[str] = []
    source = block.get("source")
    if source is not None:
        if not isinstance(source, dict) or set(source) - _SOURCE_FIELDS:
            issues.append(f"{prefix}.source has invalid fields")
        else:
            kind = source.get("kind")
            if not isinstance(kind, str) or kind not in {"manifest", "audio_folder"}:
                issues.append(f"{prefix}.source.kind must be manifest or audio_folder")
            if not source.get("path_param"):
                issues.append(f"{prefix}.source.path_param is required")
            for key, value in source.items():
                if key != "kind" and (not isinstance(value, str) or value not in params):
                    issues.append(f"{prefix}.source.{key} must name a real constructor parameter")
            if kind == "manifest" and set(source) & {"extensions_param", "recursive_param", "limit_param"}:
                issues.append(f"{prefix}.source manifest supports a single local manifest, not folder selectors")
            if kind == "audio_folder" and not source.get("limit_param"):
                issues.append(f"{prefix}.source audio_folder requires limit_param for bounded smoke")
    outputs = block.get("outputs", {})
    if not isinstance(outputs, dict):
        issues.append(f"{prefix}.outputs must map output parameters to declarations")
    else:
        primary = 0
        for param, spec in outputs.items():
            if param not in params or not isinstance(spec, dict) or set(spec) - _OUTPUT_FIELDS:
                issues.append(f"{prefix}.outputs[{param!r}] has invalid fields or parameter")
                continue
            kind = spec.get("kind", "unknown")
            if not isinstance(kind, str) or kind not in _KINDS:
                issues.append(f"{prefix}.outputs[{param!r}].kind is unsupported")
            for flag in ("primary", "relocatable", "per_item"):
                if flag in spec and not isinstance(spec[flag], bool):
                    issues.append(f"{prefix}.outputs[{param!r}].{flag} must be boolean")
            primary += spec.get("primary") is True
            if (spec.get("per_item") or spec.get("relocatable")) and spec.get("kind") != "manifest":
                issues.append(f"{prefix}.outputs[{param!r}] per_item/relocatable requires a JSONL manifest")
        if primary > 1:
            issues.append(f"{prefix}.outputs declares more than one primary artifact")
    deps = block.get("dependency_params", [])
    if not isinstance(deps, list) or any(not isinstance(p, str) or p not in params for p in deps):
        issues.append(f"{prefix}.dependency_params must list real constructor parameters")
    elif any(is_secret_key(param) for param in deps):
        issues.append(f"{prefix}.dependency_params cannot contain credential parameters")
    verified = card.get("verified")
    if not isinstance(verified, dict) or verified.get("io") != "mechanical":
        issues.append(f"{prefix} requires verified.io: mechanical")
    return issues


def io_for(ref: str) -> dict[str, Any]:
    """Return validated opt-in metadata; a malformed declaration fails closed."""
    from nemo_curator.stages.audio.audio_agent._resolve import static_contract_for
    from nemo_curator.stages.audio.audio_agent.index import get_index

    card = get_index().card(ref) or {}
    if "io" not in card:
        return {}
    issues = io_violations(ref, card, {p.name for p in static_contract_for(ref).params})
    if issues:
        raise ValueError("; ".join(issues))
    return card["io"]


def configured_values(ref: str, params: dict[str, Any]) -> dict[str, Any]:
    """Resolve constructor defaults without initializing model/runtime state."""
    from nemo_curator.stages.audio.audio_agent._resolve import static_contract_for

    return {**{p.name: p.default for p in static_contract_for(ref).params}, **params}


def planning_violations(ref: str, card: dict[str, Any], params: set[str]) -> list[str]:
    """Validate small factual planning hints, not executable predicates or prose."""
    if "planning" not in card:
        return []
    block = card["planning"]
    prefix = f"{ref}: planning"
    if not isinstance(block, dict) or set(block) - {"sample_rate_verification", "restores_continuity"}:
        return [f"{prefix} supports only sample_rate_verification and restores_continuity"]
    issues: list[str] = []
    for name, spec in block.items():
        fields = {"param", "when"} if name == "sample_rate_verification" else {"when"}
        if not isinstance(spec, dict) or set(spec) - fields:
            issues.append(f"{prefix}.{name} has invalid fields")
            continue
        if name == "sample_rate_verification" and (
            not isinstance(spec.get("param"), str) or spec["param"] not in params
        ):
            issues.append(f"{prefix}.{name}.param must name a real constructor parameter")
        when = spec.get("when", {})
        if not isinstance(when, dict) or any(
            key not in params
            or not isinstance(value, (str, int, float, bool, type(None)))
            or (isinstance(value, float) and not math.isfinite(value))
            for key, value in when.items()
        ):
            issues.append(f"{prefix}.{name}.when must map real parameters to literal scalar values")
    verified = card.get("verified")
    if not isinstance(verified, dict) or verified.get("planning") != "mechanical":
        issues.append(f"{prefix} requires verified.planning: mechanical and behavior tests")
    return issues


def planning_for(ref: str, params: dict[str, Any], *, card: dict[str, Any] | None = None) -> dict[str, Any]:
    """Configured hints whose literal preconditions hold; absent means unknown."""
    from nemo_curator.stages.audio.audio_agent._resolve import static_contract_for
    from nemo_curator.stages.audio.audio_agent.index import get_index

    if card is None:
        card = get_index().card(ref) or {}
    if "planning" not in card:
        return {}
    issues = planning_violations(ref, card, {p.name for p in static_contract_for(ref).params})
    if issues:
        raise ValueError("; ".join(issues))
    values = configured_values(ref, params)
    return {
        key: spec
        for key, spec in card["planning"].items()
        if all(values.get(param) == value for param, value in spec.get("when", {}).items())
    }


def output_paths(ref: str, params: dict[str, Any]) -> dict[str, str]:
    """Every enabled configured output, including paths omitted from the recipe."""
    from nemo_curator.stages.audio.audio_agent._resolve import resolved_contract_for

    try:
        resolved = resolved_contract_for(ref, params)
    except KeyError:
        return {}
    if not resolved.resolved or not resolved.contract.gates.writes_to_disk:
        return {}
    names = resolved.contract.gates.output_path_params or []
    from nemo_curator.stages.audio.audio_agent.input_identity import canonical_source, resolve_dataset_binding
    from nemo_curator.stages.audio.audio_agent.recipe import Recipe, StageRef

    binding = resolve_dataset_binding(Recipe(stages=[StageRef(ref, configured_values(ref, params))]))
    # Acquisition caches are inputs, not completed curated artifacts. A source
    # must opt in explicitly before its side outputs can become deliverables.
    if binding.status != "unsupported" and not io_for(ref).get("outputs"):
        return {}
    inputs = set(binding.configured_paths)
    return {
        name: value
        for name in names
        if isinstance(value := getattr(resolved.instance, name, None), str)
        and value
        and (not inputs or canonical_source(value) not in inputs)
    }


def selection_params(ref: str) -> tuple[str, str]:
    """Source selection columns, retaining the established legacy convention."""
    source = io_for(ref).get("source", {})
    if source:
        return source.get("include_files_param", ""), source.get("audio_filepath_key_param", "")
    return "include_files", "include_files_key"


def declared_paths(ref: str, params: dict[str, Any]) -> list[str]:
    """Path bindings used by the workspace gate, including renamed parameters."""
    from nemo_curator.stages.audio.audio_agent._safety import names_local_path

    paths = list(output_paths(ref, params).values())
    declared = io_for(ref)
    if declared:
        values = configured_values(ref, params)
        source = declared.get("source", {})
        for field in ("path_param", "include_files_param"):
            value = values.get(source.get(field))
            paths.extend(item for item in (value if isinstance(value, list) else [value]) if isinstance(item, str))
        paths.extend(
            value
            for param in declared.get("dependency_params", [])
            if isinstance(value := values.get(param), str) and names_local_path(value)
        )
    return paths


def generated_asset_paths(ref: str, params: dict[str, Any]) -> list[str]:
    """Configured deliverables are required even when their locations are relocatable."""
    from nemo_curator.stages.audio.audio_agent.recipe import OUTPUT_LOCATION_PARAMS

    options = io_for(ref).get("outputs", {})
    paths = output_paths(ref, params)
    return [
        path for key, path in paths.items() if len(paths) > 1 or key not in OUTPUT_LOCATION_PARAMS or key in options
    ]
