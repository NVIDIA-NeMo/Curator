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

# ruff: noqa: INP001
"""Resolve layered steward sources and render compact AGENTS.md maps."""

from __future__ import annotations

import argparse
import copy
import sys
import tomllib
from pathlib import Path
from typing import Any, NoReturn

MARKER = "<!-- generated steward map; edit source layers, not this file -->"
BACKING = {"machine": "machine-backed", "manual": "manual", "none": "none"}
LAYER_KINDS = {"base", "domain", "overlay"}
LAYER_METADATA_FIELDS = {"id", "target", "kind", "owners"}
LAYER_CONTENT_FIELDS = {"root", "steward", "judgment", "invariant"}


class ManifestError(ValueError):
    """Raised when steward layers cannot be resolved safely."""


def _invalid(message: str) -> NoReturn:
    raise ManifestError(message)


def _safe_relative(path: object) -> bool:
    if not isinstance(path, str) or not path:
        return False
    candidate = Path(path)
    return not candidate.is_absolute() and ".." not in candidate.parts


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _repo_relative(path: Path, repo: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _extend_unique(current: list[Any], incoming: list[Any]) -> list[Any]:
    output = copy.deepcopy(current)
    for item in incoming:
        if item not in output:
            output.append(copy.deepcopy(item))
    return output


def _merge_mapping(current: dict[str, Any], incoming: dict[str, Any], *, protected_path: bool = False) -> None:
    for key, value in incoming.items():
        if key not in current:
            current[key] = copy.deepcopy(value)
        elif isinstance(current[key], list) and isinstance(value, list):
            current[key] = _extend_unique(current[key], value)
        elif isinstance(current[key], dict) and isinstance(value, dict):
            _merge_mapping(current[key], value)
        elif current[key] == value:
            continue
        elif protected_path and key == "path":
            _invalid(f"overlay cannot move steward map from {current[key]} to {value}")
        else:
            # Later layers have Helm-style scalar precedence.
            current[key] = copy.deepcopy(value)


def _merge_stewards(data: dict[str, Any], incoming: list[dict[str, Any]], target: str) -> None:
    stewards = data.setdefault("steward", [])
    for addition in incoming:
        steward_id = addition.get("id")
        if steward_id != target:
            _invalid(f"layer target {target} cannot define steward {steward_id}")
        existing = next((item for item in stewards if item.get("id") == steward_id), None)
        if existing is None:
            stewards.append(copy.deepcopy(addition))
        else:
            _merge_mapping(existing, addition, protected_path=True)


def _merge_invariants(data: dict[str, Any], incoming: list[dict[str, Any]], target: str) -> None:
    invariants = data.setdefault("invariant", [])
    for raw_addition in incoming:
        addition = copy.deepcopy(raw_addition)
        if addition.get("steward") != target:
            _invalid(f"layer target {target} cannot define invariant for {addition.get('steward')}")
        override = bool(addition.pop("override", False))
        existing_index = next(
            (index for index, item in enumerate(invariants) if item.get("id") == addition.get("id")),
            None,
        )
        if existing_index is None:
            invariants.append(addition)
        elif invariants[existing_index].get("steward") != target:
            _invalid(
                f"invariant id {addition.get('id')} already belongs to {invariants[existing_index].get('steward')}"
            )
        elif override:
            invariants[existing_index] = addition
        else:
            _invalid(f"duplicate invariant {addition.get('id')}; an overlay replacement needs override = true")


def _merge_fragment(data: dict[str, Any], fragment: dict[str, Any], target: str) -> None:
    stewards = fragment.pop("steward", [])
    invariants = fragment.pop("invariant", [])
    judgment = fragment.pop("judgment", {})
    root = fragment.pop("root", {})
    if root:
        if target != "root":
            _invalid(f"layer target {target} cannot define root guidance")
        _merge_mapping(data, root)
    if stewards:
        _merge_stewards(data, stewards, target)
    if invariants:
        _merge_invariants(data, invariants, target)
    if judgment:
        unexpected = set(judgment) - {target}
        if unexpected:
            _invalid(f"layer target {target} contains judgment for {', '.join(sorted(unexpected))}")
        resolved_judgment = data.setdefault("judgment", {}).setdefault(target, {})
        _merge_mapping(resolved_judgment, judgment.get(target, {}))
    _merge_mapping(data, fragment)


def _load_layer(  # noqa: C901, PLR0912
    repo: Path,
    source: str,
    seen_layer_ids: set[str],
    seen_targets: set[object],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not _safe_relative(source):
        _invalid(f"unsafe layer source path: {source}")
    source_path = (repo / source).resolve()
    if not source_path.is_relative_to(repo):
        _invalid(f"layer source resolves outside repository: {source}")
    if not source_path.is_file():
        _invalid(f"layer source missing: {source}")
    fragment = _read_toml(source_path)
    metadata = fragment.pop("layer", None)
    if not isinstance(metadata, dict):
        _invalid(f"layer source needs [layer] metadata: {source}")
    unknown_metadata = set(metadata) - LAYER_METADATA_FIELDS
    if unknown_metadata:
        _invalid(f"layer source has unknown metadata: {', '.join(sorted(unknown_metadata))}")
    unknown_content = set(fragment) - LAYER_CONTENT_FIELDS
    if unknown_content:
        _invalid(f"layer source has unknown content: {', '.join(sorted(unknown_content))}")

    layer_id = metadata.get("id")
    target = metadata.get("target")
    kind = metadata.get("kind")
    owners = metadata.get("owners", [])
    if not isinstance(layer_id, str) or not layer_id:
        _invalid(f"layer source has no id: {source}")
    if layer_id in seen_layer_ids:
        _invalid(f"duplicate layer id: {layer_id}")
    if not isinstance(target, str) or not target:
        _invalid(f"layer {layer_id} has no target")
    if kind not in LAYER_KINDS:
        _invalid(f"layer {layer_id} has invalid kind {kind}")
    if kind == "base" and target != "root":
        _invalid(f"base layer {layer_id} must target root")
    if kind == "domain" and target == "root":
        _invalid(f"domain layer {layer_id} cannot target root")
    if not isinstance(owners, list) or not all(isinstance(owner, str) for owner in owners):
        _invalid(f"layer {layer_id} owners must be a list of GitHub handles")
    if kind == "overlay" and target not in seen_targets:
        _invalid(f"overlay {layer_id} targets unknown steward {target}")
    if kind != "overlay" and target in seen_targets:
        _invalid(f"multiple base/domain layers target steward {target}")

    record = {
        "id": layer_id,
        "target": target,
        "kind": kind,
        "source": source,
        "owners": copy.deepcopy(owners),
    }
    return fragment, record


def _record_layer(data: dict[str, Any], record: dict[str, Any]) -> None:
    target = record["target"]
    data["_layers"].append(record)
    data["_steward_sources"].setdefault(target, []).append(record["source"])
    owner_list = data["_steward_owners"].setdefault(target, [])
    data["_steward_owners"][target] = _extend_unique(owner_list, record["owners"])


def load_manifest(path: Path, repo: Path | None = None) -> dict[str, Any]:
    """Load a manifest and deterministically resolve its ordered source layers."""
    manifest_path = path.resolve()
    repo = (repo or manifest_path.parents[1]).resolve()
    raw = _read_toml(manifest_path)
    layer_sources = raw.pop("layer_sources", [])
    if not isinstance(layer_sources, list) or not all(isinstance(item, str) for item in layer_sources):
        _invalid("layer_sources must be an ordered list of repository-relative paths")

    manifest_relative = _repo_relative(manifest_path, repo)
    data = {
        **copy.deepcopy(raw),
        "_manifest_path": manifest_relative,
        "_layers": [],
        "_steward_sources": {},
        "_steward_owners": {},
    }
    seen_layer_ids: set[str] = set()
    seen_targets = {item.get("id") for item in data.get("steward", [])}
    for source in layer_sources:
        fragment, record = _load_layer(repo, source, seen_layer_ids, seen_targets)
        _merge_fragment(data, fragment, record["target"])
        seen_layer_ids.add(record["id"])
        seen_targets.add(record["target"])
        _record_layer(data, record)

    # Inline manifests remain supported for small fixtures and downstream adopters.
    for steward in data.get("steward", []):
        steward_id = steward.get("id")
        data["_steward_sources"].setdefault(steward_id, [manifest_relative])
        data["_steward_owners"].setdefault(steward_id, [])
    return data


def _proof(invariant: dict[str, Any], checks: dict[str, dict[str, Any]]) -> str:
    if invariant.get("verification") == "machine":
        check_id = invariant.get("enforced_by", "?")
        invoke = checks.get(check_id, {}).get("invoke")
        if invoke:
            escaped_invoke = invoke.replace("|", "\\|")
            return f"`{escaped_invoke}` (`{check_id}`)"
        return f"`{check_id}`"
    if invariant.get("evidence_file"):
        anchor = invariant.get("anchor", "")
        return f"{invariant['evidence_file']} · `{anchor}`"
    return "—"


def _bullets(output: list[str], heading: str, items: list[str]) -> None:
    if items:
        output.extend(["", f"## {heading}", ""])
        output.extend(f"- {item}" for item in items)


def _source_header(data: dict[str, Any], steward_id: str) -> list[str]:
    sources = data.get("_steward_sources", {}).get(steward_id, [data.get("_manifest_path", "?")])
    owners = data.get("_steward_owners", {}).get(steward_id, [])
    source_links = ", ".join(f"`{source}`" for source in sources)
    owner_text = ", ".join(owners) if owners else "repository maintainers"
    return [
        MARKER,
        f"<!-- source layers: {', '.join(sources)} -->",
        "",
        f"> **Guidance owners:** {owner_text}. Update {source_links}.",
        "> Regenerate with `python .stewards/project.py`, then run `python .stewards/verify.py --coverage`.",
    ]


def render_node(
    data: dict[str, Any],
    steward: dict[str, Any],
    invariants: list[dict[str, Any]],
    checks: dict[str, dict[str, Any]],
    judgment: dict[str, Any],
) -> str:
    output = [
        *_source_header(data, steward["id"]),
        "",
        f"# Steward: {steward['id']}",
        "",
        steward.get("point_of_view", ""),
        "",
        "Ordinary work: use this map directly with the root map and run only affected checks.",
        "Open `.stewards/` only for explicit review, audit, or steward maintenance.",
        "",
        "## Protects",
        "",
        "| Invariant | Sev | Backing | Proof / anchor |",
        "| --- | --- | --- | --- |",
    ]
    for invariant in invariants:
        statement = invariant["statement"].replace("|", "\\|")
        output.append(
            f"| {statement} | {invariant.get('severity', '')} | "
            f"{BACKING.get(invariant.get('verification'), 'none')} | {_proof(invariant, checks)} |"
        )
    _bullets(output, "Guardrails", steward.get("guardrails", []))
    edges = steward.get("edges", [])
    if edges:
        output.extend(["", "## Edges", ""])
        output.extend(f"- {edge.get('type', '?')} → **{edge.get('to')}** ({edge.get('what', '')})" for edge in edges)
    owns = steward.get("owns", {})
    if owns:
        output.extend(["", "## Owns", ""])
        for key in ("code", "tests", "docs"):
            if owns.get(key):
                values = ", ".join(f"`{value}`" for value in owns[key])
                output.append(f"- **{key}:** {values}")
    _bullets(output, "Advocate", judgment.get("advocate", []))
    _bullets(output, "Do Not", judgment.get("do_not", []))
    _bullets(output, "Serves", judgment.get("serves", []))
    return "\n".join(output).rstrip() + "\n"


def render_root(data: dict[str, Any], grouped: dict[str, list[dict[str, Any]]]) -> str:
    output = [
        *_source_header(data, "root"),
        "",
        f"# Agent Constitution — {data.get('network', 'repository')}",
        "",
        "Ordinary work: use this root map plus only scoped maps on the target path.",
        "Open `.stewards/` only for explicit review, audit, or steward maintenance.",
        "",
        "## Pillars",
        "",
    ]
    output.extend(f"- {pillar}" for pillar in data.get("pillars", []))
    _bullets(output, "Search Discipline", data.get("search_policy", []))
    _bullets(output, "Operating Rules", data.get("operating_rules", []))
    root_invariants = grouped.get("root", [])
    if root_invariants:
        output.extend(
            [
                "",
                "## Protects (constitution)",
                "",
                "| Invariant | Sev | Backing | Proof / anchor |",
                "| --- | --- | --- | --- |",
            ]
        )
        for invariant in root_invariants:
            statement = invariant["statement"].replace("|", "\\|")
            output.append(
                f"| {statement} | {invariant.get('severity', '')} | "
                f"{BACKING.get(invariant.get('verification'), 'none')} | "
                f"{_proof(invariant, data.get('check', {}))} |"
            )
    _bullets(output, "Stop & Ask", data.get("stop_and_ask", []))
    _bullets(output, "Done Criteria", data.get("done_criteria", []))
    output.extend(
        [
            "",
            "---",
            "",
            "Steward authoring: [.stewards/README.md](.stewards/README.md). "
            "Explicit review/audit: [.stewards/PROTOCOL.md](.stewards/PROTOCOL.md).",
        ]
    )
    return "\n".join(output).rstrip() + "\n"


def build_maps(data: dict[str, Any]) -> dict[str, str]:
    stewards = data.get("steward", [])
    grouped: dict[str, list[dict[str, Any]]] = {}
    for invariant in data.get("invariant", []):
        grouped.setdefault(invariant.get("steward", ""), []).append(invariant)
    maps: dict[str, str] = {}
    for steward in stewards:
        if steward["id"] == "root":
            maps[steward["path"]] = render_root(data, grouped)
        else:
            maps[steward["path"]] = render_node(
                data,
                steward,
                grouped.get(steward["id"], []),
                data.get("check", {}),
                data.get("judgment", {}).get(steward["id"], {}),
            )
    return maps


def active_map_chain(path: str, steward_paths: set[str]) -> list[str]:
    """Return generated maps that layer for work beneath the target map."""
    target = Path(path).parent
    chain = ["AGENTS.md"] if "AGENTS.md" in steward_paths else []
    for candidate in sorted(steward_paths):
        if candidate in chain or candidate == path:
            continue
        parent = Path(candidate).parent
        if parent != Path() and (parent == target or parent in target.parents):
            chain.append(candidate)
    if path not in chain:
        chain.append(path)
    return chain


def explain(data: dict[str, Any], steward_id: str) -> str:
    stewards = data.get("steward", [])
    target = next((item for item in stewards if item.get("id") == steward_id), None)
    if target is None:
        _invalid(f"unknown steward: {steward_id}")
    by_path = {item["path"]: item["id"] for item in stewards}
    lines = [f"{steward_id} -> {target['path']}", "effective map chain:"]
    for map_path in active_map_chain(target["path"], set(by_path)):
        source_id = by_path[map_path]
        sources = data.get("_steward_sources", {}).get(source_id, [])
        owners = data.get("_steward_owners", {}).get(source_id, [])
        source_text = ", ".join(sources) or data.get("_manifest_path", "?")
        owner_text = ", ".join(owners) or "repository maintainers"
        lines.append(f"- {map_path} <= {source_text} ({owner_text})")
    return "\n".join(lines) + "\n"


def main() -> int:  # noqa: PLR0911
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--print", dest="print_id")
    parser.add_argument("--explain", dest="explain_id")
    args = parser.parse_args()
    repo = args.repo.resolve()
    manifest = args.manifest or repo / ".stewards" / "manifest.toml"
    try:
        data = load_manifest(manifest, repo)
        maps = build_maps(data)
    except (ManifestError, OSError, tomllib.TOMLDecodeError) as error:
        sys.stderr.write(f"invalid steward manifest: {error}\n")
        return 2

    unsafe = [path for path in maps if not _safe_relative(path)]
    if unsafe:
        sys.stderr.write("unsafe generated map path: " + ", ".join(unsafe) + "\n")
        return 2
    if args.print_id:
        steward = next((item for item in data.get("steward", []) if item["id"] == args.print_id), None)
        if not steward:
            sys.stderr.write(f"unknown steward: {args.print_id}\n")
            return 2
        sys.stdout.write(maps[steward["path"]])
        return 0
    if args.explain_id:
        try:
            sys.stdout.write(explain(data, args.explain_id))
        except ManifestError as error:
            sys.stderr.write(f"{error}\n")
            return 2
        return 0

    stale = [
        path
        for path, content in maps.items()
        if not (repo / path).exists() or (repo / path).read_text(encoding="utf-8") != content
    ]
    if args.check:
        if stale:
            sys.stdout.write("STALE maps: " + ", ".join(stale) + "\n")
            return 1
        sys.stdout.write(f"OK all {len(maps)} maps current.\n")
        return 0
    for path, content in maps.items():
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    sys.stdout.write(f"projected {len(maps)} maps ({len(stale)} changed).\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
