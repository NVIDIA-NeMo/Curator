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
"""Verify NeMo Curator's steward graph, evidence, checks, coverage, and map budgets."""

from __future__ import annotations

import argparse
import importlib.util
import shlex
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

MARKER = "<!-- generated steward map; edit source layers, not this file -->"
LEGACY_MARKER = "<!-- generated from .stewards/manifest.toml — edit the manifest, not this file -->"
MANAGED_MARKERS = {MARKER, LEGACY_MARKER}
CODEOWNERS_MIN_FIELDS = 2
SEVERITIES = {"P0", "P1", "P2", "P3"}


def _load_projector(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("steward_projector", path)
    if spec is None or spec.loader is None:
        message = "unable to load steward projector"
        raise RuntimeError(message)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _looks_local(token: str) -> bool:
    roots = {
        ".",
        ".github",
        "benchmarking",
        "docs",
        "fern",
        "nemo_curator",
        "scripts",
        "tests",
        "tutorials",
    }
    prefixes = tuple(f"{root}/" for root in roots if root != ".")
    return token in roots or token.startswith(prefixes)


def _safe_relative(path: object) -> bool:
    if not isinstance(path, str) or not path:
        return False
    candidate = Path(path)
    return not candidate.is_absolute() and ".." not in candidate.parts


def _validate_path(errors: list[str], label: str, path: object) -> bool:
    if not _safe_relative(path):
        errors.append(f"{label}: unsafe repository-relative path {path}")
        return False
    return True


def _verify_graph(repo: Path, data: dict[str, Any], errors: list[str]) -> tuple[set[str], set[str]]:
    stewards = data.get("steward", [])
    ids = [item.get("id") for item in stewards]
    paths = [item.get("path") for item in stewards]
    if len(ids) != len(set(ids)):
        errors.append("duplicate steward id")
    if len(paths) != len(set(paths)):
        errors.append("duplicate steward path")
    steward_ids = set(ids)
    steward_paths = set(paths)
    if "root" not in steward_ids or "AGENTS.md" not in steward_paths:
        errors.append("root steward must own AGENTS.md")

    protocol = data.get("protocol")
    if not _validate_path(errors, "protocol", protocol) or not (repo / protocol).is_file():
        errors.append(f"protocol file missing: {protocol}")
    for steward in stewards:
        steward_id = steward.get("id", "?")
        _validate_path(errors, f"{steward_id} map", steward.get("path"))
        for edge in steward.get("edges", []):
            if edge.get("to") not in steward_ids:
                errors.append(f"{steward_id}: unknown edge target {edge.get('to')}")
    return steward_ids, steward_paths


def _verify_checks(repo: Path, checks: dict[str, Any], errors: list[str]) -> None:
    for check_id, check in checks.items():
        invoke = check.get("invoke")
        if not invoke:
            errors.append(f"check {check_id}: missing invoke")
        else:
            for token in shlex.split(invoke):
                if _looks_local(token) and not (repo / token).exists():
                    errors.append(f"check {check_id}: command path does not exist: {token}")

        location = check.get("location")
        location_safe = _validate_path(errors, f"check {check_id} location", location)
        if not location_safe or not (repo / location).is_file():
            errors.append(f"check {check_id}: missing location {location}")
        proof = check.get("proof_contains")
        if not proof:
            errors.append(f"check {check_id}: missing proof_contains")
        elif location_safe and proof not in (repo / location).read_text(encoding="utf-8", errors="ignore"):
            errors.append(f"check {check_id}: proof text not found in {location}: {proof}")


def _verify_layers(repo: Path, data: dict[str, Any], errors: list[str]) -> None:
    layers = data.get("_layers", [])
    if not layers:
        return
    codeowners_path = repo / ".github" / "CODEOWNERS"
    if not codeowners_path.is_file():
        errors.append("layered steward sources require .github/CODEOWNERS")
        return
    entries: dict[str, list[str]] = {}
    for raw_line in codeowners_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= CODEOWNERS_MIN_FIELDS:
            entries[parts[0].lstrip("/")] = parts[1:]

    for layer in layers:
        layer_id = layer["id"]
        source = layer["source"]
        owners = layer.get("owners", [])
        if not owners:
            errors.append(f"layer {layer_id} needs at least one guidance owner")
            continue
        invalid = [owner for owner in owners if not owner.startswith("@")]
        if invalid:
            errors.append(f"layer {layer_id} has invalid owner handles: {', '.join(invalid)}")
        registered = entries.get(source, [])
        missing = [owner for owner in owners if owner not in registered]
        if missing:
            errors.append(f"layer {layer_id} CODEOWNERS entry {source} is missing: {', '.join(missing)}")


def _verify_invariants(
    repo: Path,
    invariants: list[dict[str, Any]],
    checks: dict[str, Any],
    steward_ids: set[str],
    errors: list[str],
) -> None:
    invariant_ids: set[str] = set()
    invariant_stewards: set[str] = set()
    for invariant in invariants:
        invariant_id = invariant.get("id", "?")
        if invariant_id in invariant_ids:
            errors.append(f"duplicate invariant id: {invariant_id}")
        invariant_ids.add(invariant_id)
        steward = invariant.get("steward")
        invariant_stewards.add(steward)
        if steward not in steward_ids:
            errors.append(f"{invariant_id}: unknown steward {steward}")

        verification = invariant.get("verification")
        if verification not in {"machine", "manual", "none"}:
            errors.append(f"{invariant_id}: invalid verification {verification}")
        if verification == "machine" and invariant.get("enforced_by") not in checks:
            errors.append(f"{invariant_id}: unknown check {invariant.get('enforced_by')}")
        severity = invariant.get("severity")
        if severity not in SEVERITIES:
            errors.append(f"{invariant_id}: invalid severity {severity}")
        _verify_evidence(repo, invariant, errors)

    for steward_id in steward_ids - invariant_stewards:
        errors.append(f"steward has no invariant: {steward_id}")


def _verify_evidence(repo: Path, invariant: dict[str, Any], errors: list[str]) -> None:
    invariant_id = invariant.get("id", "?")
    evidence = invariant.get("evidence_file")
    anchor = invariant.get("anchor")
    if invariant.get("verification") == "manual" and (not evidence or not anchor):
        errors.append(f"{invariant_id}: manual invariant needs evidence_file and anchor")
    if not evidence or not _validate_path(errors, f"{invariant_id} evidence", evidence):
        return
    evidence_path = repo / evidence
    if not evidence_path.exists():
        errors.append(f"{invariant_id}: missing evidence {evidence}")
    elif anchor and anchor not in evidence_path.read_text(encoding="utf-8", errors="ignore"):
        errors.append(f"{invariant_id}: missing anchor in {evidence}: {anchor}")


def _verify_coverage(repo: Path, data: dict[str, Any], steward_paths: set[str], errors: list[str]) -> None:
    exemptions = data.get("coverage_exemptions", {})
    for root_name in data.get("coverage_roots", []):
        if not _validate_path(errors, "coverage root", root_name):
            continue
        root = repo / root_name
        if not root.is_dir():
            errors.append(f"coverage root missing: {root_name}")
            continue
        for child in sorted(item for item in root.iterdir() if item.is_dir()):
            if not any(child.rglob("*.py")):
                continue
            relative_child = child.relative_to(repo).as_posix()
            relative_map = f"{relative_child}/AGENTS.md"
            if relative_map not in steward_paths and not exemptions.get(relative_child):
                errors.append(f"uncovered code domain: {relative_child}")
    _verify_exemptions(repo, exemptions, errors)


def _verify_exemptions(repo: Path, exemptions: dict[str, str], errors: list[str]) -> None:
    for path, reason in exemptions.items():
        if not _validate_path(errors, "coverage exemption", path):
            continue
        if not reason:
            errors.append(f"coverage exemption needs a reason: {path}")
        if not (repo / path).is_dir():
            errors.append(f"coverage exemption path missing: {path}")


def _verify_maps(repo: Path, data: dict[str, Any], errors: list[str]) -> None:
    projector = _load_projector(Path(__file__).with_name("project.py"))
    maps = {path: content for path, content in projector.build_maps(data).items() if _safe_relative(path)}
    max_bytes = int(data.get("max_active_bytes", 24576))
    for path, expected in maps.items():
        target = repo / path
        if not target.exists() or target.read_text(encoding="utf-8") != expected:
            errors.append(f"stale generated map: {path}")
        chain = projector.active_map_chain(path, set(maps))
        active_bytes = sum(len(maps[item].encode()) for item in chain)
        if active_bytes > max_bytes:
            errors.append(f"active map chain exceeds budget: {path} ({active_bytes} > {max_bytes})")
    for candidate in repo.rglob("AGENTS.md"):
        relative = candidate.relative_to(repo).as_posix()
        content = candidate.read_text(encoding="utf-8", errors="ignore")
        if any(marker in content for marker in MANAGED_MARKERS) and relative not in maps:
            errors.append(f"orphan generated map: {relative}")


def verify(repo: Path, manifest_path: Path, coverage: bool) -> list[str]:
    projector = _load_projector(Path(__file__).with_name("project.py"))
    try:
        data = projector.load_manifest(manifest_path, repo)
    except (projector.ManifestError, OSError, tomllib.TOMLDecodeError) as error:
        return [f"invalid steward manifest: {error}"]
    errors: list[str] = []
    _verify_layers(repo, data, errors)
    steward_ids, steward_paths = _verify_graph(repo, data, errors)
    checks = data.get("check", {})
    invariants = data.get("invariant", [])
    _verify_checks(repo, checks, errors)
    _verify_invariants(repo, invariants, checks, steward_ids, errors)
    if coverage:
        _verify_coverage(repo, data, steward_paths, errors)
    _verify_maps(repo, data, errors)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--coverage", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    manifest = args.manifest or repo / ".stewards" / "manifest.toml"
    errors = verify(repo, manifest, args.coverage)
    if errors:
        sys.stdout.write("Steward verification failed:\n")
        sys.stdout.write("".join(f"- {error}\n" for error in errors))
        return 1
    projector = _load_projector(Path(__file__).with_name("project.py"))
    data = projector.load_manifest(manifest, repo)
    invariants = data.get("invariant", [])
    machine = sum(item.get("verification") == "machine" for item in invariants)
    manual = sum(item.get("verification") == "manual" for item in invariants)
    none = sum(item.get("verification") == "none" for item in invariants)
    sys.stdout.write(
        f"OK {len(data.get('steward', []))} stewards, {len(invariants)} invariants "
        f"({machine} machine, {manual} manual, {none} none).\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
