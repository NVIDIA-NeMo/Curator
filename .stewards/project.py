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
"""Render compact AGENTS.md maps from the NeMo Curator steward manifest."""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

MARKER = "<!-- generated from .stewards/manifest.toml — edit the manifest, not this file -->"
BACKING = {"machine": "machine-backed", "manual": "manual", "none": "none"}


def load_manifest(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _safe_relative(path: str) -> bool:
    candidate = Path(path)
    return bool(path) and not candidate.is_absolute() and ".." not in candidate.parts


def _proof(invariant: dict, checks: dict[str, dict]) -> str:
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


def render_node(
    steward: dict,
    invariants: list[dict],
    checks: dict[str, dict],
    judgment: dict,
) -> str:
    output = [
        MARKER,
        "",
        f"# Steward: {steward['id']}",
        "",
        steward.get("point_of_view", ""),
        "",
        "Ordinary work: use this map directly with the root map and run only affected checks.",
        "Do not open `.stewards/PROTOCOL.md` or `.stewards/manifest.toml` unless the task is an explicit review/audit or steward-network maintenance.",
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


def render_root(data: dict, grouped: dict[str, list[dict]]) -> str:
    output = [
        MARKER,
        "",
        f"# Agent Constitution — {data.get('network', 'repository')}",
        "",
        "Ordinary work: use this root map plus only scoped maps on the target path.",
        "Do not open `.stewards/PROTOCOL.md` or `.stewards/manifest.toml` unless the task is an explicit review/audit or steward-network maintenance.",
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
            "Explicit review/audit only: [.stewards/PROTOCOL.md](.stewards/PROTOCOL.md). "
            "Steward maintenance only: [.stewards/manifest.toml](.stewards/manifest.toml), "
            "then `python .stewards/verify.py --coverage`.",
        ]
    )
    return "\n".join(output).rstrip() + "\n"


def build_maps(data: dict) -> dict[str, str]:
    stewards = data.get("steward", [])
    grouped: dict[str, list[dict]] = {}
    for invariant in data.get("invariant", []):
        grouped.setdefault(invariant.get("steward", ""), []).append(invariant)
    maps: dict[str, str] = {}
    for steward in stewards:
        if steward["id"] == "root":
            maps[steward["path"]] = render_root(data, grouped)
        else:
            maps[steward["path"]] = render_node(
                steward,
                grouped.get(steward["id"], []),
                data.get("check", {}),
                data.get("judgment", {}).get(steward["id"], {}),
            )
    return maps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--print", dest="print_id")
    args = parser.parse_args()
    repo = args.repo.resolve()
    manifest = args.manifest or repo / ".stewards" / "manifest.toml"
    data = load_manifest(manifest)
    maps = build_maps(data)
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
    stale = [
        path for path, content in maps.items() if not (repo / path).exists() or (repo / path).read_text() != content
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
