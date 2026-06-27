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

"""Audit `[tool.uv] override-dependencies` in pyproject.toml for staleness.

For each entry, runs a fresh `uv lock` with that entry removed and classifies:
  - load-bearing: removal breaks resolution (keep the override)
  - shaping:     resolves cleanly but the locked version violates the override
                 spec (override is actively pinning -- review whether the
                 alternate version is acceptable)
  - defensive:   removal still resolves to a satisfying version today, but the
                 override's *inverse* range is independently resolvable -- so
                 the override is protecting against an upstream regression
  - stale:       removal resolves cleanly to a satisfying version AND the
                 inverse range is unsatisfiable -- the override is a true no-op

Outputs a markdown report to stdout and optionally to --output / --json.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Iterator  # noqa: TC003 -- annotation use only, but the cost is nil
from dataclasses import asdict, dataclass
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

# What `uv lock` reads from the project: the spec, the package source (for the
# dynamic version attr in nemo_curator.package_info), and README.md (referenced
# as the project's `readme`). Anything else (tests, tutorials, docs, .git, ...)
# is irrelevant to resolution. Update this list if pyproject.toml starts
# referencing additional files.
LOCK_INPUTS = ("pyproject.toml", "nemo_curator", "README.md")


def stage_repo(repo: Path, dst: Path) -> None:
    """Copy just the files `uv lock` needs from `repo` into `dst`."""
    dst.mkdir(parents=True, exist_ok=True)
    for name in LOCK_INPUTS:
        src = repo / name
        target = dst / name
        if src.is_dir():
            shutil.copytree(src, target, symlinks=True)
        else:
            shutil.copy2(src, target)


@dataclass
class Result:
    spec: str
    category: str  # load-bearing | shaping | defensive | stale | error
    detail: str
    log_excerpt: str = ""


def load_overrides(pyproject: Path) -> list[str]:
    """Return every spec listed under [tool.uv] override-dependencies."""
    data = tomllib.loads(pyproject.read_text())
    return list(data.get("tool", {}).get("uv", {}).get("override-dependencies", []))


def remove_override_line(pyproject: Path, spec: str) -> None:
    """Strip the single line containing the given override spec."""
    text = pyproject.read_text()
    pattern = re.compile(
        r'^[ \t]*"' + re.escape(spec) + r'"[ \t]*,?[ \t]*(#.*)?\n',
        re.MULTILINE,
    )
    new_text, count = pattern.subn("", text, count=1)
    if count != 1:
        msg = f"Could not locate override line for {spec!r} in {pyproject}"
        raise RuntimeError(msg)
    pyproject.write_text(new_text)


def replace_override_line(pyproject: Path, old_spec: str, new_spec: str) -> None:
    """Replace the override entry for `old_spec` with `new_spec` in pyproject.toml."""
    text = pyproject.read_text()
    pattern = re.compile(
        r'^([ \t]*)"' + re.escape(old_spec) + r'"([ \t]*,?[ \t]*(?:#.*)?\n)',
        re.MULTILINE,
    )
    new_text, count = pattern.subn(
        lambda m: f'{m.group(1)}"{new_spec}"{m.group(2)}',
        text,
        count=1,
    )
    if count != 1:
        msg = f"Could not locate override line for {old_spec!r} in {pyproject}"
        raise RuntimeError(msg)
    pyproject.write_text(new_text)


def derive_inverse_specs(spec: str) -> list[str]:
    """Return inverse specs covering ranges the override currently excludes.

    A single-bound spec like `torchcodec>=0.9.0` yields `["torchcodec<0.9.0"]`.
    A compound spec like `numpy>=2.0.0,<=2.2.0` yields both
    `["numpy<2.0.0", "numpy>2.2.0"]`. Markers are preserved.
    Returns an empty list for ban overrides (no specifier) and pin/exclusion
    operators (`==`, `!=`, `~=`) that don't have a clean single-clause inverse.
    """
    try:
        req = Requirement(spec)
    except InvalidRequirement:
        return []
    if not req.specifier:
        return []  # ban override -- already covered by the removal test

    flip = {">=": "<", ">": "<=", "<=": ">", "<": ">="}
    inverses: list[str] = []
    for clause in req.specifier:
        inv_op = flip.get(clause.operator)
        if inv_op is None:
            continue  # skip ==, !=, ~=
        body = f"{req.name}{inv_op}{clause.version}"
        if req.marker:
            body = f"{body}; {req.marker}"
        inverses.append(body)
    return inverses


@contextlib.contextmanager
def _staged_repo(repo: Path) -> Iterator[Path]:
    """Yield a temp staging copy of `repo` containing only LOCK_INPUTS."""
    with tempfile.TemporaryDirectory(prefix="audit-overrides-") as td:
        dst = Path(td) / "repo"
        stage_repo(repo, dst)
        yield dst


def run_uv_lock(workdir: Path, timeout: int) -> tuple[bool, str]:
    """Run a fresh `uv lock` in workdir and return (success, combined output)."""
    lockfile = workdir / "uv.lock"
    if lockfile.exists():
        lockfile.unlink()
    proc = subprocess.run(
        ["uv", "lock"],  # noqa: S607 -- uv is provided on PATH by the workflow / dev env
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return proc.returncode == 0, (proc.stderr + proc.stdout)


def locked_version(lockfile: Path, name: str) -> str | None:
    """Return the resolved version of `name` from uv.lock, or None if absent."""
    canonical = name.lower().replace("_", "-")
    data = tomllib.loads(lockfile.read_text())
    for pkg in data.get("package", []):
        pkg_name = pkg.get("name", "").lower().replace("_", "-")
        if pkg_name == canonical:
            return pkg.get("version")
    return None


def _categorize(req: Requirement, ver: str | None) -> tuple[str, str]:
    """Decide (category, detail) given the parsed override and the resolved version."""
    # "Ban" override (no specifier, e.g. `apex; sys_platform == 'never'`):
    # the intent is to prevent the package from resolving anywhere. If
    # removing the override pulls it into the lock, the override is load-bearing.
    if not req.specifier:
        if ver is None:
            return "stale", f"{req.name} is not in the lock without the override"
        return "load-bearing", f"removing override pulls {req.name}=={ver} into the lock"

    if ver is None:
        return "stale", f"{req.name} is not in the lock without the override"

    try:
        satisfies = Version(ver) in SpecifierSet(str(req.specifier))
    except InvalidVersion:
        return "error", f"locked version {ver!r} for {req.name} is not PEP 440 compatible"

    if satisfies:
        return "stale", f"resolves to {req.name}=={ver}, already satisfies {req.specifier}"
    return "shaping", f"resolves to {req.name}=={ver} without override (override forces {req.specifier})"


def classify(spec: str, success: bool, log: str, lockfile: Path | None) -> Result:
    """Wrap a uv lock outcome into a Result with the appropriate category."""
    if not success:
        return Result(spec, "load-bearing", "removing the override breaks resolution", log[-600:])
    if lockfile is None:
        return Result(spec, "error", "lock succeeded but no lockfile was produced")

    try:
        req = Requirement(spec)
    except Exception as e:  # noqa: BLE001
        return Result(spec, "error", f"failed to parse spec: {e}")

    category, detail = _categorize(req, locked_version(lockfile, req.name))
    return Result(spec, category, detail)


def audit_one(repo: Path, spec: str, timeout: int) -> Result:
    """Drop the override, re-lock, classify, and on `stale` run the inverse test."""
    with _staged_repo(repo) as dst:
        remove_override_line(dst / "pyproject.toml", spec)
        ok, log = run_uv_lock(dst, timeout=timeout)
        primary = classify(spec, ok, log, dst / "uv.lock" if ok else None)

    if primary.category != "stale":
        return primary

    # Differentiate "truly stale" from "defensive": replace the override with
    # its inverse and check whether any package in the graph would otherwise
    # allow that range. If yes, the override is protective.
    inverses = derive_inverse_specs(spec)
    if not inverses:
        return primary  # not invertible; leave as stale

    for inverse in inverses:
        with _staged_repo(repo) as dst:
            replace_override_line(dst / "pyproject.toml", spec, inverse)
            inverse_ok, _ = run_uv_lock(dst, timeout=timeout)
        if inverse_ok:
            return Result(
                spec,
                "defensive",
                f"natural resolution satisfies the override, but `{inverse}` is also resolvable -- override is protective",
            )

    return Result(
        spec,
        "stale",
        f"{primary.detail}; inverse range is unsatisfiable -- truly redundant",
    )


def render_markdown(results: list[Result]) -> str:
    """Render the audit results as a categorized markdown report."""
    buckets = {"stale": [], "defensive": [], "shaping": [], "load-bearing": [], "error": []}
    for r in results:
        buckets.setdefault(r.category, []).append(r)

    out = ["# Override Dependencies Audit", ""]
    out.append(
        f"Audited {len(results)} override(s): "
        f"{len(buckets['stale'])} stale, "
        f"{len(buckets['defensive'])} defensive, "
        f"{len(buckets['shaping'])} shaping, "
        f"{len(buckets['load-bearing'])} load-bearing, "
        f"{len(buckets['error'])} error.",
    )
    out.append("")

    sections = [
        ("stale", "Stale (safe to remove)"),
        ("defensive", "Defensive (redundant today, protects against upstream regression)"),
        ("shaping", "Shaping (review -- override actively constrains resolution)"),
        ("load-bearing", "Load-bearing (keep)"),
        ("error", "Errors"),
    ]
    for key, title in sections:
        if not buckets[key]:
            continue
        out.append(f"## {title}")
        out.append("")
        for r in buckets[key]:
            out.append(f"- `{r.spec}` -- {r.detail}")
        out.append("")
    return "\n".join(out)


def main() -> int:
    """Parse CLI args, run the audit over every override, and emit reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Path to the Curator repo (default: cwd)")
    parser.add_argument("--output", type=Path, help="Write markdown report to this path")
    parser.add_argument("--json", dest="json_out", type=Path, help="Write JSON report to this path")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per `uv lock` in seconds")
    parser.add_argument("--only", action="append", default=[], help="Audit only specs matching substring (repeatable)")
    parser.add_argument("--fail-on-stale", action="store_true", help="Exit non-zero if any override is stale")
    args = parser.parse_args()

    repo = args.repo.resolve()
    overrides = load_overrides(repo / "pyproject.toml")
    if args.only:
        overrides = [o for o in overrides if any(s in o for s in args.only)]
    if not overrides:
        print("No overrides matched.", file=sys.stderr)
        return 0

    results: list[Result] = []
    for i, spec in enumerate(overrides, 1):
        print(f"[{i}/{len(overrides)}] Auditing: {spec}", file=sys.stderr)
        try:
            r = audit_one(repo, spec, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            r = Result(spec, "error", f"`uv lock` exceeded timeout of {args.timeout}s")
        except Exception as e:  # noqa: BLE001
            r = Result(spec, "error", f"audit failed: {e}")
        print(f"    -> {r.category}: {r.detail}", file=sys.stderr)
        results.append(r)

    md = render_markdown(results)
    print(md)
    if args.output:
        args.output.write_text(md)
    if args.json_out:
        args.json_out.write_text(json.dumps([asdict(r) for r in results], indent=2))

    if args.fail_on_stale and any(r.category == "stale" for r in results):
        return 1
    return 0


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        sys.exit(main())
