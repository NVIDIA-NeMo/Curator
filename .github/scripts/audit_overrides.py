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
  - stale:       resolves cleanly with a satisfying version, or the package is
                 not in the lock at all (override is a no-op -- removable)

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
from dataclasses import asdict, dataclass
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

IGNORE_PATTERNS = shutil.ignore_patterns(
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "*.egg-info",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "node_modules",
    "build",
    "dist",
)


@dataclass
class Result:
    spec: str
    category: str  # load-bearing | shaping | stale | error
    detail: str
    log_excerpt: str = ""


def load_overrides(pyproject: Path) -> list[str]:
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


def run_uv_lock(workdir: Path, timeout: int) -> tuple[bool, str]:
    lockfile = workdir / "uv.lock"
    if lockfile.exists():
        lockfile.unlink()
    proc = subprocess.run(
        ["uv", "lock"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return proc.returncode == 0, (proc.stderr + proc.stdout)


def locked_version(lockfile: Path, name: str) -> str | None:
    canonical = name.lower().replace("_", "-")
    data = tomllib.loads(lockfile.read_text())
    for pkg in data.get("package", []):
        pkg_name = pkg.get("name", "").lower().replace("_", "-")
        if pkg_name == canonical:
            return pkg.get("version")
    return None


def classify(spec: str, success: bool, log: str, lockfile: Path | None) -> Result:
    if not success:
        return Result(spec, "load-bearing", "removing the override breaks resolution", log[-600:])

    assert lockfile is not None
    try:
        req = Requirement(spec)
    except Exception as e:  # noqa: BLE001
        return Result(spec, "error", f"failed to parse spec: {e}")

    ver = locked_version(lockfile, req.name)

    # "Ban" override (no specifier, e.g. `apex; sys_platform == 'never'`):
    # The intent is to prevent the package from ever resolving. If removing
    # the override pulls it into the lock, the override is load-bearing.
    if not req.specifier:
        if ver is None:
            return Result(spec, "stale", f"{req.name} is not in the lock without the override")
        return Result(spec, "load-bearing", f"removing override pulls {req.name}=={ver} into the lock")

    if ver is None:
        return Result(spec, "stale", f"{req.name} is not in the lock without the override")

    try:
        if Version(ver) in SpecifierSet(str(req.specifier)):
            return Result(spec, "stale", f"resolves to {req.name}=={ver}, already satisfies {req.specifier}")
    except InvalidVersion:
        return Result(spec, "error", f"locked version {ver!r} for {req.name} is not PEP 440 compatible")

    return Result(
        spec,
        "shaping",
        f"resolves to {req.name}=={ver} without override (override forces {req.specifier})",
    )


def audit_one(repo: Path, spec: str, timeout: int) -> Result:
    with tempfile.TemporaryDirectory(prefix="audit-overrides-") as td:
        dst = Path(td) / "repo"
        shutil.copytree(repo, dst, ignore=IGNORE_PATTERNS, symlinks=True)
        remove_override_line(dst / "pyproject.toml", spec)
        ok, log = run_uv_lock(dst, timeout=timeout)
        return classify(spec, ok, log, dst / "uv.lock" if ok else None)


def render_markdown(results: list[Result]) -> str:
    buckets = {"stale": [], "shaping": [], "load-bearing": [], "error": []}
    for r in results:
        buckets.setdefault(r.category, []).append(r)

    out = ["# Override Dependencies Audit", ""]
    out.append(
        f"Audited {len(results)} override(s): "
        f"{len(buckets['stale'])} stale, "
        f"{len(buckets['shaping'])} shaping, "
        f"{len(buckets['load-bearing'])} load-bearing, "
        f"{len(buckets['error'])} error.",
    )
    out.append("")

    sections = [
        ("stale", "Stale (safe to remove)"),
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
