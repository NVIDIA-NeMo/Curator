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

"""Regression tests for steward projection and manifest verification."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).parents[2]
PROJECTOR = PROJECT / ".stewards" / "project.py"
VERIFIER = PROJECT / ".stewards" / "verify.py"
MARKER = "<!-- generated from .stewards/manifest.toml — edit the manifest, not this file -->"


def _run(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    # The executable and script are repository-owned constants; only fixture paths vary.
    return subprocess.run(  # noqa: S603
        [sys.executable, str(script), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _fixture(root: Path, body: str, *, project: bool = False) -> Path:
    (root / ".stewards").mkdir(parents=True, exist_ok=True)
    (root / ".stewards" / "PROTOCOL.md").write_text("# Protocol\n", encoding="utf-8")
    manifest = root / ".stewards" / "manifest.toml"
    manifest.write_text(
        'network = "fixture"\nprotocol = ".stewards/PROTOCOL.md"\n' + body.strip() + "\n",
        encoding="utf-8",
    )
    if project:
        result = _run(PROJECTOR, "--repo", str(root), "--manifest", str(manifest))
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)
    return manifest


ROOT = """
[[steward]]
id = "root"
path = "AGENTS.md"

[[invariant]]
id = "root-contract"
steward = "root"
statement = "The fixture has an invariant."
severity = "P1"
verification = "none"
"""


class TestRepositoryStewardNetwork(unittest.TestCase):
    def test_repository_steward_network_is_current_and_covered(self) -> None:
        projected = _run(PROJECTOR, "--check")
        verified = _run(VERIFIER, "--coverage")

        assert projected.returncode == 0, projected.stdout + projected.stderr
        assert verified.returncode == 0, verified.stdout + verified.stderr

    def test_projector_detects_stale_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture(root, ROOT)
            (root / "AGENTS.md").write_text("stale\n", encoding="utf-8")

            result = _run(PROJECTOR, "--repo", str(root), "--manifest", str(manifest), "--check")

        assert result.returncode == 1
        assert "STALE maps" in result.stdout

    def test_root_projection_omits_maintenance_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root, ROOT, project=True)

            projected = (root / "AGENTS.md").read_text(encoding="utf-8")

        assert "## Network" not in projected
        assert "Automated backing" not in projected

    def test_projector_rejects_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture(
                root,
                ROOT.replace('path = "AGENTS.md"', 'path = "../AGENTS.md"'),
            )

            result = _run(PROJECTOR, "--repo", str(root), "--manifest", str(manifest))

        assert result.returncode == 2
        assert "unsafe generated map path" in result.stderr

    def test_verifier_rejects_path_escape_without_reading_outside_repo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture(
                root,
                ROOT.replace('path = "AGENTS.md"', 'path = "../AGENTS.md"'),
            )

            result = _run(VERIFIER, "--repo", str(root), "--manifest", str(manifest))

        assert result.returncode == 1
        assert "unsafe repository-relative path ../AGENTS.md" in result.stdout

    def test_verifier_rejects_broken_edge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture(
                root,
                ROOT.replace(
                    'path = "AGENTS.md"',
                    'path = "AGENTS.md"\nedges = [{ type = "routes", to = "missing" }]',
                ),
                project=True,
            )

            result = _run(VERIFIER, "--repo", str(root), "--manifest", str(manifest))

        assert result.returncode == 1
        assert "unknown edge target missing" in result.stdout

    def test_verifier_rejects_duplicate_ids_and_paths(self) -> None:
        duplicate = ROOT.replace(
            "[[invariant]]",
            '[[steward]]\nid = "root"\npath = "AGENTS.md"\n\n[[invariant]]',
            1,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture(root, duplicate, project=True)

            result = _run(VERIFIER, "--repo", str(root), "--manifest", str(manifest))

        assert result.returncode == 1
        assert "duplicate steward id" in result.stdout
        assert "duplicate steward path" in result.stdout

    def test_verifier_requires_manual_anchor(self) -> None:
        body = ROOT.replace(
            'verification = "none"',
            'verification = "manual"\nevidence_file = "contract.py"',
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "contract.py").write_text("CONTRACT = True\n", encoding="utf-8")
            manifest = _fixture(root, body, project=True)

            result = _run(VERIFIER, "--repo", str(root), "--manifest", str(manifest))

        assert result.returncode == 1
        assert "manual invariant needs evidence_file and anchor" in result.stdout

    def test_verifier_rejects_invalid_severity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture(root, ROOT.replace('severity = "P1"', 'severity = "critical"'), project=True)

            result = _run(VERIFIER, "--repo", str(root), "--manifest", str(manifest))

        assert result.returncode == 1
        assert "invalid severity critical" in result.stdout

    def test_verifier_rejects_false_machine_backing(self) -> None:
        body = """
[check.contract]
invoke = "python test_contract.py"
location = "test_contract.py"
proof_contains = "test_required_contract"
""" + ROOT.replace(
            'verification = "none"',
            'verification = "machine"\nenforced_by = "contract"',
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_contract.py").write_text("def test_other():\n    pass\n", encoding="utf-8")
            manifest = _fixture(root, body, project=True)

            result = _run(VERIFIER, "--repo", str(root), "--manifest", str(manifest))

        assert result.returncode == 1
        assert "proof text not found" in result.stdout

    def test_verifier_rejects_machine_check_without_proof_anchor(self) -> None:
        body = """
[check.contract]
invoke = "python test_contract.py"
location = "test_contract.py"
""" + ROOT.replace(
            'verification = "none"',
            'verification = "machine"\nenforced_by = "contract"',
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_contract.py").write_text("def test_contract():\n    pass\n", encoding="utf-8")
            manifest = _fixture(root, body, project=True)

            result = _run(VERIFIER, "--repo", str(root), "--manifest", str(manifest))

        assert result.returncode == 1
        assert "missing proof_contains" in result.stdout

    def test_verifier_rejects_uncovered_source_domain(self) -> None:
        body = 'coverage_roots = ["src/package"]\n' + ROOT
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src" / "package" / "unmapped"
            source.mkdir(parents=True)
            (source / "feature.py").write_text("VALUE = 1\n", encoding="utf-8")
            manifest = _fixture(root, body, project=True)

            result = _run(VERIFIER, "--repo", str(root), "--manifest", str(manifest), "--coverage")

        assert result.returncode == 1
        assert "uncovered code domain: src/package/unmapped" in result.stdout

    def test_verifier_accepts_reasoned_coverage_exemption(self) -> None:
        body = (
            """
coverage_roots = ["src/package"]
[coverage_exemptions]
"src/package/unmapped" = "Inherits the root contract in this fixture."
"""
            + ROOT
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src" / "package" / "unmapped"
            source.mkdir(parents=True)
            (source / "feature.py").write_text("VALUE = 1\n", encoding="utf-8")
            manifest = _fixture(root, body, project=True)

            result = _run(VERIFIER, "--repo", str(root), "--manifest", str(manifest), "--coverage")

        assert result.returncode == 0, result.stdout + result.stderr

    def test_verifier_rejects_orphan_generated_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture(root, ROOT, project=True)
            orphan = root / "orphan" / "AGENTS.md"
            orphan.parent.mkdir()
            orphan.write_text(MARKER + "\n", encoding="utf-8")

            result = _run(VERIFIER, "--repo", str(root), "--manifest", str(manifest))

        assert result.returncode == 1
        assert "orphan generated map: orphan/AGENTS.md" in result.stdout

    def test_verifier_enforces_active_context_budget(self) -> None:
        body = "max_active_bytes = 1\n" + ROOT
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture(root, body, project=True)

            result = _run(VERIFIER, "--repo", str(root), "--manifest", str(manifest))

        assert result.returncode == 1
        assert "active map chain exceeds budget" in result.stdout


if __name__ == "__main__":
    unittest.main()
