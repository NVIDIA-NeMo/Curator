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
MARKER = "<!-- generated steward map; edit source layers, not this file -->"


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


def _layered_fixture(
    root: Path,
    *,
    include_overlay: bool = False,
    include_codeowners: bool = True,
    duplicate_overlay_invariant: bool = False,
) -> Path:
    stewards = root / ".stewards"
    layers = stewards / "layers"
    layers.mkdir(parents=True)
    (stewards / "PROTOCOL.md").write_text("# Protocol\n", encoding="utf-8")
    layer_sources = [
        '".stewards/layers/repository.toml"',
        '".stewards/layers/docs.toml"',
    ]
    if include_overlay:
        layer_sources.append('".stewards/layers/docs-overlay.toml"')
    manifest = stewards / "manifest.toml"
    manifest.write_text(
        f'network = "fixture"\nprotocol = ".stewards/PROTOCOL.md"\nlayer_sources = [{", ".join(layer_sources)}]\n',
        encoding="utf-8",
    )
    (layers / "repository.toml").write_text(
        """
[layer]
id = "repository"
target = "root"
kind = "base"
owners = ["@root-team"]

[root]
pillars = ["Root guidance."]

[[steward]]
id = "root"
path = "AGENTS.md"

[[invariant]]
id = "root-contract"
steward = "root"
statement = "The fixture has a root invariant."
severity = "P1"
verification = "none"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (layers / "docs.toml").write_text(
        """
[layer]
id = "docs"
target = "docs"
kind = "domain"
owners = ["@docs-team"]

[[steward]]
id = "docs"
path = "docs/AGENTS.md"
guardrails = ["Use the canonical documentation source."]

[[invariant]]
id = "docs-contract"
steward = "docs"
statement = "Documentation guidance remains scoped."
severity = "P2"
verification = "none"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    if include_overlay:
        invariant = """

[[invariant]]
id = "docs-contract"
steward = "docs"
statement = "The overlay silently replaces the domain contract."
severity = "P1"
verification = "none"
"""
        (layers / "docs-overlay.toml").write_text(
            """
[layer]
id = "docs-special"
target = "docs"
kind = "overlay"
owners = ["@docs-team"]

[[steward]]
id = "docs"
guardrails = ["Apply the narrower documentation convention."]
""".strip()
            + (invariant if duplicate_overlay_invariant else "")
            + "\n",
            encoding="utf-8",
        )
    if include_codeowners:
        codeowners = root / ".github" / "CODEOWNERS"
        codeowners.parent.mkdir(parents=True)
        entries = [
            ".stewards/layers/repository.toml @root-team",
            ".stewards/layers/docs.toml @docs-team",
        ]
        if include_overlay:
            entries.append(".stewards/layers/docs-overlay.toml @docs-team")
        codeowners.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return manifest


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

    def test_layer_declaration_spawns_owned_scoped_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _layered_fixture(root)

            result = _run(PROJECTOR, "--repo", str(root), "--manifest", str(manifest))
            projected = (root / "docs" / "AGENTS.md").read_text(encoding="utf-8")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "**Guidance owners:** @docs-team" in projected
        assert "Update `.stewards/layers/docs.toml`" in projected
        assert "Use the canonical documentation source." in projected

    def test_explain_shows_effective_source_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _layered_fixture(root)

            result = _run(
                PROJECTOR,
                "--repo",
                str(root),
                "--manifest",
                str(manifest),
                "--explain",
                "docs",
            )

        assert result.returncode == 0, result.stdout + result.stderr
        assert "AGENTS.md <= .stewards/layers/repository.toml (@root-team)" in result.stdout
        assert "docs/AGENTS.md <= .stewards/layers/docs.toml (@docs-team)" in result.stdout

    def test_overlay_adds_guidance_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _layered_fixture(root, include_overlay=True)

            result = _run(PROJECTOR, "--repo", str(root), "--manifest", str(manifest))
            projected = (root / "docs" / "AGENTS.md").read_text(encoding="utf-8")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "Use the canonical documentation source." in projected
        assert "Apply the narrower documentation convention." in projected
        assert ".stewards/layers/docs.toml, .stewards/layers/docs-overlay.toml" in projected

    def test_overlay_cannot_silently_replace_invariant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _layered_fixture(
                root,
                include_overlay=True,
                duplicate_overlay_invariant=True,
            )

            result = _run(PROJECTOR, "--repo", str(root), "--manifest", str(manifest))

        assert result.returncode == 2
        assert "an overlay replacement needs override = true" in result.stderr

    def test_verifier_requires_codeowners_for_layer_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _layered_fixture(root, include_codeowners=False)
            projected = _run(PROJECTOR, "--repo", str(root), "--manifest", str(manifest))
            assert projected.returncode == 0, projected.stdout + projected.stderr

            result = _run(VERIFIER, "--repo", str(root), "--manifest", str(manifest))

        assert result.returncode == 1
        assert "layered steward sources require .github/CODEOWNERS" in result.stdout

    def test_projector_rejects_layer_source_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _layered_fixture(root)
            manifest.write_text(
                manifest.read_text(encoding="utf-8").replace(
                    '".stewards/layers/docs.toml"',
                    '"../docs.toml"',
                ),
                encoding="utf-8",
            )

            result = _run(PROJECTOR, "--repo", str(root), "--manifest", str(manifest))

        assert result.returncode == 2
        assert "unsafe layer source path" in result.stderr

    def test_projector_rejects_layer_symlink_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / "repo"
            root.mkdir()
            manifest = _layered_fixture(root)
            source = root / ".stewards" / "layers" / "docs.toml"
            outside = workspace / "docs.toml"
            outside.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            source.unlink()
            source.symlink_to(outside)

            result = _run(PROJECTOR, "--repo", str(root), "--manifest", str(manifest))

        assert result.returncode == 2
        assert "layer source resolves outside repository" in result.stderr

    def test_projector_rejects_unknown_layer_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _layered_fixture(root)
            source = root / ".stewards" / "layers" / "docs.toml"
            source.write_text(
                source.read_text(encoding="utf-8").replace(
                    'owners = ["@docs-team"]',
                    'owners = ["@docs-team"]\nowner = "@typo"',
                ),
                encoding="utf-8",
            )

            result = _run(PROJECTOR, "--repo", str(root), "--manifest", str(manifest))

        assert result.returncode == 2
        assert "unknown metadata: owner" in result.stderr

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
