# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stage names hardcoded in the agent, checked against the catalog they name.

Around twenty stage class names are written as string literals across eight modules --
``checks`` keying diarization behaviour, ``input_identity`` recognising manifest readers,
``verbs`` exempting a framework partitioner. Each site is deliberate and fail-closed, and
``verbs`` says why the exemption is by name: it "is a name, not a shape".

The cost of that design is that the catalog and its callers are coupled only through string
equality. Rename or drop a stage upstream and nothing here fails to import; the name simply
stops matching, and the behaviour it selected goes quiet -- a diarization special case that no
longer applies, a manifest reader no longer recognised as one. Finding every site means
grepping for it.

This does the grep, every run. It is deliberately NOT a registry: collapsing the sites into one
shared table would hide twenty different reasons behind a single list. The names stay where they
are, next to the code that explains them, and this fails when one stops being real.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from nemo_curator.audio_agent import verbs

_PACKAGE = Path(__file__).resolve().parents[2] / "nemo_curator" / "audio_agent"

# Real stage refs that do not carry the ``Stage`` suffix, so the scan below cannot spot them.
# ``ManifestReader`` is load-bearing: ``verbs`` rewrites a source to it and ``input_identity``
# dispatches on it to resolve where the data actually lives.
_UNSUFFIXED_REFS: frozenset[str] = frozenset({"ManifestReader"})

# Names that are legitimately absent from the AUDIO catalog. Each needs a reason and must still
# resolve to a real class -- an allowlist that only silences the check would defeat the point.
_NOT_IN_AUDIO_CATALOG: dict[str, tuple[str, str]] = {
    "FilePartitioningStage": (
        "nemo_curator.stages.file_partitioning",
        (
            "shared framework stage, not audio: ManifestReader delegates path discovery to it, "
            "and verbs exempts it from the AgentReady contract sweep for exactly that reason"
        ),
    ),
}


def _hardcoded_stage_names() -> dict[str, set[str]]:
    """Every stage name written as a string literal in executable code -> the modules using it.

    Docstrings are excluded: prose mentioning a stage is documentation, not a dependency, and
    including it turns a stale sentence into a failing test. Comments never reach the AST.

    Only exact ``...Stage`` identifiers count. ``index._MODULE_CATEGORY_RULES`` holds SUBSTRING
    patterns rather than refs (``"ManifestWriter"``, ``"resample"``, ``"datasets"``); none of
    them carries the suffix, so they stay out on their own. That matters most for
    ``"ManifestWriter"``, which is a perfectly correct substring rule and is not a stage name --
    checking it would fail for no reason. (``"ManifestReader"`` appears both as a rule there and
    as a real ref elsewhere, so ``index.py`` shows up among its users. Harmless: the name is
    genuinely in the catalog and genuinely load-bearing, and the extra module in a failure
    message costs nothing.)

    Known limit: this recognises a name it can see. Rewrite a call site to some entirely new
    string and the scan simply stops tracking the old one -- it catches the catalog moving out
    from under the callers, which is the failure that actually happens here, not local typos.
    """
    found: dict[str, set[str]] = {}
    for path in sorted(_PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and ast.get_docstring(node)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node) in docstrings:
                continue
            value = node.value
            is_ref = value in _UNSUFFIXED_REFS or (
                value.endswith("Stage") and value[:1].isupper() and value.isidentifier()
            )
            if is_ref:
                found.setdefault(value, set()).add(path.name)
    return found


_HARDCODED = _hardcoded_stage_names()


def test_the_scan_finds_the_names_it_is_supposed_to_guard() -> None:
    """Guard for the test itself. Tighten the AST walk by accident and this file keeps passing
    while checking nothing -- the same way a fixture with the wrong key silently checks nothing.
    """
    assert len(_HARDCODED) >= 20, f"expected ~22 hardcoded stage names, found {len(_HARDCODED)}"
    assert "ManifestReader" in _HARDCODED, "the unsuffixed refs are not being picked up"
    assert {"verbs.py", "checks.py", "input_identity.py"} <= set().union(*_HARDCODED.values())


@pytest.mark.parametrize("name", sorted(_HARDCODED))
def test_every_hardcoded_stage_name_still_exists(name: str) -> None:
    """The whole point: a rename upstream fails here, at the rename, rather than showing up
    later as a special case that quietly stopped applying."""
    catalog = {stage["stage"] for stage in verbs.discover()["stages"]}
    if name in catalog:
        return

    users = sorted(_HARDCODED[name])
    assert name in _NOT_IN_AUDIO_CATALOG, (
        f"{name!r} is hardcoded in {users} but is not in the audio catalog. Either it was "
        f"renamed or removed upstream -- in which case those modules are now selecting "
        f"behaviour on a name that never matches -- or it is a deliberate non-audio reference, "
        f"which belongs in _NOT_IN_AUDIO_CATALOG with a reason."
    )


@pytest.mark.parametrize("name", sorted(_NOT_IN_AUDIO_CATALOG))
def test_an_allowlisted_name_still_resolves_to_a_real_class(name: str) -> None:
    """An allowlist that only silenced the check would be worse than no check. A name excused
    from the audio catalog still has to be a class that exists."""
    module_path, _reason = _NOT_IN_AUDIO_CATALOG[name]
    module = pytest.importorskip(module_path)

    assert hasattr(module, name), f"{name!r} is allowlisted but no longer exists in {module_path}"


def test_the_allowlist_has_not_gone_stale() -> None:
    """If an allowlisted name joins the audio catalog, or stops being referenced at all, the
    entry and its reasoning are no longer true and should not keep vouching for it."""
    catalog = {stage["stage"] for stage in verbs.discover()["stages"]}
    for name in _NOT_IN_AUDIO_CATALOG:
        assert name not in catalog, f"{name!r} is in the audio catalog now; drop the allowlist entry"
        assert name in _HARDCODED, f"{name!r} is no longer hardcoded anywhere; drop the allowlist entry"
