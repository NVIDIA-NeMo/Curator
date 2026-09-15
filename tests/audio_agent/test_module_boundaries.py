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

"""Underscored names that are really cross-module API.

Six helpers carry a leading underscore but are imported by other modules in this package. The
naming convention says "local detail"; the import graph says otherwise, and a cleanup that
believed the convention would take three modules with it.

What makes that failure quiet rather than loud: **most of these importers are lazy**, written
inside the function that needs them. A rename therefore survives import, survives collection,
and surfaces at runtime in the middle of a delta merge or a checkpoint resume -- far from the
edit that caused it. This turns that into a failure at the moment of the rename.

The pairs below are the same list quoted in each helper's docstring. Adding a consumer means
adding it in both places, which is the point: the docstring is the explanation and this is the
enforcement.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

_PACKAGE = Path(__file__).resolve().parents[2] / "nemo_curator" / "audio_agent"

# symbol -> (home module, modules that import it)
_CROSS_MODULE_PRIVATES: dict[str, tuple[str, frozenset[str]]] = {
    "_resume_breaks_on_disk_boundary": (
        "continuation",
        frozenset({"checkpoint", "delta", "reusable_pipeline", "reuse"}),
    ),
    "_ensure_private_dir": ("run_store", frozenset({"artifacts", "calibration_store"})),
    "_write_private_json": ("run_store", frozenset({"artifacts", "calibration_store"})),
    "_clean": ("contracts", frozenset({"report"})),
    "_row_count": ("report", frozenset({"verbs"})),
    "_dedup_stage_perf": ("report", frozenset({"verbs"})),
    "_known_dataset_keys": ("reuse", frozenset({"delta"})),
}


def _importers_of(symbol: str) -> set[str]:
    """Modules with a ``from ... import <symbol>``, at any nesting depth."""
    found: set[str] = set()
    for path in sorted(_PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(a.name == symbol for a in node.names):
                found.add(path.stem)
    return found


@pytest.mark.parametrize(("symbol", "home"), [(s, h) for s, (h, _) in _CROSS_MODULE_PRIVATES.items()])
def test_the_symbol_still_exists_where_its_consumers_look_for_it(symbol: str, home: str) -> None:
    """A lazy importer cannot fail until its code path runs. This fails at the rename instead."""
    module = importlib.import_module(f"nemo_curator.audio_agent.{home}")

    assert hasattr(module, symbol), (
        f"{home}.{symbol} is gone or renamed, and "
        f"{sorted(_CROSS_MODULE_PRIVATES[symbol][1])} import it -- most of them lazily, so "
        f"nothing would have failed until one of those code paths ran"
    )


@pytest.mark.parametrize(
    ("symbol", "home", "expected"),
    [(s, h, c) for s, (h, c) in _CROSS_MODULE_PRIVATES.items()],
)
def test_the_documented_consumer_list_is_the_real_one(symbol: str, home: str, expected: frozenset[str]) -> None:
    """Keeps the docstrings honest. A new consumer that is not recorded means the next person
    reading ``{home}.{symbol}`` is told the blast radius is smaller than it is."""
    actual = _importers_of(symbol) - {home}

    assert actual == set(expected), (
        f"{home}.{symbol}: imported by {sorted(actual)}, documented as {sorted(expected)}. "
        f"Update both this table and the 'Underscored but NOT private' line in its docstring."
    )


@pytest.mark.parametrize(("symbol", "home"), [(s, h) for s, (h, _) in _CROSS_MODULE_PRIVATES.items()])
def test_each_one_says_in_its_docstring_that_it_is_not_private(symbol: str, home: str) -> None:
    """The warning has to live where someone about to move the function will read it."""
    module = importlib.import_module(f"nemo_curator.audio_agent.{home}")
    doc = getattr(module, symbol).__doc__ or ""

    assert "NOT private" in doc, f"{home}.{symbol} is imported by other modules but its docstring does not say so"
