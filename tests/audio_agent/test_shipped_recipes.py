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

"""Every shipped recipe states what "done" means, and scratch recipes have somewhere to live.

A template with no success contract teaches the next author to omit one. That is not
hypothetical: a run reasoning "the templates skip acceptance criteria, so they are optional"
shipped a dataset whose rows were empty, reported as complete.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest
import yaml

from nemo_curator.audio_agent.acceptance import parse_criteria
from nemo_curator.audio_agent.run_store import runs_dir, scratch_dir

_RECIPES = sorted((Path(__file__).resolve().parents[2] / "nemo_curator/audio_agent/recipes").glob("*.yaml"))


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _keys_the_recipe_writes(recipe: dict) -> set[str]:
    """Every key the recipe's stages declare they write, composites expanded.

    Built from the same contracts validation uses, so this asks the question the acceptance
    check will ask at runtime rather than a parallel approximation of it.
    """
    from nemo_curator.audio_agent._resolve import resolve_stage_class
    from nemo_curator.stages.audio._agent._agent_registry import build_contract
    from nemo_curator.stages.audio._agent._composite import expand_composites

    # Placeholders are passed through as the strings they are. Stripping them instead removes
    # the arguments a stage requires, and the template stops being constructible at all.
    instances = [
        resolve_stage_class(str(spec["ref"]))(**(spec.get("params") or {})) for spec in (recipe.get("stages") or [])
    ]
    written: set[str] = set()
    for leaf in expand_composites(instances).stages:
        with contextlib.suppress(Exception):  # plumbing without a contract writes nothing we can name
            contract = build_contract(leaf.stage)
            written |= set(contract.writes.data_keys) | set(contract.writes.segment_data_keys)
    return written


def test_there_are_recipes_to_check() -> None:
    assert _RECIPES, "no shipped recipe templates found"


@pytest.mark.parametrize("path", _RECIPES, ids=lambda p: p.name)
class TestEveryTemplateShipsASuccessContract:
    def test_it_declares_acceptance_criteria(self, path: Path) -> None:
        assert _load(path).get("acceptance_criteria"), f"{path.name} ships no acceptance_criteria"

    def test_the_criteria_parse_under_the_strict_schema(self, path: Path) -> None:
        criteria = parse_criteria(_load(path).get("acceptance_criteria"))
        assert criteria

    def test_at_least_one_criterion_is_binding(self, path: Path) -> None:
        # A contract of only 'nice' criteria cannot fail, so it is decoration.
        criteria = parse_criteria(_load(path).get("acceptance_criteria"))
        assert any(c.severity == "must" for c in criteria), f"{path.name} has no 'must' criterion"

    def test_every_criterion_is_machine_checkable(self, path: Path) -> None:
        # A template should not ship work for the reviewer by default.
        criteria = parse_criteria(_load(path).get("acceptance_criteria"))
        assert all(c.is_deterministic for c in criteria)

    def test_a_criterion_checks_a_field_the_recipe_actually_writes(self, path: Path) -> None:
        """A success contract naming a field nothing produces is worse than none at all.

        ``output_completeness`` asks "is this field populated on every row". Point it at a key
        no stage writes -- a typo, or a key an edit to the stage list removed -- and it reports
        every row incomplete, on a run that was fine. The contract that exists to catch an
        empty dataset then cries wolf on a good one, which is how a team learns to ignore it.
        Nothing checked the name, so it was prose pointing at a key.
        """
        recipe = _load(path)
        fields = {
            str((criterion.get("check") or {}).get("field"))
            for criterion in (recipe.get("acceptance_criteria") or [])
            if isinstance(criterion, dict) and (criterion.get("check") or {}).get("field")
        }
        if not fields:
            pytest.skip("no field-checking criterion in this template")

        written = _keys_the_recipe_writes(recipe)
        assert fields <= written, (
            f"{path.name}: acceptance criteria check field(s) {sorted(fields - written)}, "
            f"which no stage in the recipe writes; it writes {sorted(written)}"
        )


class TestScratchRecipesHaveAHome:
    def test_it_sits_under_the_run_directory(self) -> None:
        # Which is already git-ignored, and already moves with AUDIO_AGENT_RUNS_DIR -- so a
        # scratch recipe is discarded by the same gesture as the records describing its run.
        assert Path(scratch_dir()).parent == Path(runs_dir())

    def test_it_exists_after_being_asked_for(self) -> None:
        assert Path(scratch_dir()).is_dir()

    def test_it_follows_the_configured_run_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path / "elsewhere"))
        assert Path(scratch_dir()) == tmp_path / "elsewhere" / "recipes"
