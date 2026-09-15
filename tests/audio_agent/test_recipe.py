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

"""``config_hash`` is the confirm gate: what the user approved is what runs.

Its invariants were held only by the implementation. Each one below is load-bearing for a
workflow that breaks silently -- an approval that stops matching, or a gate that stops
distinguishing -- rather than for anything a type checker or a passing pipeline would catch.
"""

from __future__ import annotations

import pytest

from nemo_curator.audio_agent.recipe import Recipe, StageRef

_BASE = {
    "stages": [{"ref": "MonoConversionStage", "params": {"a": 1, "b": 2}}],
    "inputs": {"x": "/d", "y": 2},
    "preset": "p",
}


def _hash(payload: dict) -> str:
    return Recipe.from_dict(payload).freeze().config_hash


class TestTheConfirmGateAnchor:
    def test_a_round_trip_through_to_dict_keeps_the_hash(self) -> None:
        """The host is handed a hash by ``validate`` and passes the recipe back to ``run`` as a
        dict. If the trip out and back moved the hash, every approval would be refused as an
        integrity failure, and the message would blame the host for changing the plan."""
        recipe = Recipe.from_dict(_BASE).freeze()

        assert Recipe.from_dict(recipe.to_dict()).compute_hash() == recipe.compute_hash()

    def test_the_hash_does_not_depend_on_key_order(self) -> None:
        """Between the two calls the recipe passes through a language model, which re-emits
        JSON with its keys wherever it likes. Order-sensitivity here would read as a tampering
        refusal on a recipe nobody tampered with -- intermittently, which is worse."""
        shuffled = {
            "preset": "p",
            "inputs": {"y": 2, "x": "/d"},
            "stages": [{"params": {"b": 2, "a": 1}, "ref": "MonoConversionStage"}],
        }

        assert _hash(shuffled) == _hash(_BASE)

    def test_the_recomputable_annotations_do_not_touch_the_hash(self) -> None:
        """``run`` attaches these to the saved recipe itself, so they arrive back on the next
        call. The docstring on ``_canonical`` calls the hash portable and the comment above
        ``config_hash`` warns that widening it breaks already-frozen recipes: pick one of these
        up and every shipped recipe's approval stops matching, on a machine that merely planned
        or ran differently.
        """
        annotated = Recipe.from_dict(_BASE).freeze().to_dict()
        annotated.update(
            {
                "machine_plan": {"workers": 8},
                "data_derived": {"observed_sample_rate": 16000},
                "config_strategy": [{"param": "a", "source": "data_informed"}],
                "knowledge_version": "9",
                "parent_run_id": "run-2026-08-16",
            }
        )

        assert Recipe.from_dict(annotated).compute_hash() == _hash(_BASE)

    @pytest.mark.parametrize(
        ("label", "mutation"),
        [
            ("a stage param", {"stages": [{"ref": "MonoConversionStage", "params": {"a": 1, "b": 3}}]}),
            ("the stage ref", {"stages": [{"ref": "ResampleAudioStage", "params": {"a": 1, "b": 2}}]}),
            ("an input", {"inputs": {"x": "/other", "y": 2}}),
            ("the preset", {"preset": "q"}),
        ],
    )
    def test_a_change_to_what_runs_moves_the_hash(self, label: str, mutation: dict) -> None:
        """The other half of the gate. A hash that ignored any of these would let an approval
        for one pipeline authorise a different one."""
        assert _hash({**_BASE, **mutation}) != _hash(_BASE), label

    # The acceptance contract's place in the gate -- inside ``config_hash``, outside
    # ``semantic_hash`` -- is already pinned by test_reuse.py::test_acceptance_change_leaves
    # _semantic_hash_alone, which asserts all three hashes at once.

    def test_planning_preference_round_trips_without_changing_any_hash(self) -> None:
        easy = Recipe.from_dict(
            {
                **_BASE,
                "planning_preference": {
                    "schema_version": 1,
                    "curation_mode": "refine_later",
                    "source": "explicit_user_choice",
                },
            }
        ).freeze()
        fast = Recipe.from_dict(
            {
                **_BASE,
                "planning_preference": {
                    "schema_version": 1,
                    "curation_mode": "fast_first",
                    "source": "inferred_from_request",
                },
            }
        ).freeze()
        old = Recipe.from_dict(_BASE).freeze()

        assert easy.to_dict()["planning_preference"] == easy.planning_preference
        assert "planning_preference" not in old.to_dict()
        assert (
            (
                easy.config_hash,
                easy.semantic_hash,
                easy.contract_hash,
            )
            == (
                fast.config_hash,
                fast.semantic_hash,
                fast.contract_hash,
            )
            == (
                old.config_hash,
                old.semantic_hash,
                old.contract_hash,
            )
        )
        assert Recipe.from_dict(easy.to_dict()).planning_preference == easy.planning_preference

    @pytest.mark.parametrize(
        "bad",
        [
            "refine_later",
            {},
            {
                "schema_version": 2,
                "curation_mode": "refine_later",
                "source": "explicit_user_choice",
            },
            {
                "schema_version": 1,
                "curation_mode": "always_checkpoint",
                "source": "explicit_user_choice",
            },
            {
                "schema_version": 1,
                "curation_mode": "fast_first",
                "source": "guessed_from_folder",
            },
            {
                "schema_version": 1,
                "curation_mode": ["refine_later"],
                "source": "explicit_user_choice",
            },
            {
                "schema_version": 1,
                "curation_mode": "fast_first",
                "source": {"kind": "explicit_user_choice"},
            },
        ],
    )
    def test_invalid_present_planning_preference_is_actionable(
        self,
        bad: object,
    ) -> None:
        with pytest.raises(ValueError, match="planning_preference"):
            Recipe.from_dict({**_BASE, "planning_preference": bad})


class TestAMalformedStageEntrySaysWhatIsWrong:
    """``validate`` exists to hand a host something it can act on. A recipe malformed in the
    ``stages`` shape has always said so by name; ``params`` had no such check and fell through
    to whatever ``dict()`` raised -- for a string, a message about update sequence elements,
    and for a list, a ``TypeError``, which is not the type the rest of this parser raises.
    """

    @pytest.mark.parametrize("bad", ["sample_rate=16000", [1, 2], 7, ("a", "b", "c")])
    def test_params_that_are_not_a_mapping_are_named(self, bad: object) -> None:
        with pytest.raises(ValueError, match="'params' must be a mapping") as caught:
            StageRef.from_dict({"ref": "MonoConversionStage", "params": bad})

        assert "MonoConversionStage" in str(caught.value), "the host has to know which stage"

    @pytest.mark.parametrize(
        ("params", "expected"),
        [(None, {}), ({}, {}), ({"a": 1}, {"a": 1}), ([("a", 1)], {"a": 1})],
    )
    def test_everything_that_parsed_before_still_parses(self, params: object, expected: dict) -> None:
        """The conversion itself is untouched, including a list of key/value pairs, which
        ``dict()`` has always accepted."""
        assert StageRef.from_dict({"ref": "S", "params": params}).params == expected

    @pytest.mark.parametrize("bad", ["oops", [1, 2], 7])
    def test_recipe_inputs_that_are_not_a_mapping_are_named(self, bad: object) -> None:
        """``inputs`` sat one line away from ``stages`` and landed the same way ``params`` did."""
        with pytest.raises(ValueError, match="'inputs' must be a mapping"):
            Recipe.from_dict({"stages": [], "inputs": bad})

    @pytest.mark.parametrize(
        ("inputs", "expected"), [(None, {}), ({}, {}), ({"a": 1}, {"a": 1}), ([("a", 1)], {"a": 1})]
    )
    def test_inputs_that_parsed_before_still_parse(self, inputs: object, expected: dict) -> None:
        assert Recipe.from_dict({"stages": [], "inputs": inputs}).inputs == expected

    def test_every_malformed_shape_raises_one_error_type(self) -> None:
        """``acceptance.py`` calls it "one public schema-error type" and suppresses ruff's
        TRY004 to keep it. A host that catches ``ValueError`` to turn a bad recipe into a
        message would otherwise miss the two shapes that used to surface a ``TypeError``.
        """
        malformed = [
            {"stages": {"ref": "S"}},
            {"stages": ["MonoConversionStage"]},
            {"stages": [{"ref": "S", "params": [1, 2]}]},
            {"stages": [], "inputs": [1, 2]},
            {"stages": [], "acceptance_criteria": [{"nope": 1}]},
        ]
        for payload in malformed:
            with pytest.raises(ValueError):  # noqa: PT011
                Recipe.from_dict(payload)
