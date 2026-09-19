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

"""A card's composition advice has to be true, because the agent acts on it directly.

``ALMDataBuilderStage`` recommended ``PrepareModuleSegmentsStage`` upstream across two card
versions while that pairing raised ``TypeError`` on the first window -- one writes
``metrics.bandwidth`` as a per-word list, the other compares it to an int. Nothing caught it,
because the recommendation was prose naming a stage.
"""

from __future__ import annotations

from nemo_curator.audio_agent.card_conformance import (
    _blueprint_violations,
    _composite_legibility,
    _composition_violations,
    audit,
    audit_blueprints,
)
from nemo_curator.audio_agent.context import assemble


class TestCompositionEdges:
    def test_a_stage_that_does_not_exist_is_caught(self) -> None:
        card = {"composition": {"typical_upstream": ["NoSuchStage"]}}
        assert any("not a registered stage" in v for v in _composition_violations("ALMDataBuilderStage", card))

    def test_a_real_stage_passes(self) -> None:
        card = {"composition": {"typical_upstream": ["MergeAlignmentDiarizationStage"]}}
        assert _composition_violations("ALMDataBuilderStage", card) == []

    def test_a_known_bad_pairing_can_be_recorded_with_its_reason(self) -> None:
        card = {
            "composition": {
                "typical_upstream": ["MergeAlignmentDiarizationStage"],
                "incompatible_upstream": {"PrepareModuleSegmentsStage": "writes bandwidth as a list"},
            }
        }
        assert _composition_violations("ALMDataBuilderStage", card) == []

    def test_recommending_and_forbidding_the_same_stage_is_a_contradiction(self) -> None:
        card = {
            "composition": {
                "typical_upstream": ["PrepareModuleSegmentsStage"],
                "incompatible_upstream": {"PrepareModuleSegmentsStage": "crashes"},
            }
        }
        assert any(
            "both typical_upstream and incompatible_upstream" in v
            for v in _composition_violations("ALMDataBuilderStage", card)
        )

    def test_forbidding_a_stage_requires_saying_why(self) -> None:
        # "don't use X" without a reason is a rule the next reader discounts.
        card = {"composition": {"incompatible_upstream": {"PrepareModuleSegmentsStage": ""}}}
        assert any("must say WHY" in v for v in _composition_violations("ALMDataBuilderStage", card))


class TestCompositeLegibility:
    def test_a_composite_that_decomposes_is_fine(self) -> None:
        assert _composite_legibility("SplitASRAlignJoinStage") == []

    def test_a_plain_stage_is_not_subject_to_the_rule(self) -> None:
        assert _composite_legibility("ALMDataBuilderStage") == []


class TestTheShippedCardsConform:
    def test_no_card_violates_anything(self) -> None:
        # Including the two checks above: the recommendation graph is checked, not just described.
        result = audit()
        assert result["violations"] == {}, result["violations"]


class TestBlueprintConformance:
    """Blueprints are planner-facing worked examples, so their presets must be real."""

    def test_shipped_blueprints_are_clean(self) -> None:
        assert audit_blueprints()["violations"] == {}

    def test_a_preset_naming_a_nonexistent_parameter_is_caught(self) -> None:
        """Every shipped preset was wrong this way ('utmos_mos_threshold' for what the
        stage calls 'mos_threshold') and nothing caught it -- the gate covered cards only."""
        violations = _blueprint_violations(
            "bp",
            {"stages": [{"ref": "UTMOSFilterStage"}], "presets": {"tts": {"utmos_mos_threshold": 4.0}}},
        )
        assert any("utmos_mos_threshold" in v for v in violations)

    def test_a_real_parameter_passes(self) -> None:
        assert (
            _blueprint_violations(
                "bp", {"stages": [{"ref": "UTMOSFilterStage"}], "presets": {"tts": {"mos_threshold": 4.0}}}
            )
            == []
        )

    def test_an_unknown_stage_ref_is_caught(self) -> None:
        violations = _blueprint_violations("bp", {"stages": [{"ref": "NoSuchStage"}], "presets": {}})
        assert any("not a registered stage" in v for v in violations)


class TestBlueprintRetrievalHonesty:
    """An unmatched goal must not be handed arbitrary examples as though they matched."""

    def test_an_unrelated_goal_matches_nothing(self) -> None:
        context = assemble({"task": "diarize", "language": "japanese"}, include_env=False).to_dict()
        assert context["matched_blueprints"] == []
        # ...and no example's presets leak into the planning context either
        assert [k for k in context["presets"] if k.startswith("_blueprint_")] == []

    def test_a_matching_goal_still_matches(self) -> None:
        context = assemble({"task": "quality_filter", "domain": "read"}, include_env=False).to_dict()
        ids = [b.get("blueprint_id") for b in context["matched_blueprints"]]
        assert "readspeech-quality-filter" in ids

    def test_an_empty_goal_still_browses(self) -> None:
        """Nothing was asked, so offering what exists is a browse, not a claim of relevance."""
        context = assemble({}, include_env=False).to_dict()
        assert len(context["matched_blueprints"]) == 3
