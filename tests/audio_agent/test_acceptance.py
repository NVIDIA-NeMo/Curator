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

"""Unit tests for the acceptance/honesty gate (verify)."""

from __future__ import annotations

import copy
import itertools
from typing import ClassVar

import pytest

from nemo_curator import audio_agent as aa
from nemo_curator.audio_agent import acceptance
from nemo_curator.audio_agent.contracts import AcceptanceCriterion
from nemo_curator.audio_agent.recipe import Recipe


def _crit(  # noqa: PLR0913 - concise criterion fixture builder
    cid: str,
    field: str,
    op: str,
    value: float,
    ctype: str = "quality_standard",
    severity: str = "must",
) -> dict:
    return {"id": cid, "type": ctype, "severity": severity, "check": {"field": field, "op": op, "value": value}}


class TestVerify:
    def test_must_met_when_metric_satisfies(self) -> None:
        r = aa.verify([_crit("q", "mos", ">=", 3.0)], evidence={"metrics": {"mos": 3.5}})
        assert r["overall"] == "met"

    def test_must_not_met_when_metric_below_bar(self) -> None:
        r = aa.verify([_crit("q", "mos", ">=", 3.0)], evidence={"metrics": {"mos": 2.0}})
        assert r["overall"] == "not_met"

    def test_no_evidence_is_not_silently_met(self) -> None:
        r = aa.verify([_crit("q", "mos", ">=", 3.0)], evidence={})
        assert r["overall"] == "not_met"
        assert any(c.get("status") == "unverifiable" for c in r.get("criteria", []))

    def test_empty_contract_is_unverifiable_not_met(self) -> None:
        # No criteria at all -> nothing was verified: never a silent "met", but also not
        # "not_met" (which would over-reject a run that carried no explicit success bar).
        r = aa.verify([], evidence={"metrics": {"mos": 3.5}, "retained": 10, "input_count": 10})
        assert r["overall"] == "unverifiable"

    def test_nice_only_contract_stays_met(self) -> None:
        # A non-empty contract with only 'nice' criteria remains met (nice = non-blocking).
        r = aa.verify([_crit("q", "mos", ">=", 3.0, severity="nice")], evidence={"metrics": {"mos": 2.0}})
        assert r["overall"] == "met"

    @pytest.mark.parametrize("input_count", [None, 0])
    def test_relative_yield_without_positive_denominator_is_unverifiable(
        self,
        input_count: int | None,
    ) -> None:
        criterion = {
            "id": "yield",
            "type": "yield",
            "kind": "relative",
            "check": {"op": ">=", "value": 50},
        }
        result = aa.verify(
            [criterion],
            evidence={"retained": 50, "input_count": input_count},
        )
        assert result["overall"] == "not_met"
        assert result["criteria"][0]["status"] == "unverifiable"

    def test_relative_yield_rejects_inconsistent_counts(self) -> None:
        criterion = {
            "id": "yield",
            "type": "yield",
            "kind": "relative",
            "check": {"op": ">=", "value": 50},
        }
        result = aa.verify(
            [criterion],
            evidence={"retained": 11, "input_count": 10},
        )
        assert result["overall"] == "not_met"
        assert result["criteria"][0]["status"] == "unverifiable"

    def test_malformed_per_item_evidence_is_rejected(self) -> None:
        criterion = {
            "id": "output",
            "type": "output_completeness",
            "check": {"field": "pred_text"},
        }
        with pytest.raises(ValueError, match="per_item entries must be mappings"):
            aa.verify(
                [criterion],
                evidence={"per_item": [{"pred_text": "ok"}, None, "bad"]},
            )

    @pytest.mark.parametrize(
        ("evidence", "message"),
        [
            ([], "evidence must be a mapping"),
            ({"metrics": []}, "evidence.metrics must be a mapping"),
            ({"output_scan": []}, "evidence.output_scan must be a mapping"),
            (
                {"produced_roles": "transcript"},
                "evidence.produced_roles must be a collection of strings",
            ),
            (
                {"produced_keys": [1]},
                "evidence.produced_keys must contain only strings",
            ),
        ],
    )
    def test_malformed_evidence_shapes_are_rejected(
        self,
        evidence: object,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            aa.verify(
                [
                    {
                        "id": "output",
                        "type": "output_completeness",
                        "check": {"field": "pred_text"},
                    }
                ],
                evidence=evidence,
            )

    @pytest.mark.parametrize("bad_value", [True, "4.5", float("nan"), float("inf")])
    def test_invalid_aggregate_numeric_evidence_never_meets(
        self,
        bad_value: object,
    ) -> None:
        result = aa.verify(
            [_crit("q", "mos", ">=", 1.0)],
            evidence={"metrics": {"mos": bad_value}},
        )
        assert result["overall"] == "not_met"
        assert result["criteria"][0]["status"] == "unverifiable"

    @pytest.mark.parametrize("bad_value", [True, "4.5", float("nan"), float("inf")])
    def test_invalid_per_item_numeric_evidence_never_meets(
        self,
        bad_value: object,
    ) -> None:
        criterion = _crit("q", "mos", ">=", 1.0)
        criterion["check"]["scope"] = "per_retained_item"
        result = aa.verify(
            [criterion],
            evidence={"per_item": [{"mos": bad_value}]},
        )
        assert result["overall"] == "not_met"
        assert result["criteria"][0]["status"] == "not_met"


class TestHonestyGuard:
    def test_relaxed_must_forces_not_met(self) -> None:
        frozen = [_crit("q", "mos", ">=", 4.0)]
        used = [_crit("q", "mos", ">=", 3.0)]  # easier bar than confirmed
        r = aa.verify(used, evidence={"metrics": {"mos": 3.5}}, frozen_criteria=frozen)
        assert r["overall"] == "not_met"
        assert any(h.get("code") == "must_relaxed" for h in r.get("honesty", []))

    def test_conflicting_recipe_and_explicit_frozen_contract_are_rejected(self) -> None:
        recipe_criteria = [_crit("q", "mos", ">=", 4.0)]
        explicit_frozen = [_crit("q", "mos", ">=", 3.0)]

        with pytest.raises(ValueError, match="conflicts"):
            aa.verify(
                recipe_criteria,
                evidence={"metrics": {"mos": 4.5}},
                frozen_criteria=explicit_frozen,
                recipe={
                    "stages": [],
                    "acceptance_criteria": recipe_criteria,
                },
            )

    def test_dropped_must_flagged(self) -> None:
        frozen = [_crit("q", "mos", ">=", 3.0)]
        r = aa.verify([], evidence={"metrics": {"mos": 3.5}}, frozen_criteria=frozen)
        assert any(h.get("code") == "must_dropped" for h in r.get("honesty", []))

    @pytest.mark.parametrize(
        ("location", "value"),
        [
            (("type",), "distribution"),
            (("kind",), "relative"),
            (("compiles_to",), "score"),
            (("on_unachievable",), "relax_with_confirmation"),
            (("check", "scope"), "per_retained_item"),
            (("check", "method"), "reviewer_judgment"),
        ],
    )
    def test_semantic_contract_changes_are_flagged(
        self,
        location: tuple[str, ...],
        value: object,
    ) -> None:
        frozen = {
            "id": "q",
            "type": "quality_standard",
            "kind": "absolute",
            "severity": "must",
            "on_unachievable": "escalate",
            "check": {
                "scope": "aggregate",
                "field": "mos",
                "op": ">=",
                "value": 4.0,
            },
        }
        used = copy.deepcopy(frozen)
        target = used
        for key in location[:-1]:
            target = target[key]
        target[location[-1]] = value

        result = aa.verify(
            [used],
            evidence={"metrics": {"mos": 4.5}},
            frozen_criteria=[frozen],
        )

        assert result["overall"] == "not_met"
        assert any(item["code"] == "must_relaxed" for item in result["honesty"])

    def test_wider_approximation_tolerance_is_flagged(self) -> None:
        frozen = [_crit("q", "mos", "~=", 4.0)]
        frozen[0]["check"]["tolerance"] = 0.2
        used = copy.deepcopy(frozen)
        used[0]["check"]["tolerance"] = 0.5

        result = aa.verify(
            used,
            evidence={"metrics": {"mos": 4.0}},
            frozen_criteria=frozen,
        )

        assert result["overall"] == "not_met"
        assert any(item["code"] == "must_relaxed" for item in result["honesty"])

    def test_stricter_numeric_bar_is_not_goalpost_moving(self) -> None:
        frozen = [_crit("q", "mos", ">=", 4.0)]
        used = [_crit("q", "mos", ">=", 4.5)]
        result = aa.verify(
            used,
            evidence={"metrics": {"mos": 4.7}},
            frozen_criteria=frozen,
        )
        assert result["overall"] == "met"
        assert result["honesty"] == []

    def test_equivalent_explicit_defaults_do_not_trigger_honesty(self) -> None:
        frozen = [_crit("q", "mos", ">=", 4.0)]
        used = copy.deepcopy(frozen)
        used[0]["kind"] = "absolute"
        used[0]["on_unachievable"] = "escalate"
        used[0]["check"]["scope"] = "aggregate"
        used[0]["check"]["method"] = "deterministic"

        result = aa.verify(
            used,
            evidence={"metrics": {"mos": 4.5}},
            frozen_criteria=frozen,
        )

        assert result["overall"] == "met"
        assert result["honesty"] == []

    def test_description_and_source_changes_do_not_move_the_bar(self) -> None:
        frozen = [_crit("q", "mos", ">=", 4.0)]
        used = copy.deepcopy(frozen)
        used[0]["description"] = "same threshold, clearer wording"
        used[0]["source"] = {"kind": "host_explanation"}

        result = aa.verify(
            used,
            evidence={"metrics": {"mos": 4.5}},
            frozen_criteria=frozen,
        )

        assert result["honesty"] == []

    def test_narrower_approximation_tolerance_is_safe(self) -> None:
        frozen = [_crit("q", "mos", "~=", 4.0)]
        frozen[0]["check"]["tolerance"] = 0.5
        used = copy.deepcopy(frozen)
        used[0]["check"]["tolerance"] = 0.2

        result = aa.verify(
            used,
            evidence={"metrics": {"mos": 4.0}},
            frozen_criteria=frozen,
        )

        assert result["overall"] == "met"
        assert result["honesty"] == []

    def test_output_physical_field_change_is_flagged(self) -> None:
        frozen = [
            {
                "id": "transcript",
                "type": "output_completeness",
                "compiles_to": "pred_text",
                "check": {"field": "pred_text"},
            }
        ]
        used = copy.deepcopy(frozen)
        used[0]["check"]["field"] = "text"

        result = aa.verify(
            used,
            evidence={"per_item": [{"text": "hello"}]},
            frozen_criteria=frozen,
        )

        assert result["overall"] == "not_met"
        assert any(item["code"] == "must_relaxed" for item in result["honesty"])


class TestCriterionSchema:
    @pytest.mark.parametrize(
        ("criterion", "message"),
        [
            (
                {"id": "q", "type": "quality_standard", "severity": "mustt", "check": {}},
                "severity",
            ),
            (
                {"id": "q", "type": "quality_standrd", "severity": "must", "check": {}},
                "type",
            ),
            (
                {"id": "", "type": "yield", "check": {"op": ">", "value": 0}},
                "non-empty",
            ),
            (
                {"id": "y", "type": "yield", "check": {"op": "gte", "value": 0}},
                "operator",
            ),
            (
                {"id": "y", "type": "yield", "check": {"op": ">", "value": "some"}},
                "numeric",
            ),
            (
                {"id": "q", "type": "quality_standard", "check": {"field": "mos", "op": ">="}},
                "value",
            ),
            (
                {"id": "o", "type": "output_completeness", "check": {}},
                "target",
            ),
            (
                {"id": "q", "type": "quality_standard", "kind": "absolut", "check": {}},
                "kind",
            ),
            (
                {
                    "id": "q",
                    "type": "quality_standard",
                    "check": {"field": "mos", "op": ">=", "value": 3, "scpoe": "aggregate"},
                },
                "unknown check",
            ),
            (
                {
                    "id": "q",
                    "type": "quality_standard",
                    "check": {"field": "mos", "op": ">=", "value": 3},
                    "severtiy": "must",
                },
                "unknown field",
            ),
            (
                {"id": " q ", "type": "yield", "check": {"op": ">", "value": 0}},
                "surrounding whitespace",
            ),
            (
                {"id": 1, "type": "yield", "check": {"op": ">", "value": 0}},
                "non-empty string",
            ),
            (
                {"id": "q", "check": {"field": "mos", "op": ">=", "value": 3}},
                "invalid type",
            ),
            (
                {"id": "q", "type": "quality_standard", "severity": None, "check": {}},
                "severity",
            ),
            (
                {"id": "q", "type": "quality_standard", "check": []},
                "check must be a mapping",
            ),
            (
                {
                    "id": "q",
                    "type": "quality_standard",
                    "source": [],
                    "check": {"field": "mos", "op": ">=", "value": 3},
                },
                "source must be a mapping",
            ),
            (
                {
                    "id": "q",
                    "type": "quality_standard",
                    "check": {"scope": "everywhere", "field": "mos", "op": ">=", "value": 3},
                },
                "scope",
            ),
            (
                {
                    "id": "q",
                    "type": "quality_standard",
                    "check": {"method": "llm", "field": "mos", "op": ">=", "value": 3},
                },
                "method",
            ),
            (
                {
                    "id": "q",
                    "type": "quality_standard",
                    "on_unachievable": "ignore",
                    "check": {"field": "mos", "op": ">=", "value": 3},
                },
                "on_unachievable",
            ),
            (
                {"id": "y", "type": "yield", "check": {"op": ">", "value": True}},
                "numeric",
            ),
            (
                {"id": "y", "type": "yield", "check": {"op": ">", "value": float("nan")}},
                "numeric",
            ),
            (
                {"id": "y", "type": "yield", "check": {"op": ">", "value": float("inf")}},
                "numeric",
            ),
            (
                {
                    "id": "q",
                    "type": "quality_standard",
                    "check": {"field": "mos", "op": "~=", "value": 3, "tolerance": -1},
                },
                "non-negative",
            ),
            (
                {
                    "id": "q",
                    "type": "quality_standard",
                    "check": {"field": "mos", "op": ">=", "value": 3, "tolerance": 0},
                },
                "only valid",
            ),
            (
                {
                    "id": "o",
                    "type": "output_completeness",
                    "check": {"field": "pred_text", "op": "non_empty", "value": 1},
                },
                "does not support",
            ),
            (
                {
                    "id": "y",
                    "type": "yield",
                    "check": {"scope": "aggregate", "op": ">", "value": 0},
                },
                "does not support",
            ),
            (
                {
                    "id": "s",
                    "type": "semantic_fit",
                    "check": {"method": "reviewer_judgment", "op": ">=", "value": 3},
                },
                "does not support",
            ),
            (
                {
                    "id": "old",
                    "type": "quality_standard",
                    "metric": "mos",
                    "comparison": ">=",
                    "target": 3,
                },
                "unknown field",
            ),
            (
                {
                    "id": "o",
                    "type": "output_completeness",
                    "compiles_to": "producible_role",
                },
                "needs a target",
            ),
        ],
    )
    def test_invalid_criterion_is_rejected(
        self,
        criterion: dict,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            aa.verify([criterion], evidence={})

    def test_duplicate_ids_are_rejected(self) -> None:
        criterion = _crit("q", "mos", ">=", 3.0)
        with pytest.raises(ValueError, match="duplicate"):
            aa.verify([criterion, copy.deepcopy(criterion)], evidence={})

    def test_per_item_scope_alias_is_normalized_only_at_runtime(self) -> None:
        criterion = _crit("q", "mos", ">=", 3.0)
        criterion["check"]["scope"] = "per_item"
        rec = Recipe.from_dict({"stages": [], "acceptance_criteria": [criterion]})

        assert rec.acceptance_criteria[0]["check"]["scope"] == "per_item"
        result = aa.verify(
            rec.acceptance_criteria,
            evidence={"per_item": [{"mos": 3.1}, {"mos": 3.2}]},
        )
        assert result["overall"] == "met"

    def test_recipe_rejects_invalid_contract_before_hashing(self) -> None:
        with pytest.raises(ValueError, match="severity"):
            Recipe.from_dict(
                {
                    "stages": [],
                    "acceptance_criteria": [
                        {
                            "id": "q",
                            "type": "quality_standard",
                            "severity": "mustt",
                            "check": {"field": "mos", "op": ">=", "value": 3},
                        }
                    ],
                }
            )

    def test_recipe_rejects_misspelled_acceptance_contract_field(self) -> None:
        with pytest.raises(ValueError, match="acceptance_criteria"):
            Recipe.from_dict(
                {
                    "stages": [],
                    "acceptance_criterias": [
                        {
                            "id": "q",
                            "type": "quality_standard",
                            "check": {"field": "mos", "op": ">=", "value": 3},
                        }
                    ],
                }
            )

    def test_direct_criterion_object_hashes_like_its_round_trip(self) -> None:
        criterion = AcceptanceCriterion.from_dict(
            {
                "id": "quality",
                "type": "quality_standard",
                "check": {"field": "mos", "op": ">=", "value": 4.0},
            }
        )
        direct = Recipe(stages=[], acceptance_criteria=[criterion]).freeze()
        round_trip = Recipe.from_dict(direct.to_dict()).freeze()

        assert direct.config_hash == round_trip.config_hash
        assert direct.contract_hash == round_trip.contract_hash

    @pytest.mark.parametrize(
        "criterion",
        [
            {
                "id": "q",
                "type": "quality_standard",
                "kind": "absolute",
                "severity": "must",
                "check": {"scope": "aggregate", "field": "mos", "op": ">=", "value": 3},
            },
            {
                "id": "o",
                "type": "output_completeness",
                "compiles_to": "pred_text",
                "severity": "must",
            },
            {
                "id": "y",
                "type": "yield",
                "kind": "relative",
                "severity": "nice",
                "check": {"op": "~=", "value": 20, "tolerance": 2},
            },
            {
                "id": "s",
                "type": "semantic_fit",
                "severity": "nice",
                "check": {"method": "reviewer_judgment"},
            },
            {
                "id": "o",
                "type": "output_completeness",
                "compiles_to": "pred_text",
                "check": {
                    "scope": "per_item",
                    "field": "pred_text",
                    "op": "non_empty",
                },
            },
            {
                "id": "y",
                "type": "yield",
                "kind": "operational",
                "check": {"field": "retained", "op": ">", "value": 0},
            },
        ],
    )
    def test_existing_valid_criterion_shapes_remain_accepted(
        self,
        criterion: dict,
    ) -> None:
        aa.verify([criterion], evidence={})


class TestOutputCompletenessEvidence:
    _criterion: ClassVar[dict] = {
        "id": "t",
        "type": "output_completeness",
        "check": {"field": "transcript"},
        "severity": "must",
    }

    def test_declaration_without_observed_values_is_unverifiable(self) -> None:
        result = aa.verify(
            [self._criterion],
            evidence={
                "produced_roles": ["transcript"],
                "produced_keys": ["pred_text"],
            },
        )
        assert result["overall"] == "not_met"
        assert result["criteria"][0]["status"] == "unverifiable"

    def test_role_alias_with_nonempty_observed_values_is_met(self) -> None:
        result = aa.verify(
            [self._criterion],
            evidence={
                "produced_roles": ["transcript"],
                "produced_keys": ["pred_text"],
                "per_item": [{"pred_text": "hello"}, {"pred_text": "world"}],
            },
        )
        assert result["overall"] == "met"
        assert "pred_text" in result["criteria"][0]["evidence"]

    def test_role_alias_with_empty_observed_values_is_not_met(self) -> None:
        result = aa.verify(
            [self._criterion],
            evidence={
                "produced_roles": ["transcript"],
                "produced_keys": ["pred_text"],
                "per_item": [{"pred_text": ""}, {"pred_text": None}],
            },
        )
        assert result["overall"] == "not_met"
        assert result["criteria"][0]["status"] == "not_met"

    def test_some_rows_missing_the_field_is_not_met(self) -> None:
        result = aa.verify(
            [self._criterion],
            evidence={
                "per_item": [{"pred_text": "hello"}, {"other": "world"}],
            },
        )
        assert result["overall"] == "not_met"
        assert "only 1/2" in result["criteria"][0]["evidence"]

    def test_whitespace_and_empty_containers_are_empty(self) -> None:
        for value in ("  ", [], {}):
            result = aa.verify(
                [self._criterion],
                evidence={"per_item": [{"pred_text": value}]},
            )
            assert result["overall"] == "not_met"

    def test_zero_and_false_are_valid_non_empty_values(self) -> None:
        for value in (0, False):
            result = aa.verify(
                [self._criterion],
                evidence={"per_item": [{"pred_text": value}]},
            )
            assert result["overall"] == "met"

    def test_unmapped_role_is_unverifiable_not_declaration_met(self) -> None:
        criterion = {
            "id": "custom",
            "type": "output_completeness",
            "compiles_to": "custom_semantic_role",
        }
        result = aa.verify(
            [criterion],
            evidence={
                "produced_roles": ["custom_semantic_role"],
                "per_item": [{"custom_serialized_key": "value"}],
            },
        )
        assert result["overall"] == "not_met"
        assert result["criteria"][0]["status"] == "unverifiable"

    def test_different_physical_field_cannot_prove_unrelated_semantic_target(
        self,
    ) -> None:
        criterion = {
            "id": "speaker",
            "type": "output_completeness",
            "compiles_to": "speaker_id",
            "check": {"field": "pred_text"},
        }
        result = aa.verify(
            [criterion],
            evidence={
                "produced_roles": ["pred_text"],
                "produced_keys": ["pred_text"],
                "per_item": [{"pred_text": "hello"}],
            },
        )
        assert result["overall"] == "not_met"
        assert result["criteria"][0]["status"] == "unverifiable"

    @pytest.mark.parametrize("expected_rows", [1, 3])
    def test_terminal_row_count_must_equal_serializer_proof(
        self,
        expected_rows: int,
    ) -> None:
        result = aa.verify(
            [self._criterion],
            evidence={
                "expected_output_rows": expected_rows,
                "output_scan": {
                    "status": "complete",
                    "field_scope": "top_level",
                    "valid_rows": 2,
                    "fields": {
                        "pred_text": {
                            "present": 2,
                            "non_empty": 2,
                            "numeric": 0,
                        }
                    },
                },
            },
        )
        assert result["overall"] == "not_met"
        assert "expected_output_rows=" in result["criteria"][0]["evidence"]

    def test_unknown_writer_cardinality_is_unverifiable_not_a_false_mismatch(
        self,
    ) -> None:
        result = aa.verify(
            [self._criterion],
            evidence={
                # SnippetManifestWriterStage can legitimately serialize fewer
                # rows than it returns because origin stubs pass through.
                "retained": 3,
                "output_scan": {
                    "status": "complete",
                    "field_scope": "top_level",
                    "valid_rows": 2,
                    "fields": {
                        "pred_text": {
                            "present": 2,
                            "non_empty": 2,
                            "numeric": 0,
                        }
                    },
                },
            },
        )
        assert result["criteria"][0]["status"] == "unverifiable"
        assert "trustworthy serialized-row count" in result["criteria"][0]["note"]

    @pytest.mark.parametrize(
        "output_scan",
        [
            {"status": "missing", "valid_rows": 0, "fields": {}},
            {
                "status": "partial",
                "valid_rows": 1,
                "read_errors": 1,
                "fields": {
                    "pred_text": {
                        "present": 1,
                        "non_empty": 1,
                        "numeric": 0,
                    }
                },
            },
        ],
    )
    def test_aggregate_metric_cannot_override_bad_terminal_evidence(
        self,
        output_scan: dict,
    ) -> None:
        result = aa.verify(
            [self._criterion],
            evidence={
                "metrics": {"pred_text": "aggregate-placeholder"},
                "produced_keys": ["pred_text"],
                "retained": 1,
                "output_scan": output_scan,
            },
        )
        assert result["overall"] == "not_met"
        assert result["criteria"][0]["status"] == "unverifiable"


class TestExhaustivePerItemEvidence:
    _criterion: ClassVar[dict] = {
        "id": "q",
        "type": "quality_standard",
        "check": {
            "scope": "per_retained_item",
            "field": "mos",
            "op": ">=",
            "value": 4.0,
        },
    }

    def test_complete_scan_catches_a_field_missing_beyond_the_preview(self) -> None:
        result = aa.verify(
            [self._criterion],
            evidence={
                "per_item": [{"mos": 4.5}, {"mos": 4.6}],
                "output_scan": {
                    "status": "complete",
                    "field_scope": "top_level",
                    "valid_rows": 3,
                    "fields": {
                        "mos": {
                            "present": 2,
                            "non_empty": 2,
                            "numeric": 2,
                            "min": 4.5,
                            "max": 4.6,
                        }
                    },
                },
                "expected_output_rows": 3,
            },
        )
        assert result["overall"] == "not_met"
        assert "only 2/3" in result["criteria"][0]["evidence"]

    def test_complete_scan_range_catches_a_late_threshold_failure(self) -> None:
        result = aa.verify(
            [self._criterion],
            evidence={
                "per_item": [{"mos": 4.5}, {"mos": 4.6}],
                "output_scan": {
                    "status": "complete",
                    "field_scope": "top_level",
                    "valid_rows": 3,
                    "fields": {
                        "mos": {
                            "present": 3,
                            "non_empty": 3,
                            "numeric": 3,
                            "min": 3.0,
                            "max": 4.6,
                        }
                    },
                },
                "expected_output_rows": 3,
            },
        )
        assert result["overall"] == "not_met"
        assert "all 3 terminal" in result["criteria"][0]["evidence"]

    def test_truncated_terminal_scan_cannot_meet_per_item_quality(self) -> None:
        result = aa.verify(
            [self._criterion],
            evidence={
                "expected_output_rows": 100,
                "output_scan": {
                    "status": "complete",
                    "field_scope": "top_level",
                    "valid_rows": 1,
                    "fields": {
                        "mos": {
                            "present": 1,
                            "non_empty": 1,
                            "numeric": 1,
                            "min": 4.5,
                            "max": 4.5,
                        }
                    },
                },
            },
        )
        assert result["overall"] == "not_met"
        assert "expected_output_rows=100" in result["criteria"][0]["evidence"]

    def test_truncated_explicit_per_item_evidence_cannot_meet_quality(self) -> None:
        result = aa.verify(
            [self._criterion],
            evidence={
                "expected_output_rows": 100,
                "per_item": [{"mos": 4.5}],
            },
        )

        assert result["overall"] == "not_met"
        assert "expected_output_rows=100" in result["criteria"][0]["evidence"]


class TestTheTerminalScanShortcutMatchesCheckingEveryRow:
    """``_per_item_scan_result`` answers "does every row satisfy this?" from the field's min and
    max alone, rather than from the rows. That shortcut is only sound if it agrees with
    ``_cmp`` applied to each value, for every operator -- and where min/max genuinely cannot
    decide (``!=`` over a range that straddles the target), it has to decline rather than guess.

    The operators are a hand-written ladder of comparisons against ``low`` or ``high``, and
    picking the wrong bound for one of them is a one-character mistake that yields a confident,
    wrong success verdict. Nothing else in the suite compares the two implementations.
    """

    _VALUES: ClassVar[tuple[float, ...]] = (0.0, 1.0, 2.0, 3.0)
    _TARGETS: ClassVar[tuple[float, ...]] = (0.0, 1.0, 1.5, 2.0, 3.0)
    _OPS: ClassVar[tuple[str, ...]] = (">=", ">", "<=", "<", "==", "!=", "~=")

    @staticmethod
    def _shortcut(values: list[float], op: str, target: float, tol: float) -> tuple[str, str, str]:
        scan = {
            "status": "ok",
            "valid_rows": len(values),
            "field_scope": "top_level",
            "fields": {
                "duration": {
                    "present": len(values),
                    "numeric": len(values),
                    "min": min(values),
                    "max": max(values),
                }
            },
        }
        result = acceptance._per_item_scan_result("duration", op, target, tol, scan, len(values))
        assert result is not None
        return result

    def test_the_fixture_actually_reaches_the_comparison(self) -> None:
        """Guard for the tests below rather than for the code. Get the scan shape wrong -- the
        row count lives under ``valid_rows`` -- and every call returns "unverifiable" from an
        early guard. A differential that skips those then compares nothing at all and passes.
        """
        verdict, detail, _ = self._shortcut([2.0, 3.0], ">=", 1.0, 0)

        assert verdict == "met"
        assert "span" in detail, "the reason must come from the range comparison, not a guard"

    @pytest.mark.parametrize("op", _OPS)
    def test_it_agrees_with_per_row_truth_or_declines(self, op: str) -> None:
        tol = 0.5 if op == "~=" else 0
        checked = 0
        for size in (1, 2, 3):
            for values in itertools.combinations_with_replacement(self._VALUES, size):
                for target in self._TARGETS:
                    verdict = self._shortcut(list(values), op, target, tol)[0]
                    if verdict == "unverifiable":
                        continue  # declining to decide from a range is always safe
                    truth = all(acceptance._cmp(v, op, target, tol) for v in values)
                    assert (verdict == "met") is truth, (
                        f"{list(values)} {op} {target}: scan says {verdict!r}, row-by-row says "
                        f"{'met' if truth else 'not_met'}"
                    )
                    checked += 1
        assert checked, f"no decidable case exercised for {op!r}"

    def test_a_straddling_range_declines_rather_than_guessing(self) -> None:
        """``!=`` is the one operator min/max cannot always settle: with values spanning the
        target, whether some row equals it is simply not in the range. Guessing either way
        would be a fabricated verdict."""
        assert self._shortcut([0.0, 3.0], "!=", 1.5, 0)[0] == "unverifiable"
        assert self._shortcut([2.0, 2.0], "!=", 1.5, 0)[0] == "met"
        assert self._shortcut([1.5, 1.5], "!=", 1.5, 0)[0] == "not_met"
