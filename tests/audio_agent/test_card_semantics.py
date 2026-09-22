# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Shape checks for advisory card semantics.

These tests intentionally do not assert a vocabulary for scope or meaning.  The
core makes the prose retrievable; the host LLM remains responsible for applying
it to user intent.
"""

# Existing historical regression names intentionally repeat later in this module.

from copy import deepcopy

import pytest

from nemo_curator.audio_agent.card_conformance import (
    _decision_violations,
    _semantic_fact_violations,
    check_card,
)
from nemo_curator.audio_agent.index import get_index

_VALID_DECISION = {
    "separable_from_producer": True,
    "score_key_param": "wer_key",
    "score_key_default": "wer_pct",
    "value_type": "number",
    "scope": "task",
    "selector": {
        "stage_id": "PreserveByValueStage",
        "key_param": "input_value_key",
        "value_param": "target_value",
        "operator_param": "operator",
        "allowed_operators": ["lt", "le", "eq", "ne", "ge", "gt"],
    },
    "missing_score_policy": "selector_error",
    "monotonic_direction": "lower_better",
    "atomic": True,
}


def test_shipped_decisions_are_valid_and_limited_to_proven_producers() -> None:
    cards = get_index().all_cards()
    decision_cards = {stage_id: card for stage_id, card in cards.items() if "decision" in card}

    assert set(decision_cards) == {
        "GetPairwiseWerStage",
        "GetAudioDurationStage",
        "SIGMOSFilterStage",
        "UTMOSFilterStage",
    }
    for stage_id, card in decision_cards.items():
        assert _decision_violations(stage_id, card["decision"]) == []
        assert card["verified"]["decision"] == "mechanical"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"threshold": 25.0}, "unknown key 'threshold'"),
        ({"separable_from_producer": False}, "separable_from_producer must be true"),
        ({"score_key_param": "not_a_param"}, "is not a constructor param"),
        ({"score_key_param": "text_key", "score_key_default": "text"}, "does not control a declared producer write"),
        ({"score_key_default": "wrong"}, "does not match"),
        ({"value_type": "mapping"}, "value_type"),
        ({"scope": "segment"}, "scope must be 'task'"),
        ({"missing_score_policy": "keep"}, "missing_score_policy"),
        ({"monotonic_direction": "sideways"}, "monotonic_direction"),
        ({"atomic": False}, "atomic must be true"),
    ],
)
def test_decision_rejects_malformed_producer_contracts(
    updates: dict[str, object],
    message: str,
) -> None:
    decision = deepcopy(_VALID_DECISION)
    decision.update(updates)

    assert any(message in violation for violation in _decision_violations("GetPairwiseWerStage", decision))


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"stage_id": "NoSuchStage"}, "is not a registered stage"),
        ({"stage_id": "GetAudioDurationStage"}, "must be 'PreserveByValueStage'"),
        ({"key_param": "not_a_param"}, "is not a constructor param"),
        ({"key_param": "target_value"}, "key_param must be 'input_value_key'"),
        ({"allowed_operators": ["le", "between"]}, "unsupported operators"),
        ({"allowed_operators": ["le", "le"]}, "must not contain duplicates"),
        ({"unexpected": "ignored"}, "unknown key 'unexpected'"),
    ],
)
def test_decision_rejects_malformed_selector_contracts(
    updates: dict[str, object],
    message: str,
) -> None:
    decision = deepcopy(_VALID_DECISION)
    decision["selector"].update(updates)

    assert any(message in violation for violation in _decision_violations("GetPairwiseWerStage", decision))


def test_decision_requires_mechanical_verification() -> None:
    card = {
        "category": "quality",
        "summary": "Pairwise WER annotation.",
        "provenance": {"card_version": 1},
        "verified": {"decision": "best_guess"},
        "decision": deepcopy(_VALID_DECISION),
    }

    violations = check_card("GetPairwiseWerStage", card)

    assert any("verified.decision: mechanical" in violation for violation in violations)


def test_decision_is_rejected_for_an_unproven_producer() -> None:
    violations = _decision_violations("BandFilterStage", deepcopy(_VALID_DECISION))

    assert any("only supported for" in violation for violation in violations)


def test_utmos_decision_requires_exact_task_annotation_and_drop_selector() -> None:
    decision = deepcopy(get_index().card("UTMOSFilterStage")["decision"])
    decision["producer_constraints"]["mode"] = "auto"
    decision["selector"]["required_missing_policy"] = "error"

    violations = _decision_violations("UTMOSFilterStage", decision)

    assert any("producer-declared safe settings" in violation for violation in violations)
    assert any("requires missing policy 'drop'" in violation for violation in violations)


def test_sigmos_compound_decision_must_cover_every_declared_dimension() -> None:
    decision = deepcopy(get_index().card("SIGMOSFilterStage")["decision"])
    decision["dimensions"].pop()

    violations = _decision_violations("SIGMOSFilterStage", decision)

    assert any("must exactly cover" in violation for violation in violations)


def test_sigmos_compound_decision_requires_atomic_and_selector() -> None:
    decision = deepcopy(get_index().card("SIGMOSFilterStage")["decision"])
    decision["selector"]["stage_id"] = "PreserveByValueStage"

    violations = _decision_violations("SIGMOSFilterStage", decision)

    assert any("PreserveByValueConditionsStage" in violation for violation in violations)


@pytest.mark.parametrize(
    ("stage_id", "scope"),
    [
        ("SIGMOSFilterStage", "task"),
        ("SIGMOSFilterStage", "segments"),
        ("UTMOSFilterStage", "segments"),
    ],
)
def test_model_condition_selectors_explicitly_require_and(
    stage_id: str,
    scope: str,
) -> None:
    decision = deepcopy(get_index().card(stage_id)["decision"])
    declaration = (
        decision
        if decision["scope"] == scope
        else next(variant for variant in decision["variants"] if variant["scope"] == scope)
    )
    selector = declaration["selector"]

    assert selector["condition_logic_param"] == "condition_logic"
    assert selector["required_condition_logic"] == "and"

    selector["required_condition_logic"] = "or"
    violations = _decision_violations(stage_id, decision)
    assert any("required_condition_logic must be 'and'" in violation for violation in violations)


@pytest.mark.parametrize("stage_id", ["UTMOSFilterStage", "SIGMOSFilterStage"])
def test_model_decision_cards_declare_conformant_segment_variants(stage_id: str) -> None:
    decision = deepcopy(get_index().card(stage_id)["decision"])
    segment = decision["variants"][0]

    assert segment["scope"] == "segments"
    assert segment["producer_constraints"] == {"action": "annotate", "mode": "segments"}
    assert segment["selector"]["items_key_param"] == "items_key"
    assert segment["selector"]["items_key_source_param"] == "segments_key"
    assert segment["selector"]["required_empty_policy"] is True
    assert segment["selector"]["required_condition_logic"] == "and"
    assert not _decision_violations(stage_id, decision)


def test_segment_decision_rejects_unbound_items_key_and_empty_parent_policy() -> None:
    decision = deepcopy(get_index().card("SIGMOSFilterStage")["decision"])
    segment = decision["variants"][0]
    segment["selector"]["items_key_source_param"] = "not_a_param"
    segment["selector"]["required_empty_policy"] = False

    violations = _decision_violations("SIGMOSFilterStage", decision)

    assert any("not_a_param" in violation for violation in violations)
    assert any("required_empty_policy must be true" in violation for violation in violations)


def test_compound_selector_card_documents_flat_logic_and_resolvable_upstreams() -> None:
    card = get_index().card("PreserveByValueConditionsStage")
    expected_upstream = {
        "SIGMOSFilterStage",
        "UTMOSFilterStage",
        "BandFilterStage",
        "ChannelCountStage",
        "SampleRateFilterStage",
        "InferenceSortformerStage",
        "PyAnnoteDiarizationStage",
        "ManifestCheckpointStage",
    }

    assert card["params_of_note"]["condition_logic"].startswith("'and' (default)")
    assert "arbitrary-length" in card["summary"]
    assert "dot paths are not supported" in card["summary"]
    assert set(card["composition"]["typical_upstream"]) == expected_upstream
    assert check_card("PreserveByValueConditionsStage", card) == []


def test_semantic_facts_accept_compact_and_rich_prose() -> None:
    assert not _semantic_fact_violations(
        "ExampleStage",
        {
            "score": "A compact factual statement.",
            "duration": {
                "meaning": "Duration of the original recording.",
                "unit": "seconds",
                "provenance": "Computed from the decoded source.",
                "scope": "original recording, copied to emitted children",
                "propagation": "Copied unchanged across fan-out.",
                "counterexamples": ["It is not recomputed for each child."],
            },
        },
    )


def test_semantic_facts_reject_unreadable_shapes_without_interpreting_meaning() -> None:
    violations = _semantic_fact_violations(
        "ExampleStage",
        {
            "": "missing anchor",
            "score": {"meaning": "", "counterexamples": "not-a-list"},
            "duration": 3,
        },
    )
    assert any("keys must be non-empty" in item for item in violations)
    assert any(".meaning must be a non-empty string" in item for item in violations)
    assert any(".counterexamples must be a non-empty list" in item for item in violations)
    assert any("semantic_facts['duration'] must be prose or a mapping" in item for item in violations)


def test_semantic_facts_require_an_honest_evidence_tier() -> None:
    violations = check_card(
        "GetAudioDurationStage",
        {
            "category": "export",
            "summary": "Duration evidence.",
            "verified": {"params": "mechanical"},
            "semantic_facts": {"duration": "Duration of the selected audio."},
        },
    )

    assert any("verified.semantic_facts" in item for item in violations)


def test_a_card_key_nobody_reads_is_a_violation_not_a_shrug() -> None:
    """An unknown top-level key fails silently in the worst way: no error anywhere, the card
    still passes conformance, and its content simply never reaches the host critic. Two shipped
    cards wrote ``gotchas`` and ``relationships`` for what the readers call ``counterexamples``
    and ``comparison``, so the disambiguation prose written to stop a stage being confused with
    its neighbour was read by nobody at all.
    """
    violations = check_card(
        "GetAudioDurationStage",
        {
            "category": "export",
            "summary": "Duration evidence.",
            "verified": {"params": "mechanical"},
            "gotchas": ["what the readers call counterexamples"],
            "relationships": {"OtherStage": "what the readers call comparison"},
        },
    )

    assert any("unknown top-level field 'gotchas'" in item for item in violations)
    assert any("unknown top-level field 'relationships'" in item for item in violations)


def test_a_row_dropping_stage_cannot_ship_a_card_that_omits_the_filter_tag() -> None:
    """The contract is the stricter statement and the one an author writes alone, having just
    made the stage drop rows. Without the tag nothing assembling a recipe knows it can, so a
    stage silently discarding most of a corpus reads as a pass-through exactly where the
    decision to include it is made.
    """
    violations = check_card(
        "SampleRateFilterStage",
        {
            "category": "preprocess",
            "summary": "Selects rows by sample rate.",
            "verified": {"params": "mechanical"},
            "tags": [],
        },
    )

    assert any("cardinality='filter'" in item and "is_filter" in item for item in violations)


def test_filtering_within_a_row_is_not_required_to_claim_a_row_cardinality() -> None:
    """The tag is the broader planner-facing notion. ``OverlapFilterStage`` shrinks a segment
    list while every row survives, so tag-without-cardinality is a correct pairing; demanding
    the converse would make it declare a row cardinality it does not have.
    """
    violations = check_card(
        "OverlapFilterStage",
        {
            "category": "preprocess",
            "summary": "Drops overlapping segments within a row.",
            "verified": {"params": "mechanical"},
            "tags": ["is_filter"],
        },
    )

    assert not any("is_filter" in item for item in violations)


def test_verified_trust_metadata_must_be_a_mapping() -> None:
    violations = check_card(
        "GetAudioDurationStage",
        {
            "category": "export",
            "summary": "Duration evidence.",
            "verified": "mechanical",
            "semantic_facts": {"duration": "Duration of the selected audio."},
        },
    )

    assert any("verified must be a mapping" in item for item in violations)


def test_a_stage_advertising_a_score_must_say_what_the_score_means() -> None:
    """The metrics checks validate a block only IF one is present, so a scorer with no block at
    all passed silently. ``BandwidthEstimationStage`` shipped ``produces_score`` with no scale,
    range or direction; nothing downstream could then derive a comparison operator, and the
    generic filter the resolver emits for an annotate-only scorer is where an inverted filter
    comes from.
    """
    violations = check_card(
        "BandwidthEstimationStage",
        {
            "category": "quality",
            "summary": "Estimates effective bandwidth.",
            "verified": {"params": "mechanical"},
            "provenance": {"card_version": 1},
            "tags": ["produces_score"],
        },
    )

    assert any("produces_score" in item and "no metrics block" in item for item in violations)


def test_a_categorical_score_is_not_required_to_invent_a_numeric_direction() -> None:
    """The guard requires a metrics BLOCK, not a direction. ``BandFilterStage`` predicts
    full_band/narrow_band -- there is no 'higher is better' to declare -- so demanding a scale
    would force a card to state something untrue to pass a gate.
    """
    violations = check_card(
        "BandFilterStage",
        {
            "category": "quality",
            "summary": "Classifies bandwidth and filters.",
            "verified": {"params": "mechanical"},
            "provenance": {"card_version": 1},
            "tags": ["produces_score", "is_filter"],
            "metrics": {"band_prediction": {"threshold_param": "band_value"}},
        },
    )

    assert not any("produces_score" in item for item in violations)


def test_a_card_must_date_itself_so_a_stale_guess_is_distinguishable_from_a_fresh_one() -> None:
    """A card is read as current. Without ``provenance`` there is nothing to say when its
    ``best_guess`` facts were last checked against the code. 47 of 49 cards carried it by
    convention and the two that did not were simply the newest -- how an unenforced convention
    always fails.
    """
    violations = check_card(
        "GetAudioDurationStage",
        {
            "category": "export",
            "summary": "Duration evidence.",
            "verified": {"params": "mechanical"},
        },
    )

    assert any("missing required field 'provenance'" in item for item in violations)


def test_a_stage_shipping_without_a_card_fails_the_gate_unless_waived(monkeypatch) -> None:  # noqa: ANN001
    """An uncarded stage was reported and never failed, so a stage could ship plannable with no
    card semantics behind it: ``discover`` lists it, the planner may pick it, and the host critic
    gets no meaning, scope or counterexample to reason from. ``--allow-uncarded`` stays as the
    deliberate waiver for a work-in-progress stage.
    """
    from nemo_curator.audio_agent import card_conformance as cc

    monkeypatch.setattr(
        cc,
        "audit",
        lambda: {
            "violations": {},
            "orphan_cards": [],
            "uncarded_stages": ["AStageNobodyCarded"],
            "gate_unverified": [],
            "carded_count": 1,
            "stage_count": 2,
        },
    )
    monkeypatch.setattr(cc, "audit_blueprints", lambda: {"violations": {}, "blueprint_count": 0})

    assert cc.main([]) == 1
    assert cc.main(["--allow-uncarded"]) == 0


def test_the_checkpoint_stage_card_describes_a_writer_that_is_not_a_sink() -> None:
    """ManifestCheckpointStage writes to disk without ending the pipeline, and its card must say so.

    Relocated from tests/stages/audio/test_common.py: the subject is a stage this branch added,
    but what is asserted is its CARD -- agent knowledge, not module behaviour, so a module owner
    should not have to maintain it. ``card_conformance`` checks cards are well formed; this
    checks this card says the right thing.
    """
    from nemo_curator.audio_agent.index import KnowledgeIndex
    from nemo_curator.stages.audio._agent._catalog import get_agent_ready_stage_class
    from nemo_curator.stages.audio.common import ManifestCheckpointStage

    assert get_agent_ready_stage_class("ManifestCheckpointStage") is ManifestCheckpointStage

    card = KnowledgeIndex().card("ManifestCheckpointStage")
    assert card is not None
    assert card["tags"] == ["writes_disk"], "a checkpoint writes, but is not a terminal sink"
    assert "sink" not in card["tags"]
    assert "metadata checkpoint" in card["summary"]
    assert any("Waveform tensors are illegal" in note for note in card["notes"])
