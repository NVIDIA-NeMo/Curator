# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance binds to the declared primary without losing sibling outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, StageContract
from nemo_curator.stages.audio.audio_agent import _capabilities, _verb_evidence
from nemo_curator.stages.audio.audio_agent.card_conformance import check_card
from nemo_curator.stages.audio.audio_agent.recipe import Recipe, StageRef, build_stages
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


@pytest.fixture
def multi_writer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    from nemo_curator.stages import base
    from nemo_curator.stages.audio._agent import _catalog
    from nemo_curator.stages.audio.audio_agent import index

    _catalog.list_agent_ready_stages()
    monkeypatch.setattr(base, "_STAGE_REGISTRY", dict(base._STAGE_REGISTRY))

    @dataclass
    class MultiManifestFixtureWriter(AgentReady, ProcessingStage[AudioTask, AudioTask]):
        records_file: str
        review_file: str
        include_records: bool = True
        include_review: bool = True
        drop_empty: bool = False
        name: str = "MultiManifestFixtureWriter"

        def describe(self) -> StageContract:
            return StageContract(
                cardinality="filter" if self.drop_empty else "1:1",
                gates=Gates(
                    writes_to_disk=True,
                    output_path_params=[
                        name
                        for name, enabled in (
                            ("records_file", self.include_records),
                            ("review_file", self.include_review),
                        )
                        if enabled
                    ],
                    requires_serializable_input=True,
                    per_row_independent=False,
                ),
            )

        def process(self, task: AudioTask) -> AudioTask | None:
            if self.drop_empty and not task.data:
                return None
            if self.include_records:
                with open(self.records_file, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(task.data) + "\n")
            if self.include_review:
                with open(self.review_file, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"review": "pending"}) + "\n")
            return task

    card = {
        "stage_id": "MultiManifestFixtureWriter",
        "category": "export",
        "summary": "Save each returned record and a separate review-index row.",
        "tags": ["writes_disk", "sink"],
        "io": {
            "outputs": {
                "records_file": {"kind": "manifest", "per_item": True, "primary": True},
                "review_file": {"kind": "manifest", "per_item": True},
            }
        },
        "verified": {"params": "mechanical", "io": "mechanical"},
        "provenance": {"card_version": 1},
    }
    knowledge = index.KnowledgeIndex()
    knowledge._cards_by_stage[card["stage_id"]] = card
    monkeypatch.setattr(index, "get_index", lambda: knowledge)
    monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path / "state"))
    rows = [{"score": score} for score in (0.1, 0.2, 0.3)]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    recipe = Recipe(
        stages=[
            StageRef("ManifestReader", {"manifest_path": str(source)}),
            StageRef(
                card["stage_id"],
                {"records_file": str(tmp_path / "records.jsonl"), "review_file": str(tmp_path / "review.jsonl")},
            ),
        ],
        acceptance_criteria=[
            {"id": "scores", "type": "output_completeness", "check": {"field": "score"}},
        ],
    ).freeze()
    stages, errors = build_stages(recipe)
    assert not errors
    for number, row in enumerate(rows):
        stages[-1].process(AudioTask(task_id=str(number), data=row))
    assert check_card(card["stage_id"], card) == []
    return SimpleNamespace(
        recipe=recipe,
        card=card,
        records=recipe.stages[-1].params["records_file"],
        review=recipe.stages[-1].params["review_file"],
        report=SimpleNamespace(accepted=len(rows), input_count=len(rows)),
    )


@pytest.mark.parametrize("reverse_order", [False, True])
def test_explicit_primary_selects_evidence_independently_of_card_order(
    multi_writer: SimpleNamespace, reverse_order: bool
) -> None:
    outputs = multi_writer.card["io"]["outputs"]
    if reverse_order:
        multi_writer.card["io"]["outputs"] = dict(reversed(list(outputs.items())))
    assert _verb_evidence._terminal_evidence_outputs(multi_writer.recipe, [multi_writer.review]) == (
        [multi_writer.records],
        True,
    )
    assert set(_verb_evidence._recipe_outputs(multi_writer.recipe, None)) == {
        multi_writer.records,
        multi_writer.review,
    }
    assert set(
        _capabilities.generated_asset_paths(multi_writer.recipe.stages[-1].ref, multi_writer.recipe.stages[-1].params)
    ) == {
        multi_writer.records,
        multi_writer.review,
    }


def test_primary_acceptance_does_not_trust_a_scan_of_its_sibling(multi_writer: SimpleNamespace) -> None:
    _, sibling_scan = _verb_evidence._scan_terminal_output([multi_writer.review], limit=0)
    result = _verb_evidence._acceptance_result(
        multi_writer.recipe,
        multi_writer.report,
        [],
        ["score"],
        [multi_writer.records, multi_writer.review],
        output_scan=sibling_scan,
    )
    assert result["overall"] == "met", result


@pytest.mark.parametrize("bad_rows", [[{"score": 0.1}], [{}, {}, {}], [{"score": None}] * 3])
def test_primary_selection_does_not_hide_missing_rows_or_values(
    multi_writer: SimpleNamespace, bad_rows: list[dict]
) -> None:
    Path(multi_writer.records).write_text("".join(json.dumps(row) + "\n" for row in bad_rows), encoding="utf-8")
    result = _verb_evidence._acceptance_result(multi_writer.recipe, multi_writer.report, [], ["score"])
    assert result["overall"] != "met", result


@pytest.mark.parametrize("change", ["no_primary", "filter", "nonterminal"])
def test_ambiguous_or_nonterminal_writers_do_not_prove_final_coverage(
    multi_writer: SimpleNamespace, change: str
) -> None:
    recipe = multi_writer.recipe
    if change == "no_primary":
        del multi_writer.card["io"]["outputs"]["records_file"]["primary"]
    elif change == "filter":
        recipe.stages[-1].params["drop_empty"] = True
    else:
        recipe.stages.append(
            StageRef("PreserveByValueStage", {"input_value_key": "score", "operator": "lt", "target_value": 0.2})
        )
    _, proven = _verb_evidence._terminal_evidence_outputs(recipe, [])
    assert not proven
    result = _verb_evidence._acceptance_result(recipe, multi_writer.report, [], ["score"])
    assert result["overall"] != "met", result


@pytest.mark.parametrize("excluded", ["disabled", "not_per_item"])
def test_primary_must_be_enabled_and_per_item(multi_writer: SimpleNamespace, excluded: str) -> None:
    if excluded == "disabled":
        multi_writer.recipe.stages[-1].params["include_records"] = False
    else:
        multi_writer.card["io"]["outputs"]["records_file"]["per_item"] = False
    assert _verb_evidence._terminal_evidence_outputs(multi_writer.recipe, []) == ([multi_writer.review], True)


def test_two_primaries_fail_card_validation(multi_writer: SimpleNamespace) -> None:
    multi_writer.card["io"]["outputs"]["review_file"]["primary"] = True
    with pytest.raises(ValueError, match="more than one primary"):
        _verb_evidence._terminal_evidence_outputs(multi_writer.recipe, [])
