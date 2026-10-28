# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Broken cards must remain visible to the release gate, not disappear silently."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from nemo_curator.stages.audio.audio_agent import card_conformance, index

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("contents", ["broken: [", "- not a card", ""])
def test_invalid_card_files_are_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, contents: str) -> None:
    cards = tmp_path / "cards"
    cards.mkdir()
    (cards / "Broken.yaml").write_text(contents, encoding="utf-8")
    monkeypatch.setattr(index, "_KNOWLEDGE_DIR", str(tmp_path))
    knowledge = index.KnowledgeIndex()
    assert knowledge.load_errors
    result = card_conformance.audit(knowledge)
    assert result["knowledge_errors"] == knowledge.load_errors


def test_duplicate_ids_are_reported_without_restricting_filenames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cards = tmp_path / "cards"
    cards.mkdir()
    for name in ("First", "Second"):
        (cards / f"{name}.yaml").write_text("stage_id: Same\n", encoding="utf-8")
    monkeypatch.setattr(index, "_KNOWLEDGE_DIR", str(tmp_path))
    knowledge = index.KnowledgeIndex()
    assert any("duplicate card stage_id" in error for error in knowledge.load_errors)
    assert len(knowledge.load_errors) == 1


def test_card_gate_fails_on_loader_errors_even_when_inventory_is_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cards = tmp_path / "cards"
    cards.mkdir()
    (cards / "Broken.yaml").write_text("broken: [", encoding="utf-8")
    monkeypatch.setattr(index, "_KNOWLEDGE_DIR", str(tmp_path))
    knowledge = index.KnowledgeIndex()
    knowledge.stage_names = list
    knowledge._blueprints = {}
    monkeypatch.setattr(index, "get_index", lambda: knowledge)
    assert card_conformance.main([]) == 1
