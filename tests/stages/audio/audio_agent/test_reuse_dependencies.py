# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File-pair consumers cannot reuse a fingerprint of only the primary file."""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest

from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, IOSpec, StageContract
from nemo_curator.stages.audio.audio_agent import artifacts, delta, reuse
from nemo_curator.stages.audio.audio_agent._reuse_dependencies import untracked_audio_input
from nemo_curator.stages.audio.audio_agent.recipe import Recipe, StageRef
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


@pytest.fixture
def paired_recipe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Recipe:
    from nemo_curator.stages import base
    from nemo_curator.stages.audio._agent import _catalog

    _catalog.list_agent_ready_stages()
    monkeypatch.setattr(base, "_STAGE_REGISTRY", dict(base._STAGE_REGISTRY))

    @dataclass
    class FuturePairMetric(AgentReady, ProcessingStage[AudioTask, AudioTask]):
        audio_filepath_key: str = "audio_filepath"
        reference_key: str = "reference"
        nested: bool = False
        optional: bool = False
        KEY_ROLE_OVERRIDES: ClassVar[dict] = {"reference_key": "audio_filepath"}

        def describe(self) -> StageContract:
            spec = IOSpec(**{("segment_data_keys" if self.nested else "data_keys"): [self.reference_key]})
            return StageContract(
                reads=IOSpec(data_keys=[self.audio_filepath_key]) if self.optional else spec,
                optional_reads=spec if self.optional else IOSpec(),
                gates=Gates(per_row_independent=True),
            )

        def process(self, task: AudioTask) -> AudioTask:
            return task

    return Recipe(
        stages=[
            StageRef("ManifestReader", {"manifest_path": str(tmp_path / "source.jsonl")}),
            StageRef("ManifestCheckpointStage", {"output_path": str(tmp_path / "prefix.jsonl")}),
            StageRef("FuturePairMetric"),
            StageRef("ManifestWriterStage", {"output_path": str(tmp_path / "final.jsonl")}),
        ]
    )


@pytest.mark.parametrize("params", [{}, {"reference_key": "clean_pcm"}, {"nested": True}, {"optional": True}])
def test_untracked_reference_blocks_only_its_suffix(paired_recipe: Recipe, params: dict) -> None:
    paired_recipe.stages[2].params.update(params)
    boundary = untracked_audio_input(paired_recipe)
    assert boundary is not None
    assert boundary[0] == 2
    assert "not covered" in boundary[1]
    plans = artifacts.plan_steps(paired_recipe, "data")
    assert not any(p.reuse_block for p in plans[:2])
    assert all(p.reuse_block for p in plans[2:])
    assert delta.region(paired_recipe, upto=4)[0] == 2
    assert reuse.verified_point(paired_recipe, 4, dataset_key="data")[0] is None


def test_primary_alias_is_already_covered(paired_recipe: Recipe) -> None:
    paired_recipe.stages[2].params["reference_key"] = "audio_filepath"
    assert untracked_audio_input(paired_recipe) is None


def test_generated_reference_is_covered_by_persisted_provenance(paired_recipe: Recipe, tmp_path: Path) -> None:
    paired_recipe.stages.insert(
        2,
        StageRef(
            "ResampleAudioStage",
            {
                "resampled_audio_dir": str(tmp_path / "audio"),
                "resampled_audio_filepath_key": "reference",
            },
        ),
    )
    assert untracked_audio_input(paired_recipe) is None


def test_reference_change_cannot_recover_a_legacy_completed_suffix(paired_recipe: Recipe) -> None:
    from nemo_curator.stages.audio.audio_agent._verb_continuation import _merge_plans

    plan = {"mode": "already_done", "reuse_stages": [s.ref for s in paired_recipe.stages]}
    scan = {"decision": "fresh"}
    merged = _merge_plans(paired_recipe, plan, scan, dataset_key="data")
    assert merged["mode"] == "full_rerun"
    assert "dependency proof" in merged["reason"]


def test_checkpoint_advice_stops_at_untracked_inputs(paired_recipe: Recipe) -> None:
    from nemo_curator.stages.audio.audio_agent import checkpoint

    spot, reason = checkpoint.at(paired_recipe, index=3)
    assert spot is None
    assert "not covered" in reason


def test_source_aliases_require_every_selected_row(paired_recipe: Recipe, tmp_path: Path) -> None:
    import json

    manifest = Path(paired_recipe.stages[0].params["manifest_path"])
    primary = str(tmp_path / "a.wav")
    secondary = str(tmp_path / "b.wav")
    manifest.write_text(json.dumps({"audio_filepath": primary, "reference": primary}) + "\n")
    assert untracked_audio_input(paired_recipe) is None
    with manifest.open("a") as stream:
        for _ in range(2001):
            stream.write(json.dumps({"audio_filepath": primary, "reference": primary}) + "\n")
        stream.write(json.dumps({"audio_filepath": secondary, "reference": primary}) + "\n")
    assert untracked_audio_input(paired_recipe)[0] == 2
    paired_recipe.stages[0].params["include_files"] = [primary]
    assert untracked_audio_input(paired_recipe) is None
