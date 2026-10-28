# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from nemo_curator.stages.audio import audio_agent as aa
from nemo_curator.stages.audio.audio_agent.checks import _missing_source_audio_keys
from nemo_curator.stages.audio.audio_agent.recipe import Recipe, build_stages


@pytest.mark.parametrize("waveform_key", ["waveform", "pcm"])
def test_unknown_rewrite_forgets_only_its_own_rate_evidence(
    waveform_key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nemo_curator.stages import base
    from nemo_curator.stages.audio._agent import _catalog
    from nemo_curator.stages.audio._agent._agent_ready import AgentReady, IOSpec, StageContract
    from nemo_curator.stages.audio.audio_agent.checks import CheckContext, _sample_rate_provenance
    from nemo_curator.stages.audio.audio_agent.recipe import StageRef
    from nemo_curator.stages.base import ProcessingStage
    from nemo_curator.tasks import AudioTask

    _catalog.list_agent_ready_stages()
    monkeypatch.setattr(base, "_STAGE_REGISTRY", dict(base._STAGE_REGISTRY))

    @dataclass
    class FutureResidentRewrite(AgentReady, ProcessingStage[AudioTask, AudioTask]):
        waveform_key: str = "waveform"
        sample_rate_key: str = "sample_rate"

        def describe(self) -> StageContract:
            return StageContract(writes=IOSpec(data_keys=[self.waveform_key, self.sample_rate_key]))

        def process(self, task: AudioTask) -> AudioTask:
            return task

    recipe = Recipe(
        stages=[
            StageRef("ManifestReader", {"manifest_path": "source.jsonl"}),
            StageRef(
                "ResampleAudioStage",
                {
                    "resampled_audio_dir": "unused",
                    "target_sample_rate": 48000,
                    "write_to_disk": False,
                    "keep_waveform_in_task": True,
                    "waveform_key": waveform_key,
                },
            ),
            StageRef("FutureResidentRewrite", {"waveform_key": waveform_key}),
            StageRef("UTMOSFilterStage", {"input_residency": "waveform", "waveform_key": waveform_key}),
        ]
    )
    built, errors = build_stages(recipe)
    assert not errors
    ctx = CheckContext(recipe, built, None, None, set(), set(), 0)
    rates = _sample_rate_provenance(ctx, {48000})
    assert rates[2] == {48000}
    assert rates[3] == set()  # unknown, not stale 48 kHz
    recipe.stages[2].params["waveform_key"] = "unrelated"
    built, errors = build_stages(recipe)
    assert not errors
    ctx.stages = built
    assert _sample_rate_provenance(ctx, {48000})[3] == set()  # the rate field is still shared
    recipe.stages[2].params["sample_rate_key"] = "unrelated_rate"
    built, errors = build_stages(recipe)
    assert not errors
    ctx.stages = built
    assert _sample_rate_provenance(ctx, {48000})[3] == {48000}


def test_card_declared_rejoiner_requires_the_configured_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_curator.stages import base
    from nemo_curator.stages.audio._agent import _catalog
    from nemo_curator.stages.audio._agent._agent_ready import AgentReady, StageContract
    from nemo_curator.stages.audio.audio_agent import checks, index
    from nemo_curator.stages.audio.audio_agent.checks import CheckContext
    from nemo_curator.stages.audio.audio_agent.recipe import StageRef
    from nemo_curator.stages.base import ProcessingStage
    from nemo_curator.tasks import AudioTask

    _catalog.list_agent_ready_stages()
    monkeypatch.setattr(base, "_STAGE_REGISTRY", dict(base._STAGE_REGISTRY))

    @dataclass
    class FutureTimelineRestore(AgentReady, ProcessingStage[AudioTask, AudioTask]):
        restore: bool = True

        def describe(self) -> StageContract:
            return StageContract(cardinality="N:1")

        def process(self, task: AudioTask) -> AudioTask:
            return task

    knowledge = index.KnowledgeIndex()
    card = {"planning": {"restores_continuity": {"when": {"restore": True}}}, "verified": {"planning": "mechanical"}}
    knowledge._cards_by_stage["FutureTimelineRestore"] = card
    monkeypatch.setattr(checks, "get_index", lambda: knowledge)
    recipe = Recipe(
        stages=[
            StageRef("VADSegmentationStage"),
            StageRef("FutureTimelineRestore"),
            StageRef("InferenceSortformerStage"),
        ]
    )
    ctx = CheckContext(recipe, [], None, None, set(), set(), 0)
    assert checks._check_diarization_continuity(ctx).issues == []
    recipe.stages[1].params["restore"] = False
    assert checks._check_diarization_continuity(ctx).issues
    recipe.stages[1].params.clear()
    card.pop("planning")
    assert checks._check_diarization_continuity(ctx).issues  # N:1 is not sufficient


def _recipe(tmp_path: Path, stages: list[dict], row: dict) -> dict:
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(row) + "\n")
    return {
        "stages": [
            {"ref": "ManifestReader", "params": {"manifest_path": str(source)}},
            *stages,
            {"ref": "ManifestWriterStage", "params": {"output_path": str(tmp_path / "output.jsonl")}},
        ]
    }


def test_source_key_check_accepts_the_existing_duration_stage() -> None:
    from nemo_curator.stages.audio.common import GetAudioDurationStage

    assert _missing_source_audio_keys([object(), GetAudioDurationStage()], {"audio_filepath"}) == set()
    assert _missing_source_audio_keys([object(), GetAudioDurationStage()], set()) == {"audio_filepath"}


@pytest.mark.parametrize(
    "stage",
    [
        {"ref": "ChineseConversionStage", "params": {}},
        {"ref": "InverseTextNormalizationStage", "params": {}},
        {"ref": "PreserveByValueStage", "params": {"input_value_key": "score", "target_value": 1}},
    ],
)
def test_text_and_metadata_consumers_do_not_require_audio(tmp_path: Path, stage: dict) -> None:
    recipe = _recipe(tmp_path, [stage], {"id": "a", "segments": [], "score": 1})
    result = aa.validate(recipe)
    assert result["runnable"], result["issues"]
    assert not (tmp_path / "output.jsonl").exists()


@pytest.mark.parametrize("residency", ["file", "auto"])
def test_audio_consumers_still_refuse_missing_carriers(tmp_path: Path, residency: str) -> None:
    recipe = _recipe(
        tmp_path,
        [
            {"ref": "ChannelCountStage", "params": {"action": "annotate", "input_residency": residency}},
        ],
        {"id": "a", "text": "hello"},
    )
    result = aa.validate(recipe)
    assert not result["runnable"]
    assert any(issue["code"] == "source_schema_mismatch" for issue in result["issues"])


def test_available_resident_pair_is_not_forced_to_have_a_path(tmp_path: Path) -> None:
    recipe = Recipe.from_dict(
        _recipe(
            tmp_path,
            [
                {"ref": "ChannelCountStage", "params": {"action": "annotate", "input_residency": "auto"}},
            ],
            {"waveform": [0.0], "sample_rate": 16000},
        )
    )
    stages, errors = build_stages(recipe)
    assert not errors
    assert not _missing_source_audio_keys(stages, {"waveform", "sample_rate"})
    assert _missing_source_audio_keys(stages, {"sample_rate"})


@pytest.mark.parametrize("columns", [{"recording"}, {"audio_filepath"}])
def test_source_paths_must_match_the_configured_consumer(tmp_path: Path, columns: set[str]) -> None:
    recipe = Recipe.from_dict(
        _recipe(
            tmp_path,
            [
                {
                    "ref": "ChannelCountStage",
                    "params": {
                        "action": "annotate",
                        "input_residency": "file",
                        "audio_filepath_key": "recording",
                    },
                },
            ],
            dict.fromkeys(columns, "/clip.wav"),
        )
    )
    stages, errors = build_stages(recipe)
    assert not errors
    assert _missing_source_audio_keys(stages, columns) == (set() if "recording" in columns else {"recording"})
    result = aa.validate(recipe)
    assert any(issue["code"] == "source_schema_mismatch" for issue in result["issues"]) is ("recording" not in columns)


@pytest.mark.parametrize("rebuild", [False, True])
def test_conditional_composite_outputs_must_survive_the_suffix(tmp_path: Path, rebuild: bool) -> None:
    stages = [{"ref": "SplitASRAlignJoinStage", "params": {}}]
    if rebuild:
        stages.append({"ref": "ALMDataBuilderStage", "params": {}})
    recipe = _recipe(tmp_path, stages, {"audio_filepath": "/clip.wav", "segments": []})

    result = aa.validate(recipe, expected_outputs=["alignment"])

    missing = [issue for issue in result["issues"] if issue["code"] == "missing_output_producer"]
    assert bool(missing) is rebuild


def test_acceptance_rejects_unrelated_role_field_binding_before_execution(tmp_path: Path) -> None:
    recipe = _recipe(tmp_path, [], {"audio_filepath": "/clip.wav", "segments": []})
    recipe["acceptance_criteria"] = [
        {
            "id": "words",
            "type": "output_completeness",
            "compiles_to": "words",
            "check": {"field": "segments"},
        }
    ]
    result = aa.validate(recipe)
    assert not result["runnable"]
    assert any(issue["code"] == "acceptance_field_mismatch" for issue in result["issues"])


def test_reviewer_requirements_are_explained_at_validation(tmp_path: Path) -> None:
    recipe = _recipe(tmp_path, [], {"id": "a", "segments": []})
    recipe["acceptance_criteria"] = [{"id": "meaning", "type": "semantic_fit", "description": "correct meaning"}]
    result = aa.validate(recipe)
    issue = next(issue for issue in result["issues"] if issue["code"] == "acceptance_requires_review")
    assert issue["escalate_to"] == "user"
    assert "unverifiable" in issue["message"]


def test_directory_sink_reports_unproven_acceptance_before_execution(tmp_path: Path) -> None:
    recipe = _recipe(tmp_path, [], {"audio_filepath": "/clip.wav"})
    recipe["stages"][-1] = {
        "ref": "ResampleAudioStage",
        "params": {
            "resampled_audio_dir": str(tmp_path / "audio"),
            "write_to_disk": True,
        },
    }
    recipe["acceptance_criteria"] = [
        {
            "id": "paths",
            "type": "output_completeness",
            "check": {"field": "audio_filepath"},
        }
    ]
    result = aa.validate(recipe)
    assert any(issue["code"] == "acceptance_output_evidence_unproven" for issue in result["issues"])
    assert not (tmp_path / "audio").exists()


def test_agent_refuses_extraction_that_cannot_preserve_source_identity(tmp_path: Path) -> None:
    recipe = _recipe(
        tmp_path,
        [
            {
                "ref": "SnippetExtractionStage",
                "params": {
                    "output_dir": str(tmp_path / "snips"),
                    "output_audio_tar_path": str(tmp_path / "snips.tar"),
                    "audio_filepath_key": "id",
                    "dry_run": True,
                },
            }
        ],
        {"id": str(tmp_path / "source.wav"), "_snippet_plan": []},
    )
    recipe["stages"][0]["params"]["include_files_key"] = "id"

    result = aa.validate(recipe)

    assert not result["runnable"]
    assert any(issue["code"] == "unwrappable_stage" and issue["severity"] == "error" for issue in result["issues"])
    assert not (tmp_path / "snips").exists()
    assert not (tmp_path / "snips.tar").exists()
