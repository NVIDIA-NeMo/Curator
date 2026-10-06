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

"""Configured persistence determines which outputs can become reuse artifacts."""

from pathlib import Path

import pytest

from nemo_curator.stages.audio.audio_agent import artifacts, reuse, verbs
from nemo_curator.stages.audio.audio_agent.recipe import Recipe, StageRef, build_stages


@pytest.mark.parametrize("write_to_disk", [False, True])
def test_resampler_artifact_requires_enabled_disk_output(tmp_path: Path, write_to_disk: bool) -> None:
    output = tmp_path / "audio"
    output.mkdir()
    (output / "old.wav").write_bytes(b"old content")
    stage = StageRef(
        ref="ResampleAudioStage",
        params={"resampled_audio_dir": str(output), "write_to_disk": write_to_disk, "keep_waveform_in_task": True},
    )
    assert artifacts.output_uri(stage) == ((str(output), "audio_dir") if write_to_disk else ("", "unknown"))


def test_disabled_sink_cannot_publish_unrelated_existing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(tmp_path))
    output = tmp_path / "disabled"
    output.mkdir()
    (output / "old.wav").write_bytes(b"old content")
    recipe = Recipe.from_dict(
        {
            "stages": [
                {"ref": "ManifestReader", "params": {"manifest_path": str(tmp_path / "input.jsonl")}},
                {
                    "ref": "ResampleAudioStage",
                    "params": {
                        "resampled_audio_dir": str(output),
                        "write_to_disk": False,
                        "keep_waveform_in_task": True,
                    },
                },
            ]
        }
    ).freeze()
    stages, issues = build_stages(recipe)
    assert stages
    assert not issues
    assert not artifacts.plan_steps(recipe, "stat:fixture")[-1].persists()
    published = verbs._publish_artifacts(
        recipe,
        stages,
        dataset_key="stat:fixture",
        fingerprint_tier="stat",
        per_stage={},
        run_id="fixture",
        input_count=1,
        data_profile=None,
        started_at="",
        ended_at="",
    )
    assert published == []
    assert not artifacts.list_artifacts()
    assert reuse.scan(recipe, dataset_key="stat:fixture")["decision"] != "already_done"


def test_dry_run_snippet_archive_is_not_an_artifact(tmp_path: Path) -> None:
    stage = StageRef(
        ref="SnippetExtractionStage",
        params={"output_dir": str(tmp_path), "output_audio_tar_path": str(tmp_path / "old.tar"), "dry_run": True},
    )
    assert artifacts.output_uri(stage) == ("", "unknown")


def test_uninspectable_stage_cannot_claim_an_artifact(tmp_path: Path) -> None:
    assert artifacts.output_uri(StageRef(ref="UnknownStage", params={"output_path": str(tmp_path / "old.jsonl")})) == (
        "",
        "unknown",
    )
