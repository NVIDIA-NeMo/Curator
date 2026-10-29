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

"""Smoke input bounds preserve the configured source's own selection limit."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_curator.stages.audio.audio_agent._verb_smoke_inputs import _bound_recipe
from nemo_curator.stages.audio.audio_agent.contracts import SmokeReport
from nemo_curator.stages.audio.audio_agent.recipe import Recipe, build_stages


@pytest.mark.parametrize("limit", [None, -1, 0, 1, 20])
def test_folder_smoke_never_reads_beyond_the_recipe(tmp_path: Path, limit: int | None) -> None:
    for name in ("a.wav", "b.wav", "c.wav"):
        (tmp_path / name).write_bytes(b"fixture")
    recipe = Recipe.from_dict(
        {
            "stages": [
                {
                    "ref": "CreateInitialManifestAudioFolderStage",
                    "params": {"data_dir": str(tmp_path), "max_samples": limit},
                }
            ]
        }
    )
    source, errors = build_stages(recipe)
    assert not errors
    authored = source[0].process(None)
    bound = _bound_recipe(recipe, 2, SmokeReport(sample=2), SimpleNamespace(generated=False))
    stages, errors = build_stages(bound.recipe)
    assert not errors
    sampled = stages[0].process(None)
    assert [task.data["audio_filepath"] for task in sampled] == [task.data["audio_filepath"] for task in authored][:2]
    assert recipe.stages[0].params["max_samples"] == limit


def test_readspeech_smoke_respects_an_omitted_default_limit(tmp_path: Path) -> None:
    recipe = Recipe.from_dict(
        {
            "stages": [
                {
                    "ref": "CreateInitialManifestReadSpeechStage",
                    "params": {"raw_data_dir": str(tmp_path)},
                }
            ]
        }
    )
    source, errors = build_stages(recipe)
    assert not errors
    bound = _bound_recipe(recipe, 10000, SmokeReport(sample=10000), SimpleNamespace(generated=False))
    assert bound.recipe.stages[0].params["max_samples"] == source[0].max_samples
    assert "max_samples" not in recipe.stages[0].params
