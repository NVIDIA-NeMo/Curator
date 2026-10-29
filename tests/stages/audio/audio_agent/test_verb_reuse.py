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

"""Checkpoint keys use the same selected source as execution and reuse scans."""

import json
from pathlib import Path

import pytest

from nemo_curator.stages.audio.audio_agent import _verb_reuse
from nemo_curator.stages.audio.audio_agent.recipe import Recipe


@pytest.mark.parametrize("source_kind", ["folder", "manifest"])
def test_explicit_data_preserves_selected_recipe_identity(tmp_path: Path, source_kind: str) -> None:
    for name in ("a.wav", "b.wav"):
        (tmp_path / name).write_bytes(b"fixture")
    selected = str(tmp_path / "b.wav")
    if source_kind == "folder":
        data = tmp_path
        source = {
            "ref": "CreateInitialManifestAudioFolderStage",
            "params": {"data_dir": str(data), "include_files": [selected], "max_samples": 1},
        }
    else:
        data = tmp_path / "input.jsonl"
        data.write_text("".join(json.dumps({"recording": str(tmp_path / name)}) + "\n" for name in ("a.wav", "b.wav")))
        source = {
            "ref": "ManifestReader",
            "params": {"manifest_path": str(data), "include_files": [selected], "include_files_key": "recording"},
        }
    recipe = Recipe.from_dict({"stages": [source]}).freeze()
    expected = _verb_reuse.reuse_scan(recipe, data=str(data))["dataset_key"]
    assert expected
    assert _verb_reuse._recipe_dataset_key(recipe, str(data)) == expected
    assert _verb_reuse._recipe_dataset_key(recipe, None) == expected
    assert _verb_reuse._recipe_dataset_key(recipe, str(tmp_path / "other")) == ""
    (tmp_path / "a.wav").write_bytes(b"excluded content changes size")
    assert _verb_reuse._recipe_dataset_key(recipe, str(data)) == expected
    (tmp_path / "b.wav").write_bytes(b"selected content changes size")
    assert _verb_reuse._recipe_dataset_key(recipe, str(data)) != expected
