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

"""Coverage for the audio-agent package layout."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


def test_audio_agent_is_nested_under_the_audio_package() -> None:
    from nemo_curator.stages.audio import audio_agent as canonical_from_parent

    canonical = importlib.import_module("nemo_curator.stages.audio.audio_agent")
    package = Path(canonical.__file__).resolve().parent
    assert canonical_from_parent is canonical
    assert package.parts[-3:] == ("stages", "audio", "audio_agent")
    assert not (package.parents[2] / "audio_agent").exists()


def test_nested_module_entrypoint_serves_the_cli() -> None:
    result = subprocess.run(  # noqa: S603 - fixed interpreter and module name
        [sys.executable, "-m", "nemo_curator.stages.audio.audio_agent", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Audio Agent tool surface" in result.stdout
    assert "discover" in result.stdout
    assert "run" in result.stdout


def test_resources_and_source_checkout_detection_follow_the_new_package() -> None:
    from nemo_curator.stages.audio.audio_agent import env_health, skills_dir

    package = Path(env_health.__file__).resolve().parent
    assert package.parts[-3:] == ("stages", "audio", "audio_agent")
    assert Path(skills_dir()).resolve() == package / "skills"
    assert (package / "knowledge" / "CARD_SCHEMA.md").is_file()
    assert (package / "recipes" / "alm_windowing.yaml").is_file()
    assert env_health._from_source_checkout() is True
