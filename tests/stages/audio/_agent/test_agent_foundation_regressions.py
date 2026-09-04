# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Regressions for foundation behavior exposed to audio-pipeline agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import soundfile as sf
import torch

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages import audio
from nemo_curator.stages.audio import agent
from nemo_curator.stages.audio._agent._agent_registry import build_contract, static_contract
from nemo_curator.stages.audio._agent._catalog import unavailable_modules
from nemo_curator.stages.audio._agent._composite import expand_composites
from nemo_curator.stages.audio._agent._planning import validate_pipeline
from nemo_curator.stages.audio._agent._residency import write_audio_stable
from nemo_curator.stages.audio.common import ManifestReader, ManifestWriterStage

if TYPE_CHECKING:
    from pathlib import Path


def test_stable_audio_names_include_layout_and_written_short_stereo_shape(tmp_path: Path) -> None:
    """Different channel layouts with identical samples need distinct artifacts."""
    output_dir = str(tmp_path)
    mono = torch.zeros(1, 32000)
    stereo = torch.zeros(2, 16000)

    mono_path = write_audio_stable(mono, 16000, output_dir=output_dir, stem="audio")
    stereo_path = write_audio_stable(stereo, 16000, output_dir=output_dir, stem="audio")

    assert mono_path != stereo_path
    assert sf.info(mono_path).channels == 1
    assert sf.info(stereo_path).channels == 2

    short_stereo_path = write_audio_stable(
        torch.tensor([[0.25], [0.75]]),
        16000,
        output_dir=output_dir,
        stem="short",
    )
    short_info = sf.info(short_stereo_path)
    assert (short_info.frames, short_info.channels) == (1, 2)


def test_nested_composite_is_reported_as_unrunnable(monkeypatch) -> None:  # noqa: ANN001
    """A shape rejected by Pipeline must not be downgraded to opaque."""
    stage = ManifestReader("manifest.jsonl")
    nested_children = [ManifestReader("one.jsonl"), ManifestReader("two.jsonl")]
    monkeypatch.setattr(stage, "decompose_and_apply_with", lambda: nested_children)

    expansion = expand_composites([stage])
    assert expansion.stages == []
    assert 0 not in expansion.opaque
    assert "nested composition" in expansion.unrunnable[0]

    report = validate_pipeline([stage])
    assert not report.ok
    assert any(issue.code == "composite_unrunnable" for issue in report.issues)

    # Parity with the executor is the whole claim: the bug was validation approving a shape
    # ``Pipeline.build()`` refuses. Asserting only our own verdict would let the two drift
    # apart again -- if the executor ever accepted nesting, this error would become a false
    # alarm, and the test above would keep passing.
    with pytest.raises(TypeError, match="Nested composition is not supported"):
        Pipeline(name="nested-composite-parity", stages=[stage]).build()


def test_manifest_writer_static_contract_exposes_invariant_sink_gates(tmp_path: Path) -> None:
    """Static discovery must not describe a required-path JSONL sink as pure."""
    static = static_contract(ManifestWriterStage)
    configured = build_contract(ManifestWriterStage(output_path=str(tmp_path / "out.jsonl")))

    assert static.gates == configured.gates


def test_public_facade_exposes_unavailable_modules_and_folder_source() -> None:
    """The documented public layer must expose foundation discovery features."""
    from nemo_curator.stages.audio import CreateInitialManifestAudioFolderStage
    from nemo_curator.stages.audio.common import CreateInitialManifestAudioFolderStage as FolderSource

    assert agent.unavailable_modules is unavailable_modules
    assert CreateInitialManifestAudioFolderStage is FolderSource
    assert "CreateInitialManifestAudioFolderStage" in audio.__all__
