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


from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from nemo_curator.stages.audio._agent._agent_registry import build_contract, stage_params, static_contract
from nemo_curator.stages.audio.agent import pipeline_identity
from nemo_curator.stages.audio.common import (
    GetAudioDurationStage,
    ManifestWriterStage,
    PreserveByValueStage,
)
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from pathlib import Path

    from nemo_curator.stages.base import ProcessingStage


def test_pipeline_identity_tracks_configured_semantics() -> None:
    assert pipeline_identity([GetAudioDurationStage()]) == pipeline_identity([GetAudioDurationStage()])
    assert pipeline_identity([GetAudioDurationStage()]) != pipeline_identity(
        [GetAudioDurationStage(duration_key="seconds")]
    )
    assert pipeline_identity(
        [GetAudioDurationStage(), GetAudioDurationStage(duration_key="seconds")]
    ) != pipeline_identity([GetAudioDurationStage(duration_key="seconds"), GetAudioDurationStage()])


def test_pipeline_identity_rejects_unserializable_configuration() -> None:
    with pytest.raises(TypeError, match="Cannot fingerprint"):
        pipeline_identity([GetAudioDurationStage(duration_key=object())])


@pytest.mark.parametrize("comparison", ["lt", "le", "eq", "ne", "ge", "gt"])
def test_pipeline_identity_supports_value_filters(comparison: str) -> None:
    from nemo_curator.stages.audio.common import ManifestWriterStage

    def identity(operator: str, target: float = 1) -> str:
        return pipeline_identity(
            [
                GetAudioDurationStage(),
                PreserveByValueStage("duration", target, operator=operator),
                ManifestWriterStage("output.jsonl"),
            ]
        )

    assert identity(comparison) == identity(comparison)
    assert identity(comparison) != identity(comparison, 2)
    assert identity(comparison) != identity("ne" if comparison == "eq" else "eq")


def test_pipeline_identity_rejects_arbitrary_callable() -> None:
    with pytest.raises(TypeError, match="Cannot fingerprint"):
        pipeline_identity([GetAudioDurationStage(duration_key=lambda: "duration")])


def test_composite_identity_follows_resolved_configuration(tmp_path: Path) -> None:
    module = pytest.importorskip("nemo_curator.stages.audio.advanced_pipelines.audio_data_filter")
    cls = module.AudioDataFilterStage
    config = tmp_path / "config.yaml"
    config.write_text("mono_conversion:\n  output_sample_rate: 16000\n")
    first = pipeline_identity([cls(config_path=config)])
    assert first == pipeline_identity([cls(config={"mono_conversion": {"output_sample_rate": 16000}})])
    config.write_text("mono_conversion:\n  output_sample_rate: 48000\n")
    assert first != pipeline_identity([cls(config_path=config)])


def test_identity_rejects_unrunnable_composite() -> None:
    from nemo_curator.stages.base import CompositeStage

    class SingleChildAudioComposite(CompositeStage[AudioTask, AudioTask]):
        def decompose(self) -> list[ProcessingStage]:
            return [GetAudioDurationStage()]

    with pytest.raises(ValueError, match="unresolved composite"):
        pipeline_identity([SingleChildAudioComposite()])


@dataclass
class _AgentParamMetadataFixture:
    visible: str = "public"
    runtime_only: object | None = field(default=None, metadata={"agent_param": False})


@dataclass
class _AgentRequiredMetadataFixture:
    required_for_agent: str = field(default="", metadata={"agent_required": True})


def test_stage_params_respects_field_level_agent_exclusion() -> None:
    assert [param.name for param in stage_params(_AgentParamMetadataFixture)] == ["visible"]


def test_stage_params_can_require_a_runtime_default_for_agent_configuration() -> None:
    param = stage_params(_AgentRequiredMetadataFixture)[0]

    assert param.default == ""
    assert param.required is True


def test_manifest_writer_static_contract_exposes_invariant_sink_gates(tmp_path: Path) -> None:
    """Static discovery must not describe a required-path JSONL sink as pure."""
    static = static_contract(ManifestWriterStage)
    configured = build_contract(ManifestWriterStage(output_path=str(tmp_path / "out.jsonl")))

    assert static.gates == configured.gates
