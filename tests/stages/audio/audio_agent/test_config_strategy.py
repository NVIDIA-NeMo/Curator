# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A conversion target is not a rate-verification threshold, even with the same spelling."""

from dataclasses import dataclass

import pytest

from nemo_curator.stages.audio._agent._agent_ready import AgentReady, IOSpec, StageContract
from nemo_curator.stages.audio.audio_agent import config_strategy
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


def test_new_converter_does_not_adopt_the_source_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_curator.stages import base
    from nemo_curator.stages.audio._agent import _catalog

    _catalog.list_agent_ready_stages()
    monkeypatch.setattr(base, "_STAGE_REGISTRY", dict(base._STAGE_REGISTRY))

    @dataclass
    class FutureRateConverter(AgentReady, ProcessingStage[AudioTask, AudioTask]):
        output_sample_rate: int = 16000

        def describe(self) -> StageContract:
            return StageContract()

        def process(self, task: AudioTask) -> AudioTask:
            return task

    derived = config_strategy.resolve_from_data("FutureRateConverter", {"sample_rates": {"48000": 2}})
    assert derived["params"] == {}
    assert derived["strategy"] == []
    assert FutureRateConverter().output_sample_rate == 16000


def test_declared_verification_respects_its_configured_mode() -> None:
    profile = {"sample_rates": {"16000": 2}}
    assert config_strategy.resolve_from_data("MonoConversionStage", profile)["params"] == {"output_sample_rate": 16000}
    disabled = config_strategy.resolve_from_data(
        "MonoConversionStage", profile, configured={"strict_sample_rate": False}
    )
    assert disabled["params"] == {}
    assert disabled["asks"] == []
    explicit = config_strategy.resolve_from_data("MonoConversionStage", profile, already_set={"output_sample_rate"})
    assert explicit["params"] == {}
    assert explicit["strategy"] == []


def test_new_verifier_can_bind_a_renamed_parameter_from_its_card(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_curator.stages import base
    from nemo_curator.stages.audio._agent import _catalog
    from nemo_curator.stages.audio.audio_agent import index

    _catalog.list_agent_ready_stages()
    monkeypatch.setattr(base, "_STAGE_REGISTRY", dict(base._STAGE_REGISTRY))

    @dataclass
    class FutureRateGate(AgentReady, ProcessingStage[AudioTask, AudioTask]):
        expected_rate: int = 32000
        enforce: bool = True

        def describe(self) -> StageContract:
            return StageContract(reads=IOSpec(data_keys=["sample_rate"]), cardinality="filter")

        def process(self, task: AudioTask) -> AudioTask | list[AudioTask]:
            return [] if self.enforce and task.data["sample_rate"] != self.expected_rate else task

    knowledge = index.KnowledgeIndex()
    knowledge._cards_by_stage["FutureRateGate"] = {
        "planning": {"sample_rate_verification": {"param": "expected_rate", "when": {"enforce": True}}},
        "verified": {"planning": "mechanical"},
    }
    monkeypatch.setattr(index, "get_index", lambda: knowledge)
    profile = {"sample_rates": {"16000": 1}}
    derived = config_strategy.resolve_from_data("FutureRateGate", profile)
    assert derived["params"] == {"expected_rate": 16000}
    task = AudioTask(data={"sample_rate": 16000})
    assert FutureRateGate().process(task) == []
    assert FutureRateGate(**derived["params"]).process(task) is task
    assert config_strategy.resolve_from_data("FutureRateGate", profile, configured={"enforce": False})["params"] == {}
