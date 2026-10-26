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

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

from nemo_curator.stages.audio._agent._agent_ready import AgentReady, IOSpec, StageContract
from nemo_curator.stages.audio._agent._agent_registry import build_contract
from nemo_curator.stages.audio._agent._conformance import assert_agent_ready
from nemo_curator.stages.audio._agent._planning import validate_pipeline
from nemo_curator.stages.audio.tagging.text.chinese_conversion import ChineseConversionStage
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


class TestChineseConversionStage:
    def test_converts_traditional_to_simplified(self, audio_task: Callable[..., AudioTask]) -> None:
        stage = ChineseConversionStage(text_key="text", convert_type="t2s")
        stage.setup()
        task = audio_task(
            segments=[
                {"text": "漢字", "start": 0.0, "end": 1.0},
            ],
        )
        result = stage.process(task)
        out = result.data
        assert out["segments"][0]["text_simplified"] == "汉字"
        assert out["segments"][0]["text"] == "漢字"

    def test_segment_without_text_key_is_skipped(self, audio_task: Callable[..., AudioTask]) -> None:
        stage = ChineseConversionStage(text_key="text")
        stage.setup()
        task = audio_task(
            segments=[
                {"start": 0.0, "end": 1.0},
            ],
        )
        result = stage.process(task)
        out = result.data
        assert "text_simplified" not in out["segments"][0]

    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            ({"segments": [{}]}, None),
            ({"segments": []}, None),
            ({"segments": [{"text": "漢字"}]}, "汉字"),
        ],
        ids=["missing-text", "empty-segments", "populated"],
    )
    def test_agent_ready_conditional_nested_output(
        self,
        data: dict,
        expected: str | None,
    ) -> None:
        stage = ChineseConversionStage()
        converter = MagicMock()
        converter.convert.return_value = "汉字"
        stage._converter = converter
        task = AudioTask(dataset_name="test", data=data)

        contract = assert_agent_ready(
            stage,
            lambda: task,
            segments_key="segments",
        )

        assert contract.reads.data_keys == ["segments"]
        assert contract.writes.segment_data_keys == []
        assert len(contract.conditional_writes) == 1
        assert contract.conditional_writes[0].writes.segment_data_keys == ["text_simplified"]
        if expected is None:
            assert all("text_simplified" not in segment for segment in task.data.get("segments", []))
        else:
            assert task.data["segments"][0]["text_simplified"] == expected

    def test_planner_does_not_guarantee_conditional_nested_output(self) -> None:
        report = validate_pipeline(
            [ChineseConversionStage()],
            initial_roles={"segments"},
            initial_keys={"segments"},
            initial_task_type="AudioTask",
        )

        assert report.ok
        assert "text_simplified" not in report.produced_keys


def test_chinese_legacy_positional_signature_still_binds() -> None:
    """The agent-added keys are keyword-only, so legacy positional slots keep their meaning."""
    stage = ChineseConversionStage("t", "s2t", "MyName")
    assert stage.text_key == "t"
    assert stage.convert_type == "s2t"
    assert stage.name == "MyName"
    assert stage.segments_key == "segments"
    assert stage.output_suffix == "_simplified"


@pytest.mark.parametrize("has_text", [False, True])
def test_renamed_nested_text_contract_and_planner(has_text: bool) -> None:
    stage = ChineseConversionStage(text_key="transcript", output_suffix="_simplified")
    contract = build_contract(stage)
    assert contract.optional_reads.segment_data_keys == ["transcript"]
    assert contract.conditional_writes[0].requires_keys == ["transcript"]
    assert contract.key_roles["transcript_simplified"] == "text"

    class TextConsumer(AgentReady, ProcessingStage):
        def process(self, task: AudioTask) -> AudioTask:
            return task

        def describe(self) -> StageContract:
            return StageContract(
                reads=IOSpec(segment_data_keys=["transcript_simplified"]),
                key_roles={"transcript_simplified": "text"},
            )

    report = validate_pipeline(
        [stage, TextConsumer()],
        initial_roles={"segments"},
        initial_keys={"segments"},
        initial_segment_keys={"transcript"} if has_text else set(),
        initial_task_type="AudioTask",
    )
    assert report.ok == has_text
    assert any(issue.code == "conditional_read" for issue in report.issues) == has_text
