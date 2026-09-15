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

from nemo_curator.stages.audio._agent._conformance import assert_agent_ready
from nemo_curator.stages.audio._agent._planning import validate_pipeline
from nemo_curator.stages.audio.tagging.text.itn import InverseTextNormalizationStage
from nemo_curator.tasks import AudioTask


class TestInverseTextNormalizationStage:
    """Tests for InverseTextNormalizationStage."""

    def test_process(self, audio_task: Callable[..., AudioTask]) -> None:
        stage = InverseTextNormalizationStage(language="en", text_key="text")
        stage.setup()
        task = audio_task(
            segments=[
                {"text": "hello", "start": 0.0, "end": 0.5},
                {"text": "the answer is forty two", "start": 0.5, "end": 1.0},
            ],
        )
        result = stage.process(task)
        assert stage._normalizer is not None
        out = result.data
        assert len(out["segments"]) == 2
        assert out["segments"][0]["text_ITN"] == "hello"
        assert out["segments"][1]["text_ITN"] == "the answer is 42"

    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            ({"segments": [{}]}, None),
            ({"segments": [{"text": ""}]}, None),
            ({"segments": [{"text": "forty two"}]}, "42"),
        ],
        ids=["missing-text", "empty-text", "populated"],
    )
    def test_agent_ready_conditional_nested_output(
        self,
        data: dict,
        expected: str | None,
    ) -> None:
        stage = InverseTextNormalizationStage()
        normalizer = MagicMock()
        normalizer.split_text_into_sentences.side_effect = lambda text: [text]
        normalizer.normalize_list.return_value = ["42"]
        stage._normalizer = normalizer
        task = AudioTask(dataset_name="test", data=data)

        contract = assert_agent_ready(
            stage,
            lambda: task,
            segments_key="segments",
        )

        assert contract.reads.data_keys == ["segments"]
        assert contract.writes.segment_data_keys == []
        assert len(contract.conditional_writes) == 1
        assert contract.conditional_writes[0].writes.segment_data_keys == ["text_ITN"]
        if expected is None:
            assert all("text_ITN" not in segment for segment in task.data.get("segments", []))
        else:
            assert task.data["segments"][0]["text_ITN"] == expected

    def test_planner_does_not_guarantee_conditional_nested_output(self) -> None:
        report = validate_pipeline(
            [InverseTextNormalizationStage()],
            initial_roles={"segments"},
            initial_keys={"segments"},
            initial_task_type="AudioTask",
        )

        assert report.ok
        assert "text_ITN" not in report.produced_keys
