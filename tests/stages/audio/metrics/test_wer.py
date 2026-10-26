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

import pytest

from nemo_curator.stages.audio._agent._planning import validate_pipeline
from nemo_curator.stages.audio.metrics.wer import ComputeWERStage


@pytest.mark.parametrize("segments_key", ["segments", "turns"])
def test_nested_wer_cannot_use_unrelated_top_level_text(segments_key: str) -> None:
    stage = ComputeWERStage(
        segments_key=segments_key, hypothesis_text_key="prediction", reference_text_key="reference"
    )
    report = validate_pipeline([stage], initial_keys={segments_key, "prediction", "reference"})

    assert not report.ok
    assert any(issue.code == "unsatisfied_reads" for issue in report.issues)


@pytest.mark.parametrize("segments_key", ["segments", "turns"])
def test_nested_wer_accepts_text_in_selected_scope(segments_key: str) -> None:
    stage = ComputeWERStage(
        segments_key=segments_key, hypothesis_text_key="prediction", reference_text_key="reference"
    )
    report = validate_pipeline([stage], initial_keys={segments_key}, initial_segment_keys={"prediction", "reference"})

    assert report.ok


def test_task_wer_accepts_top_level_text_without_segment_container() -> None:
    report = validate_pipeline([ComputeWERStage()], initial_keys={"text", "text_ref"})

    assert report.ok
