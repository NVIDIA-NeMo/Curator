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

from nemo_curator.stages.audio.audio_agent.recipe import Recipe
from nemo_curator.stages.audio.audio_agent.report import build_run_report, source_rejection_count
from nemo_curator.tasks import AudioTask


@pytest.mark.parametrize(
    "processing",
    [
        {"ref": "VADSegmentationStage", "params": {"nested": False}},
        {"ref": "InferenceSortformerStage", "params": {"fanout": True}},
        {"ref": "AudioToDocumentStage"},
    ],
)
def test_rejection_counts_remain_unknown_after_fanout_or_aggregation(processing: dict) -> None:
    recipe = Recipe.from_dict(
        {
            "stages": [
                {"ref": "CreateInitialManifestAudioFolderStage", "params": {"data_dir": "/fixture"}},
                processing,
            ]
        }
    )
    result = build_run_report(
        recipe=recipe, result_tasks=[AudioTask(data={}) for _ in range(3)], data_profile={"num_files": 3}
    )

    assert result.accepted == 3
    assert result.source_items == 3
    assert result.rejected is None


def test_one_to_one_pipeline_still_reports_source_rejections() -> None:
    recipe = Recipe.from_dict(
        {
            "stages": [
                {"ref": "CreateInitialManifestAudioFolderStage", "params": {"data_dir": "/fixture"}},
                {"ref": "GetAudioDurationStage"},
                {
                    "ref": "PreserveByValueStage",
                    "params": {"input_value_key": "duration", "target_value": 1, "operator": "ge"},
                },
            ]
        }
    )
    result = build_run_report(
        recipe=recipe, result_tasks=[AudioTask(data={}) for _ in range(2)], data_profile={"num_files": 3}
    )

    assert result.rejected == 1
    assert source_rejection_count(recipe, 3, 4) is None
