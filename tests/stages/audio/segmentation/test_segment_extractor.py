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

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.audio.segmentation import SegmentExtractorStage
from nemo_curator.tasks import AudioTask
from nemo_curator.utils.performance_utils import StagePerfStats


def test_segment_extractor_fans_out_sed_events_without_writing_audio() -> None:
    stage = SegmentExtractorStage()
    task = AudioTask(
        task_id="utt",
        dataset_name="dataset",
        filepath_key="audio_filepath",
        data={
            "audio_filepath": "input.wav",
            "text": "hello",
            "sed_events": [
                {"start_time": 0.5, "end_time": 1.0, "mean_confidence": 0.7},
                {"start_time": 2.0, "end_time": 3.5},
            ],
        },
        _metadata={"source": "manifest.jsonl"},
        _stage_perf=[StagePerfStats(stage_name="upstream", process_time=1.0)],
    )

    outputs = stage.process(task)

    assert len(outputs) == 2
    assert outputs[0].task_id == "utt_seg_0"
    assert outputs[0].data["audio_filepath"] == "input.wav"
    assert outputs[0].data["segment_start"] == 0.5
    assert outputs[0].data["segment_end"] == 1.0
    assert outputs[0].data["segment_idx"] == 0
    assert outputs[0].data["segment_confidence"] == 0.7
    assert outputs[0]._metadata == {"source": "manifest.jsonl"}
    assert outputs[0]._stage_perf[0].stage_name == "upstream"

    assert outputs[1].data["segment_start"] == 2.0
    assert outputs[1].data["segment_end"] == 3.5
    assert outputs[1].data["segment_confidence"] == 0.0


def test_segment_extractor_returns_empty_for_no_events() -> None:
    stage = SegmentExtractorStage()
    task = AudioTask(data={"sed_events": []})

    assert stage.process(task) == []


def test_segment_extractor_marks_ray_fanout() -> None:
    stage = SegmentExtractorStage()

    assert stage.ray_stage_spec() == {RayStageSpecKeys.IS_FANOUT_STAGE: True}
