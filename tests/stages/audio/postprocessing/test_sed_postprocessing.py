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

import numpy as np

from nemo_curator.stages.audio.inference.sed import SEDInferenceStage
from nemo_curator.stages.audio.postprocessing.sed_postprocessing import SEDPostprocessingStage
from nemo_curator.stages.audio.postprocessing.sed_utils import aggregate_speech_probs, framewise_to_events
from nemo_curator.tasks import AudioTask


def test_sed_inference_import_does_not_load_optional_model_deps() -> None:
    stage = SEDInferenceStage(num_workers_override=2)

    assert stage.num_workers() == 2
    assert stage.xenna_stage_spec() == {"num_workers": 2}


def test_framewise_to_events_threshold_and_merge() -> None:
    probs = np.array([0.1, 0.8, 0.9, 0.1, 0.7, 0.8], dtype=np.float32)

    events = framewise_to_events(probs, fps=10.0, threshold=0.5, min_duration_sec=0.1, merge_gap_sec=0.2)

    assert events == [
        {
            "start_time": 0.1,
            "end_time": 0.6,
            "mean_confidence": float(np.mean(probs[1:6])),
            "max_confidence": float(np.max(probs[1:6])),
        }
    ]


def test_aggregate_speech_probs_tolerates_small_class_matrix() -> None:
    framewise = np.zeros((4, 2), dtype=np.float32)
    framewise[:, 0] = 0.5
    framewise[:, 1] = 0.5

    probs = aggregate_speech_probs(framewise, [0, 1, 10], mode="noisy_or")

    np.testing.assert_allclose(probs, np.full(4, 0.75, dtype=np.float32))


def test_sed_postprocessing_labels_events_and_drops_framewise() -> None:
    framewise = np.zeros((10, 7), dtype=np.float32)
    framewise[2:7, 0] = 0.9
    task = AudioTask(data={
        "_sed_framewise": framewise,
        "sed_valid_frames": 10,
        "sed_fps": 10.0,
    })
    stage = SEDPostprocessingStage(threshold=0.5, min_duration_sec=0.1)

    result = stage.process(task)

    assert "_sed_framewise" not in result.data
    assert result.data["sed_events"][0]["label"] == "speech"
    assert result.data["sed_events"][0]["start_time"] == 0.2
    assert result.data["sed_events"][0]["end_time"] == 0.7
