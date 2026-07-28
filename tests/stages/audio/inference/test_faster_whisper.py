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

from unittest.mock import MagicMock

import numpy as np

from nemo_curator.stages.audio.inference.faster_whisper import InferenceFasterWhisperStage
from nemo_curator.tasks import AudioTask


def test_stage_routes_supported_languages_and_marks_unsupported() -> None:
    stage = InferenceFasterWhisperStage()
    stage._model = MagicMock()
    stage._model.generate.return_value = (["kumusta"], ["tl"])
    tasks = [
        AudioTask(data={"waveform": np.zeros(10), "sampling_rate": 16000, "source_lang": "fil"}),
        AudioTask(data={"waveform": np.zeros(10), "sampling_rate": 16000, "source_lang": "xx"}),
    ]

    stage.process_batch(tasks)

    assert tasks[0].data["asr_prediction"] == "kumusta"
    assert tasks[0].data["asr_language"] == "tl"
    assert "waveform" not in tasks[0].data
    assert tasks[1].data["asr_prediction"] == ""
    assert tasks[1].data["additional_notes"]["asr_prediction"] == "lang_not_supported:xx"


def test_stage_can_preserve_waveforms_for_recovery() -> None:
    stage = InferenceFasterWhisperStage(keep_waveform=True)
    stage._model = MagicMock()
    stage._model.generate.return_value = (["hello"], ["en"])
    task = AudioTask(data={"waveform": np.zeros(10), "sampling_rate": 16000, "source_lang": "en"})

    stage.process_batch([task])

    assert "waveform" in task.data
