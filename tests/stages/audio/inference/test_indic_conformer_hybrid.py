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

from nemo_curator.stages.audio.inference.indic_conformer_hybrid import (
    InferenceIndicConformerHybridStage,
)
from nemo_curator.tasks import AudioTask


def test_stage_routes_only_supported_indic_languages() -> None:
    stage = InferenceIndicConformerHybridStage()
    stage._model = MagicMock()
    stage._model.generate.return_value = (["नमस्ते"], ["hi"])
    tasks = [
        AudioTask(data={"waveform": np.zeros(10), "sampling_rate": 16000, "source_lang": "hi"}),
        AudioTask(data={"waveform": np.zeros(10), "sampling_rate": 16000, "source_lang": "en"}),
    ]

    stage.process_batch(tasks)

    assert tasks[0].data["asr_prediction"] == "नमस्ते"
    assert tasks[0].data["asr_language"] == "hi"
    assert tasks[1].data["asr_prediction"] == ""
    assert tasks[1].data["additional_notes"]["asr_prediction"] == "lang_not_supported:en"
    assert "waveform" not in tasks[0].data


def test_stage_preserves_waveform_for_a_recovery_model() -> None:
    stage = InferenceIndicConformerHybridStage(keep_waveform=True)
    stage._model = MagicMock()
    stage._model.generate.return_value = (["தமிழ்"], ["ta"])
    task = AudioTask(data={"waveform": np.zeros(10), "sampling_rate": 16000, "source_lang": "ta"})

    stage.process_batch([task])

    assert "waveform" in task.data
