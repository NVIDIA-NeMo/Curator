# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from pathlib import Path

import pytest

from nemo_curator.stages.audio.inference.vad.whisperx_vad import WhisperXVADStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


class TestWhisperXVADStage:
    @pytest.mark.gpu
    def test_process(self, wav_filepath: Path) -> None:
        stage = WhisperXVADStage(
            min_length=0.5,
            max_length=40.0,
            segments_key="vad_segments",
            resources=Resources(gpus=1),
        )
        stage.setup()

        entry = {
            "resampled_audio_filepath": str(wav_filepath),
            "duration": 60.0,
        }
        task = AudioTask(data=entry)
        result = stage.process(task)
        out = result.data
        assert "vad_segments" in out
        assert isinstance(out["vad_segments"], list)
        assert len(out["vad_segments"]) == 2
