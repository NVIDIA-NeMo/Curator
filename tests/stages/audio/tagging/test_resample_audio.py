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

import tempfile
from collections.abc import Callable
from pathlib import Path

from nemo_curator.stages.audio.tagging.resample_audio import ResampleAudioStage
from nemo_curator.tasks import AudioTask


class TestResampleAudioStage:
    """Tests for ResampleAudioStage."""

    def test_process(self, audio_task: Callable[..., AudioTask], audio_filepath: Path) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stage = ResampleAudioStage(resampled_audio_dir=tmpdir)
            stage.setup()
            task = audio_task(
                audio_filepath=str(audio_filepath),
                audio_item_id="id_1",
            )
            result = stage.process(task)
            out = result.data
            assert out.get("audio_filepath") == str(audio_filepath)
            assert out.get("resampled_audio_filepath") == f"{tmpdir}/id_1.wav"
            assert out.get("duration") == 60.0
