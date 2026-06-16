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
from math import isclose
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

    def test_process_emit_waveform_without_writing_audio(
        self,
        audio_task: Callable[..., AudioTask],
        audio_filepath: Path,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stage = ResampleAudioStage(
                resampled_audio_dir=tmpdir,
                write_resampled_audio=False,
                emit_waveform=True,
                target_sample_rate=16000,
                target_nchannels=1,
            )
            stage.setup_on_node()
            task = audio_task(
                audio_filepath=str(audio_filepath),
                audio_item_id="id_1",
            )

            result = stage.process(task)

            out = result.data
            assert out.get("audio_filepath") == str(audio_filepath)
            assert "resampled_audio_filepath" not in out
            assert out.get("sample_rate") == 16000
            assert out.get("is_mono") is True
            assert out["waveform"].shape[0] == 1
            assert out.get("num_samples") == out["waveform"].shape[-1]
            assert isclose(out.get("duration"), 60.0)
            assert not (Path(tmpdir) / "id_1.wav").exists()
