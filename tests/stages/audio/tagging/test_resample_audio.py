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

import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import pytest

import nemo_curator.stages.audio.tagging.resample_audio as resample_audio_module
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

    def test_process_removes_partial_output_after_ffmpeg_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        audio_task: Callable[..., AudioTask],
        audio_filepath: Path,
    ) -> None:
        temporary_paths: list[Path] = []

        def fail_after_partial_write(cmd: list[str], *, check: bool, capture_output: bool, text: bool) -> None:
            assert check is True
            assert capture_output is True
            assert text is True
            temporary_path = Path(cmd[-1])
            temporary_path.write_bytes(b"partial")
            temporary_paths.append(temporary_path)
            raise subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr(resample_audio_module.subprocess, "run", fail_after_partial_write)
        stage = ResampleAudioStage(resampled_audio_dir=str(tmp_path))
        task = audio_task(audio_filepath=str(audio_filepath), audio_item_id="id_1")

        with pytest.raises(RuntimeError, match="Error converting"):
            stage.process(task)

        assert len(temporary_paths) == 1
        assert not temporary_paths[0].exists()
        assert not (tmp_path / "id_1.wav").exists()
