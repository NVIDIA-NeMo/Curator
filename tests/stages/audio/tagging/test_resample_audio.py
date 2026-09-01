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
from unittest.mock import patch

import pytest

from nemo_curator.backends.ray_data.utils import is_actor_stage
from nemo_curator.stages.audio.tagging.resample_audio import ResampleAudioStage
from nemo_curator.tasks import AudioTask


class TestResampleAudioStage:
    """Tests for ResampleAudioStage."""

    def test_setup_on_node_does_not_preflight_worker_executable(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        stage = ResampleAudioStage(resampled_audio_dir=str(output_dir))

        with patch("nemo_curator.stages.audio.tagging.resample_audio.shutil.which") as mock_which:
            stage.setup_on_node()

        mock_which.assert_not_called()
        assert output_dir.is_dir()

    def test_remains_ray_data_task_stage(self, tmp_path: Path) -> None:
        stage = ResampleAudioStage(resampled_audio_dir=str(tmp_path))

        assert is_actor_stage(stage) is False

    def test_preserves_existing_positional_field_order(self, tmp_path: Path) -> None:
        stage = ResampleAudioStage(
            str(tmp_path),
            "flac",
            22050,
            "wav",
            2,
            "source_path",
            "converted_path",
            "clip_duration",
            "clip_id",
            "CustomResample",
        )

        assert stage.audio_filepath_key == "source_path"
        assert stage.resampled_audio_filepath_key == "converted_path"
        assert stage.duration_key == "clip_duration"
        assert stage.audio_item_id_key == "clip_id"
        assert stage.name == "CustomResample"
        assert stage.ffmpeg_executable == "ffmpeg"

    def test_resolve_accepts_explicit_executable_path_with_spaces(self, tmp_path: Path) -> None:
        executable = tmp_path / "user tools" / "ffmpeg custom"
        executable.parent.mkdir()
        executable.touch()
        executable.chmod(0o755)
        stage = ResampleAudioStage(
            resampled_audio_dir=str(tmp_path / "output"),
            ffmpeg_executable=str(executable),
        )

        resolved_executable = stage._resolve_ffmpeg_executable()

        assert resolved_executable == str(executable.resolve())

    def test_resolve_expands_user_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        home_dir = tmp_path / "home"
        executable = home_dir / ".local" / "bin" / "ffmpeg"
        executable.parent.mkdir(parents=True)
        executable.touch()
        executable.chmod(0o755)
        monkeypatch.setenv("HOME", str(home_dir))
        stage = ResampleAudioStage(
            resampled_audio_dir=str(tmp_path / "output"),
            ffmpeg_executable="~/.local/bin/ffmpeg",
        )

        resolved_executable = stage._resolve_ffmpeg_executable()

        assert resolved_executable == str(executable.resolve())

    def test_resolve_makes_relative_path_stable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        working_dir = tmp_path / "working"
        executable = working_dir / "tools" / "ffmpeg"
        executable.parent.mkdir(parents=True)
        executable.touch()
        executable.chmod(0o755)
        monkeypatch.chdir(working_dir)
        stage = ResampleAudioStage(
            resampled_audio_dir=str(tmp_path / "output"),
            ffmpeg_executable="tools/ffmpeg",
        )

        resolved_executable = stage._resolve_ffmpeg_executable()
        monkeypatch.chdir(tmp_path)

        assert resolved_executable == str(executable.resolve())

    def test_resolve_reports_actionable_error_for_missing_executable(self, tmp_path: Path) -> None:
        stage = ResampleAudioStage(
            resampled_audio_dir=str(tmp_path),
            ffmpeg_executable=str(tmp_path / "missing" / "ffmpeg"),
        )

        with pytest.raises(RuntimeError) as exc_info:
            stage._resolve_ffmpeg_executable()

        message = str(exc_info.value)
        assert str(tmp_path / "missing" / "ffmpeg") in message
        assert "user-writable environment" in message
        assert "ffmpeg_executable" in message
        assert "every executor node" in message

    def test_process_uses_configured_executable_and_publishes_output(
        self,
        tmp_path: Path,
        audio_task: Callable[..., AudioTask],
    ) -> None:
        executable = "/opt/user tools/ffmpeg custom"
        input_path = tmp_path / "input.wav"
        input_path.touch()
        output_dir = tmp_path / "output"
        stage = ResampleAudioStage(
            resampled_audio_dir=str(output_dir),
            ffmpeg_executable=executable,
        )
        stage.setup_on_node()

        def create_mock_output(cmd: list[str], **_kwargs: object) -> None:
            Path(cmd[-1]).touch()

        with (
            patch("nemo_curator.stages.audio.tagging.resample_audio.shutil.which", return_value=executable),
            patch(
                "nemo_curator.stages.audio.tagging.resample_audio.subprocess.run",
                side_effect=create_mock_output,
            ) as mock_run,
            patch("nemo_curator.stages.audio.tagging.resample_audio.get_audio_duration", return_value=1.0),
        ):
            stage.process(audio_task(audio_filepath=str(input_path), audio_item_id="id_1"))

        assert output_dir.is_dir()
        mock_run.assert_called_once()
        command = mock_run.call_args.args[0]
        assert command[:-1] == [
            executable,
            "-v",
            "error",
            "-i",
            str(input_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
        ]
        assert Path(command[-1]).parent == output_dir
        assert mock_run.call_args.kwargs == {"check": True, "capture_output": True, "text": True}
        assert (output_dir / "id_1.wav").is_file()

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
