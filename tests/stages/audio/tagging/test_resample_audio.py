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

import hashlib
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import torch
from nemo_curator.stages.audio._agent._agent_registry import build_contract, static_contract

import nemo_curator.stages.audio.tagging.resample_audio as resample_audio_module
from nemo_curator.stages.audio._agent._conformance import assert_agent_ready
from nemo_curator.stages.audio._agent._residency import resolve_audio
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

    def test_a_file_input_keeps_the_name_it_has_always_had(self, audio_filepath: Path) -> None:
        """Every tutorial reads real files off disk; their output names must not move."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stage = ResampleAudioStage(resampled_audio_dir=tmpdir)
            stage.setup()
            stage.process(AudioTask(task_id="t", dataset_name="d", data={"audio_filepath": str(audio_filepath)}))

            path_hash = hashlib.sha256(str(audio_filepath).encode()).hexdigest()[:8]
            assert os.listdir(tmpdir) == [f"{audio_filepath.stem}_{path_hash}.wav"]

    def test_a_waveform_input_writes_one_file_however_often_it_is_rerun(self) -> None:
        waveform = torch.sin(torch.arange(0, 16000 * 2) * 0.01).unsqueeze(0)
        with tempfile.TemporaryDirectory() as tmpdir:
            for _ in range(3):
                stage = ResampleAudioStage(resampled_audio_dir=tmpdir, input_residency="waveform")
                stage.setup()
                stage.process(
                    AudioTask(
                        task_id="t",
                        dataset_name="d",
                        data={"waveform": waveform.clone(), "sample_rate": 16000},
                    )
                )

            assert len(os.listdir(tmpdir)) == 1, "the same audio must not pile up a file per run"

    def test_changing_the_target_rate_does_not_reuse_the_old_conversion(self) -> None:
        """The name carries the settings, so 'it already exists, skip it' cannot serve 48 kHz for 16."""
        waveform = torch.sin(torch.arange(0, 16000 * 2) * 0.01).unsqueeze(0)
        with tempfile.TemporaryDirectory() as tmpdir:
            for rate in (16000, 8000):
                stage = ResampleAudioStage(
                    resampled_audio_dir=tmpdir, input_residency="waveform", target_sample_rate=rate
                )
                stage.setup()
                stage.process(
                    AudioTask(
                        task_id="t",
                        dataset_name="d",
                        data={"waveform": waveform.clone(), "sample_rate": 16000},
                    )
                )

            assert len(os.listdir(tmpdir)) == 2, "a different target rate must not answer from the old file"

    def test_a_second_run_at_a_new_rate_does_not_serve_the_old_file(self) -> None:
        import numpy as np
        import soundfile

        with tempfile.TemporaryDirectory() as srcdir, tempfile.TemporaryDirectory() as out:
            src = os.path.join(srcdir, "a.wav")
            soundfile.write(src, np.sin(np.arange(48000) * 0.01).astype("float32"), 48000)

            for rate in (16000, 8000):
                stage = ResampleAudioStage(resampled_audio_dir=out, write_to_disk=True, target_sample_rate=rate)
                stage.setup()
                stage.process(AudioTask(task_id="t", dataset_name="d", data={"audio_filepath": src}))

                written = os.path.join(out, os.listdir(out)[0])
                assert soundfile.info(written).samplerate == rate, "served audio at the previous run's rate"

    def test_segments_sharing_a_parent_id_each_get_their_own_file(self) -> None:
        waveform = torch.sin(torch.arange(0, 16000) * 0.01).unsqueeze(0)
        with tempfile.TemporaryDirectory() as tmpdir:
            stage = ResampleAudioStage(resampled_audio_dir=tmpdir, input_residency="waveform")
            stage.setup()
            for segment in range(3):
                stage.process(
                    AudioTask(
                        task_id="t",
                        dataset_name="d",
                        data={
                            # What VAD hands every child: the parent's id, identical across siblings.
                            "audio_item_id": "utt1",
                            "waveform": (waveform * (segment + 1)).clone(),
                            "sample_rate": 16000,
                        },
                    )
                )

            assert len(os.listdir(tmpdir)) == 3, "sibling segments collapsed onto one filename"

    def test_process_batch_accepts_every_advertised_residency(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(resample_audio_module.subprocess, "run", _fake_ffmpeg_copy)
        source = tmp_path / "source.wav"
        sf.write(source, np.zeros(16000, dtype=np.float32), 16000)
        waveform = torch.ones(1, 16000)
        cases = [
            ("file", {"audio_filepath": str(source)}),
            ("waveform", {"waveform": waveform, "sample_rate": 16000}),
            ("auto", {"audio_filepath": str(source)}),
            ("auto", {"waveform": waveform, "sample_rate": 16000}),
        ]

        for index, (residency, data) in enumerate(cases):
            stage = ResampleAudioStage(
                resampled_audio_dir=str(tmp_path / f"unused-{index}"),
                input_residency=residency,
                write_to_disk=False,
                keep_waveform_in_task=True,
            )
            result = stage.process_batch([AudioTask(dataset_name="d", data=dict(data))])
            assert len(result) == 1
            assert result[0].data["sample_rate"] == 16000

    def test_process_batch_rejects_incomplete_residencies(self, tmp_path: Path) -> None:
        waveform = torch.ones(1, 16)
        cases = [
            ("file", {}),
            ("file", {"audio_filepath": None}),
            ("waveform", {"waveform": waveform}),
            ("waveform", {"sample_rate": 16000}),
            ("waveform", {"waveform": None, "sample_rate": 16000}),
            ("auto", {}),
            ("auto", {"waveform": waveform}),
        ]

        for residency, data in cases:
            stage = ResampleAudioStage(
                resampled_audio_dir=str(tmp_path / "unused"),
                input_residency=residency,
                write_to_disk=False,
                keep_waveform_in_task=True,
            )
            with pytest.raises(ValueError, match="failed validation"):
                stage.process_batch([AudioTask(dataset_name="d", data=data)])

    @pytest.mark.parametrize("write_to_disk", [False, True])
    def test_output_cleanup_when_loading_converted_audio_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        write_to_disk: bool,
    ) -> None:
        monkeypatch.setattr(resample_audio_module.subprocess, "run", _fake_ffmpeg_copy)
        source = tmp_path / "source.wav"
        sf.write(source, np.zeros(16000, dtype=np.float32), 16000)
        attempted_outputs: list[str] = []

        def fail_load(path: str, *, mono: bool) -> tuple[torch.Tensor, int]:
            assert mono is False
            attempted_outputs.append(path)
            message = "cannot load converted output"
            raise OSError(message)

        monkeypatch.setattr(resample_audio_module, "load_audio_file", fail_load)
        stage = ResampleAudioStage(
            resampled_audio_dir=str(tmp_path / "out"),
            write_to_disk=write_to_disk,
            keep_waveform_in_task=True,
        )

        with pytest.raises(OSError, match="cannot load"):
            stage.process(
                AudioTask(
                    dataset_name="d",
                    data={"audio_filepath": str(source), "audio_item_id": "failure"},
                )
            )

        assert len(attempted_outputs) == 1
        assert os.path.exists(attempted_outputs[0]) is write_to_disk

    def test_disk_only_conversion_removes_stale_resident_audio(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(resample_audio_module.subprocess, "run", _fake_ffmpeg_convert)
        source = tmp_path / "source.wav"
        sf.write(source, np.zeros((8000, 2), dtype=np.float32), 8000)
        task = AudioTask(
            dataset_name="d",
            data={
                "audio_filepath": str(source),
                "waveform": torch.stack([torch.zeros(8000), torch.ones(8000)]),
                "sample_rate": 8000,
            },
        )
        stage = ResampleAudioStage(
            resampled_audio_dir=str(tmp_path / "out"),
            input_residency="waveform",
            target_sample_rate=16000,
            target_nchannels=1,
            write_to_disk=True,
            keep_waveform_in_task=False,
            update_audio_filepath=True,
        )

        result = stage.process(task)

        assert "waveform" not in result.data
        assert "sample_rate" not in result.data
        assert set(build_contract(stage).removes_keys) == {"waveform", "sample_rate"}
        consumed = resolve_audio(result.data, residency="auto", mono=False)
        assert consumed is not None
        converted, sample_rate = consumed
        assert sample_rate == 16000
        assert tuple(converted.shape[:1]) == (1,)

    def test_sink_contracts_and_static_gates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(resample_audio_module.subprocess, "run", _fake_ffmpeg_copy)
        source = tmp_path / "source.wav"
        sf.write(source, np.zeros(16000, dtype=np.float32), 16000)

        with pytest.raises(ValueError, match="keep_waveform_in_task or write_to_disk"):
            ResampleAudioStage(resampled_audio_dir=str(tmp_path), write_to_disk=False)
        with pytest.raises(ValueError, match="update_audio_filepath"):
            ResampleAudioStage(
                resampled_audio_dir=str(tmp_path),
                write_to_disk=False,
                keep_waveform_in_task=True,
                update_audio_filepath=True,
            )

        supported = [
            {"write_to_disk": True, "keep_waveform_in_task": False},
            {"write_to_disk": False, "keep_waveform_in_task": True},
            {"write_to_disk": True, "keep_waveform_in_task": True},
            {
                "write_to_disk": True,
                "keep_waveform_in_task": False,
                "update_audio_filepath": True,
            },
        ]
        for index, config in enumerate(supported):
            stage = ResampleAudioStage(resampled_audio_dir=str(tmp_path / f"out-{index}"), **config)
            assert_agent_ready(
                stage,
                lambda: AudioTask(dataset_name="d", data={"audio_filepath": str(source)}),
                available_keys={"audio_filepath"},
            )

        replacement = build_contract(
            ResampleAudioStage(
                resampled_audio_dir=str(tmp_path / "replacement"),
                update_audio_filepath=True,
            )
        )
        assert replacement.writes.data_keys.count("audio_filepath") == 1
        assert [write.writes.data_keys for write in replacement.conditional_writes] == [
            ["original_audio_filepath"]
        ]

        static = static_contract(ResampleAudioStage)
        configured = build_contract(ResampleAudioStage(resampled_audio_dir=str(tmp_path / "configured")))
        assert static.gates == configured.gates


def _fake_ffmpeg_copy(cmd: list[str], **_: Any) -> SimpleNamespace:  # noqa: ANN401
    """Stand in for the ffmpeg call by copying the source to the requested output."""
    src = cmd[cmd.index("-i") + 1]
    dst = cmd[-1]
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    return SimpleNamespace(returncode=0)


def _fake_ffmpeg_convert(cmd: list[str], **_: Any) -> SimpleNamespace:  # noqa: ANN401
    """Apply the requested rate/channel header changes without invoking FFmpeg."""
    src = cmd[cmd.index("-i") + 1]
    dst = cmd[-1]
    target_rate = int(cmd[cmd.index("-ar") + 1])
    target_channels = int(cmd[cmd.index("-ac") + 1])
    samples, source_rate = sf.read(src, always_2d=True)
    if target_channels == 1:
        samples = samples.mean(axis=1, keepdims=True)
    output_frames = round(len(samples) * target_rate / source_rate)
    old_positions = np.linspace(0.0, 1.0, len(samples), endpoint=False)
    new_positions = np.linspace(0.0, 1.0, output_frames, endpoint=False)
    converted = np.stack(
        [np.interp(new_positions, old_positions, samples[:, channel]) for channel in range(target_channels)],
        axis=1,
    )
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    sf.write(dst, converted, target_rate)
    return SimpleNamespace(returncode=0)


class TestSkippingExistingOutput:
    """Re-running is idempotent on disk, but an in-memory run must never wrongly skip.

    Lifted from tests/stages/audio/test_agent_simulation_pipelines.py: it drives only
    ResampleAudioStage, and counts ffmpeg invocations -- a property the naming tests above
    do not cover, since they count output FILES rather than conversions.
    """

    def test_disk_output_is_converted_once_but_memory_output_every_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nemo_curator.stages.audio.tagging import resample_audio as resample_module

        calls = {"n": 0}

        def counting_ffmpeg(cmd: list[str], **kwargs: Any) -> SimpleNamespace:  # noqa: ANN401
            calls["n"] += 1
            return _fake_ffmpeg_copy(cmd, **kwargs)

        monkeypatch.setattr(resample_module.subprocess, "run", counting_ffmpeg)
        source = tmp_path / "src.wav"
        sf.write(source, torch.linspace(-0.25, 0.25, 16000).numpy(), 16000)

        disk_stage = ResampleAudioStage(
            resampled_audio_dir=str(tmp_path / "out"),
            input_residency="file",
            write_to_disk=True,
            keep_waveform_in_task=False,
        )

        def disk_task() -> AudioTask:
            return AudioTask(dataset_name="t", data={"audio_filepath": str(source), "audio_item_id": "fixed_id"})

        disk_stage.process(disk_task())
        assert calls["n"] == 1
        disk_stage.process(disk_task())
        assert calls["n"] == 1, "the output already exists on disk, so it must be skipped"

        # write_to_disk=False writes to a fresh temp path each run, so it must always convert.
        calls["n"] = 0
        mem_stage = ResampleAudioStage(
            resampled_audio_dir=str(tmp_path / "unused"),
            input_residency="file",
            write_to_disk=False,
            keep_waveform_in_task=True,
        )
        for _ in range(2):
            mem_stage.process(AudioTask(dataset_name="t", data={"audio_filepath": str(source), "audio_item_id": "m"}))
        assert calls["n"] == 2, "an in-memory run has no durable output to skip"

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
