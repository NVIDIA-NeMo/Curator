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

"""Real-extraction tests for ``SnippetExtractionStage`` (CPU only).

Generates a short synthesized sine WAV (mono and stereo variants),
feeds it through ``SnippetExtractionStage`` with a hand-rolled snippet
plan, and checks the resulting audio files match expected sample
rate, channel count, and duration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import soundfile as sf

from nemo_curator.stages.audio.alm.pretrain import SnippetExtractionStage
from nemo_curator.stages.audio.alm.pretrain.stages import _PLAN_DATA_KEY
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from pathlib import Path


def _make_wav(path: Path, duration_sec: float, sample_rate: int, channels: int = 1) -> None:
    n = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n, endpoint=False, dtype=np.float32)
    mono = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    data = mono if channels == 1 else np.stack([mono] * channels, axis=-1)
    sf.write(str(path), data, sample_rate, subtype="PCM_16")


def _seg(start: float, end: float) -> dict:
    return {"speaker": "A", "start": start, "end": end, "text": "x", "text_ITN": "x", "words": []}


def _task_with_plan(audio_path: Path, plan: list[dict], extras: dict | None = None) -> AudioTask:
    data = {"id": "X", "audio_filepath": str(audio_path), _PLAN_DATA_KEY: plan}
    if extras:
        data.update(extras)
    return AudioTask(task_id="t1", dataset_name="ds", data=data)


# ----------------------------------------------------------------------
# Real extraction
# ----------------------------------------------------------------------


class TestSnippetExtractionStageReal:
    def test_writes_one_file_per_planned_snippet(self, tmp_path: Path) -> None:
        src = tmp_path / "src.wav"
        _make_wav(src, duration_sec=10.0, sample_rate=16000)

        plan = [
            {"start": 0.0, "end": 3.0, "segments": [_seg(0.0, 3.0)]},
            {"start": 4.0, "end": 8.0, "segments": [_seg(4.0, 8.0)]},
        ]
        out_dir = tmp_path / "snips"
        stage = SnippetExtractionStage(output_dir=str(out_dir), target_sample_rate=16000, output_format="wav")
        stage.__post_init__()
        stage.setup_on_node()
        stage.setup()

        out = stage.process(_task_with_plan(src, plan))
        assert len(out) == 2
        files = sorted(out_dir.glob("*.wav"))
        assert len(files) == 2

        # File 1: 3.0s @ 16k = 48000 frames
        info1 = sf.info(str(files[0]))
        assert info1.samplerate == 16000
        assert info1.channels == 1
        assert info1.frames == pytest.approx(48000, abs=2)

    def test_resamples_when_source_rate_differs(self, tmp_path: Path) -> None:
        src = tmp_path / "src22k.wav"
        _make_wav(src, duration_sec=4.0, sample_rate=22050)

        plan = [{"start": 0.0, "end": 2.0, "segments": [_seg(0.0, 2.0)]}]
        stage = SnippetExtractionStage(
            output_dir=str(tmp_path / "snips"), target_sample_rate=16000, output_format="wav"
        )
        stage.__post_init__()
        stage.setup_on_node()
        stage.setup()

        out = stage.process(_task_with_plan(src, plan))
        assert len(out) == 1
        info = sf.info(out[0].data["audio_filepath"])
        assert info.samplerate == 16000
        assert info.channels == 1
        # 2s @ 16k = 32000 frames (allow tiny resample rounding)
        assert info.frames == pytest.approx(32000, abs=4)

    def test_channel_averages_to_mono_for_stereo_source(self, tmp_path: Path) -> None:
        src = tmp_path / "stereo.wav"
        _make_wav(src, duration_sec=2.0, sample_rate=16000, channels=2)

        plan = [{"start": 0.0, "end": 1.0, "segments": [_seg(0.0, 1.0)]}]
        stage = SnippetExtractionStage(
            output_dir=str(tmp_path / "snips"), target_sample_rate=16000, output_format="wav"
        )
        stage.__post_init__()
        stage.setup_on_node()
        stage.setup()

        out = stage.process(_task_with_plan(src, plan))
        info = sf.info(out[0].data["audio_filepath"])
        assert info.channels == 1

    def test_emitted_metadata_matches_extracted_file(self, tmp_path: Path) -> None:
        src = tmp_path / "src.wav"
        _make_wav(src, duration_sec=5.0, sample_rate=16000)

        plan = [{"start": 1.0, "end": 4.0, "segments": [_seg(1.0, 4.0)]}]
        stage = SnippetExtractionStage(
            output_dir=str(tmp_path / "snips"), target_sample_rate=16000, output_format="flac"
        )
        stage.__post_init__()
        stage.setup_on_node()
        stage.setup()

        # Source row has audio_sample_rate / audio_num_channels populated, so
        # the extractor's conditional updates fire.
        out = stage.process(
            _task_with_plan(
                src, plan, extras={"audio_sample_rate": 22050, "audio_num_channels": 2}
            )
        )
        d = out[0].data
        # Snippet ID + path follow the spec
        assert d["snippet_id"] == "X_1.000_4.000"
        assert d["audio_filepath"].endswith("X_1.000_4.000.flac")
        # Metadata reflects post-cut audio (overwritten because the source had these keys)
        assert d["audio_sample_rate"] == 16000
        assert d["audio_num_channels"] == 1
        # Duration approximately matches plan (within one frame at target sr)
        assert d["duration"] == pytest.approx(3.0, abs=1.0 / 16000)

    def test_missing_source_emits_stub(self, tmp_path: Path) -> None:
        plan = [{"start": 0.0, "end": 1.0, "segments": [_seg(0.0, 1.0)]}]
        task = _task_with_plan(tmp_path / "does_not_exist.wav", plan)
        stage = SnippetExtractionStage(
            output_dir=str(tmp_path / "snips"), target_sample_rate=16000, output_format="wav"
        )
        stage.__post_init__()
        stage.setup_on_node()
        stage.setup()
        out = stage.process(task)
        assert len(out) == 1
        assert out[0].data["snippet_id"] is None
