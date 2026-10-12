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

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from nemo_curator.stages.audio.tts.audio_io import (
    MISSING_ENHANCED_AUDIO,
    SegmentAudioError,
    TarAudioConfig,
    load_segment_audio,
)
from nemo_curator.stages.audio.tts.bandwidth import BandwidthAnnotationStage
from nemo_curator.stages.audio.tts.enhance_reuse import ENHANCE_FAILED, ResembleEnhanceStage
from nemo_curator.tasks import AudioTask
from tests.stages.audio.tts.test_audio_io import make_tar_segments


class FakeWorker:
    """Stands in for the resemble venv: writes a quieter 44.1 kHz copy; fails inputs listed in ``fail``."""

    def __init__(self, fail: int = -1) -> None:
        self.fail = fail
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str], env: dict, timeout: float) -> str:
        self.calls.append(cmd)
        manifest = Path(cmd[cmd.index("--manifest") + 1])
        lines = []
        for i, line in enumerate(manifest.read_text().splitlines()):
            item = json.loads(line)
            if i == self.fail:
                lines.append(json.dumps({"in": item["in"], "out": item["out"], "ok": False, "error": "boom"}))
                continue
            audio, sr = sf.read(item["in"], dtype="float32")
            n = round(len(audio) * 44100 / sr)
            out = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32) * 0.5
            Path(item["out"]).parent.mkdir(parents=True, exist_ok=True)
            sf.write(item["out"], out, 44100)
            lines.append(json.dumps({**item, "ok": True, "sampling_rate": 44100, "num_samples": n, "error": None}))
        return "\n".join(lines)


def test_enhances_segments_and_records_annotation(tmp_path: Path) -> None:
    segments = make_tar_segments(tmp_path, [("a.wav", 1.0, 48000), ("b.wav", 0.5, 16000), ("c.wav", 0.5, 16000)])
    segments[2]["GranaryHifi"]["AudioMetadata"]["tar_member_offset"] = None
    worker = FakeWorker(fail=1)
    stage = ResembleEnhanceStage(output_dir=str(tmp_path / "enh"), command_runner=worker, nfe=16)
    task = stage.process(AudioTask(data={"orig_sample_id": "vid", "segments": segments}))
    first, second, third = (s["enhanced_audio"] for s in task.data["segments"])
    assert first["error"] is None
    assert first["method"] == "resemble_enhance"
    assert first["sampling_rate"] == 44100
    assert first["duration"] == pytest.approx(1.0, abs=1e-3)
    assert Path(first["audio_filepath"]) == tmp_path / "enh" / "vid" / "a.wav"
    assert second["error"] == ENHANCE_FAILED
    assert second["error_detail"] == "boom"
    assert third["error"] == "missing_tar_offsets"
    assert "--nfe" in worker.calls[0]
    assert worker.calls[0][worker.calls[0].index("--nfe") + 1] == "16"

    # idempotent: a second run reuses the existing output without calling the worker for it
    again = FakeWorker()
    ResembleEnhanceStage(output_dir=str(tmp_path / "enh"), command_runner=again).process(
        AudioTask(data={"orig_sample_id": "vid", "segments": segments[:1]})
    )
    assert again.calls == []


def test_audio_source_switch_reads_enhanced_copy(tmp_path: Path) -> None:
    segments = make_tar_segments(tmp_path, [("a.wav", 1.0, 48000), ("b.wav", 0.5, 16000)])
    ResembleEnhanceStage(output_dir=str(tmp_path / "enh"), command_runner=FakeWorker(fail=1)).process(
        AudioTask(data={"orig_sample_id": "vid", "segments": segments})
    )
    original, sr0 = load_segment_audio(segments[0])
    enhanced, sr1 = load_segment_audio(segments[0], None, TarAudioConfig(audio_source="enhanced"))
    assert (sr0, sr1) == (48000, 44100)
    assert np.abs(enhanced).max() < np.abs(original).max()

    with pytest.raises(SegmentAudioError) as exc_info:
        load_segment_audio(segments[1], None, TarAudioConfig(audio_source="enhanced"))
    assert exc_info.value.code == MISSING_ENHANCED_AUDIO
    _, sr_fallback = load_segment_audio(
        segments[1], None, TarAudioConfig(audio_source="enhanced", enhanced_fallback=True)
    )
    assert sr_fallback == 16000

    with pytest.raises(ValueError, match="audio_source"):
        TarAudioConfig(audio_source="studio")


def test_downstream_stage_follows_audio_source(tmp_path: Path) -> None:
    segments = make_tar_segments(tmp_path, [("a.wav", 1.0, 48000)])
    ResembleEnhanceStage(output_dir=str(tmp_path / "enh"), command_runner=FakeWorker()).process(
        AudioTask(data={"orig_sample_id": "vid", "segments": segments})
    )
    task = AudioTask(data={"orig_sample_id": "vid", "segments": segments})
    BandwidthAnnotationStage(output_key="bw_orig").process(task)
    BandwidthAnnotationStage(output_key="bw_enh", audio_source="enhanced").process(task)
    segment = task.data["segments"][0]
    assert segment["bw_orig"]["bandwidth_sample_rate"] == 48000
    assert segment["bw_enh"]["bandwidth_sample_rate"] == 44100
