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

"""Agent-style pipeline tests that move real WAV data through audio stages.

The broader agent-simulation suite mocks heavy model boundaries. These tests use
small deterministic WAV files on disk so we also verify real audio file I/O,
waveform residency, JSONL manifest output, and timestamp handoffs.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
pytest.importorskip("torch")

from nemo_curator.stages.audio.common import (  # noqa: E402
    GetAudioDurationStage,
    ManifestReaderStage,
    ManifestWriterStage,
    PreserveByValueStage,
    load_audio_file,
)
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage  # noqa: E402
from nemo_curator.stages.audio.postprocessing.timestamp_mapper import TimestampMapperStage  # noqa: E402
from nemo_curator.stages.audio.preprocessing.concatenation import SegmentConcatenationStage  # noqa: E402
from nemo_curator.stages.audio.preprocessing.mono_conversion import MonoConversionStage  # noqa: E402
from nemo_curator.stages.audio.tagging.resample_audio import ResampleAudioStage  # noqa: E402
from nemo_curator.tasks import AudioTask, FileGroupTask  # noqa: E402


def _write_stereo_wav(path: Path, *, sample_rate: int = 16000, duration_sec: float = 0.5) -> Path:
    samples = int(sample_rate * duration_sec)
    t = np.arange(samples, dtype=np.float32) / sample_rate
    left = 0.35 * np.sin(2 * math.pi * 440 * t)
    right = 0.20 * np.sin(2 * math.pi * 660 * t)
    audio = np.stack([left, right], axis=1).astype(np.float32)
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")
    return path


def _write_manifest(path: Path, row: dict) -> Path:
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


def test_actual_audio_manifest_mono_writer_document_pipeline_custom_keys(tmp_path: Path) -> None:
    audio_path = _write_stereo_wav(
        tmp_path / "input_stereo.wav",
        sample_rate=16000,
        duration_sec=0.5,
    )
    manifest_path = _write_manifest(
        tmp_path / "input.jsonl",
        {
            "agent_audio_path": str(audio_path),
            "language": "en",
            "source_id": "real-audio-001",
        },
    )

    reader = ManifestReaderStage()
    tasks = reader.process(FileGroupTask(dataset_name="actual_audio", data=[str(manifest_path)]))
    assert len(tasks) == 1
    task = tasks[0]
    assert task.data["agent_audio_path"] == str(audio_path)

    duration = GetAudioDurationStage(
        audio_filepath_key="agent_audio_path",
        duration_key="agent_duration",
    )
    task = duration.process(task)
    assert isinstance(task, AudioTask)
    assert task.data["agent_duration"] == pytest.approx(0.5, abs=0.02)

    keep_long_enough = PreserveByValueStage(
        input_value_key="agent_duration",
        target_value=0,
        operator="ge",
    )
    assert keep_long_enough.process_batch([task]) == [task]

    drop_too_short = PreserveByValueStage(
        input_value_key="agent_duration",
        target_value=1,
        operator="ge",
    )
    assert drop_too_short.process_batch([task]) == []

    mono = MonoConversionStage(
        output_sample_rate=16000,
        audio_filepath_key="agent_audio_path",
        waveform_key="agent_waveform",
        sample_rate_key="agent_sr",
        is_mono_key="agent_is_mono",
        duration_key="agent_mono_duration",
        num_samples_key="agent_num_samples",
        output_audio_filepath_key="agent_mono_path",
        original_audio_filepath_key="agent_original_path",
        strict_sample_rate=True,
        keep_waveform_in_task=False,
        write_to_disk=True,
        update_audio_filepath=True,
        output_dir=str(tmp_path / "mono"),
    )
    task = mono.process(task)
    assert isinstance(task, AudioTask)
    assert "agent_waveform" not in task.data
    assert task.data["agent_original_path"] == str(audio_path)
    assert task.data["agent_audio_path"] == task.data["agent_mono_path"]
    assert task.data["agent_is_mono"] is True
    assert task.data["agent_num_samples"] == 8000

    mono_info = sf.info(task.data["agent_mono_path"])
    assert mono_info.channels == 1
    assert mono_info.samplerate == 16000

    writer_path = tmp_path / "output.jsonl"
    writer = ManifestWriterStage(output_path=str(writer_path))
    writer.setup_on_node()
    writer.setup()
    written_task = writer.process(task)
    assert isinstance(written_task, AudioTask)

    written_rows = [json.loads(line) for line in writer_path.read_text(encoding="utf-8").splitlines()]
    assert len(written_rows) == 1
    assert written_rows[0]["agent_audio_path"] == task.data["agent_mono_path"]
    assert written_rows[0]["agent_original_path"] == str(audio_path)

    docs = AudioToDocumentStage(
        keep_keys=["agent_audio_path", "agent_original_path", "agent_duration", "language", "source_id"]
    ).process_batch([task])
    assert len(docs) == 1
    df = docs[0].data
    assert list(df["source_id"]) == ["real-audio-001"]
    assert "agent_waveform" not in df.columns


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ResampleAudioStage requires ffmpeg")
def test_actual_audio_waveform_residency_resample_and_document_sanitization(tmp_path: Path) -> None:
    audio_path = _write_stereo_wav(
        tmp_path / "input_48k.wav",
        sample_rate=48000,
        duration_sec=0.25,
    )
    task = AudioTask(
        dataset_name="actual_audio",
        data={"agent_audio_path": str(audio_path), "source_id": "waveform-residency"},
    )

    mono = MonoConversionStage(
        output_sample_rate=48000,
        audio_filepath_key="agent_audio_path",
        waveform_key="agent_waveform",
        sample_rate_key="agent_sr",
        duration_key="agent_duration",
        num_samples_key="agent_num_samples",
        is_mono_key="agent_is_mono",
        strict_sample_rate=True,
        keep_waveform_in_task=True,
        write_to_disk=False,
    )
    task = mono.process(task)
    assert isinstance(task, AudioTask)
    assert task.data["agent_waveform"].shape == (1, 12000)
    assert task.data["agent_sr"] == 48000

    resample = ResampleAudioStage(
        resampled_audio_dir=str(tmp_path / "unused-resampled-dir"),
        target_sample_rate=16000,
        target_nchannels=1,
        audio_filepath_key="agent_audio_path",
        resampled_audio_filepath_key="agent_resampled_path",
        waveform_key="agent_waveform",
        sample_rate_key="agent_sr",
        duration_key="agent_resampled_duration",
        audio_item_id_key="agent_audio_id",
        input_residency="waveform",
        keep_waveform_in_task=True,
        write_to_disk=False,
        update_audio_filepath=False,
    )
    resample.setup_on_node()
    task = resample.process(task)

    assert "agent_resampled_path" not in task.data
    assert task.data["agent_audio_path"] == str(audio_path)
    assert task.data["agent_sr"] == 16000
    assert task.data["agent_waveform"].shape[0] == 1
    assert task.data["agent_resampled_duration"] == pytest.approx(0.25, abs=0.03)

    docs = AudioToDocumentStage().process_batch([task])
    df = docs[0].data
    assert "agent_waveform" not in df.columns
    assert list(df["source_id"]) == ["waveform-residency"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ResampleAudioStage requires ffmpeg")
def test_actual_audio_disk_resample_updates_canonical_path_pipeline(tmp_path: Path) -> None:
    audio_path = _write_stereo_wav(
        tmp_path / "input_disk_resample.wav",
        sample_rate=48000,
        duration_sec=0.3,
    )
    task = AudioTask(
        dataset_name="actual_audio",
        data={"agent_audio_path": str(audio_path), "source_id": "disk-resample"},
    )

    resample = ResampleAudioStage(
        resampled_audio_dir=str(tmp_path / "resampled"),
        target_sample_rate=16000,
        target_nchannels=1,
        audio_filepath_key="agent_audio_path",
        resampled_audio_filepath_key="agent_resampled_path",
        duration_key="agent_resampled_duration",
        audio_item_id_key="agent_audio_id",
        original_audio_filepath_key="agent_original_path",
        write_to_disk=True,
        keep_waveform_in_task=False,
        update_audio_filepath=True,
    )
    resample.setup_on_node()
    task = resample.process(task)

    assert task.data["agent_original_path"] == str(audio_path)
    assert task.data["agent_audio_path"] == task.data["agent_resampled_path"]
    assert Path(task.data["agent_resampled_path"]).exists()
    assert task.data["agent_resampled_duration"] == pytest.approx(0.3, abs=0.03)

    resampled_info = sf.info(task.data["agent_resampled_path"])
    assert resampled_info.channels == 1
    assert resampled_info.samplerate == 16000

    duration = GetAudioDurationStage(
        audio_filepath_key="agent_audio_path",
        duration_key="agent_checked_duration",
    )
    task = duration.process(task)
    assert isinstance(task, AudioTask)
    assert task.data["agent_checked_duration"] == pytest.approx(0.3, abs=0.03)

    writer_path = tmp_path / "disk_resample_output.jsonl"
    writer = ManifestWriterStage(output_path=str(writer_path))
    writer.setup_on_node()
    writer.setup()
    writer.process(task)
    written = json.loads(writer_path.read_text(encoding="utf-8").strip())
    assert written["agent_audio_path"] == task.data["agent_resampled_path"]
    assert written["agent_original_path"] == str(audio_path)

    docs = AudioToDocumentStage(
        keep_keys=["agent_audio_path", "agent_original_path", "agent_checked_duration", "source_id"]
    ).process_batch([task])
    assert list(docs[0].data["source_id"]) == ["disk-resample"]


def test_actual_audio_segment_concat_then_timestamp_mapper_preserves_handoffs(tmp_path: Path) -> None:
    audio_path = _write_stereo_wav(
        tmp_path / "segments_source.wav",
        sample_rate=16000,
        duration_sec=0.4,
    )
    waveform, sample_rate = load_audio_file(str(audio_path), mono=True)
    samples_per_segment = int(0.1 * sample_rate)

    parent = AudioTask(
        dataset_name="actual_audio",
        data={
            "agent_segments": [
                {
                    "waveform": waveform[:, :samples_per_segment].clone(),
                    "sample_rate": sample_rate,
                    "original_file": str(audio_path),
                    "start_ms": 0,
                    "end_ms": 100,
                    "segment_num": 0,
                },
                {
                    "waveform": waveform[:, 2 * samples_per_segment : 3 * samples_per_segment].clone(),
                    "sample_rate": sample_rate,
                    "original_file": str(audio_path),
                    "start_ms": 200,
                    "end_ms": 300,
                    "segment_num": 1,
                },
            ],
            "source_id": "segment-map",
        },
        _metadata={"pipeline": "actual-audio"},
        _stage_perf=[{"stage": "upstream", "process_time": 0.1}],
    )

    concat = SegmentConcatenationStage(
        silence_duration_sec=0.05,
        segments_key="agent_segments",
        waveform_key="waveform",
        sample_rate_key="sample_rate",
    )
    combined = concat.process(parent)
    assert isinstance(combined, AudioTask)
    assert combined.data["num_segments"] == 2
    assert combined.data["sample_rate"] == sample_rate
    assert combined.data["waveform"].shape == (1, int(0.25 * sample_rate))
    assert combined._metadata["pipeline"] == "actual-audio"
    assert len(combined._metadata["segment_mappings"]) == 2
    assert combined._stage_perf == parent._stage_perf

    combined.data.update({"start_ms": 0, "end_ms": 100, "source_id": "segment-map"})
    mapper = TimestampMapperStage(passthrough_keys=["source_id"])
    mapped = mapper.process(combined)
    assert isinstance(mapped, AudioTask)
    assert mapped.data == {
        "original_file": str(audio_path),
        "original_start_ms": 0,
        "original_end_ms": 100,
        "duration_ms": 100,
        "duration": 0.1,
        "source_id": "segment-map",
    }
