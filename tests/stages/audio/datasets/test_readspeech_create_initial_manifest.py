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

from pathlib import Path
from unittest.mock import patch

from nemo_curator.stages.audio.datasets.readspeech.create_initial_manifest import (
    CreateInitialManifestReadSpeechStage,
)
from nemo_curator.tasks import AudioTask, _EmptyTask


def test_ray_stage_spec(tmp_path: Path) -> None:
    from nemo_curator.backends.utils import RayStageSpecKeys

    stage = CreateInitialManifestReadSpeechStage(raw_data_dir=str(tmp_path), auto_download=False)
    spec = stage.ray_stage_spec()
    assert spec[RayStageSpecKeys.IS_FANOUT_STAGE] is True


def test_inputs_outputs(tmp_path: Path) -> None:
    stage = CreateInitialManifestReadSpeechStage(raw_data_dir=str(tmp_path), auto_download=False)
    assert stage.inputs() == ([], [])
    assert stage.outputs() == ([], ["audio_filepath", "text"])


def test_parse_filename_standard(tmp_path: Path) -> None:
    stage = CreateInitialManifestReadSpeechStage(raw_data_dir=str(tmp_path), auto_download=False)
    result = stage.parse_filename("book_00025_chp_0019_reader_04069_0_seg_1_seg1.wav")
    assert result["book_id"] == "00025"
    assert result["chapter"] == "0019"
    assert result["reader_id"] == "04069"


def test_parse_filename_short(tmp_path: Path) -> None:
    stage = CreateInitialManifestReadSpeechStage(raw_data_dir=str(tmp_path), auto_download=False)
    result = stage.parse_filename("short.wav")
    assert result["book_id"] == ""
    assert result["reader_id"] == ""


def test_collect_audio_files(tmp_path: Path) -> None:
    wav_dir = tmp_path / "read_speech"
    wav_dir.mkdir()
    (wav_dir / "book_00000_chp_0001_reader_00100_0_seg_1_seg1.wav").write_bytes(b"\x00")
    (wav_dir / "book_00000_chp_0001_reader_00100_0_seg_2_seg1.wav").write_bytes(b"\x00")
    (wav_dir / "notes.txt").write_text("not audio")

    stage = CreateInitialManifestReadSpeechStage(raw_data_dir=str(tmp_path), auto_download=False)
    entries = stage.collect_audio_files(str(wav_dir))
    assert len(entries) == 2
    assert all(e["sample_rate"] == 48000 for e in entries)
    assert entries[0]["book_id"] == "00000"


def test_select_samples_limits(tmp_path: Path) -> None:
    stage = CreateInitialManifestReadSpeechStage(raw_data_dir=str(tmp_path), max_samples=3, auto_download=False)
    entries = [{"file": f"f{i}.wav"} for i in range(10)]
    assert len(stage.select_samples(entries)) == 3

    stage.max_samples = -1
    assert len(stage.select_samples(entries)) == 10


def test_process_end_to_end(tmp_path: Path) -> None:
    wav_dir = tmp_path / "dns_data" / "read_speech"
    wav_dir.mkdir(parents=True)
    (wav_dir / "book_00000_chp_0001_reader_00100_0_seg_1_seg1.wav").write_bytes(b"\x00")
    (wav_dir / "book_00001_chp_0002_reader_00200_0_seg_1_seg1.wav").write_bytes(b"\x00")

    stage = CreateInitialManifestReadSpeechStage(
        raw_data_dir=str(tmp_path / "dns_data"), max_samples=-1, auto_download=False,
    )
    results = stage.process(_EmptyTask(task_id="empty", dataset_name="test", data=None))
    assert len(results) == 2
    assert all(isinstance(r, AudioTask) for r in results)
    assert results[0].task_id == "readspeech_0"
    assert results[0].dataset_name == "DNS-ReadSpeech"


def test_process_empty_dir(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    stage = CreateInitialManifestReadSpeechStage(raw_data_dir=str(empty_dir), auto_download=False)
    results = stage.process(_EmptyTask(task_id="e", dataset_name="t", data=None))
    assert results == []


def test_auto_download_calls_download(tmp_path: Path) -> None:
    stage = CreateInitialManifestReadSpeechStage(raw_data_dir=str(tmp_path / "dns_data"), auto_download=True)
    wav_dir = tmp_path / "dns_data" / "read_speech"
    wav_dir.mkdir(parents=True)
    (wav_dir / "book_00000_chp_0001_reader_00100_0_seg_1_seg1.wav").write_bytes(b"\x00")

    with patch.object(stage, "download_and_extract", return_value=str(wav_dir)) as mock_dl:
        results = stage.process(_EmptyTask(task_id="e", dataset_name="t", data=None))
        mock_dl.assert_called_once()
        assert len(results) == 1
