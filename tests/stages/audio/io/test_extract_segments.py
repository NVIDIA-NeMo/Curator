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

from nemo_curator.stages.audio.io.extract_segments import (
    SegmentExtractionStage,
    detect_combo,
    load_manifest,
    load_manifests,
)
from nemo_curator.tasks import AudioTask


class TestDetectCombo:
    def test_empty_returns_2(self) -> None:
        assert detect_combo([]) == 2

    def test_timestamps_only_returns_2(self) -> None:
        entries = [{"original_start_ms": 0, "original_end_ms": 1000}]
        assert detect_combo(entries) == 2

    def test_speaker_diar_returns_3(self) -> None:
        entries = [{"speaker_id": "speaker_0", "diar_segments": [[0, 1]]}]
        assert detect_combo(entries) == 3

    def test_speaker_only_returns_4(self) -> None:
        entries = [{"speaker_id": "speaker_0", "original_start_ms": 0}]
        assert detect_combo(entries) == 4


class TestLoadManifest:
    def test_loads_jsonl(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.jsonl"
        entries = [{"text": "hello"}, {"text": "world"}]
        manifest.write_text("\n".join(json.dumps(e) for e in entries))
        result = load_manifest(str(manifest))
        assert len(result) == 2
        assert result[0]["text"] == "hello"

    def test_skips_empty_lines(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text('{"text": "hello"}\n\n{"text": "world"}\n')
        result = load_manifest(str(manifest))
        assert len(result) == 2


class TestLoadManifests:
    def test_single_file(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text('{"text": "hello"}\n')
        result = load_manifests(str(manifest), str(tmp_path))
        assert len(result) == 1

    def test_directory(self, tmp_path: Path) -> None:
        for i in range(3):
            (tmp_path / f"manifest_{i}.jsonl").write_text(f'{{"text": "entry_{i}"}}\n')
        result = load_manifests(str(tmp_path), str(tmp_path))
        assert len(result) == 3

    def test_save_combined_flag(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "a.jsonl").write_text('{"text": "hello"}\n')
        output_dir = tmp_path / "output"

        load_manifests(str(input_dir), str(output_dir), save_combined=True)
        assert (output_dir / "manifest.jsonl").exists()

    def test_no_save_combined_by_default(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "a.jsonl").write_text('{"text": "hello"}\n')
        output_dir = tmp_path / "output"

        load_manifests(str(input_dir), str(output_dir))
        assert not (output_dir / "manifest.jsonl").exists()


class TestSegmentExtractionStage:
    def test_requires_output_dir(self) -> None:
        with pytest.raises(ValueError, match="output_dir"):
            SegmentExtractionStage(output_dir="")

    def test_invalid_format(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="output_format"):
            SegmentExtractionStage(output_dir=str(tmp_path), output_format="mp3")

    def test_process_raises_not_implemented(self, tmp_path: Path) -> None:
        stage = SegmentExtractionStage(output_dir=str(tmp_path))
        task = AudioTask(data={"original_file": "test.wav"})
        with pytest.raises(NotImplementedError):
            stage.process(task)

    def test_empty_batch(self, tmp_path: Path) -> None:
        stage = SegmentExtractionStage(output_dir=str(tmp_path))
        assert stage.process_batch([]) == []

    def test_extracts_segment(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(42)
        audio = rng.standard_normal(32000).astype(np.float32)
        audio_path = str(tmp_path / "source.wav")
        sf.write(audio_path, audio, 16000)

        output_dir = tmp_path / "output"
        stage = SegmentExtractionStage(output_dir=str(output_dir), output_format="wav")

        task = AudioTask(data={
            "original_file": audio_path,
            "original_start_ms": 0,
            "original_end_ms": 1000,
            "duration": 1.0,
        })
        results = stage.process_batch([task])
        assert len(results) == 1

        wav_files = list(output_dir.glob("*.wav"))
        assert len(wav_files) == 1

    def test_metadata_csv_written(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(42)
        audio = rng.standard_normal(32000).astype(np.float32)
        audio_path = str(tmp_path / "source.wav")
        sf.write(audio_path, audio, 16000)

        output_dir = tmp_path / "output"
        stage = SegmentExtractionStage(output_dir=str(output_dir))

        task = AudioTask(data={
            "original_file": audio_path,
            "original_start_ms": 0,
            "original_end_ms": 1000,
            "duration": 1.0,
        })
        stage.process_batch([task])
        assert (output_dir / "metadata.csv").exists()

    def test_inputs_outputs(self, tmp_path: Path) -> None:
        stage = SegmentExtractionStage(output_dir=str(tmp_path))
        _, optional = stage.inputs()
        assert "original_file" in optional
