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
from datasets import Dataset

from nemo_curator.stages.audio.asr.datasets import HuggingFaceASRDatasetHandler
from nemo_curator.tasks import _EmptyTask
from tests.stages.audio.asr.datasets.conftest import INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE


def _write_audio(path: Path) -> None:
    samples = np.linspace(0.0, 1.0, INPUT_SAMPLE_RATE // 20, endpoint=False, dtype=np.float32)
    tone = 0.1 * np.sin(2 * np.pi * 220 * samples)
    stereo = np.stack([tone, tone * 0.5], axis=1)
    sf.write(path, stereo, INPUT_SAMPLE_RATE, subtype="PCM_16")


def _save_hf_split(raw_root: Path, lang: str, split: str, rows: dict[str, list[object]]) -> Path:
    split_dir = raw_root / lang / split
    dataset = Dataset.from_dict(rows)
    dataset.save_to_disk(str(split_dir))
    return split_dir


def test_huggingface_handler_ingests_saved_hf_dataset_and_writes_manifest(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    audio_dir = tmp_path / "source_audio"
    audio_dir.mkdir(parents=True)
    audio_paths = []
    for i in range(3):
        audio_path = audio_dir / f"sample_{i}.wav"
        _write_audio(audio_path)
        audio_paths.append(str(audio_path))
    _save_hf_split(
        raw_root,
        "gu",
        "train",
        {
            "audio_filepath": audio_paths,
            "text": ["ગુજરાતી વાક્ય"] * len(audio_paths),
            "speaker_id": [f"speaker_{i}" for i in range(len(audio_paths))],
            "fname": [f"utt_{i}.wav" for i in range(len(audio_paths))],
        },
    )
    output_dir = tmp_path / "out"
    stage = HuggingFaceASRDatasetHandler(
        raw_data_dir=str(raw_root),
        output_dir=str(output_dir),
        langs=["gu"],
        source_name="UnitHF",
        native_splits=["train"],
        split_dir_pattern="{lang}/{split}",
        extra_keys=["speaker_id", "fname"],
        extraction_workers=2,
        write_manifest=True,
        manifest_splits=["train"],
    )
    stage.setup_on_node()
    stage.setup()

    tasks = stage.process(_EmptyTask(dataset_name="test", data=None))
    metrics = stage._consume_custom_metrics()
    stage.teardown()

    assert len(tasks) == 3
    assert metrics["input_rows"] == 3
    assert metrics["emitted_tasks"] == 3
    for i, task in enumerate(tasks):
        assert task.data["audio_filepath"] == str(output_dir / "gu" / "train" / "audio" / f"utt_{i}.wav")
        audio_info = sf.info(task.data["audio_filepath"])
        assert audio_info.samplerate == OUTPUT_SAMPLE_RATE
        assert audio_info.channels == 1
        assert audio_info.subtype == "PCM_16"
        assert task.data["text"] == "ગુજરાતી વાક્ય"
        assert task.data["duration"] == pytest.approx(audio_info.frames / OUTPUT_SAMPLE_RATE)
        assert task.data["lang"] == "gu"
        assert task.data["split_type"] == "train"
        assert task.data["source"] == "UnitHF"
        assert task.data["orig_sample_rate"] == INPUT_SAMPLE_RATE
        assert task.data["orig_num_channels"] == 2
        assert task.data["speaker_id"] == f"speaker_{i}"
        assert task.data["fname"] == f"utt_{i}.wav"

    with (output_dir / "gu" / "train.jsonl").open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert rows == [task.data for task in tasks]


def test_huggingface_handler_reports_skipped_rows_and_maps_splits(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    audio_dir = tmp_path / "source_audio"
    valid_audio = audio_dir / "valid.wav"
    missing_text_audio = audio_dir / "missing_text.wav"
    audio_dir.mkdir(parents=True)
    _write_audio(valid_audio)
    _write_audio(missing_text_audio)
    _save_hf_split(
        raw_root,
        "gu",
        "valid",
        {
            "audio_filepath": [
                str(valid_audio),
                str(missing_text_audio),
                None,
                str(audio_dir / "does_not_exist.wav"),
            ],
            "text": ["valid text", None, "missing audio", "bad audio"],
        },
    )
    stage = HuggingFaceASRDatasetHandler(
        raw_data_dir=str(raw_root),
        output_dir=str(tmp_path / "out"),
        langs=["gu"],
        source_name="UnitHF",
        native_splits=["valid"],
        split_mapping={"valid": "dev"},
        extraction_workers=1,
    )
    stage.setup()

    tasks = stage.process(_EmptyTask(dataset_name="test", data=None))
    metrics = stage._consume_custom_metrics()

    assert len(tasks) == 1
    assert tasks[0].data["split_type"] == "dev"
    assert metrics["input_rows"] == 4
    assert metrics["emitted_tasks"] == 1
    assert metrics["skipped_missing_text"] == 1
    assert metrics["skipped_missing_audio"] == 1
    assert metrics["skipped_audio_load"] == 1


def test_huggingface_handler_accepts_dataset_source_names(tmp_path: Path) -> None:
    kwargs = {
        "raw_data_dir": str(tmp_path / "raw"),
        "output_dir": str(tmp_path / "out"),
        "langs": ["gu"],
        "native_splits": ["train", "valid"],
        "split_dir_pattern": "{lang}/{split}",
    }

    kathbath = HuggingFaceASRDatasetHandler(source_name="Kathbath", **kwargs)
    shrutilipi = HuggingFaceASRDatasetHandler(source_name="Shrutilipi", **kwargs)

    assert kathbath.source_name == "Kathbath"
    assert kathbath.native_splits == ["train", "valid"]
    assert kathbath.split_dir_pattern == "{lang}/{split}"
    assert shrutilipi.source_name == "Shrutilipi"
    assert shrutilipi.native_splits == ["train", "valid"]
    assert shrutilipi.split_dir_pattern == "{lang}/{split}"


def test_huggingface_handler_uses_known_source_field_mappings(tmp_path: Path) -> None:
    kwargs = {
        "raw_data_dir": str(tmp_path / "raw"),
        "output_dir": str(tmp_path / "out"),
        "langs": ["gu"],
    }

    kathbath = HuggingFaceASRDatasetHandler(source_name="Kathbath", **kwargs)
    shrutilipi = HuggingFaceASRDatasetHandler(source_name="Shrutilipi", **kwargs)

    assert kathbath._source_field_mapping() == {
        "fname": "fname",
        "gender": "gender",
        "speaker_id": "speaker_id",
    }
    assert shrutilipi._source_field_mapping() == {}
