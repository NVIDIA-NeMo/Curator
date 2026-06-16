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
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from datasets import Dataset

from nemo_curator.stages.audio.asr.datasets import huggingface as huggingface_module
from nemo_curator.stages.audio.asr.datasets.huggingface import HuggingFaceASRDatasetHandler
from nemo_curator.tasks import _EmptyTask
from tests.stages.audio.asr.datasets.conftest import INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE


def _indicvoices_stage(**kwargs: object) -> HuggingFaceASRDatasetHandler:
    return HuggingFaceASRDatasetHandler(
        source_name="IndicVoices",
        valid_split_strategy="dev_test",
        **kwargs,
    )


def test_indicvoices_handler_ingests_realistic_hf_dataset(
    tmp_path: Path,
    indicvoices_raw_dataset: tuple[Path, int],
) -> None:
    raw_root, total_rows = indicvoices_raw_dataset
    output_dir = tmp_path / "out"
    stage = _indicvoices_stage(
        raw_data_dir=str(raw_root),
        output_dir=str(output_dir),
        langs=["gu"],
        native_splits=["valid"],
        split_dir_pattern="{split}",
        extraction_workers=2,
    )
    stage.setup()
    tasks = stage.process(_EmptyTask(dataset_name="test", data=None))

    assert len(tasks) == total_rows
    split_helper = _indicvoices_stage(raw_data_dir=str(raw_root), output_dir=str(output_dir), langs=["gu"])
    expected_counts = Counter(split_helper.assign_split("valid", f"gu_valid_{i}") for i in range(total_rows))
    actual_counts = Counter(task.data["split_type"] for task in tasks)
    assert actual_counts == expected_counts
    assert actual_counts["dev"] + actual_counts["test"] == total_rows
    assert 0.55 <= actual_counts["dev"] / total_rows <= 0.65

    for task in tasks:
        audio_path = Path(task.data["audio_filepath"])
        assert audio_path.parent == output_dir / "gu" / task.data["split_type"] / "audio"
        wav_info = sf.info(audio_path)
        assert wav_info.samplerate == OUTPUT_SAMPLE_RATE
        assert wav_info.channels == 1
        assert wav_info.subtype == "PCM_16"
        assert task.data["orig_sample_rate"] == INPUT_SAMPLE_RATE
        assert task.data["orig_num_channels"] == 2
        assert task.data["lang"] == "gu"
        assert task.data["source"] == "IndicVoices"
        assert task.data["speaker_id"].startswith("speaker_")
        assert task.data["gender"] in {"Female", "Male"}
        assert task.data["age"] == "30-45"
        assert "age_group" not in task.data
        assert task.data["scenario"] == "Extempore"


def test_indicvoices_handler_reports_skipped_rows_from_realistic_hf_dataset(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    audio_dir = tmp_path / "source_audio"
    valid_dir = raw_root / "valid"
    audio_dir.mkdir(parents=True)

    valid_audio = audio_dir / "valid.wav"
    missing_text_audio = audio_dir / "missing_text.wav"
    sf.write(valid_audio, np.zeros(INPUT_SAMPLE_RATE // 20, dtype=np.float32), INPUT_SAMPLE_RATE)
    sf.write(missing_text_audio, np.zeros(INPUT_SAMPLE_RATE // 20, dtype=np.float32), INPUT_SAMPLE_RATE)

    dataset = Dataset.from_dict(
        {
            "audio_filepath": [
                str(valid_audio),
                str(missing_text_audio),
                None,
                str(audio_dir / "does_not_exist.wav"),
            ],
            "text": ["valid text", None, "missing audio", "bad audio"],
            "speaker_id": ["speaker_0", "speaker_1", "speaker_2", "speaker_3"],
        }
    )
    dataset.save_to_disk(str(valid_dir))

    stage = _indicvoices_stage(
        raw_data_dir=str(raw_root),
        output_dir=str(tmp_path / "out"),
        langs=["gu"],
        native_splits=["valid"],
        split_dir_pattern="{split}",
        extraction_workers=1,
    )
    stage.setup()
    tasks = stage.process(_EmptyTask(dataset_name="test", data=None))
    metrics = stage._consume_custom_metrics()

    assert len(tasks) == 1
    assert metrics["input_rows"] == 4
    assert metrics["emitted_tasks"] == 1
    assert metrics["skipped_missing_text"] == 1
    assert metrics["skipped_missing_audio"] == 1
    assert metrics["skipped_audio_load"] == 1


def test_indicvoices_handler_writes_manifests_when_enabled(
    tmp_path: Path,
    indicvoices_raw_dataset: tuple[Path, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_messages = []
    monkeypatch.setattr(huggingface_module.logger, "info", log_messages.append)
    raw_root, total_rows = indicvoices_raw_dataset
    output_dir = tmp_path / "out"
    stage = _indicvoices_stage(
        raw_data_dir=str(raw_root),
        output_dir=str(output_dir),
        langs=["gu"],
        native_splits=["valid"],
        split_dir_pattern="{split}",
        extraction_workers=2,
        write_manifest=True,
    )
    stage.setup_on_node()
    stage.setup()

    tasks = stage.process(_EmptyTask(dataset_name="test", data=None))
    metrics = stage._consume_custom_metrics()
    stage.teardown()

    assert len(tasks) == total_rows
    actual_counts = Counter(task.data["split_type"] for task in tasks)
    expected_durations = {
        "train": 0.0,
        "dev": sum(task.data["duration"] for task in tasks if task.data["split_type"] == "dev"),
        "test": sum(task.data["duration"] for task in tasks if task.data["split_type"] == "test"),
    }
    assert metrics["duration_train_seconds"] == pytest.approx(expected_durations["train"])
    assert metrics["duration_dev_seconds"] == pytest.approx(expected_durations["dev"])
    assert metrics["duration_test_seconds"] == pytest.approx(expected_durations["test"])
    assert metrics["duration_train_hours"] == pytest.approx(expected_durations["train"] / 3600)
    assert metrics["duration_dev_hours"] == pytest.approx(expected_durations["dev"] / 3600)
    assert metrics["duration_test_hours"] == pytest.approx(expected_durations["test"] / 3600)
    assert any("duration_by_split_hours=(train=0.00h, dev=0.00h, test=0.00h)" in msg for msg in log_messages)
    assert (output_dir / "gu" / "dev" / "audio").is_dir()
    assert (output_dir / "gu" / "test" / "audio").is_dir()

    for split in ["dev", "test"]:
        manifest_path = output_dir / "gu" / f"{split}.jsonl"
        assert manifest_path.exists()
        with manifest_path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert len(rows) == actual_counts[split]
        assert all(row["split_type"] == split for row in rows)
        assert all(Path(row["audio_filepath"]).parent == output_dir / "gu" / split / "audio" for row in rows)
