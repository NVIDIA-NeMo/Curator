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
from omegaconf import OmegaConf

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.config.run import create_pipeline_from_yaml
from nemo_curator.stages.audio.asr.datasets.indicvoices import IndicVoicesHandler
from nemo_curator.tasks import _EmptyTask

_NUM_REALISTIC_ROWS = 100
_INPUT_SAMPLE_RATE = 8000
_OUTPUT_SAMPLE_RATE = 16000
_CONFIGS_DIR = Path(__file__).parent / "configs"


@pytest.fixture
def indicvoices_raw_dataset(tmp_path: Path) -> tuple[Path, int]:
    """Create a tiny on-disk HF dataset shaped like the raw IndicVoices split."""
    raw_root = tmp_path / "raw"
    audio_dir = tmp_path / "source_audio"
    valid_dir = raw_root / "valid"
    audio_dir.mkdir(parents=True)

    filepaths = []
    for i in range(_NUM_REALISTIC_ROWS):
        samples = np.linspace(0.0, 1.0, _INPUT_SAMPLE_RATE // 20, endpoint=False, dtype=np.float32)
        tone = 0.1 * np.sin(2 * np.pi * (220 + i) * samples)
        stereo = np.stack([tone, tone * 0.5], axis=1)
        path = audio_dir / f"sample_{i}.wav"
        sf.write(path, stereo, _INPUT_SAMPLE_RATE, subtype="PCM_16")
        filepaths.append(str(path))

    dataset = Dataset.from_dict(
        {
            "audio_filepath": filepaths,
            "text": [f"ગુજરાતી વાક્ય {i}" for i in range(_NUM_REALISTIC_ROWS)],
            "duration": [0.05] * _NUM_REALISTIC_ROWS,
            "lang": ["gu"] * _NUM_REALISTIC_ROWS,
            "speaker_id": [f"speaker_{i % 3}" for i in range(_NUM_REALISTIC_ROWS)],
            "gender": ["Female" if i % 2 else "Male" for i in range(_NUM_REALISTIC_ROWS)],
            "age_group": ["30-45"] * _NUM_REALISTIC_ROWS,
            "scenario": ["Extempore"] * _NUM_REALISTIC_ROWS,
            "task_name": ["Unit Test"] * _NUM_REALISTIC_ROWS,
            "state": ["Gujarat"] * _NUM_REALISTIC_ROWS,
            "district": ["Ahmedabad"] * _NUM_REALISTIC_ROWS,
            "normalized": [f"ગુજરાતી વાક્ય {i}" for i in range(_NUM_REALISTIC_ROWS)],
        }
    )
    dataset.save_to_disk(str(valid_dir))
    return raw_root, _NUM_REALISTIC_ROWS


def test_indicvoices_handler(
    tmp_path: Path,
    indicvoices_raw_dataset: tuple[Path, int],
) -> None:
    raw_root, total_rows = indicvoices_raw_dataset
    output_dir = tmp_path / "out"
    cfg = OmegaConf.load(_CONFIGS_DIR / "indicvoices.yaml")
    cfg.raw_data_dir = str(raw_root)
    cfg.output_dir = str(output_dir)

    pipeline = create_pipeline_from_yaml(cfg)
    tasks = pipeline.run(XennaExecutor(config={"execution_mode": "batch"}))

    assert len(tasks) == total_rows
    split_helper = IndicVoicesHandler(raw_data_dir=str(raw_root), output_dir=str(output_dir), langs=["gu"])
    expected_counts = Counter(split_helper.assign_split("valid", f"gu_valid_{i}") for i in range(total_rows))
    actual_counts = Counter(task.data["split_type"] for task in tasks)
    assert actual_counts == expected_counts
    assert actual_counts["dev"] + actual_counts["test"] == total_rows
    assert 0.55 <= actual_counts["dev"] / total_rows <= 0.65

    for task in tasks:
        audio_path = Path(task.data["audio_filepath"])
        assert audio_path.parent == output_dir / "gu" / task.data["split_type"] / "audio"
        wav_info = sf.info(audio_path)
        assert wav_info.samplerate == _OUTPUT_SAMPLE_RATE
        assert wav_info.channels == 1
        assert wav_info.subtype == "PCM_16"
        assert task.data["orig_sample_rate"] == _INPUT_SAMPLE_RATE
        assert task.data["orig_num_channels"] == 2
        assert task.data["lang"] == "gu"
        assert task.data["source"] == "IndicVoices"
        assert task.data["speaker_id"].startswith("speaker_")

    manifest_counts = {}
    for split in ["train", "dev", "test"]:
        path = output_dir / "gu" / f"{split}.jsonl"
        assert path.exists()
        with path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        manifest_counts[split] = len(rows)
        assert all(row["split_type"] == split for row in rows)

    assert manifest_counts == {"train": 0, "dev": actual_counts["dev"], "test": actual_counts["test"]}


def test_indicvoices_handler_reports_skipped_rows_from_realistic_hf_dataset(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    audio_dir = tmp_path / "source_audio"
    valid_dir = raw_root / "valid"
    audio_dir.mkdir(parents=True)

    valid_audio = audio_dir / "valid.wav"
    missing_text_audio = audio_dir / "missing_text.wav"
    sf.write(valid_audio, np.zeros(_INPUT_SAMPLE_RATE // 20, dtype=np.float32), _INPUT_SAMPLE_RATE)
    sf.write(missing_text_audio, np.zeros(_INPUT_SAMPLE_RATE // 20, dtype=np.float32), _INPUT_SAMPLE_RATE)

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

    stage = IndicVoicesHandler(
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
