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

"""End-to-end test for IndicVoices ASR dataset ingestion.

Runs the YAML-configured pipeline:
  IndicVoicesHandler -> SplitAwareManifestWriter
"""

import json
from collections import Counter
from pathlib import Path

import soundfile as sf
from omegaconf import OmegaConf

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.config.run import create_pipeline_from_yaml
from nemo_curator.stages.audio.asr.datasets.indicvoices import IndicVoicesHandler
from tests.stages.audio.asr.datasets.conftest import INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE

CONFIGS_DIR = Path(__file__).parent / "configs"


def test_indicvoices_pipeline_e2e(
    tmp_path: Path,
    indicvoices_raw_dataset: tuple[Path, int],
) -> None:
    raw_root, total_rows = indicvoices_raw_dataset
    output_dir = tmp_path / "out"
    cfg = OmegaConf.load(CONFIGS_DIR / "indicvoices_pipeline.yaml")
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
        assert wav_info.samplerate == OUTPUT_SAMPLE_RATE
        assert wav_info.channels == 1
        assert wav_info.subtype == "PCM_16"
        assert task.data["orig_sample_rate"] == INPUT_SAMPLE_RATE
        assert task.data["orig_num_channels"] == 2
        assert task.data["lang"] == "gu"
        assert task.data["source"] == "IndicVoices"
        assert task.data["speaker_id"].startswith("speaker_")
        assert task.data["text_original"] == task.data["text"]
        assert task.data["unknown_chars"] == {}
        assert task.data["transcript_error"] is False

    stats_metrics = [
        perf.custom_metrics for task in tasks for perf in task._stage_perf if perf.stage_name == "transcript_stats"
    ]
    assert len(stats_metrics) == total_rows
    assert max(metric["input_tasks"] for metric in stats_metrics) == total_rows
    assert max(metric["emitted_tasks"] for metric in stats_metrics) == total_rows
    assert max(metric["valid_transcripts"] for metric in stats_metrics) == total_rows
    assert max(metric["invalid_transcripts"] for metric in stats_metrics) == 0
    assert max(metric["unique_unknown_chars"] for metric in stats_metrics) == 0
    assert max(metric["unique_unknown_char_rate"] for metric in stats_metrics) == 0

    manifest_counts = {}
    for split in ["train", "dev", "test"]:
        path = output_dir / "gu" / f"{split}_normalized.jsonl"
        assert path.exists()
        with path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        manifest_counts[split] = len(rows)
        assert all(row["split_type"] == split for row in rows)
        assert all(row["text_original"] == row["text"] for row in rows)
        assert all(row["unknown_chars"] == {} for row in rows)
        assert all(row["transcript_error"] is False for row in rows)

    assert manifest_counts == {"train": 0, "dev": actual_counts["dev"], "test": actual_counts["test"]}
