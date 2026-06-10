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

import pytest

from nemo_curator.stages.audio.asr.normalization import TranscriptStatsStage
from nemo_curator.tasks import AudioTask


def _task(
    text: str,
    duration: float,
    split_type: str,
    transcript_error: bool,
    extra: dict | None = None,
) -> AudioTask:
    data = {
        "text": text,
        "lang": "gu",
        "duration": duration,
        "split_type": split_type,
        "transcript_error": transcript_error,
        "unknown_chars": {},
    }
    data.update(extra or {})
    return AudioTask(data=data)


def test_transcript_stats_aggregates_valid_invalid_unknowns_and_splits() -> None:
    stage = TranscriptStatsStage()
    tasks = [
        _task("abc", 1.0, "train", False),
        _task("ગુજરાતી", 2.0, "dev", True, {"unknown_chars": {"x": 2, "y": 1}}),
        _task("શબ્દ", 3.0, "test", False),
    ]

    results = [stage.process(task) for task in tasks]
    summary = stage.summary()
    metrics = stage._consume_custom_metrics()

    assert results == tasks
    assert summary["total_transcripts"] == 3
    assert summary["valid_transcripts"] == 2
    assert summary["invalid_transcripts"] == 1
    assert summary["valid_transcript_rate"] == pytest.approx(2 / 3)
    assert summary["invalid_transcript_rate"] == pytest.approx(1 / 3)
    assert summary["total_duration_hours"] == pytest.approx(6.0 / 3600)
    assert summary["valid_duration_hours"] == pytest.approx(4.0 / 3600)
    assert summary["invalid_duration_hours"] == pytest.approx(2.0 / 3600)
    assert summary["total_chars"] == len("abcગુજરાતીશબ્દ")
    assert summary["unique_known_chars"] == len(set("abcગુજરાતીશબ્દ"))
    assert summary["unique_known_char_rate"] == pytest.approx(len(set("abcગુજરાતીશબ્દ")) / len("abcગુજરાતીશબ્દ"))
    assert summary["unique_unknown_chars"] == 2
    assert summary["unique_unknown_char_rate"] == pytest.approx(2 / len("abcગુજરાતીશબ્દ"))
    assert "અ" in summary["alpha_minus_known_chars"]
    assert "ગ" not in summary["alpha_minus_known_chars"]
    assert summary["split_counts"] == {
        "train": {"total": 1, "valid": 1, "invalid": 0},
        "dev": {"total": 1, "valid": 0, "invalid": 1},
        "test": {"total": 1, "valid": 1, "invalid": 0},
    }
    assert summary["split_hours"] == {
        "train": {"total": pytest.approx(1.0 / 3600), "valid": pytest.approx(1.0 / 3600), "invalid": 0.0},
        "dev": {"total": pytest.approx(2.0 / 3600), "valid": 0.0, "invalid": pytest.approx(2.0 / 3600)},
        "test": {"total": pytest.approx(3.0 / 3600), "valid": pytest.approx(3.0 / 3600), "invalid": 0.0},
    }
    assert metrics["input_tasks"] == 3
    assert metrics["emitted_tasks"] == 3
    assert metrics["valid_transcripts"] == 2
    assert metrics["invalid_transcripts"] == 1
    assert metrics["total_duration_hours"] == pytest.approx(6.0 / 3600)
    assert metrics["valid_duration_hours"] == pytest.approx(4.0 / 3600)
    assert metrics["unique_known_chars"] == len(set("abcગુજરાતીશબ્દ"))
    assert metrics["unique_known_char_rate"] == pytest.approx(len(set("abcગુજરાતીશબ્દ")) / len("abcગુજરાતીશબ્દ"))
    assert metrics["unique_unknown_chars"] == 2
    assert metrics["unique_unknown_char_rate"] == pytest.approx(2 / len("abcગુજરાતીશબ્દ"))


def test_transcript_stats_can_drop_invalid_after_counting() -> None:
    stage = TranscriptStatsStage(drop_invalid=True)
    valid = _task("abc", 1.0, "train", False)
    invalid = _task("abcx", 2.0, "train", True, {"unknown_chars": {"x": 1}})

    assert stage.process(valid) is valid
    assert stage.process(invalid) is None

    summary = stage.summary()
    metrics = stage._consume_custom_metrics()
    assert summary["total_transcripts"] == 2
    assert summary["valid_transcripts"] == 1
    assert summary["invalid_transcripts"] == 1
    assert metrics["input_tasks"] == 2
    assert metrics["emitted_tasks"] == 1
    assert metrics["dropped_invalid"] == 1


def test_transcript_stats_rejects_multiple_languages() -> None:
    stage = TranscriptStatsStage()
    stage.process(_task("abc", 1.0, "train", False))
    with pytest.raises(ValueError, match="expects one language per dataset"):
        stage.process(_task("शब्द", 1.0, "train", False, {"lang": "hi"}))


def test_transcript_stats_runs_as_single_worker_for_exact_dataset_summary() -> None:
    assert TranscriptStatsStage().num_workers() == 1


def test_transcript_stats_writes_summary_during_processing(tmp_path: Path) -> None:
    summary_path = tmp_path / "stats" / "summary.json"
    stage = TranscriptStatsStage(output_summary_path=str(summary_path))
    stage.setup_on_node()

    assert summary_path.exists()

    stage.process(_task("ગુજરાતી", 1.0, "dev", False))
    with summary_path.open(encoding="utf-8") as f:
        first_summary = json.load(f)
    assert first_summary["total_transcripts"] == 1
    assert first_summary["valid_transcripts"] == 1

    stage.process(_task("શબ્દ", 2.0, "test", False))

    raw_summary = summary_path.read_text(encoding="utf-8")
    assert raw_summary.count('"total_transcripts"') == 1
    with summary_path.open(encoding="utf-8") as f:
        final_summary = json.load(f)
    assert final_summary["total_transcripts"] == 2
    assert final_summary["valid_transcripts"] == 2
    assert final_summary["split_counts"] == {
        "dev": {"total": 1, "valid": 1, "invalid": 0},
        "test": {"total": 1, "valid": 1, "invalid": 0},
    }
    stage.teardown()
