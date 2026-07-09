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
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from nemo_curator.tasks import AudioTask
from nemo_curator.utils.performance_utils import StagePerfStats

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "benchmarking" / "scripts"))

import audio_tagging_benchmark as benchmark  # noqa: E402
from utils import RepeatEntriesStage  # noqa: E402


def _valid_segment() -> dict[str, Any]:
    return {
        "speaker": "speaker_0",
        "start": 0.0,
        "end": 2.0,
        "text": "hello world",
        "text_2": "hello world",
        "words": [{"word": "hello", "start": 0.0, "end": 1.0}],
        "metrics": {"wer": {"wer": 0.0}},
    }


def _valid_task() -> AudioTask:
    perf = [
        StagePerfStats(stage_name=stage_name, num_items_processed=1) for stage_name in benchmark._REQUIRED_STAGE_NAMES
    ]
    return AudioTask(data={"duration": 10.0, "segments": [_valid_segment()]}, _stage_perf=perf)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_repeat_entries_assigns_unique_ids_without_owning_task_ids() -> None:
    task = AudioTask(
        task_id="source",
        dataset_name="ami",
        data={"audio_filepath": "/data/audio.wav", "audio_item_id": "meeting"},
    )

    repeated = RepeatEntriesStage(repeat_factor=2, unique_id_key="audio_item_id").process(task)

    assert [item.data["audio_item_id"] for item in repeated] == ["meeting_repeat_0", "meeting_repeat_1"]
    assert all(item.task_id == "" for item in repeated)
    assert all(item.dataset_name == "ami" for item in repeated)


def test_repeat_entries_rejects_invalid_factor_and_missing_id() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        RepeatEntriesStage(repeat_factor=0)
    with pytest.raises(ValueError, match="audio_item_id"):
        RepeatEntriesStage(repeat_factor=2, unique_id_key="audio_item_id").process(AudioTask())


def test_count_jsonl_rows_rejects_empty_invalid_and_non_object_rows(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.touch()
    with pytest.raises(RuntimeError, match="missing or empty"):
        benchmark._count_jsonl_rows(path, "Input manifest")

    path.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid JSON"):
        benchmark._count_jsonl_rows(path, "Input manifest")

    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(TypeError, match="not a JSON object"):
        benchmark._count_jsonl_rows(path, "Input manifest")


def test_output_validation_enforces_input_output_and_manifest_row_equality(tmp_path: Path) -> None:
    manifest = tmp_path / "output.jsonl"
    _write_jsonl(manifest, [{}])

    with pytest.raises(RuntimeError, match="returned 0 rows for 1 input rows"):
        benchmark._validate_outputs([], manifest, num_input_rows=1)

    with pytest.raises(RuntimeError, match="returned 1 rows for 2 input rows"):
        benchmark._validate_outputs([_valid_task()], manifest, num_input_rows=2)

    _write_jsonl(manifest, [{}, {}])
    with pytest.raises(RuntimeError, match="contains 2 rows for 1 input rows"):
        benchmark._validate_outputs([_valid_task()], manifest, num_input_rows=1)


def test_output_validation_counts_only_complete_segments_and_every_stage(tmp_path: Path) -> None:
    manifest = tmp_path / "output.jsonl"
    _write_jsonl(manifest, [{}])
    task = _valid_task()
    incomplete = _valid_segment()
    incomplete.pop("text_2")
    incomplete["metrics"].pop("wer")
    task.data["segments"].append(incomplete)

    metrics = benchmark._validate_outputs([task], manifest, num_input_rows=1)

    assert metrics["input_output_row_count_match"] is True
    assert metrics["num_input_rows"] == metrics["num_output_rows"] == metrics["num_manifest_rows"] == 1
    assert metrics["num_segments_emitted"] == 2
    assert metrics["num_segments_processed"] == 1
    assert metrics["segment_output_coverage_ratio"] == 0.5

    task._stage_perf = [perf for perf in task._stage_perf if perf.stage_name != "ASRAlignment2"]
    with pytest.raises(RuntimeError, match="ASRAlignment2"):
        benchmark._validate_outputs([task], manifest, num_input_rows=1)


def test_run_uses_manifest_local_model_and_complete_stage_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_manifest = tmp_path / "input.jsonl"
    _write_jsonl(input_manifest, [{"audio_filepath": "/data/meeting.wav", "audio_item_id": "meeting"}])
    model_path = tmp_path / "pyannote-model"
    model_path.mkdir()
    recorded: list[Any] = []

    class RecordingPipeline:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.stages: list[Any] = []
            recorded.append(self)

        def add_stage(self, stage: object) -> None:
            self.stages.append(stage)

        def describe(self) -> str:
            return "recording pipeline"

        def run(self, _executor: object) -> list[AudioTask]:
            _write_jsonl(tmp_path / "results" / "tagging_output.jsonl", [{}])
            return [_valid_task()]

    monkeypatch.setattr(benchmark, "Pipeline", RecordingPipeline)
    monkeypatch.setattr(benchmark, "setup_executor", lambda *_args, **_kwargs: object())
    elapsed = iter([10.0, 12.0])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(elapsed))

    result = benchmark.run_audio_tagging_benchmark(
        benchmark_results_path=str(tmp_path),
        input_manifest=str(input_manifest),
        diarization_model_path=str(model_path),
        repeat_factor=1,
        max_segment_length=40.0,
        asr_batch_size=50,
        asr_transcribe_batch_size=2,
        squim_compute_batch_size=2,
        use_cuda_graphs=False,
        executor="xenna",
    )

    stages = recorded[0].stages
    assert [stage.name for stage in stages] == ["manifest_reader", *benchmark._REQUIRED_STAGE_NAMES]
    diarization = stages[2]
    assert (diarization.model_name, diarization.hf_token) == (str(model_path), None)
    assert (stages[4].transcribe_batch_size, stages[4].use_cuda_graphs) == (2, False)
    assert (stages[8].compute_batch_size, stages[10].infer_segment_only) == (2, True)
    assert result["metrics"]["input_output_row_count_match"] is True


def test_nightly_entries_use_real_prestaged_ami_and_no_hf_token() -> None:
    config = yaml.safe_load((_REPO_ROOT / "benchmarking" / "nightly-benchmark.yaml").read_text())
    entries = {entry["name"]: entry for entry in config["entries"]}

    for name, expected_rows, minimum_segments, minimum_hours in (
        ("audio_tagging_tts_xenna", 3, 100, 1.5),
        ("audio_tagging_tts_xenna_repeat", 6, 200, 3.0),
    ):
        entry = entries[name]
        assert "--input-manifest={dataset:audio_tagging_ami_sdm,files}/manifest.jsonl" in entry["args"]
        assert "--diarization-model-path={model_weights_path}/audio_tagging/" in entry["args"]
        assert "HF_TOKEN" not in entry["args"]
        requirements = {item["metric"]: item for item in entry["requirements"]}
        assert requirements["input_output_row_count_match"]["exact_value"] is True
        assert requirements["num_input_rows"]["exact_value"] == expected_rows
        assert requirements["num_output_rows"]["exact_value"] == expected_rows
        assert requirements["num_segments_processed"]["min_value"] == minimum_segments
        assert requirements["stage_execution_coverage_ratio"]["exact_value"] == 1.0
        assert requirements["total_audio_duration_hours"]["min_value"] == minimum_hours

    benchmark_source = (_REPO_ROOT / "benchmarking" / "scripts" / "audio_tagging_benchmark.py").read_text()
    run_script = (_REPO_ROOT / "benchmarking" / "tools" / "run.sh").read_text()
    assert "HF_TOKEN" not in benchmark_source
    assert "--hf-token" not in benchmark_source
    assert "HF_TOKEN" not in run_script
