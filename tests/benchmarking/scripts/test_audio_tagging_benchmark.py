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

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from loguru import logger

from nemo_curator.tasks import AudioTask
from nemo_curator.utils.performance_utils import StagePerfStats

if TYPE_CHECKING:
    from collections.abc import Callable

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "benchmarking" / "scripts"))

import audio_tagging_benchmark as benchmark  # noqa: E402
from utils import RepeatEntriesStage  # noqa: E402


class _RecordingPipeline:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.stages: list[object] = []

    def add_stage(self, stage: object) -> None:
        self.stages.append(stage)

    def describe(self) -> str:
        return "recording pipeline"


class _DummyStage:
    def with_(self, **_kwargs: object) -> _DummyStage:
        return self


def _valid_segment(start: float = 0.0, end: float = 2.0) -> dict[str, Any]:
    return {
        "speaker": "speaker_0",
        "start": start,
        "end": end,
        "text": "hello world",
        "text_2": "hello world",
        "words": [
            {"word": "hello", "start": start, "end": start + 0.8},
            {"word": "world", "start": start + 1.0, "end": end},
        ],
        "metrics": {"bandwidth": [8000, 8000], "wer": {"wer": 0.0}},
    }


def _valid_task(duration: float = 10.0) -> AudioTask:
    stage_perf = [
        StagePerfStats(stage_name=stage_name, num_items_processed=1) for stage_name in benchmark._REQUIRED_STAGE_NAMES
    ]
    return AudioTask(data={"duration": duration, "segments": [_valid_segment()]}, _stage_perf=stage_perf)


def test_repeat_entries_assigns_unique_audio_ids() -> None:
    task = AudioTask(
        task_id="source-task",
        dataset_name="fleurs_en_us_dev",
        filepath_key="audio_filepath",
        data={"audio_filepath": "/data/audio.wav", "audio_item_id": "audio"},
    )

    repeated = RepeatEntriesStage(repeat_factor=2, unique_id_key="audio_item_id").process(task)

    assert [item.data["audio_item_id"] for item in repeated] == ["audio_repeat_0", "audio_repeat_1"]
    assert [item.task_id for item in repeated] == ["source-task_repeat_0", "source-task_repeat_1"]
    assert all(item.dataset_name == task.dataset_name for item in repeated)
    assert all(item.filepath_key == task.filepath_key for item in repeated)


def test_repeat_entries_requires_configured_unique_id() -> None:
    stage = RepeatEntriesStage(repeat_factor=2, unique_id_key="audio_item_id")

    with pytest.raises(ValueError, match="audio_item_id"):
        stage.process(AudioTask(data={"audio_filepath": "/data/audio.wav"}))


def test_repeat_entries_accepts_numeric_zero_id() -> None:
    task = AudioTask(task_id="0", data={"audio_filepath": "/data/audio.wav", "audio_item_id": 0})

    repeated = RepeatEntriesStage(repeat_factor=2, unique_id_key="audio_item_id").process(task)

    assert [item.data["audio_item_id"] for item in repeated] == ["0_repeat_0", "0_repeat_1"]


def test_repeat_entries_rejects_non_positive_factor() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        RepeatEntriesStage(repeat_factor=0)


def test_arg_parser_requires_exactly_one_source(tmp_path: Path) -> None:
    parser = benchmark.build_arg_parser()
    output_path = str(tmp_path / "results")

    with pytest.raises(SystemExit):
        parser.parse_args(["--benchmark-results-path", output_path])

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--benchmark-results-path",
                output_path,
                "--input-manifest",
                "input.jsonl",
                "--raw-data-dir",
                "fleurs",
            ]
        )


def test_arg_parser_has_no_token_argument(tmp_path: Path) -> None:
    parser = benchmark.build_arg_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--benchmark-results-path",
                str(tmp_path / "results"),
                "--input-manifest",
                "input.jsonl",
                "--hf-token",
                "do-not-put-secrets-on-the-command-line",
            ]
        )


def test_resolve_hf_token_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_SECRET_KEY", "legacy-secret")
    assert benchmark._resolve_hf_token() == "legacy-secret"

    monkeypatch.setenv("HF_TOKEN", "preferred-token")
    assert benchmark._resolve_hf_token() == "preferred-token"


def test_resolve_hf_token_requires_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_SECRET_KEY", raising=False)

    with pytest.raises(RuntimeError, match="Set HF_TOKEN"):
        benchmark._resolve_hf_token()


def test_sanitize_sensitive_params_recursively() -> None:
    params = {
        "hf_token": "secret-1",
        "nested": {"api-key": "secret-2", "ordinary": "visible"},
        "passwords": ["secret-3"],
    }

    assert benchmark._sanitize_sensitive_params(params) == {
        "hf_token": "<redacted>",
        "nested": {"api-key": "<redacted>", "ordinary": "visible"},
        "passwords": "<redacted>",
    }


@pytest.mark.parametrize(("input_manifest", "raw_data_dir"), [(None, None), ("a", "b")])
def test_add_input_stage_requires_exactly_one_source(input_manifest: str | None, raw_data_dir: str | None) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        benchmark._add_input_stage(_RecordingPipeline(), input_manifest, raw_data_dir, "en_us", "dev", False, None)


def test_add_input_stage_rejects_missing_or_empty_manifest(tmp_path: Path) -> None:
    pipeline = _RecordingPipeline()
    missing = tmp_path / "missing.jsonl"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        benchmark._add_input_stage(pipeline, str(missing), None, "en_us", "dev", False, None)

    empty = tmp_path / "empty.jsonl"
    empty.touch()
    with pytest.raises(ValueError, match="empty"):
        benchmark._add_input_stage(pipeline, str(empty), None, "en_us", "dev", False, None)


def test_add_input_stage_accepts_nonempty_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "input.jsonl"
    manifest.write_text('{"audio_filepath": "audio.wav"}\n', encoding="utf-8")
    pipeline = _RecordingPipeline()

    benchmark._add_input_stage(pipeline, str(manifest), None, "en_us", "dev", False, None)

    assert len(pipeline.stages) == 1
    assert isinstance(pipeline.stages[0], benchmark.ManifestReader)


def test_add_input_stage_requires_prestaged_fleurs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="transcript not found"):
        benchmark._add_input_stage(_RecordingPipeline(), None, str(tmp_path), "en_us", "dev", False, None)


def test_add_input_stage_accepts_prestaged_fleurs(tmp_path: Path) -> None:
    language_dir = tmp_path / "en_us"
    (language_dir / "dev").mkdir(parents=True)
    (language_dir / "dev.tsv").write_text("0\taudio.wav\thello\n", encoding="utf-8")
    pipeline = _RecordingPipeline()

    benchmark._add_input_stage(pipeline, None, str(tmp_path), "en_us", "dev", False, None)

    assert len(pipeline.stages) == 1
    assert isinstance(pipeline.stages[0], benchmark.CreateInitialManifestFleursStage)


def test_validate_written_manifest_rejects_missing_empty_and_invalid_json(tmp_path: Path) -> None:
    manifest = tmp_path / "output.jsonl"

    with pytest.raises(RuntimeError, match="missing or empty"):
        benchmark._validate_written_manifest(str(manifest), 1)

    manifest.touch()
    with pytest.raises(RuntimeError, match="missing or empty"):
        benchmark._validate_written_manifest(str(manifest), 1)

    manifest.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid JSON"):
        benchmark._validate_written_manifest(str(manifest), 1)


def test_validate_written_manifest_checks_json_objects_and_row_count(tmp_path: Path) -> None:
    manifest = tmp_path / "output.jsonl"
    manifest.write_text("[]\n", encoding="utf-8")
    with pytest.raises(TypeError, match="not a JSON object"):
        benchmark._validate_written_manifest(str(manifest), 1)

    manifest.write_text('{"row": 1}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="contains 1 rows; expected 2"):
        benchmark._validate_written_manifest(str(manifest), 2)

    manifest.write_text('{"row": 1}\n{"row": 2}\n', encoding="utf-8")
    benchmark._validate_written_manifest(str(manifest), 2)


def test_second_pass_validation_rejects_skipped_asr_and_wer() -> None:
    missing_second_pass = _valid_task()
    missing_second_pass.data["segments"][0].pop("text_2")
    with pytest.raises(RuntimeError, match="ASRAlignment2 may have been skipped"):
        benchmark._validate_second_pass_outputs([missing_second_pass])

    missing_wer = _valid_task()
    missing_wer.data["segments"][0]["metrics"].pop("wer")
    with pytest.raises(RuntimeError, match="WER stage may have been skipped"):
        benchmark._validate_second_pass_outputs([missing_wer])


def test_stage_execution_validation_rejects_any_skipped_stage() -> None:
    task = _valid_task()
    task._stage_perf = [perf for perf in task._stage_perf if perf.stage_name != "ASRAlignment2"]

    with pytest.raises(RuntimeError, match="ASRAlignment2"):
        benchmark._validate_required_stages_executed([task])


def test_run_audio_tagging_benchmark_reports_validated_success_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_manifest = tmp_path / "input.jsonl"
    input_manifest.write_text('{"audio_filepath": "audio.wav"}\n', encoding="utf-8")
    output_manifest = tmp_path / "results" / "tagging_output.jsonl"
    task = _valid_task()

    class _SuccessfulPipeline(_RecordingPipeline):
        def run(self, _executor: object) -> list[AudioTask]:
            output_manifest.parent.mkdir(parents=True)
            output_manifest.write_text(json.dumps(task.data) + "\n", encoding="utf-8")
            return [task]

    monkeypatch.setattr(benchmark, "Pipeline", _SuccessfulPipeline)
    monkeypatch.setattr(benchmark, "setup_executor", lambda *_args, **_kwargs: object())
    created_stages: list[tuple[str, dict[str, object]]] = []

    def stage_factory(class_name: str) -> Callable[..., _DummyStage]:
        def create_stage(*_args: object, **kwargs: object) -> _DummyStage:
            created_stages.append((class_name, dict(kwargs)))
            return _DummyStage()

        return create_stage

    for stage_name in (
        "ManifestReader",
        "ResampleAudioStage",
        "PyAnnoteDiarizationStage",
        "SplitLongAudioStage",
        "NeMoASRAlignerStage",
        "JoinSplitAudioMetadataStage",
        "MergeAlignmentDiarizationStage",
        "BandwidthEstimationStage",
        "TorchSquimQualityMetricsStage",
        "PrepareModuleSegmentsStage",
        "ComputeWERStage",
        "ManifestWriterStage",
    ):
        monkeypatch.setattr(benchmark, stage_name, stage_factory(stage_name))
    elapsed_times = iter([10.0, 12.0])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(elapsed_times))

    result = benchmark.run_audio_tagging_benchmark(
        benchmark_results_path=str(tmp_path),
        repeat_factor=1,
        hf_token=str(tmp_path / "runtime-credential"),
        max_segment_length=40.0,
        asr_batch_size=50,
        executor="xenna",
        input_manifest=str(input_manifest),
    )

    assert result["tasks"] == [task]
    assert [class_name for class_name, _kwargs in created_stages] == [
        "ManifestReader",
        "ResampleAudioStage",
        "PyAnnoteDiarizationStage",
        "SplitLongAudioStage",
        "NeMoASRAlignerStage",
        "JoinSplitAudioMetadataStage",
        "MergeAlignmentDiarizationStage",
        "BandwidthEstimationStage",
        "TorchSquimQualityMetricsStage",
        "PrepareModuleSegmentsStage",
        "NeMoASRAlignerStage",
        "ComputeWERStage",
        "ManifestWriterStage",
    ]
    asr_stage_configs = [kwargs for class_name, kwargs in created_stages if class_name == "NeMoASRAlignerStage"]
    assert [config["name"] for config in asr_stage_configs] == ["ASRAlignment", "ASRAlignment2"]
    assert asr_stage_configs[1] == {
        "name": "ASRAlignment2",
        "model_name": "nvidia/stt_en_conformer_ctc_large",
        "is_fastconformer": False,
        "decoder_type": "ctc",
        "batch_size": 64,
        "split_batch_size": 100,
        "text_key": "text_2",
        "infer_segment_only": True,
        "compute_timestamps": False,
    }
    assert result["metrics"] == pytest.approx(
        {
            "is_success": True,
            "time_taken_s": 2.0,
            "num_tasks_processed": 1,
            "num_tasks_with_segments": 1,
            "num_segments_processed": 1,
            "segment_task_coverage_ratio": 1.0,
            "num_segments_with_second_pass_asr": 1,
            "second_pass_asr_segment_coverage_ratio": 1.0,
            "num_segments_with_wer": 1,
            "wer_segment_coverage_ratio": 1.0,
            "num_required_stages": 12,
            "num_executed_required_stages": 12,
            "stage_execution_coverage_ratio": 1.0,
            "total_audio_duration_hours": 10.0 / 3600,
            "tagged_audio_duration_hours": 2.0 / 3600,
            "throughput_tasks_per_sec": 0.5,
            "throughput_audio_hours_per_hour": 5.0,
        }
    )


def test_main_redacts_environment_token_from_logs_and_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = "PR2187_SENTINEL_HF_TOKEN"
    output_path = tmp_path / "benchmark-output"
    input_manifest = tmp_path / "input.jsonl"
    input_manifest.write_text('{"audio_filepath": "audio.wav"}\n', encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", sentinel)
    monkeypatch.delenv("HF_SECRET_KEY", raising=False)
    monkeypatch.setattr(
        benchmark,
        "run_audio_tagging_benchmark",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError(f"model request failed for {sentinel}")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audio_tagging_benchmark.py",
            "--benchmark-results-path",
            str(output_path),
            "--input-manifest",
            str(input_manifest),
        ],
    )
    log_messages: list[str] = []
    sink_id = logger.add(lambda message: log_messages.append(str(message)), format="{message}")
    try:
        assert benchmark.main() == 1
    finally:
        logger.remove(sink_id)

    artifact_bytes = b"".join(path.read_bytes() for path in output_path.iterdir() if path.is_file())
    assert sentinel not in "".join(log_messages)
    assert sentinel.encode() not in artifact_bytes
    assert json.loads((output_path / "metrics.json").read_text())["error_message"] == (
        "model request failed for <redacted>"
    )
    assert all("token" not in key.lower() for key in json.loads((output_path / "params.json").read_text()))


def test_nightly_audio_tagging_entries_use_bounded_english_fleurs() -> None:
    config_path = _REPO_ROOT / "benchmarking" / "nightly-benchmark.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    entries = {entry["name"]: entry for entry in config["entries"]}

    expected_counts = {
        "audio_tagging_tts_xenna": (394, 197, 1.0),
        "audio_tagging_tts_xenna_repeat": (788, 394, 2.0),
    }
    for name, (expected_tasks, minimum_tagged_tasks, minimum_hours) in expected_counts.items():
        entry = entries[name]
        assert "--raw-data-dir={dataset:fleurs_en_us,files}" in entry["args"]
        assert "--lang=en_us" in entry["args"]
        assert "--split=dev" in entry["args"]
        assert "--no-auto-download" in entry["args"]
        assert "--hf-token" not in entry["args"]
        assert "HF_SECRET_KEY" not in entry["args"]

        serialized_runner_command = json.dumps(shlex.split(entry["args"]))
        assert "PR2187_SENTINEL_HF_TOKEN" not in serialized_runner_command

        requirements = {requirement["metric"]: requirement for requirement in entry["requirements"]}
        assert requirements["is_success"]["exact_value"] is True
        assert requirements["num_tasks_processed"]["exact_value"] == expected_tasks
        assert requirements["num_tasks_with_segments"]["min_value"] == minimum_tagged_tasks
        assert requirements["segment_task_coverage_ratio"]["min_value"] == 0.5
        assert requirements["num_segments_processed"]["min_value"] == minimum_tagged_tasks
        assert requirements["num_segments_with_second_pass_asr"]["min_value"] == minimum_tagged_tasks
        assert requirements["second_pass_asr_segment_coverage_ratio"]["exact_value"] == 1.0
        assert requirements["num_segments_with_wer"]["min_value"] == minimum_tagged_tasks
        assert requirements["wer_segment_coverage_ratio"]["exact_value"] == 1.0
        assert requirements["num_required_stages"]["exact_value"] == 12
        assert requirements["num_executed_required_stages"]["exact_value"] == 12
        assert requirements["stage_execution_coverage_ratio"]["exact_value"] == 1.0
        assert requirements["total_audio_duration_hours"]["min_value"] == minimum_hours
        assert requirements["throughput_audio_hours_per_hour"]["min_value"] == 1.0


def test_benchmark_container_passes_hf_token_without_embedding_its_value() -> None:
    run_script = (_REPO_ROOT / "benchmarking" / "tools" / "run.sh").read_text(encoding="utf-8")

    assert "export HF_TOKEN" in run_script
    assert "--env=HF_TOKEN \\" in run_script
    assert "--env=HF_TOKEN=" not in run_script
