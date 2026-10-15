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
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking" / "scripts"))

import audio_granary_metadata_benchmark as benchmark


def _write_rows(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


@pytest.fixture
def input_manifest(tmp_path: Path) -> Path:
    audio = tmp_path / "input.wav"
    sf.write(audio, np.zeros(32_000, dtype=np.float32), 16_000)
    return _write_rows(tmp_path / "input.jsonl", [{"audio_filepath": str(audio), "duration": 2.0}])


def _segment_row(output_dir: Path, original: str, *, start: int = 0, end: int = 1000) -> dict:
    audio = output_dir / "audio" / f"clip_{start}_{end}.opus"
    audio.parent.mkdir(parents=True, exist_ok=True)
    sf.write(audio, np.zeros((end - start) * 16, dtype=np.float32), 16_000, format="OGG", subtype="OPUS")
    rttm = output_dir / "rttm" / "clip.rttm"
    rttm.parent.mkdir(parents=True, exist_ok=True)
    rttm.write_text("SPEAKER clip 1 0.000 2.000 <NA> <NA> speaker_0 <NA> <NA>\n")
    return {
        "original_file": original,
        "audio_filepath": str(audio),
        "start_ms": start,
        "end_ms": end,
        "segment_num": start // 1000,
        "duration": (end - start) / 1000,
        "source_lang": "en",
        "_skipme": "",
        "additional_notes": {},
        "sed_events": [],
        "num_speakers": 1,
        "rttm_filepath": rttm.relative_to(output_dir / "rttm").as_posix(),
    }


def test_fanout_and_silent_parent_are_counted_separately(tmp_path: Path) -> None:
    rows = [_segment_row(tmp_path, "/original.wav", start=start, end=start + 1000) for start in (0, 1000)]
    rows[1].update(source_lang="skipped", _skipme="skipped due to disagreement between langID models.")
    rows.append({"audio_filepath": "/silent.wav", "duration": 3.0, "vad_empty": True, "_skipme": "vad_empty"})
    path = _write_rows(tmp_path / "metadata.jsonl", rows)
    metrics = benchmark._collect_output_metrics(path, {"/original.wav": 2.0, "/silent.wav": 3.0})

    assert metrics == {
        "num_input_recordings": 2,
        "num_tasks_processed": 3,
        "num_speech_segments": 2,
        "num_vad_empty_recordings": 1,
        "num_skipped_tasks": 2,
        "num_language_rejected_segments": 1,
        "input_coverage_ratio": 1.0,
        "total_audio_duration_hours": 5 / 3600,
        "total_segment_duration_hours": 2 / 3600,
    }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"original_file": "/unknown.wav"}, "Unknown output source"),
        ({"start_ms": -1}, "Invalid speech interval"),
        ({"end_ms": 2002, "duration": 2.002}, "Invalid speech interval"),
        ({"end_ms": float("inf")}, "Invalid speech interval"),
        ({"duration": 0}, "Invalid output duration"),
        ({"duration": True}, "Invalid output duration"),
        ({"duration": float("nan")}, "Invalid output duration"),
        ({"duration": 1.5}, "Invalid speech interval"),
        ({"segment_num": -1}, "Invalid segment_num"),
        ({"read_error": True}, "Audio read error"),
        ({"_skipme": "audio_load_error"}, "Unexpected skip reason"),
        ({"_skipme": "arbitrary failure"}, "Unexpected skip reason"),
        ({"source_lang": ""}, "Invalid language"),
        ({"num_speakers": True}, "Invalid language"),
        ({"sed_events": None}, "Invalid language"),
        ({"audio_filepath": "/missing.opus"}, "Opus segment"),
        ({"rttm_filepath": "/missing.rttm"}, "RTTM file"),
    ],
)
def test_invalid_output_fails(tmp_path: Path, change: dict, message: str) -> None:
    row = _segment_row(tmp_path, "/original.wav")
    row.update(change)
    path = _write_rows(tmp_path / "metadata.jsonl", [row])
    with pytest.raises(ValueError, match=message):
        benchmark._collect_output_metrics(path, {"/original.wav": 2.0})


@pytest.mark.parametrize("fault", ["corrupt", "duration", "stereo", "wrong_format"])
def test_segment_audio_must_match_metadata(tmp_path: Path, fault: str) -> None:
    row = _segment_row(tmp_path, "/original.wav")
    audio = Path(row["audio_filepath"])
    if fault == "corrupt":
        audio.write_bytes(b"not audio")
    elif fault == "wrong_format":
        sf.write(audio, np.zeros(16_000), 16_000, format="WAV")
    else:
        samples = np.zeros(8_000) if fault == "duration" else np.zeros((16_000, 2))
        sf.write(audio, samples, 16_000, format="OGG", subtype="OPUS")
    path = _write_rows(tmp_path / "metadata.jsonl", [row])
    with pytest.raises(ValueError, match=r"Unreadable Opus|do not match segment metadata"):
        benchmark._collect_output_metrics(path, {"/original.wav": 2.0})


@pytest.mark.parametrize("fault", ["duplicate", "missing_parent", "empty_and_speech", "duplicate_empty"])
def test_invalid_coverage_fails(tmp_path: Path, fault: str) -> None:
    row = _segment_row(tmp_path, "/original.wav")
    empty = {"audio_filepath": "/original.wav", "duration": 2.0, "vad_empty": True}
    rows = [row]
    inputs = {"/original.wav": 2.0}
    if fault == "duplicate":
        rows.append(row)
    elif fault == "missing_parent":
        inputs["/missing.wav"] = 2.0
    elif fault == "empty_and_speech":
        rows.append(empty)
    else:
        rows = [empty, empty]
    path = _write_rows(tmp_path / "metadata.jsonl", rows)
    with pytest.raises(ValueError, match=r"Duplicate|coverage|duplicate"):
        benchmark._collect_output_metrics(path, inputs)


@pytest.mark.parametrize("reason", sorted(benchmark._LID_REJECTIONS))
def test_language_rejections_are_audited(tmp_path: Path, reason: str) -> None:
    row = _segment_row(tmp_path, "/original.wav")
    row.update(source_lang="", _skipme=reason)
    path = _write_rows(tmp_path / "metadata.jsonl", [row])
    metrics = benchmark._collect_output_metrics(path, {"/original.wav": 2.0})
    assert metrics["num_language_rejected_segments"] == 1


def test_input_duration_is_checked_against_audio(input_manifest: Path) -> None:
    row = json.loads(input_manifest.read_text())
    assert benchmark._read_input_durations(input_manifest) == {row["audio_filepath"]: 2.0}
    row["duration"] = 20.0
    _write_rows(input_manifest, [row])
    with pytest.raises(ValueError, match="does not match decoded audio"):
        benchmark._read_input_durations(input_manifest)


def test_duplicate_inputs_fail(input_manifest: Path) -> None:
    row = json.loads(input_manifest.read_text())
    _write_rows(input_manifest, [row, row])
    with pytest.raises(ValueError, match="Duplicate input"):
        benchmark._read_input_durations(input_manifest)


@pytest.mark.parametrize("item_id", ["../escape", "/absolute", "..", "nested\\name", "", 3, None])
def test_unsafe_audio_ids_fail(input_manifest: Path, item_id: object) -> None:
    row = json.loads(input_manifest.read_text())
    row["audio_item_id"] = item_id
    _write_rows(input_manifest, [row])
    with pytest.raises(ValueError, match="plain filename"):
        benchmark._read_input_durations(input_manifest)


def test_duplicate_audio_ids_cannot_reuse_other_audio(input_manifest: Path, tmp_path: Path) -> None:
    row = json.loads(input_manifest.read_text())
    other = tmp_path / "other.wav"
    other.write_bytes(Path(row["audio_filepath"]).read_bytes())
    _write_rows(
        input_manifest,
        [{**row, "audio_item_id": "same"}, {**row, "audio_item_id": "same", "audio_filepath": str(other)}],
    )
    with pytest.raises(ValueError, match="Duplicate audio_item_id"):
        benchmark._read_input_durations(input_manifest)


@pytest.mark.parametrize("key", ["text", "transcript"])
def test_parent_transcripts_are_not_segment_labels(input_manifest: Path, key: str) -> None:
    row = json.loads(input_manifest.read_text())
    row[key] = "Whole recording transcript"
    _write_rows(input_manifest, [row])
    with pytest.raises(ValueError, match="not aligned to VAD segments"):
        benchmark._read_input_durations(input_manifest)


@pytest.mark.parametrize("contents", ["", "[]\n", "invalid json\n", '{"audio_filepath":"relative.wav"}\n'])
def test_malformed_input_fails(tmp_path: Path, contents: str) -> None:
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(contents)
    with pytest.raises(ValueError, match=r"contains no rows|JSON|absolute path"):
        benchmark._read_input_durations(manifest)


def test_main_records_failure_without_touching_input(
    input_manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = input_manifest.read_bytes()
    pipeline = Mock()
    pipeline.run.side_effect = RuntimeError("worker failed")
    monkeypatch.setattr(benchmark, "create_pipeline_from_yaml", Mock(return_value=pipeline))
    monkeypatch.setattr(benchmark, "create_executor_from_yaml", Mock())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--input-manifest",
            str(input_manifest),
            "--model-dir",
            str(tmp_path),
            "--benchmark-results-path",
            str(tmp_path / "run"),
        ],
    )
    assert benchmark.main() == 1
    metrics = json.loads((tmp_path / "run/metrics.json").read_text())
    assert metrics["is_success"] is False
    assert metrics["error_message"] == "worker failed"
    assert input_manifest.read_bytes() == original


@pytest.mark.parametrize("backend", [None, "xenna", "ray_data"])
def test_shared_yaml_settings_and_runner_metadata_are_preserved(
    input_manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str | None
) -> None:
    results = tmp_path / "run"
    results.mkdir()
    (results / "scratch").mkdir()
    (results / "params.json").write_text('{"ray_num_gpus":8}')
    expected = OmegaConf.load(benchmark._PIPELINE_CONFIG)
    expected.manifest_path = str(input_manifest)
    expected.output_dir = str(results / "results")
    expected.model_dir = str(tmp_path)
    if backend is not None:
        expected.backend = backend

    def create_pipeline(cfg: DictConfig) -> Mock:
        assert OmegaConf.to_container(cfg, resolve=True) == OmegaConf.to_container(expected, resolve=True)
        pipeline = Mock()
        original = json.loads(input_manifest.read_text())["audio_filepath"]
        pipeline.run.side_effect = lambda **_kwargs: _write_rows(
            Path(cfg.output_dir) / "metadata.jsonl", [_segment_row(Path(cfg.output_dir), original)]
        )
        return pipeline

    monkeypatch.setattr(benchmark, "create_pipeline_from_yaml", create_pipeline)
    monkeypatch.setattr(benchmark, "create_executor_from_yaml", Mock())
    args = [
        "benchmark",
        "--input-manifest",
        str(input_manifest),
        "--model-dir",
        str(tmp_path),
        "--benchmark-results-path",
        str(results),
    ]
    if backend is not None:
        args.extend(["--executor", backend])
    monkeypatch.setattr(sys, "argv", args)
    assert benchmark.main() == 0
    params = json.loads((results / "params.json").read_text())
    assert params["ray_num_gpus"] == 8
    assert params["pipeline_config"] == OmegaConf.to_container(expected, resolve=True)
    assert json.loads((results / "metrics.json").read_text())["input_coverage_ratio"] == 1.0


@pytest.mark.parametrize("existing", ["results", "scratch/audio.wav", "metrics.json", "tasks.pkl"])
def test_previous_outputs_are_never_overwritten(
    tmp_path: Path, existing: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous = tmp_path / existing
    previous.parent.mkdir(parents=True, exist_ok=True)
    previous.write_bytes(b"existing result")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--input-manifest",
            "/unused",
            "--model-dir",
            "/unused",
            "--benchmark-results-path",
            str(tmp_path),
        ],
    )
    assert benchmark.main() == 1
    assert previous.read_bytes() == b"existing result"
    assert [path for path in tmp_path.rglob("*") if path.is_file()] == [previous]
