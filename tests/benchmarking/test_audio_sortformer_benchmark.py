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

import importlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from pytest import MonkeyPatch

from benchmarking.scripts import audio_sortformer_contract as contract

pytest.importorskip("nemo.collections.asr")


@pytest.fixture(scope="module")
def benchmark_module():
    scripts_dir = Path(__file__).parents[2] / "benchmarking" / "scripts"
    import sys

    sys.path.insert(0, str(scripts_dir))
    try:
        yield importlib.import_module("audio_sortformer_benchmark")
    finally:
        sys.path.remove(str(scripts_dir))
        sys.modules.pop("audio_sortformer_benchmark", None)


def _manifest_rows() -> list[dict]:
    return [
        {
            "audio_filepath": f"/container/audio/{filename}",
            "audio_item_id": filename.removesuffix(".wav"),
            "session_name": filename.removesuffix(".wav"),
            "duration": contract.DATASET_PUBLISHED_MEAN_DURATION_S,
            "timestamps_start": [0.0, 1.0, 2.0],
            "timestamps_end": [1.0, 2.0, 3.0],
            "speakers": ["speaker_a", "speaker_b", "speaker_c"],
        }
        for filename in contract.EXPECTED_AUDIO_FILENAMES
    ]


def test_load_and_rewrite_manifest_uses_mounted_audio_paths(
    tmp_path: Path, monkeypatch: MonkeyPatch, benchmark_module: ModuleType
) -> None:
    raw_data_dir = tmp_path / "input"
    audio_dir = raw_data_dir / "audio"
    audio_dir.mkdir(parents=True)
    rows = _manifest_rows()
    for row in rows:
        audio_path = audio_dir / Path(row["audio_filepath"]).name
        audio_path.write_bytes(b"audio")
    (raw_data_dir / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(
        benchmark_module.contract,
        "AUDIO_CORPUS_SHA256",
        benchmark_module.contract.audio_corpus_sha256(audio_dir),
    )
    monkeypatch.setattr(
        benchmark_module.contract,
        "REFERENCE_ANNOTATIONS_SHA256",
        benchmark_module.contract.reference_annotations_sha256(rows),
    )
    monkeypatch.setattr(
        benchmark_module.contract.sf,
        "info",
        lambda _path: SimpleNamespace(
            samplerate=16_000,
            channels=1,
            frames=int(contract.DATASET_PUBLISHED_MEAN_DURATION_S * 16_000),
        ),
    )

    target_manifest = tmp_path / "output" / "manifest.jsonl"
    rewritten = benchmark_module._load_and_rewrite_manifest(raw_data_dir, target_manifest)

    assert len(rewritten) == 34
    assert all(Path(row["audio_filepath"]).parent == audio_dir.resolve() for row in rewritten)
    assert target_manifest.is_file()

    (audio_dir / contract.EXPECTED_AUDIO_FILENAMES[0]).write_bytes(b"other")
    with pytest.raises(RuntimeError, match="audio corpus SHA-256 mismatch"):
        benchmark_module._load_and_rewrite_manifest(raw_data_dir, target_manifest)

    (audio_dir / contract.EXPECTED_AUDIO_FILENAMES[0]).write_bytes(b"audio")
    rows[0]["speakers"][0] = "tampered_speaker"
    (raw_data_dir / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(RuntimeError, match="reference annotation SHA-256 mismatch"):
        benchmark_module._load_and_rewrite_manifest(raw_data_dir, target_manifest)


def test_collect_metrics_requires_complete_unique_segment_output(benchmark_module: ModuleType) -> None:
    tasks = [
        SimpleNamespace(
            data={
                "duration": 10.0,
                "session_name": f"session_{index}",
                "audio_item_id": f"source_{index}",
                "diar_segments": [{"start": 0.0, "end": 1.0, "speaker": "speaker_0"}],
                "timestamps_start": [0.0],
                "timestamps_end": [1.0],
                "speakers": ["reference_0"],
            }
        )
        for index in range(2)
    ]

    metrics = benchmark_module._collect_diarization_metrics(
        tasks,
        elapsed_s=1.0,
        num_input_files=2,
        source_num_files=2,
        source_audio_duration_s=20.0,
    )

    assert metrics["is_success"] is True
    assert metrics["input_output_row_count_match"] is True
    assert metrics["output_session_names_unique"] is True
    assert metrics["num_files_with_segments"] == 2
    assert metrics["num_source_files_scored_for_der"] == 2
    assert metrics["total_segments_detected"] == 2
    assert metrics["total_audio_duration_hours"] == pytest.approx(20 / 3600, abs=1e-6)


def test_collect_metrics_rejects_duplicate_sessions_and_malformed_segments(benchmark_module: ModuleType) -> None:
    tasks = [
        SimpleNamespace(
            data={
                "duration": 10.0,
                "session_name": "duplicate",
                "audio_item_id": "source_0",
                "diar_segments": [{"start": 1.0, "end": 0.0, "speaker": "speaker_0"}],
                "timestamps_start": [0.0],
                "timestamps_end": [1.0],
                "speakers": ["reference_0"],
            }
        ),
        SimpleNamespace(
            data={
                "duration": 10.0,
                "session_name": "duplicate",
                "audio_item_id": "source_0",
                "diar_segments": [{"start": 0.0, "end": 1.0, "speaker": "speaker_0"}],
                "timestamps_start": [0.0],
                "timestamps_end": [1.0],
                "speakers": ["reference_0"],
            }
        ),
    ]

    metrics = benchmark_module._collect_diarization_metrics(
        tasks,
        elapsed_s=1.0,
        num_input_files=2,
        source_num_files=2,
        source_audio_duration_s=20.0,
    )

    assert metrics["is_success"] is False
    assert metrics["output_session_names_unique"] is False
    assert metrics["malformed_segments"] == 1


def test_collect_metrics_reports_semantically_wrong_output(benchmark_module: ModuleType) -> None:
    task = SimpleNamespace(
        data={
            "duration": 2.0,
            "session_name": "meeting",
            "audio_item_id": "meeting",
            "timestamps_start": [0.0, 1.0],
            "timestamps_end": [1.0, 2.0],
            "speakers": ["speaker_a", "speaker_b"],
            "diar_segments": [{"start": 0.0, "end": 2.0, "speaker": "one_speaker"}],
        }
    )

    metrics = benchmark_module._collect_diarization_metrics(
        [task], 1.0, num_input_files=1, source_num_files=1, source_audio_duration_s=2.0
    )

    assert metrics["malformed_segments"] == 0
    assert metrics["diarization_error_rate_percent"] == pytest.approx(50.0)
    assert metrics["is_success"] is True


def test_build_pipeline_uses_eight_one_gpu_workers(tmp_path: Path, benchmark_module: ModuleType) -> None:
    pipeline = benchmark_module._build_pipeline(
        manifest_path=tmp_path / "manifest.jsonl",
        model_path=tmp_path / "model.nemo",
        gpu_stage_num_workers=8,
        chunk_len=6,
        chunk_left_context=1,
        chunk_right_context=7,
        fifo_len=188,
        spkcache_update_period=144,
        spkcache_len=188,
        rttm_out_dir=str(tmp_path / "rttm"),
    )

    assert len(pipeline.stages) == 2
    _, inference = pipeline.stages
    assert inference.resources.gpus == 1
    assert inference.num_workers() == 8
    assert inference.chunk_len == 6
    assert inference.chunk_left_context == 1
    assert inference.chunk_right_context == 7
    assert inference.fifo_len == 188
    assert inference.spkcache_update_period == 144
    assert inference.spkcache_len == 188
