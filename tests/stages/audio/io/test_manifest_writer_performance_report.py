# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import tracemalloc
from pathlib import Path

import pytest
from fsspec.core import url_to_fs

from nemo_curator.stages.audio.common import ManifestWriterStage, _append_slurm_shard_suffix
from nemo_curator.utils.performance_utils import StagePerfStats
from nemo_curator.utils.stage_perf_collector import PerformanceRecordStore


def _report_context(*, slurm_array: dict[str, int] | None = None) -> dict[str, object]:
    return {
        "pipeline_name": "qwen-omni",
        "run_id": "run-1",
        "executor": "RayDataExecutor",
        "pipeline": {
            "pipeline_name": "qwen-omni",
            "stages": [{"stage_id": "002:ASR"}],
        },
        "slurm_array": slurm_array,
    }


def test_manifest_writer_rejects_equivalent_local_manifest_and_report_paths(tmp_path: Path) -> None:
    output_path = tmp_path / "shared.json"

    with pytest.raises(ValueError, match="must not resolve to the manifest output_path"):
        ManifestWriterStage(
            output_path=str(output_path),
            performance_report_path=output_path.as_uri(),
        )


def test_manifest_writer_rejects_equivalent_memory_filesystem_destination() -> None:
    with pytest.raises(ValueError, match="must not resolve to the manifest output_path"):
        ManifestWriterStage(
            output_path="memory://pr2296-path-check/shared.json",
            performance_report_path="memory:///pr2296-path-check/shared.json",
        )


def test_manifest_writer_report_path_requests_collection(tmp_path: Path) -> None:
    writer = ManifestWriterStage(
        output_path=str(tmp_path / "manifest.jsonl"),
        performance_report_path=str(tmp_path / "performance.json"),
    )

    assert writer.requests_performance_records() is True


def test_manifest_writer_persists_all_performance_records_through_fsspec() -> None:
    report_path = "memory://performance/qwen.json"
    writer = ManifestWriterStage(
        output_path="memory://performance/qwen.jsonl",
        performance_report_path=report_path,
    )
    records = [
        StagePerfStats(
            stage_name="ASR",
            stage_id="002:ASR",
            invocation_id="invocation-1",
            process_time=1.5,
            custom_metrics={"audio_duration_s": 12.0},
        )
    ]

    record_store = PerformanceRecordStore.from_records(records)
    writer.finalize_performance_report(
        performance_records=record_store,
        wall_time_s=2.0,
        report_context=_report_context(),
    )

    fs, resolved_path = url_to_fs(report_path)
    with fs.open(resolved_path, encoding="utf-8") as report_file:
        report = json.load(report_file)

    assert report["schema_version"] == 1
    assert report["run_id"] == "run-1"
    assert report["executor"] == "RayDataExecutor"
    assert report["wall_time_s"] == 2.0
    assert report["record_count"] == 1
    assert report["records"][0]["custom_metrics"] == {"audio_duration_s": 12.0}
    record_store.cleanup()


def test_manifest_writer_streams_high_cardinality_record_store(tmp_path: Path) -> None:
    report_path = tmp_path / "performance.json"
    writer = ManifestWriterStage(
        output_path=str(tmp_path / "output.jsonl"),
        performance_report_path=str(report_path),
    )
    record = StagePerfStats(
        stage_name="ASR",
        invocation_id="invocation",
        process_time=1.5,
        custom_metrics={"audio_duration_s": 12.0},
    )
    record_store = PerformanceRecordStore.from_records(record for _ in range(50_000))

    tracemalloc.start()
    try:
        writer.finalize_performance_report(
            performance_records=record_store,
            wall_time_s=2.0,
            report_context=_report_context(),
        )
        _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak_bytes < 2 * 1024 * 1024
    with report_path.open(encoding="utf-8") as report_file:
        report = json.load(report_file)
    assert report["record_count"] == 50_000
    assert len(report["records"]) == 50_000
    record_store.cleanup()


def test_manifest_writer_derives_unique_slurm_shard_destination(tmp_path: Path) -> None:
    report_path = tmp_path / "performance.json"
    writer = ManifestWriterStage(
        output_path=str(tmp_path / "manifest.jsonl"),
        performance_report_path=str(report_path),
    )
    record_store = PerformanceRecordStore()

    writer.finalize_performance_report(
        performance_records=record_store,
        wall_time_s=1.0,
        report_context=_report_context(slurm_array={"shard_index": 7, "total_shards": 20}),
    )

    sharded_report_path = Path(_append_slurm_shard_suffix(str(report_path), 7, 20))
    assert not report_path.exists()
    assert sharded_report_path.is_file()
    report = json.loads(sharded_report_path.read_text(encoding="utf-8"))
    assert report["slurm_array"] == {"shard_index": 7, "total_shards": 20}


def test_manifest_writer_rejects_effective_slurm_report_collision_without_data_loss(tmp_path: Path) -> None:
    output_path = tmp_path / "performance.shard-00007-of-00020.json"
    writer = ManifestWriterStage(
        output_path=str(output_path),
        performance_report_path=str(tmp_path / "performance.json"),
    )
    manifest_contents = '{"audio_filepath": "sample.wav"}\n'
    output_path.write_text(manifest_contents, encoding="utf-8")

    with pytest.raises(ValueError, match="effective performance report path"):
        writer.finalize_performance_report(
            performance_records=PerformanceRecordStore(),
            wall_time_s=1.0,
            report_context=_report_context(slurm_array={"shard_index": 7, "total_shards": 20}),
        )

    assert output_path.read_text(encoding="utf-8") == manifest_contents


def test_manifest_writer_rejects_effective_slurm_collision_on_memory_filesystem() -> None:
    output_path = "memory://pr2296-slurm-check/performance.shard-00007-of-00020.json"
    report_path = "memory:///pr2296-slurm-check/performance.json"
    writer = ManifestWriterStage(output_path=output_path, performance_report_path=report_path)
    output_fs, resolved_output_path = url_to_fs(output_path)
    output_fs.makedirs("/pr2296-slurm-check", exist_ok=True)
    output_fs.write_text(resolved_output_path, "manifest-data", encoding="utf-8")

    with pytest.raises(ValueError, match="effective performance report path"):
        writer.finalize_performance_report(
            performance_records=PerformanceRecordStore(),
            wall_time_s=1.0,
            report_context=_report_context(slurm_array={"shard_index": 7, "total_shards": 20}),
        )

    assert output_fs.read_text(resolved_output_path, encoding="utf-8") == "manifest-data"
