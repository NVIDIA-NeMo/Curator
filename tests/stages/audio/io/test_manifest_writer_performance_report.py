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
from unittest.mock import MagicMock, patch

import pytest
from fsspec.core import url_to_fs

from nemo_curator.stages.audio.common import ManifestWriterStage, _append_slurm_shard_suffix
from nemo_curator.utils.performance_utils import StagePerfStats
from nemo_curator.utils.stage_perf_collector import PerformanceRecordStore, performance_collection_enabled


def test_manifest_writer_rejects_equivalent_local_manifest_and_report_paths(tmp_path: Path) -> None:
    output_path = tmp_path / "shared.json"

    with pytest.raises(ValueError, match="must not resolve to the manifest output_path"):
        ManifestWriterStage(
            output_path=str(output_path),
            performance_report_path=output_path.as_uri(),
        )


def test_manifest_writer_rejects_same_mocked_remote_destination() -> None:
    remote_fs = MagicMock()
    remote_fs.protocol = "s3"
    remote_fs.storage_options = {"endpoint_url": "https://example.test"}

    with (
        patch(
            "nemo_curator.stages.audio.common.url_to_fs",
            side_effect=[(remote_fs, "bucket/shared.json"), (remote_fs, "bucket/shared.json")],
        ),
        pytest.raises(ValueError, match="must not resolve to the manifest output_path"),
    ):
        ManifestWriterStage(
            output_path="s3://bucket/manifest.json",
            performance_report_path="s3://bucket/performance.json",
        )


def test_manifest_writer_report_path_automatically_requests_collection(tmp_path: Path) -> None:
    writer = ManifestWriterStage(
        output_path=str(tmp_path / "manifest.jsonl"),
        performance_report_path=str(tmp_path / "performance.json"),
    )

    assert writer.requests_performance_records() is True
    assert performance_collection_enabled([writer]) is True


def test_manifest_writer_persists_all_performance_records_through_fsspec() -> None:
    report_path = "memory://performance/qwen.json"
    writer = ManifestWriterStage(
        output_path="memory://performance/qwen.jsonl",
        performance_report_path=report_path,
    )
    writer._curator_run_id = "run-1"
    writer._curator_executor = "RayDataExecutor"
    writer._curator_pipeline_metadata = {
        "pipeline_name": "qwen-omni",
        "stages": [{"stage_id": "002:ASR"}],
    }
    records = [
        StagePerfStats(
            stage_name="ASR",
            stage_id="002:ASR",
            invocation_id="invocation-1",
            process_time=1.5,
            custom_metrics={"audio_duration_s": 12.0},
            gpu_indices=[0, 1],
        )
    ]

    record_store = PerformanceRecordStore.from_records(records)
    writer.finalize_performance_report([], performance_records=record_store, wall_time_s=2.0)

    fs, resolved_path = url_to_fs(report_path)
    with fs.open(resolved_path, encoding="utf-8") as report_file:
        report = json.load(report_file)

    assert report["schema_version"] == 1
    assert report["run_id"] == "run-1"
    assert report["executor"] == "RayDataExecutor"
    assert report["wall_time_s"] == 2.0
    assert report["record_count"] == 1
    assert report["records"][0]["custom_metrics"] == {"audio_duration_s": 12.0}
    assert report["records"][0]["gpu_indices"] == [0, 1]
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
        writer.finalize_performance_report([], performance_records=record_store, wall_time_s=2.0)
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
    writer._curator_slurm_array_shard_index = 7
    writer._curator_slurm_array_total_shards = 20
    record_store = PerformanceRecordStore()

    writer.finalize_performance_report([], performance_records=record_store, wall_time_s=1.0)

    sharded_report_path = Path(_append_slurm_shard_suffix(str(report_path), 7, 20))
    assert not report_path.exists()
    assert sharded_report_path.is_file()
    report = json.loads(sharded_report_path.read_text(encoding="utf-8"))
    assert report["slurm_array"] == {"shard_index": 7, "total_shards": 20}
