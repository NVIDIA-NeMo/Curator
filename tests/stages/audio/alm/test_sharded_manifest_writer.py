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

import numpy as np

from nemo_curator.stages.audio.alm.sharded_manifest_writer import ShardedManifestWriterStage
from nemo_curator.tasks import AudioTask
from nemo_curator.utils.performance_utils import StagePerfStats


def _task(index: int, total: int) -> AudioTask:
    return AudioTask(
        task_id=str(index),
        data={
            "text": f"row {index}",
            "score": np.float32(index),
            "duration": 2.0,
            "waveform": np.zeros(8, dtype=np.float32),
        },
        _metadata={"_shard_key": "corpus/en/manifest_0", "_shard_total": total},
    )


def test_batch_writes_jsonl_without_waveform_and_finalizes_done(tmp_path: Path) -> None:
    writer = ShardedManifestWriterStage(output_dir=str(tmp_path))
    writer.setup_on_node()

    writer.process_batch([_task(1, 2), _task(2, 2)])

    jsonl = tmp_path / "corpus" / "en" / "manifest_0.jsonl"
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert [row["text"] for row in rows] == ["row 1", "row 2"]
    assert all("waveform" not in row for row in rows)
    assert (tmp_path / "corpus" / "en" / "manifest_0.jsonl.done").read_text().strip() == "2"


def test_setup_recovers_partial_line_count(tmp_path: Path) -> None:
    partial = tmp_path / "corpus" / "manifest.jsonl"
    partial.parent.mkdir()
    partial.write_text('{"row": 1}\n{"row": 2}\n', encoding="utf-8")
    writer = ShardedManifestWriterStage(output_dir=str(tmp_path))

    writer.setup()

    assert writer._shard_counts["corpus/manifest"] == 2


def test_perf_summary_uses_shared_metrics_and_accepts_external_telemetry(tmp_path: Path) -> None:
    writer = ShardedManifestWriterStage(
        output_dir=str(tmp_path),
        write_perf_stats=True,
        perf_run_id="run-123",
        perf_executor="RayDataExecutor",
        perf_pipeline_metadata={"pipeline_name": "granary-v2", "backend": "ray_data"},
    )
    writer.setup_on_node()
    writer.setup()

    writer.process_batch([_task(1, 2), _task(2, 2)])

    summary_path = tmp_path / "perf_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["rows_out"] == 2.0
    assert summary["output_hours"] == 4.0 / 3600.0
    assert summary["run_id"] == "run-123"
    assert summary["executor"] == "RayDataExecutor"
    assert summary["pipeline"] == {"pipeline_name": "granary-v2", "backend": "ray_data"}
    writer_summary = summary["stages"]["sharded_manifest_writer"]
    assert writer_summary["total_items_processed"] == 2.0
    assert writer_summary["invocation_count"] == 1.0
    assert writer_summary["custom_metrics_sum"]["pipeline_output_rows"] == 2.0
    assert writer._writer_metrics.shard_count("corpus/en/manifest_0") == 2

    assert writer.record_external_stage_perf(
        StagePerfStats(
            stage_name="pipeline_hardware_sampler",
            process_time=3.0,
            num_items_processed=1,
            custom_metrics={"gpu_utilization_mean": 75.0},
        )
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    external = summary["stages"]["pipeline_hardware_sampler"]
    assert external["total_process_time_s"] == 3.0
    assert external["custom_metrics_sum"]["gpu_utilization_mean"] == 75.0
