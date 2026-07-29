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

from fsspec.core import url_to_fs

from nemo_curator.stages.audio.common import ManifestWriterStage
from nemo_curator.utils.performance_utils import StagePerfStats


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

    writer.finalize_performance_report([], performance_records=records, wall_time_s=2.0)

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
