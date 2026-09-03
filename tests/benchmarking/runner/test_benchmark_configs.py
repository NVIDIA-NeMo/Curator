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

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking"))

from runner.utils import merge_config_files

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "benchmarking"


def _entries(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["name"]: entry for entry in config["entries"]}


def _requirement(entry: dict[str, Any], metric: str) -> dict[str, Any]:
    return next(requirement for requirement in entry["requirements"] if requirement["metric"] == metric)


def test_benchmarks_yaml_is_complete_default_8xh100_config() -> None:
    config = merge_config_files([_CONFIG_DIR / "benchmarks.yaml"])
    entries = _entries(config)

    assert config["ray"] == {"num_cpus": 128, "num_gpus": 8, "enable_object_spilling": False}
    assert config["object_store_size"] == 536870912000
    assert config["max_timeout_s"] == 14340
    assert "audio_tagging_tts_xenna_repeat" not in entries
    for entry_name in ("audio_tagging_tts_xenna", "audio_tagging_tts_raydata"):
        assert "--gpu-stage-num-workers" not in entries[entry_name]["args"]
    replica_8_arg = '--autoscaling-config=\'{"min_replicas": 8, "max_replicas": 8}\''
    assert replica_8_arg in entries["ndd_dynamo"]["args"]


def test_4xgb200_64cpu_override_updates_resources_and_caps_timeouts() -> None:
    config = merge_config_files([_CONFIG_DIR / "benchmarks.yaml", _CONFIG_DIR / "4xGB200-64CPU.yaml"])
    entries = _entries(config)

    assert config["ray"] == {"num_cpus": 64, "num_gpus": 4, "enable_object_spilling": False}
    assert config["object_store_size"] == 429496729600
    assert config["default_timeout_s"] == 14340
    assert config["max_timeout_s"] == 14340
    assert all(
        entry.get("timeout_s", config["default_timeout_s"]) <= config["max_timeout_s"] for entry in entries.values()
    )

    assert "audio_tagging_tts_xenna_repeat" not in entries
    for entry_name in ("audio_tagging_tts_xenna", "audio_tagging_tts_raydata"):
        assert "--gpu-stage-num-workers" not in entries[entry_name]["args"]
        assert entries[entry_name]["timeout_s"] == 7200

    for entry_name in ("ndd_dynamo", "ndd_ray_serve"):
        replica_4_arg = '--autoscaling-config=\'{"min_replicas": 4, "max_replicas": 4}\''
        assert replica_4_arg in entries[entry_name]["args"]
        assert entries[entry_name]["ray"]["num_cpus"] == 16


def test_4xgb200_64cpu_override_sets_known_video_performance_baselines() -> None:
    config = merge_config_files([_CONFIG_DIR / "benchmarks.yaml", _CONFIG_DIR / "4xGB200-64CPU.yaml"])
    entries = _entries(config)

    expected_min_values = {
        "video_embedding_xenna": 4.0,
        "video_embedding_raydata": 2.0,
        "video_transcoding_xenna": 5.0,
        "video_transcoding_raydata": 3.0,
        "video_captioning_xenna": 0.25,
        "video_captioning_raydata": 0.25,
        "video_transnetv2_motion_aesthetic_filter_embeddings_xenna": 0.25,
        "video_transnetv2_motion_aesthetic_filter_embeddings_raydata": 0.1,
    }

    for entry_name, min_value in expected_min_values.items():
        requirement = _requirement(entries[entry_name], "throughput_clips_per_sec")
        assert requirement["min_value"] == min_value

    # Correctness/count requirements remain inherited from benchmarks.yaml.
    assert _requirement(entries["video_embedding_xenna"], "num_clips_generated")["exact_value"] == 7600
    assert _requirement(entries["video_captioning_xenna"], "num_clips_generated")["exact_value"] == 1013
