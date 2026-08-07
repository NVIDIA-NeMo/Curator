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

from nemo_curator.utils.performance_utils import StagePerfStats


def test_legacy_and_extended_serialization_are_separate() -> None:
    perf = StagePerfStats(
        stage_name="stage",
        process_time=1.5,
        actor_idle_time=0.25,
        input_data_size_mb=2.0,
        num_items_processed=3,
        custom_metrics={"io": 2.0},
        stage_id="002:stage",
        invocation_id="invocation-1",
        window_start_s=10.0,
        window_end_s=11.5,
    )

    assert perf.to_dict() == {
        "stage_name": "stage",
        "process_time": 1.5,
        "actor_idle_time": 0.25,
        "input_data_size_mb": 2.0,
        "num_items_processed": 3,
        "custom_metrics": {"io": 2.0},
    }
    assert perf.to_extended_dict() == {
        "stage_name": "stage",
        "process_time": 1.5,
        "actor_idle_time": 0.25,
        "input_data_size_mb": 2.0,
        "num_items_processed": 3,
        "custom_metrics": {"io": 2.0},
        "stage_id": "002:stage",
        "invocation_id": "invocation-1",
        "window_start_s": 10.0,
        "window_end_s": 11.5,
    }


def test_aggregate_retains_matching_stage_and_window_but_not_invocation() -> None:
    first = StagePerfStats(
        stage_name="stage",
        process_time=1.0,
        stage_id="001:stage",
        invocation_id="first",
        window_start_s=10.0,
        window_end_s=11.0,
    )
    second = StagePerfStats(
        stage_name="stage",
        process_time=2.0,
        stage_id="001:stage",
        invocation_id="second",
        window_start_s=11.0,
        window_end_s=13.0,
    )

    combined = first + second

    assert combined.process_time == 3.0
    assert combined.stage_id == "001:stage"
    assert combined.invocation_id == ""
    assert combined.window_start_s == 10.0
    assert combined.window_end_s == 13.0


def test_aggregate_drops_mismatched_stage_identity() -> None:
    first = StagePerfStats(stage_name="stage", stage_id="001:stage")
    second = StagePerfStats(stage_name="stage", stage_id="002:stage")

    assert (first + second).stage_id == ""


def test_reset_clears_invocation_identity_and_window() -> None:
    perf = StagePerfStats(
        stage_name="stage",
        process_time=1.0,
        stage_id="001:stage",
        invocation_id="invocation",
        window_start_s=10.0,
        window_end_s=11.0,
    )

    perf.reset()

    assert perf.process_time == 0.0
    assert perf.stage_id == ""
    assert perf.invocation_id == ""
    assert perf.window_start_s == 0.0
    assert perf.window_end_s == 0.0
