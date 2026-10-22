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

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking"))

from nemo_curator_benchmarking.config import SlurmTimeoutPlan, build_benchmark_config_plan, plan_entry_slurm_timeout


@pytest.mark.parametrize(
    ("entry_timeout", "overrides", "expected"),
    [
        (14340, {}, SlurmTimeoutPlan(13740, 14400, 13740, True)),
        (7200, {}, SlurmTimeoutPlan(7200, 7860, 13740, False)),
        (10, {"startup_timeout_s": 20, "cleanup_timeout_s": 30}, SlurmTimeoutPlan(10, 600, 14340, False)),
    ],
)
def test_plan_entry_slurm_timeout(entry_timeout: int, overrides: dict, expected: SlurmTimeoutPlan) -> None:
    plan = build_benchmark_config_plan({"entries": [{"name": "benchmark", "timeout_s": entry_timeout}], **overrides})
    assert plan_entry_slurm_timeout(plan.entries[0], plan) == expected
