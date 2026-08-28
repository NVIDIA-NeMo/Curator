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

"""GPU visibility policy: a MASKED GPU (present but unreachable from a sandboxed
process) must be treated as "re-verify with full device access", never as a hard
"no GPU" that blocks validate/plan. Only a definitively GPU-less host (a CPU-only
torch build) is a real, blocking gap.

These tests are the regression guard that was missing when the fix was first
attempted (profiler flag + skill prose only), which let the gating layer keep
hard-failing on a masked GPU across users/machines.
"""

from nemo_curator.audio_agent import planner
from nemo_curator.audio_agent.contracts import EnvProfile
from nemo_curator.audio_agent.diagnostics import environment_preflight
from nemo_curator.audio_agent.recipe import Recipe, build_stages
from nemo_curator.stages.audio import agent as foundation

# A GPU-required recipe (InferenceSortformerStage is bound=gpu / not gpu_optional).
_GPU_RECIPE = {
    "stages": [
        {"ref": "ManifestReader", "params": {"manifest_path": "/tmp/m.jsonl"}},  # noqa: S108
        {"ref": "InferenceSortformerStage", "params": {"resources": {"gpus": 1}}},
    ]
}


def _masked_env() -> EnvProfile:
    # torch can't see a device, but hardware/driver signals say one is present.
    return EnvProfile(
        has_gpu=False,
        gpu_count=0,
        gpu_visibility="torch_cuda_unavailable",
        gpu_possibly_masked=True,
        torch_cuda_built=True,
        nvidia_device_nodes=1,
        total_cpus=8,
        total_ram_gb=32.0,
    )


def _absent_env() -> EnvProfile:
    # A CPU-only torch build cannot use a GPU regardless of hardware -> real block.
    return EnvProfile(
        has_gpu=False,
        gpu_count=0,
        gpu_visibility="cpu_only_torch",
        gpu_possibly_masked=False,
        torch_cuda_built=False,
        total_cpus=8,
        total_ram_gb=32.0,
    )


def _available_env() -> EnvProfile:
    return EnvProfile(
        has_gpu=True,
        gpu_count=1,
        gpu_mem_gb=24.0,
        gpu_visibility="available",
        cuda_runtime_version="12.9",
        cuda_driver_max_version="12.9",
        total_cpus=8,
        total_ram_gb=32.0,
    )


def _built():  # noqa: ANN202
    built, _issues = build_stages(Recipe.from_dict(_GPU_RECIPE))
    assert built is not None
    return built


class TestGpuStatusClassification:
    def test_available(self) -> None:
        assert EnvProfile(has_gpu=True, gpu_count=1).gpu_status == "available"

    def test_possibly_masked(self) -> None:
        assert _masked_env().gpu_status == "possibly_masked"

    def test_absent_is_only_cpu_only_torch(self) -> None:
        assert _absent_env().gpu_status == "absent"

    def test_unknown_when_no_visibility_facts(self) -> None:
        assert EnvProfile(has_gpu=False, gpu_visibility="unknown").gpu_status == "unknown"
        assert EnvProfile(has_gpu=False, gpu_visibility="torch_unavailable").gpu_status == "unknown"

    def test_status_is_exposed_in_dict(self) -> None:
        assert _masked_env().to_dict()["gpu_status"] == "possibly_masked"


class TestPreflightGating:
    def test_masked_gpu_does_not_block(self) -> None:
        d = environment_preflight(_built(), _masked_env(), operation="validate")
        assert d["can_execute"] is True
        assert d["decision_required"] is False
        gpu_issue = next(i for i in d["issues"] if i["code"] == "gpu_possibly_masked")
        assert gpu_issue["blocking"] is False
        # the remedy is to re-verify with full device access, not to give up / go CPU
        summaries = " ".join(o.get("summary", "") for o in gpu_issue["options"]).lower()
        assert "full device access" in summaries

    def test_absent_gpu_blocks(self) -> None:
        d = environment_preflight(_built(), _absent_env(), operation="validate")
        assert d["can_execute"] is False
        assert d["decision_required"] is True
        gpu_issue = next(i for i in d["issues"] if i["code"] == "gpu_unavailable")
        assert gpu_issue["blocking"] is True

    def test_available_gpu_has_no_gpu_gap(self) -> None:
        d = environment_preflight(_built(), _available_env(), operation="validate")
        codes = {i["code"] for i in d["issues"]}
        assert "gpu_unavailable" not in codes
        assert "gpu_possibly_masked" not in codes
        assert d["can_execute"] is True


class TestPlannerFeasibility:
    def _plan(self, env: EnvProfile):  # noqa: ANN202
        built = _built()
        contracts = [foundation.build_contract(s) for s in built]
        return planner.plan(built, contracts, env)

    def test_masked_gpu_stays_feasible(self) -> None:
        p = self._plan(_masked_env())
        assert p.feasible is True
        assert any("mask" in n.lower() for n in p.notes)

    def test_absent_gpu_is_infeasible(self) -> None:
        p = self._plan(_absent_env())
        assert p.feasible is False
        assert any("GPU" in e for e in p.escalations)
