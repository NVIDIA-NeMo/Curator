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

"""Existing Ray connections determine preflight and planning ownership."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import ray

from nemo_curator.stages.audio import audio_agent as aa
from nemo_curator.stages.audio.audio_agent import _verb_runtime, profiler
from nemo_curator.stages.audio.audio_agent.contracts import EnvProfile


@pytest.mark.parametrize("configured", [None, "auto"])
@pytest.mark.parametrize("operation", ["validate", "run", "smoke"])
def test_preconnected_remote_cluster_does_not_use_driver_gpu_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured: str | None, operation: str
) -> None:
    address = "10.20.30.40:6379"
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    if configured:
        monkeypatch.setenv("RAY_ADDRESS", configured)
    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.delenv("AUDIO_AGENT_REQUIRE_SMOKE", raising=False)
    monkeypatch.delenv("AUDIO_AGENT_WORKSPACE", raising=False)
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray, "get_runtime_context", lambda: SimpleNamespace(gcs_address=address))
    monkeypatch.setattr(ray, "cluster_resources", lambda: {"CPU": 16, "GPU": 1, "memory": 64 * 1024**3})
    env = EnvProfile(
        has_gpu=False,
        gpu_count=0,
        gpu_visibility="not_detected",
        torch_cuda_built=False,
        python_supported=True,
        total_cpus=1,
        total_ram_gb=2,
    )
    monkeypatch.setattr(profiler, "probe_env", lambda: env)
    executed = []

    def execute(*_args, **_kwargs):  # noqa: ANN202
        executed.append(True)
        raise RuntimeError("execution reached remote boundary")  # noqa: EM101

    monkeypatch.setattr(_verb_runtime, "_run_pipeline", execute)
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(json.dumps({"audio_filepath": str(tmp_path / "clip.wav")}) + "\n")
    recipe = {
        "stages": [
            {"ref": "ManifestReader", "params": {"manifest_path": str(manifest)}},
            {
                "ref": "ASRStage",
                "params": {
                    "adapter_target": "nemo_curator.models.asr.nemo_asr.NeMoASRAdapter",
                    "model_id": "nvidia/parakeet-tdt-0.6b-v2",
                    "max_audio_sec_per_actor": 2400.0,
                },
            },
        ]
    }
    result = getattr(aa, operation)(recipe, **({"confirm": True} if operation == "run" else {}))
    assert result["environment_decision"]["execution_target"] == "external_ray", result
    assert result["environment_decision"]["can_execute"] is True
    if operation != "validate":
        assert executed == [True], result
