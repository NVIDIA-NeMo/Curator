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

"""Recipe-aware environment decisions and LLM hand-off regressions."""

from __future__ import annotations

from pathlib import Path

import pytest

from nemo_curator import audio_agent as aa
from nemo_curator.audio_agent import env_health, verbs
from nemo_curator.audio_agent.contracts import EnvProfile
from nemo_curator.audio_agent.diagnostics import (
    diagnose_failure,
    environment_preflight,
)
from nemo_curator.audio_agent.recipe import Recipe, build_stages

_MANIFEST = Path(__file__).resolve().parents[1] / "fixtures/audio/alm/sample_input.jsonl"


def _healthy_env(**updates: object) -> EnvProfile:
    env = EnvProfile(
        has_gpu=True,
        gpu_count=1,
        gpu_names=["Test GPU"],
        gpu_mem_gb=24,
        gpu_visibility="available",
        nvidia_smi_status="ok",
        nvidia_smi_gpu_count=1,
        nvidia_device_nodes=1,
        torch_version="2.test",
        torch_cuda_built=True,
        total_cpus=16,
        total_ram_gb=64,
        free_disk_gb=50,
        has_ffmpeg=True,
        available_secrets=["HF_TOKEN"],
        python_version="3.13.1",
        python_supported=True,
        cuda_runtime_version="12.9",
        cuda_driver_max_version="12.9",
        cuda_compatible=True,
    )
    for key, value in updates.items():
        setattr(env, key, value)
    return env


def _cuda_mismatch() -> EnvProfile:
    return _healthy_env(
        cuda_runtime_version="12.9",
        cuda_driver_max_version="12.6",
        cuda_compatible=False,
    )


def _build(*refs: tuple[str, dict]) -> list[object]:
    stages, issues = build_stages(
        Recipe.from_dict({"stages": [{"ref": ref, "params": params} for ref, params in refs]})
    )
    assert issues == []
    assert stages is not None
    return list(stages)


def _recipe(output: Path, *middle: dict) -> dict:
    return {
        "stages": [
            {"ref": "ManifestReader", "params": {"manifest_path": str(_MANIFEST)}},
            *middle,
            {"ref": "ManifestWriterStage", "params": {"output_path": str(output)}},
        ]
    }


def _choice(decision: dict, option_id: str) -> dict:
    return next(item for item in decision["choices"] if item["id"] == option_id)


def test_cpu_recipe_ignores_machine_wide_cuda_mismatch() -> None:
    stages = _build(("GetAudioDurationStage", {}))

    decision = environment_preflight(stages, _cuda_mismatch())

    assert decision["status"] == "ready"
    assert decision["can_execute"] is True
    assert "cuda_driver_toolkit" in decision["ignored_machine_checks"]


def test_non_jit_gpu_recipe_warns_and_offers_bounded_smoke_before_host_change() -> None:
    stages = _build(
        (
            "UTMOSFilterStage",
            {"resources": {"cpus": 1, "gpus": 1}},
        )
    )

    decision = environment_preflight(stages, _cuda_mismatch(), operation="smoke")

    assert decision["status"] == "degraded"
    assert decision["can_execute"] is True
    assert [issue["code"] for issue in decision["issues"]].count("cuda_driver_toolkit_mismatch") == 1
    assert decision["recommended"] == "verify_gpu_stage_with_bounded_smoke"
    cpu = _choice(decision, "use_cpu_recipe_variant")
    assert cpu["availability"] == "conditional"
    assert cpu["preserves_recipe"] is False
    assert "validate" in " ".join(cpu["steps"])
    assert all(choice["id"] != "use_ctc_alignment_variant" for choice in decision["choices"])


def test_tdt_gpu_recipe_blocks_on_cuda_mismatch() -> None:
    stages = _build(
        (
            "ASRStage",
            {
                "adapter_target": "nemo_curator.models.asr.nemo_asr.NeMoASRAdapter",
                "model_id": "nvidia/parakeet-tdt-0.6b-v2",
                "resources": {"cpus": 1, "gpus": 1},
            },
        )
    )

    decision = environment_preflight(stages, _cuda_mismatch(), operation="smoke")

    assert decision["status"] == "action_required"
    assert decision["can_execute"] is False
    assert [issue["code"] for issue in decision["issues"]].count("cuda_driver_toolkit") == 1
    assert all(choice["id"] != "use_ctc_alignment_variant" for choice in decision["choices"])


def test_mixed_pipeline_never_claims_cpu_when_one_leaf_is_gpu_only() -> None:
    stages = _build(
        ("UTMOSFilterStage", {"resources": {"cpus": 1, "gpus": 1}}),
        ("InferenceSortformerStage", {"resources": {"cpus": 1, "gpus": 1}}),
    )

    decision = environment_preflight(stages, _cuda_mismatch())

    cpu = _choice(decision, "use_cpu_recipe_variant")
    assert cpu["availability"] == "unavailable"
    assert "InferenceSortformerStage" in cpu["reason"]


def test_external_ray_does_not_treat_local_cuda_mismatch_as_worker_fact() -> None:
    stages = _build(
        ("UTMOSFilterStage", {"resources": {"cpus": 1, "gpus": 1}}),
    )

    decision = environment_preflight(
        stages,
        _cuda_mismatch(),
        execution_target="external_ray",
    )

    assert decision["can_execute"] is True
    assert "cuda_driver_toolkit" not in {issue["code"] for issue in decision["issues"]}
    assert "remote_worker_environment_unverified" in {issue["code"] for issue in decision["issues"]}


def test_external_ray_does_not_treat_other_driver_facts_as_worker_facts(
    tmp_path: Path,
) -> None:
    stages = _build(
        ("ResampleAudioStage", {"resampled_audio_dir": str(tmp_path / "rs")}),
        (
            "PyAnnoteDiarizationStage",
            {"hf_token": "", "resources": {"cpus": 1, "gpus": 1}},
        ),
    )
    env = _cuda_mismatch()
    env.has_ffmpeg = False
    env.available_secrets = []
    env.python_supported = False
    env.free_disk_gb = 0

    decision = environment_preflight(
        stages,
        env,
        execution_target="external_ray",
    )

    assert decision["can_execute"] is True
    assert {issue["code"] for issue in decision["issues"]} == {"remote_worker_environment_unverified"}


def test_a_genuinely_missing_package_is_caught_earlier_than_preflight() -> None:
    """A per-stage package table used to gate this, and could never observe the condition.

    ``missing_packages`` comes from ``importlib.util.find_spec`` -- the SAME probe the
    catalog uses to decide which stage modules imported. So a package that is really absent
    removes its stages from the catalog, the recipe cannot reference them at all, and the
    per-stage gate never saw a recipe it could act on. Every new module still had to be
    added to that table to keep it honest.

    What actually protects the caller is checked here: the stage is not registered, and
    ``discover()`` reports the import error that explains why.
    """
    import json
    import subprocess
    import sys

    script = """
import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "whisperx":
            raise ImportError("No module named 'whisperx'")
        return None
sys.meta_path.insert(0, Block())
from nemo_curator import audio_agent as aa
out = aa.validate({"stages": [{"ref": "WhisperXVADStage", "params": {}}]})
print("RESULT" + json.dumps({
    "codes": [i["code"] for i in out["issues"]],
    "unavailable": bool(aa.discover().get("unavailable")),
}))
""".replace("json.dumps", "__import__('json').dumps")

    proc = subprocess.run(  # noqa: S603 - fixed argv, this interpreter, no shell
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    payload = next(line[len("RESULT") :] for line in proc.stdout.splitlines() if line.startswith("RESULT"))
    result = json.loads(payload)

    assert "unknown_stage" in result["codes"], "the stage is not registered when its package is absent"
    assert result["unavailable"], "discover() reports WHY the module is unavailable"


def test_missing_unselected_audio_package_does_not_block_recipe() -> None:
    decision = environment_preflight(
        _build(("GetAudioDurationStage", {})),
        _healthy_env(missing_packages=["nemo_toolkit[asr]"]),
    )

    assert decision["status"] == "ready"
    assert decision["can_execute"] is True
    assert "audio_extras" in decision["ignored_machine_checks"]


def test_external_ray_does_not_project_driver_missing_package_to_workers() -> None:
    stages = _build(
        (
            "ASRStage",
            {
                "adapter_target": "nemo_curator.models.asr.nemo_asr.NeMoASRAdapter",
                "model_id": "nvidia/parakeet-tdt-0.6b-v2",
                "resources": {"cpus": 1, "gpus": 1},
            },
        )
    )

    decision = environment_preflight(
        stages,
        _healthy_env(missing_packages=["nemo_toolkit[asr]"]),
        execution_target="external_ray",
    )

    assert decision["can_execute"] is True
    assert {issue["code"] for issue in decision["issues"]} == {"remote_worker_environment_unverified"}


def test_ambiguous_uv_dependencies_stay_unverified_not_healthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        env_health,
        "_uv_run_cmdline",
        lambda: ["uv", "run", "--with", "requests", "python", "-m", "x"],
    )

    decision = environment_preflight(
        _build(("GetAudioDurationStage", {})),
        _healthy_env(),
    )

    assert decision["can_execute"] is True
    assert "worker_env_unverified" in {issue["code"] for issue in decision["issues"]}


def test_ffmpeg_and_secret_block_only_relevant_selected_stages(tmp_path: Path) -> None:
    env = _healthy_env(has_ffmpeg=False, available_secrets=[])
    duration = environment_preflight(_build(("GetAudioDurationStage", {})), env)
    resample = environment_preflight(
        _build(("ResampleAudioStage", {"resampled_audio_dir": str(tmp_path / "rs")})),
        env,
    )
    pyannote_stages = _build(
        (
            "PyAnnoteDiarizationStage",
            {"hf_token": "unused-test-value", "resources": {"cpus": 1, "gpus": 1}},
        )
    )
    pyannote_stages[0].hf_token = ""
    pyannote = environment_preflight(pyannote_stages, env)

    assert duration["can_execute"] is True
    assert "ffmpeg_missing" not in {issue["code"] for issue in duration["issues"]}
    assert "ffmpeg_missing" in {issue["code"] for issue in resample["issues"]}
    assert "missing_secret" in {issue["code"] for issue in pyannote["issues"]}
    secret = next(choice for choice in pyannote["choices"] if choice["kind"] == "credential")
    assert "outside the chat" in secret["label"]


def test_explicit_stage_credential_satisfies_gate_without_exposing_value() -> None:
    env = _healthy_env(available_secrets=[])
    stages = _build(
        (
            "PyAnnoteDiarizationStage",
            {"hf_token": "configured-but-never-returned", "resources": {"cpus": 1, "gpus": 1}},
        )
    )

    decision = environment_preflight(stages, env)

    assert "missing_secret" not in {issue["code"] for issue in decision["issues"]}
    assert "configured-but-never-returned" not in str(decision)


def test_multiple_blockers_do_not_claim_one_choice_fixes_everything() -> None:
    env = _cuda_mismatch()
    env.available_secrets = []
    stages = _build(
        (
            "ASRStage",
            {
                "adapter_target": "nemo_curator.models.asr.nemo_asr.NeMoASRAdapter",
                "model_id": "nvidia/parakeet-tdt-0.6b-v2",
                "resources": {"cpus": 1, "gpus": 1},
            },
        ),
        (
            "PyAnnoteDiarizationStage",
            {"hf_token": "unused-test-value", "resources": {"cpus": 1, "gpus": 1}},
        ),
    )
    stages[1].hf_token = ""

    decision = environment_preflight(stages, env)

    driver = _choice(decision, "upgrade_nvidia_driver")
    assert driver["remaining_blockers"] == ["missing_secret:HF_TOKEN"]
    cpu = _choice(decision, "use_cpu_recipe_variant")
    assert "missing_secret:HF_TOKEN" in cpu["remaining_blockers"]
    assert decision["recommended"] is None


def test_runtime_diagnosis_never_recommends_a_partial_multi_blocker_fix() -> None:
    env = _cuda_mismatch()
    env.available_secrets = []
    stages = _build(
        (
            "ASRStage",
            {
                "adapter_target": "nemo_curator.models.asr.nemo_asr.NeMoASRAdapter",
                "model_id": "nvidia/parakeet-tdt-0.6b-v2",
                "resources": {"cpus": 1, "gpus": 1},
            },
        ),
        (
            "PyAnnoteDiarizationStage",
            {"hf_token": "placeholder", "resources": {"cpus": 1, "gpus": 1}},
        ),
    )
    stages[1].hf_token = ""

    diagnosis = diagnose_failure(
        "OSError: [Errno 28] No space left on device",
        stages=stages,
        env=env,
    )

    assert diagnosis["recommended"] is None
    assert all("remaining_blockers" in choice for choice in diagnosis["choices"])
    assert all(not choice["recommended"] or not choice["remaining_blockers"] for choice in diagnosis["choices"])
    assert _choice(diagnosis, "upgrade_nvidia_driver")["remaining_blockers"] == ["missing_secret:HF_TOKEN"]
    assert set(_choice(diagnosis, "recover_disk_full")["remaining_blockers"]) == {
        "cuda_driver_toolkit",
        "missing_secret:HF_TOKEN",
    }


def test_validate_projects_recipe_environment_blocker_without_touching_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recipe = _recipe(
        tmp_path / "out.jsonl",
        {
            "ref": "ASRStage",
            "params": {
                "adapter_target": "nemo_curator.models.asr.nemo_asr.NeMoASRAdapter",
                "model_id": "nvidia/parakeet-tdt-0.6b-v2",
                "audio_filepath_key": "audio_filepath",
                "resources": {"cpus": 1, "gpus": 1},
            },
        },
    )
    frozen_before = Recipe.from_dict(recipe).freeze()
    monkeypatch.setattr(verbs, "probe_env", _cuda_mismatch)

    result = aa.validate(recipe)

    assert result["runnable"] is False
    assert result["status"] == "fail"
    assert result["environment_decision"]["decision_required"] is True
    assert [item["code"] for item in result["gate_flags"]].count("cuda_driver_toolkit") == 1
    assert Recipe.from_dict(recipe).freeze().config_hash == frozen_before.config_hash


def test_validate_keeps_non_jit_gpu_recipe_runnable_for_bounded_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recipe = _recipe(
        tmp_path / "out.jsonl",
        {
            "ref": "UTMOSFilterStage",
            "params": {"resources": {"cpus": 1, "gpus": 1}},
        },
    )
    monkeypatch.setattr(verbs, "probe_env", _cuda_mismatch)

    result = aa.validate(recipe)

    assert result["runnable"] is True
    assert result["environment_decision"]["status"] == "degraded"
    assert result["environment_decision"]["recommended"] == ("verify_gpu_stage_with_bounded_smoke")


def test_validate_respects_external_ray_instead_of_local_machine_gates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recipe = _recipe(
        tmp_path / "out.jsonl",
        {
            "ref": "ResampleAudioStage",
            "params": {"resampled_audio_dir": str(tmp_path / "rs")},
        },
    )
    local_env = _healthy_env(
        has_gpu=False,
        gpu_count=0,
        gpu_visibility="not_detected",
        has_ffmpeg=False,
        available_secrets=[],
    )
    monkeypatch.setenv("RAY_ADDRESS", "ray://remote.example:10001")
    monkeypatch.setattr(verbs, "probe_env", lambda: local_env)

    result = aa.validate(recipe)

    assert result["runnable"] is True
    assert result["environment_decision"]["execution_target"] == "external_ray"
    assert "ffmpeg_missing" not in {item["code"] for item in result["gate_flags"]}


@pytest.mark.parametrize(
    "address",
    [
        "auto",
        "127.0.0.1:6379",
        "127.42.0.1:6379",
        "localhost:6379",
        "ray://localhost:10001",
        "[::1]:6379",
        "ray://[::1]:10001",
    ],
)
def test_validate_keeps_local_gates_for_ray_discovery_and_loopback(
    address: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recipe = _recipe(
        tmp_path / "out.jsonl",
        {
            "ref": "ResampleAudioStage",
            "params": {"resampled_audio_dir": str(tmp_path / "rs")},
        },
    )
    local_env = _healthy_env(has_ffmpeg=False)
    monkeypatch.setenv("RAY_ADDRESS", address)
    monkeypatch.setattr(verbs, "probe_env", lambda: local_env)

    result = aa.validate(recipe)

    assert result["runnable"] is False
    assert result["environment_decision"]["execution_target"] == "local"
    assert "ffmpeg_missing" in {item["code"] for item in result["gate_flags"]}


def test_hard_gpu_leaf_checks_cuda_even_if_recipe_reservation_is_zero() -> None:
    stages = _build(
        (
            "ASRStage",
            {
                "adapter_target": "nemo_curator.models.asr.nemo_asr.NeMoASRAdapter",
                "model_id": "nvidia/parakeet-tdt-0.6b-v2",
                "resources": {"cpus": 1, "gpus": 0},
            },
        ),
    )

    decision = environment_preflight(stages, _cuda_mismatch())

    assert decision["can_execute"] is False
    assert "cuda_driver_toolkit" in {issue["code"] for issue in decision["issues"]}


@pytest.mark.parametrize("verb", ["smoke", "run"])
def test_execution_refuses_before_ray_or_output_preparation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    verb: str,
) -> None:
    output = tmp_path / f"{verb}.jsonl"
    output.write_bytes(b"existing-output-must-survive\n")
    recipe_dict = _recipe(
        output,
        {
            "ref": "ASRStage",
            "params": {
                "adapter_target": "nemo_curator.models.asr.nemo_asr.NeMoASRAdapter",
                "model_id": "nvidia/parakeet-tdt-0.6b-v2",
                "audio_filepath_key": "audio_filepath",
                "resources": {"cpus": 1, "gpus": 1},
            },
        },
    )
    monkeypatch.setattr(verbs, "probe_env", _cuda_mismatch)

    def must_not_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("execution preparation ran before the environment gate")  # noqa: EM101

    monkeypatch.setattr(verbs, "_bootstrap_ray", must_not_run)
    monkeypatch.setattr(verbs, "_apply_ray_cluster_capacity", must_not_run)
    monkeypatch.setattr(verbs, "_plan_resources", must_not_run)
    monkeypatch.setattr(verbs, "_make_executor", must_not_run)
    monkeypatch.setattr(verbs, "_run_pipeline_autofallback", must_not_run)
    if verb == "smoke":
        monkeypatch.setattr(verbs, "_bound_recipe", must_not_run)
        result = aa.smoke(recipe_dict, sample=1, bootstrap_ray=True)
    else:
        frozen = Recipe.from_dict(recipe_dict).freeze()
        result = aa.run(
            recipe_dict,
            confirm=frozen.config_hash,
            bootstrap_ray=True,
        )

    assert result["status"] == "refused"
    assert result["reason_code"] == "environment_action_required"
    assert result["environment_decision"]["prompt_user"] is True
    assert output.read_bytes() == b"existing-output-must-survive\n"


def test_unknown_failure_stays_unknown_and_redacts_secret() -> None:
    diagnosis = diagnose_failure(
        "UnmappedNativeError: credential=super-secret-value",
        env=_healthy_env(),
        operation="smoke",
        phase="stage_setup",
    )

    assert diagnosis["status"] == "unknown"
    assert diagnosis["failure"]["code"] == "unknown_failure"
    assert "super-secret-value" not in diagnosis["failure"]["evidence"]
    assert diagnosis["recommended"] == "collect_minimal_diagnostics"
    assert all(choice["kind"] == "diagnostic" for choice in diagnosis["choices"])


def test_diagnosis_evidence_redacts_transport_credentials() -> None:
    basic = "dXNlcjpwYXNzd29yZA=="
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"

    diagnosis = diagnose_failure(
        f"TransportError Authorization: Basic {basic} url=https://alice:correct-horse@example.test/v2 assertion={jwt}",
        env=_healthy_env(),
    )

    rendered = str(diagnosis)
    assert basic not in rendered
    assert "alice:correct-horse" not in rendered
    assert jwt not in rendered
    assert rendered.count("<redacted-secret>") >= 3


def test_public_smoke_failure_is_structured_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recipe = _recipe(tmp_path / "out.jsonl")
    monkeypatch.setattr(verbs, "probe_env", _healthy_env)

    def fail_with_secret(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("HF_TOKEN=plain-runtime-token Permission denied")  # noqa: EM101

    monkeypatch.setattr(verbs, "_run_pipeline_autofallback", fail_with_secret)

    result = aa.smoke(recipe, sample=1, executor=object())

    rendered = str(result)
    assert result["status"] == "error"
    assert result["diagnosis"]["failure"]["code"] == "permission_denied"
    assert "plain-runtime-token" not in rendered
    assert "<redacted-secret>" in rendered
    assert result["smoke_token_status"].startswith("not_issued:")


def test_continuation_never_reintroduces_recipe_secrets_after_run_redaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_value = "hf_continuation_secret_value"  # noqa: S105
    materialized = Recipe.from_dict(
        {
            "stages": [
                {
                    "ref": "PyAnnoteDiarizationStage",
                    "params": {"hf_token": secret_value},
                }
            ]
        }
    )

    class Continuation:
        @staticmethod
        def materialize(*_args: object, **_kwargs: object) -> tuple[Recipe, str]:
            return materialized, ""

    monkeypatch.setattr(verbs, "validate", lambda *_args, **_kwargs: {"runnable": True})
    monkeypatch.setattr(verbs, "run", lambda *_args, **_kwargs: {"status": "completed"})
    monkeypatch.setattr(verbs, "_continuation_context_from_plan", lambda *_args: {})
    result = verbs._execute_plan(
        Recipe.from_dict({"stages": []}),
        {
            "mode": "incremental",
            "reuse_point": {"uri": "memory://artifact", "stage_index": 0},
            "reuse_stages": [],
        },
        choice="extend",
        data=None,
        confirm=True,
        output_dir=None,
        bootstrap_ray=False,
        goal=None,
        parent=object(),
        continuation_mod=Continuation,
    )

    assert secret_value not in str(result)
    assert result["recipe"]["stages"][0]["params"]["hf_token"] == "<redacted-secret>"  # noqa: S105


@pytest.mark.parametrize(
    ("error", "code"),
    [
        ("OSError: [Errno 28] No space left on device", "disk_full"),
        ("PermissionError: [Errno 13] Permission denied", "permission_denied"),
        ("ImportError: libfoo.so: undefined symbol: ABI_1", "native_library_mismatch"),
        (
            "CUDA error 804: forward compatibility was attempted on non supported HW",
            "cuda_driver_runtime",
        ),
        ("SSLError: certificate verify failed", "tls_certificate"),
    ],
)
def test_common_environment_failures_get_grounded_user_choices(
    error: str,
    code: str,
) -> None:
    diagnosis = diagnose_failure(error, env=_healthy_env())

    assert diagnosis["status"] == "action_required"
    assert diagnosis["failure"]["code"] == code
    assert diagnosis["choices"]
    assert all(choice["requires_confirmation"] for choice in diagnosis["choices"])
    assert diagnosis["prompt_user"] is True


def test_known_ptx_failure_is_classified_and_does_not_repeat_attempted_action() -> None:
    diagnosis = diagnose_failure(
        "CUDA_ERROR_UNSUPPORTED_PTX_VERSION (error 222)",
        env=_cuda_mismatch(),
        attempted_actions=["recover_cuda_runtime_jit_failure"],
    )

    assert diagnosis["failure"]["code"] == "cuda_runtime_jit_failure"
    attempted = next(choice for choice in diagnosis["choices"] if choice["id"] == "recover_cuda_runtime_jit_failure")
    assert attempted["availability"] == "unavailable"
    assert attempted["recommended"] is False


def test_ptx_diagnosis_never_offers_ctc_for_pure_tdt_transcription() -> None:
    stages = _build(
        (
            "ASRStage",
            {
                "adapter_target": "nemo_curator.models.asr.nemo_asr.NeMoASRAdapter",
                "model_id": "nvidia/parakeet-tdt-0.6b-v2",
                "resources": {"cpus": 1, "gpus": 1},
            },
        )
    )

    diagnosis = diagnose_failure(
        "CUDA_ERROR_UNSUPPORTED_PTX_VERSION (error 222)",
        stages=stages,
        env=_cuda_mismatch(),
    )

    assert diagnosis["failure"]["code"] == "asr_decoder_cuda_graph"
    assert all(choice["id"] != "use_ctc_alignment_variant" for choice in diagnosis["choices"])


def test_ptx_diagnosis_for_non_asr_stage_removes_asr_workarounds() -> None:
    stages = _build(
        ("UTMOSFilterStage", {"resources": {"cpus": 1, "gpus": 1}}),
    )

    diagnosis = diagnose_failure(
        "CUDA_ERROR_UNSUPPORTED_PTX_VERSION (error 222)",
        stages=stages,
        env=_cuda_mismatch(),
    )

    assert diagnosis["failure"]["code"] == "cuda_runtime_jit_failure"
    assert all(
        choice["id"] not in {"inspect_asr_cuda_stack", "use_ctc_alignment_variant"} for choice in diagnosis["choices"]
    )
    assert "CTC" not in " ".join(step for choice in diagnosis["choices"] for step in choice["steps"])


def test_gpu_probe_explains_masked_device_instead_of_saying_no_hardware() -> None:
    check = env_health._gpu(
        _healthy_env(
            has_gpu=False,
            gpu_count=0,
            gpu_names=[],
            gpu_visibility="masked_by_cuda_visible_devices",
            cuda_visible_devices="masked",
        )
    )

    assert check.status == "warn"
    assert "CUDA_VISIBLE_DEVICES" in check.finding
    assert check.options[0].id == "request_or_expose_gpu"


def test_cpu_only_torch_without_hardware_does_not_recommend_cuda_install() -> None:
    env = _healthy_env(
        has_gpu=False,
        gpu_count=0,
        gpu_names=[],
        gpu_visibility="cpu_only_torch",
        nvidia_smi_status="missing",
        nvidia_smi_gpu_count=0,
        nvidia_device_nodes=0,
        torch_cuda_built=False,
        cuda_runtime_version="",
        cuda_driver_max_version="",
    )
    stages = _build(
        ("UTMOSFilterStage", {"resources": {"cpus": 1, "gpus": 1}}),
    )

    decision = environment_preflight(stages, env)

    assert decision["recommended"] == "use_gpu_host"
    gpu_env = _choice(decision, "sync_gpu_project_environment")
    assert gpu_env["availability"] == "conditional"
    assert gpu_env["recommended"] is False


def test_zero_free_disk_is_a_warning_not_probe_unknown() -> None:
    check = env_health._disk(_healthy_env(free_disk_gb=0.0))

    assert check.status == "warn"
    assert "0.0 GB" in check.finding
    assert check.confidence == "high"
