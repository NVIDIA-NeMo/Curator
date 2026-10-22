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

"""Bounded pipeline execution and smoke evidence."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from nemo_curator.stages.audio.audio_agent import (
    _safety,
    _verb_evidence,
    _verb_inputs,
    _verb_pretrain,
    _verb_runtime,
    _verb_smoke_inputs,
    _verb_validation,
    profiler,
)
from nemo_curator.stages.audio.audio_agent import (
    report as _report_data,
)
from nemo_curator.stages.audio.audio_agent.contracts import SmokeReport
from nemo_curator.stages.audio.audio_agent.recipe import build_stages
from nemo_curator.stages.audio.audio_agent.report import _row_count

if TYPE_CHECKING:
    from nemo_curator.stages.audio.audio_agent.recipe import (
        Recipe,
    )


@_verb_runtime._stops_a_head_it_started
def smoke(  # noqa: C901, PLR0911, PLR0912, PLR0913, PLR0915
    recipe: Recipe | dict[str, Any],
    *,
    sample: int = 10,
    data: str | None = None,
    executor: Any = None,  # noqa: ANN401
    output_dir: str | None = None,
    bootstrap_ray: bool = False,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the recipe on a bounded sample and return structured evidence.

    ``bootstrap_ray`` opts into auto-starting a correctly-configured local Ray
    head when none is reachable (see ``_ray.ensure_cluster``). The result carries a
    ``calibration`` block — measured per-stage resources extracted from this run
    (1C.2) — to feed the next ``run``/``smoke`` planner; an optional ``calibration``
    input seeds mode selection from a prior smoke.

    ``data`` is an optional assertion about the configured first source stage,
    not an input override. A mismatch is refused before execution.
    """
    rec = _verb_inputs._as_recipe(recipe).freeze()
    if isinstance(sample, bool) or not isinstance(sample, int) or sample <= 0:
        return {
            "status": "refused",
            "reason": "smoke sample must be a positive integer",
            "sample": sample,
            "config_hash": rec.config_hash,
        }
    checkpoint_refusal = _verb_validation._checkpoint_decision_refusal(rec)
    if checkpoint_refusal is not None:
        return _safety.redact(checkpoint_refusal)
    pviol = _safety.path_violations([data, output_dir, *_safety.recipe_path_params(rec)])
    if pviol:
        return {"status": "refused", "reason": "path(s) resolve outside the allowed workspace", "violations": pviol}
    binding = _verb_inputs._dataset_binding(rec, data)
    if _verb_inputs._binding_blocks_execution(binding, data):
        return _safety.redact(_verb_inputs._binding_refusal(binding))
    dp_obj = _verb_inputs._profile_binding(binding)
    if dp_obj is not None and _verb_inputs._profile_error(dp_obj):
        return _safety.redact(_verb_inputs._profile_refusal(binding, dp_obj))
    preflight_stages, preflight_build_issues = build_stages(rec)
    execution_target = (
        "custom_executor"
        if executor is not None
        else _verb_inputs._ray_execution_target(os.environ.get("RAY_ADDRESS"))
    )
    preflight_env = profiler.probe_env()
    if preflight_stages is None:
        from nemo_curator.stages.audio.audio_agent.diagnostics import diagnose_failure

        build_error = "; ".join(
            str(issue.get("message") or issue.get("code") or "") for issue in preflight_build_issues
        )
        return _safety.redact(
            {
                "status": "error",
                "reason": "smoke recipe could not be constructed",
                "issues": preflight_build_issues,
                "diagnosis": diagnose_failure(
                    build_error,
                    env=preflight_env,
                    operation="smoke",
                    phase="stage_construction",
                    execution_target=execution_target,
                ),
                "sample": sample,
                "config_hash": rec.config_hash,
                "data_binding": binding.to_dict(),
            }
        )
    from nemo_curator.stages.audio.audio_agent.diagnostics import environment_preflight

    environment_decision = environment_preflight(
        preflight_stages,
        preflight_env,
        operation="smoke",
        execution_target=execution_target,
    )
    if not environment_decision.get("can_execute", False):
        return _safety.redact(
            {
                "status": "refused",
                "reason_code": "environment_action_required",
                "reason": environment_decision.get("summary"),
                "environment_decision": environment_decision,
                "sample": sample,
                "config_hash": rec.config_hash,
                "data_binding": binding.to_dict(),
            }
        )
    rpt = SmokeReport(sample=sample)
    bound = _verb_smoke_inputs._bound_recipe(rec, sample, rpt, binding)
    if bound.recipe is not None:
        bound = _verb_smoke_inputs._isolate_smoke_outputs(bound, rpt)
    if bound.recipe is None:
        _verb_smoke_inputs._cleanup(list(bound.tmp_paths))
        return _safety.redact(
            {
                "status": "refused",
                "reason": f"smoke input cannot be safely bounded: {bound.error}",
                "sample": sample,
                "config_hash": rec.config_hash,
                "data_binding": binding.to_dict(),
            }
        )
    bounded = bound.recipe
    tmp_paths = list(bound.tmp_paths)

    stages, issues = build_stages(bounded)
    if stages is None:
        rpt.errors.extend(i.get("message", "") for i in issues)
        _verb_smoke_inputs._cleanup(tmp_paths)
        out = rpt.to_dict()
        out["status"] = "error"
        out["reason"] = "smoke recipe could not be constructed"
        out["config_hash"] = rec.config_hash
        out["data_binding"] = binding.to_dict()
        from nemo_curator.stages.audio.audio_agent.diagnostics import diagnose_failure

        out["diagnosis"] = diagnose_failure(
            "; ".join(rpt.errors),
            env=preflight_env,
            operation="smoke",
            phase="bounded_stage_construction",
            execution_target=execution_target,
        )
        return _safety.redact(out)
    try:
        write_issues = _verb_smoke_inputs._smoke_write_issues(stages, bound.output_root)
    except Exception as exc:  # noqa: BLE001 - output proof must fail closed
        write_issues = [f"disk-write contracts could not be fully inspected ({type(exc).__name__}: {exc})"]
    if write_issues:
        _verb_smoke_inputs._cleanup(tmp_paths)
        return _safety.redact(
            {
                "status": "refused",
                "reason": "smoke output isolation could not be proven",
                "issues": write_issues,
                "sample": sample,
                "config_hash": rec.config_hash,
                "data_binding": binding.to_dict(),
            }
        )
    pretrain_finalizer, pretrain_error = _verb_pretrain._pretrain_finalizer(stages)
    if pretrain_error:
        _verb_smoke_inputs._cleanup(tmp_paths)
        return _safety.redact(
            {
                "status": "refused",
                "reason": pretrain_error,
                "sample": sample,
                "config_hash": rec.config_hash,
                "data_binding": binding.to_dict(),
            }
        )

    data_profile = dp_obj.to_dict() if dp_obj else None
    profile_count = int((data_profile or {}).get("num_files", 0))
    rpt.input_count = (
        bound.input_count
        if bound.input_count is not None
        else min(sample, profile_count)
        if data_profile is not None
        else sample
    )
    caller_executor = executor
    owned_ray_address: str | None = None
    ray_address = os.environ.get("RAY_ADDRESS") if caller_executor is None else None
    if bootstrap_ray and caller_executor is None:
        try:
            from nemo_curator.stages.audio.audio_agent._ray import owns_cluster

            owned_before = owns_cluster()
            ray_address = _verb_runtime._bootstrap_ray()
            if not owned_before and owns_cluster(ray_address):
                owned_ray_address = ray_address
            rpt.notes.append("ray_head=" + ray_address)
        except Exception as exc:  # noqa: BLE001 - structured preflight failure
            _verb_smoke_inputs._cleanup(tmp_paths)
            from nemo_curator.stages.audio.audio_agent.diagnostics import diagnose_failure

            error_text = f"{type(exc).__name__}: {exc}"
            return _safety.redact(
                {
                    "status": "error",
                    "reason": ("Ray bootstrap failed before resource planning: " + error_text),
                    "diagnosis": diagnose_failure(
                        error_text,
                        stages=stages,
                        env=preflight_env,
                        operation="smoke",
                        phase="ray_bootstrap",
                        execution_target=execution_target,
                    ),
                    "sample": sample,
                    "config_hash": rec.config_hash,
                    "data_binding": binding.to_dict(),
                }
            )
    try:
        env_obj = _verb_runtime._resource_environment(
            preflight_env,
            ray_address,
            execution_target,
        )
        rplan = _verb_runtime._plan_resources(
            stages,
            env_obj,
            data_profile,
            calibration=calibration,
        )
        _verb_runtime._adapt_resource_plan_for_target(
            rplan,
            execution_target=execution_target,
            operation="smoke",
        )
    except Exception as exc:  # noqa: BLE001 - structured failure + guaranteed cleanup
        ray_cleanup = _verb_runtime._shutdown_owned_ray(owned_ray_address)
        _verb_smoke_inputs._cleanup(tmp_paths)
        from nemo_curator.stages.audio.audio_agent.diagnostics import diagnose_failure

        error_text = f"{type(exc).__name__}: {exc}"
        return _safety.redact(
            {
                "status": "error",
                "reason": ("smoke resource planning failed: " + error_text),
                "diagnosis": diagnose_failure(
                    error_text,
                    stages=stages,
                    env=preflight_env,
                    operation="smoke",
                    phase="resource_planning",
                    execution_target=execution_target,
                ),
                "sample": sample,
                "config_hash": rec.config_hash,
                "data_binding": binding.to_dict(),
                **({"ray_bootstrap_cleanup": ray_cleanup} if ray_cleanup is not None else {}),
            }
        )
    rpt.notes.append(f"resource_plan_mode={rplan.mode}")
    if rplan.escalations:
        rpt.notes.append("resource_escalations=" + "; ".join(rplan.escalations))
    if not getattr(rplan, "feasible", True):
        ray_cleanup = _verb_runtime._shutdown_owned_ray(owned_ray_address)
        _verb_smoke_inputs._cleanup(tmp_paths)
        from nemo_curator.stages.audio.audio_agent.diagnostics import diagnose_failure

        infeasible = "; ".join(rplan.escalations) or "resource plan infeasible"
        return _safety.redact(
            {
                "status": "refused",
                "reason": "smoke resource plan is infeasible on the selected execution target",
                "sample": sample,
                "config_hash": rec.config_hash,
                "escalations": list(rplan.escalations),
                "machine_plan": rplan.to_dict(),
                "diagnosis": diagnose_failure(
                    infeasible,
                    stages=stages,
                    env=env_obj,
                    operation="smoke",
                    phase="resource_planning",
                    execution_target=execution_target,
                ),
                "data_binding": binding.to_dict(),
                **({"ray_bootstrap_cleanup": ray_cleanup} if ray_cleanup is not None else {}),
            }
        )
    t0 = time.perf_counter()
    pretrain_prepared = False
    pretrain_output_rows: int | None = None
    runtime_diagnosis: dict[str, Any] | None = None
    try:
        if pretrain_finalizer is not None:
            pretrain_finalizer.prepare()
            pretrain_prepared = True
            rpt.notes.append("alm_pretrain_prepare=completed")
        results, _used_mode = _verb_runtime._run_pipeline_autofallback(
            stages,
            rplan.mode,
            caller_executor,
            note=rpt.notes.append,
            before_retry=_verb_runtime._automatic_retry_reset_hook(
                stages,
                extra_reset=(pretrain_finalizer.prepare if pretrain_finalizer is not None else None),
            ),
        )
        rpt.ran = True
        if pretrain_finalizer is not None:
            finalized_rows = pretrain_finalizer.finalize()
            if isinstance(finalized_rows, int) and not isinstance(finalized_rows, bool):
                pretrain_output_rows = max(0, finalized_rows)
            rpt.notes.append("alm_pretrain_finalize=completed")
        # ALM returns one origin stub when every snippet was filtered.  Only its
        # manifest is a serialized output, so count that instead of return
        # carriers or a zero-output smoke could receive a success token.
        rpt.retained = pretrain_output_rows if pretrain_output_rows is not None else _row_count(results)
        rpt.rejected = _report_data.source_rejection_count(
            bounded, min(sample, rpt.input_count), rpt.retained, stages=stages
        )
        rpt.per_stage_metrics = _verb_evidence._stage_metrics(results)
        sampled_rows = list(_verb_evidence._result_rows(results))
        rpt.examples = _verb_evidence._examples_from_rows(sampled_rows, limit=3)
        from nemo_curator.stages.audio.audio_agent.acceptance import parse_criteria, permits_empty_result

        rpt.goals_met = rpt.retained > 0 or permits_empty_result(
            parse_criteria(rec.acceptance_criteria), input_count=min(sample, rpt.input_count)
        )
        if rpt.retained == 0 and rpt.goals_met:
            rpt.notes.append("empty_sample_permitted_by_acceptance_contract; full-dataset yield is not established")
        incomplete_outputs = _verb_evidence._empty_required_outputs(rec, sampled_rows)
        if incomplete_outputs:
            # A stage may retain rows while omitting a required field or filling it
            # with no content (e.g. an ASR that emits blank transcripts).
            # retained>0 alone would wrongly read as success.
            rpt.goals_met = False
            rpt.notes.append("required output(s) MISSING or EMPTY in sampled rows: " + ", ".join(incomplete_outputs))
    except Exception as e:  # noqa: BLE001 - classify any execution failure for the critic
        if pretrain_finalizer is not None and pretrain_prepared:
            # Deliberately NOT finalizing here. ``finalize_audio_pretrain_outputs`` merges
            # unconditionally, so a failed attempt that produced no shards would replace a
            # previously completed output bundle with empty files. Abandoning this attempt's
            # partial shards is the lesser loss; say so, since the shards are left on disk.
            rpt.notes.append("alm_pretrain_partial_finalize=skipped_to_preserve_prior_output")
        error_text = f"{type(e).__name__}: {e}"
        rpt.errors.append(error_text)
        from nemo_curator.stages.audio.audio_agent.diagnostics import diagnose_failure

        runtime_diagnosis = diagnose_failure(
            error_text,
            stages=stages,
            env=env_obj,
            operation="smoke",
            phase="pipeline_execution",
            execution_target=execution_target,
        )
        rpt.notes.append(
            "failure_code=" + str((runtime_diagnosis.get("failure") or {}).get("code") or "unknown_failure")
        )
    finally:
        rpt.notes.append(f"elapsed_sec={round(time.perf_counter() - t0, 3)}")
        ray_cleanup = _verb_runtime._shutdown_owned_ray(owned_ray_address)
        if ray_cleanup is not None:
            rpt.notes.append("ray_bootstrap_cleanup=" + ("completed" if ray_cleanup else "failed"))
        _verb_smoke_inputs._cleanup(tmp_paths)
    out = rpt.to_dict()
    out["status"] = "completed" if rpt.ran and not rpt.errors else "error"
    out["config_hash"] = rec.config_hash
    out["data_binding"] = binding.to_dict()
    out["environment_decision"] = environment_decision
    if runtime_diagnosis is not None:
        out["diagnosis"] = runtime_diagnosis
    from nemo_curator.stages.audio.audio_agent import calibration as _cal

    out["calibration"] = _cal.from_smoke(out, machine_fingerprint=rplan.machine_fingerprint)
    # Keep the measurements where the next run can find them, so a plan is informed whether or
    # not the caller remembers --calibration. Meeting the sampled goals is deliberately NOT a
    # condition: what a stage used is a fact about resources, independent of whether the
    # sampled data satisfied the recipe.
    if rpt.ran and not rpt.errors and out["calibration"]:
        from nemo_curator.stages.audio.audio_agent import calibration_store

        out["calibration_stored"] = bool(
            calibration_store.save(
                rec.config_hash,
                out["calibration"],
                machine_fingerprint=rplan.machine_fingerprint,
            )
        )
    redacted = _verb_evidence._redact_result(out, rec)
    if output_dir:
        redacted["warnings"] = [
            "output_dir is a legacy no-op; smoke outputs are always redirected "
            "to an ephemeral sandbox. Configure output paths on recipe stages."
        ]
    # Surface the smoke token AFTER redaction: it is evidence the host must hand back
    # to run() (AUDIO_AGENT_REQUIRE_SMOKE), not a secret to hide from the host. redact()
    # would otherwise strip it because the key contains "token".
    if rpt.ran and rpt.goals_met is True and not rpt.errors:
        redacted["smoke_token"] = _safety.smoke_token(rec.config_hash)
    else:
        redacted["smoke_token_status"] = "not_issued: smoke must run without errors and meet sampled goals"  # noqa: S105
    return redacted
