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

"""Confirmed full pipeline execution."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from nemo_curator.stages.audio.audio_agent.recipe import build_stages

if TYPE_CHECKING:
    from nemo_curator.stages.audio.audio_agent import _verb_continuation

from nemo_curator.stages.audio.audio_agent import (
    _safety,
    _verb_evidence,
    _verb_inputs,
    _verb_outputs,
    _verb_pretrain,
    _verb_publish,
    _verb_runtime,
    _verb_validation,
    profiler,
)
from nemo_curator.stages.audio.audio_agent import (
    report as _report_data,
)
from nemo_curator.stages.audio.audio_agent.recipe import (
    Recipe,
)


@_verb_runtime._stops_a_head_it_started
def run(  # noqa: PLR0913, C901, PLR0911, PLR0912, PLR0915 - one verb, one keyword per execution knob (kept flat on purpose)
    recipe: Recipe | dict[str, Any],
    *,
    confirm: bool | str = False,
    data: str | None = None,
    executor: Any = None,  # noqa: ANN401
    output_dir: str | None = None,
    checkpoint_path: str | None = None,
    bootstrap_ray: bool = False,
    smoke_token: str | None = None,
    calibration: dict[str, Any] | None = None,
    goal: dict[str, Any] | None = None,
    reuse: dict[str, Any] | None = None,
    _continuation_context: _verb_continuation._ContinuationRunContext | None = None,
) -> dict[str, Any]:
    """Confirm-gated full run. Refuses without explicit confirmation (0 silent runs).

    ``confirm`` may be ``True`` or the recipe's ``config_hash`` (integrity: what
    was approved is what runs). Returns a refusal-with-estimate until confirmed.
    ``checkpoint_path`` enables partial-run recovery (resume completed source
    partitions on a rerun) when the pipeline's stages are resumability-safe.
    Atomically staged sinks refuse source resume because completed-source state
    is not transactional with publication of their output.
    ``bootstrap_ray`` opts into auto-starting a local Ray head when none is reachable.
    ``calibration`` is optional because a prior ``smoke`` of this exact recipe already
    stored its measurements: they are applied when none is passed, and the resource plan
    records that it planned from them.

    On success every step that persisted output is published as a content-addressed
    artifact, so a later request can reuse it instead of recomputing it
    (``REUSE_ARCHITECTURE.md``). ``goal`` records what the run was FOR (it is what makes a
    reuse candidate legible to a human later), and ``reuse`` carries the lineage when this
    run is itself the tail of a materialized continuation. Continuation identity
    is accepted only through an internal context: the core verifies one complete
    artifact against the original logical recipe and current source, then derives
    the publication tuples and complete run-record chain itself.

    ``data`` is an optional assertion about the configured first source stage,
    not an input override. A mismatch is refused before execution.
    """
    from nemo_curator.stages.audio.audio_agent import _verb_continuation

    rec = _verb_inputs._as_recipe(recipe).freeze()
    checkpoint_refusal = _verb_validation._checkpoint_decision_refusal(rec)
    if checkpoint_refusal is not None:
        return _safety.redact(checkpoint_refusal)
    if goal is not None and not isinstance(goal, dict):
        return {
            "status": "refused",
            "reason": (f"goal must be a JSON object/mapping, got {type(goal).__name__}"),
            "recipe_id": rec.recipe_id,
            "config_hash": rec.config_hash,
        }
    pviol = _safety.path_violations([data, output_dir, checkpoint_path, *_safety.recipe_path_params(rec)])
    if pviol:
        return {
            "status": "refused",
            "reason": "path(s) resolve outside the allowed workspace",
            "recipe_id": rec.recipe_id,
            "violations": pviol,
        }
    # Profiles before checking _binding_blocks_execution; ``smoke`` does the opposite. Safe
    # either way: _profile_binding returns None unless the binding resolved or is ambiguous, so
    # a missing/mismatched one reads nothing despite being profiled first. The orders differ
    # only for an ambiguous binding that also blocks -- wasted I/O, never a different answer.
    # Eight verbs share this preamble and they are not all identical.
    binding = _verb_inputs._dataset_binding(rec, data)
    dp_obj = _verb_inputs._profile_binding(binding)
    data_profile = dp_obj.to_dict() if dp_obj else None
    if dp_obj is not None and _verb_inputs._profile_error(dp_obj):
        return _safety.redact(_verb_inputs._profile_refusal(binding, dp_obj))

    approval_error = _verb_validation._approval_error(rec, confirm=confirm, smoke_token=smoke_token)
    if approval_error is not None:
        return {
            "status": "refused",
            "recipe_id": rec.recipe_id,
            "config_hash": rec.config_hash,
            "estimate": _verb_evidence._estimate(data_profile),
            "data_binding": binding.to_dict(),
            **approval_error,
        }
    checkpoint_refusals = _verb_validation._checkpoint_output_refusals(rec)
    if checkpoint_refusals:
        return {
            "status": "refused",
            "reason": "dedicated checkpoint output is not safe to create",
            "violations": checkpoint_refusals,
            "recipe_id": rec.recipe_id,
            "config_hash": rec.config_hash,
            "data_binding": binding.to_dict(),
        }
    if _verb_inputs._binding_blocks_execution(binding, data):
        return _safety.redact(_verb_inputs._binding_refusal(binding))

    logical_identity, lineage_error = _verb_continuation._verify_continuation_context(
        rec,
        binding,
        _continuation_context,
    )
    if lineage_error:
        return {
            "status": "refused",
            "reason": lineage_error,
            "data_binding": binding.to_dict(),
        }
    data_fp = None if logical_identity else (dp_obj.fingerprint() if dp_obj else None)
    dataset_key = (
        str(logical_identity["dataset_key"]) if logical_identity else (dp_obj.dataset_key() if dp_obj else "")
    )
    fingerprint_tier = (
        str(logical_identity["fingerprint_tier"]) if logical_identity else (dp_obj.fingerprint_tier if dp_obj else "")
    )
    recorded_data = logical_identity.get("data_source") if logical_identity else (binding.primary_path or data)

    stages, issues = build_stages(rec)
    if stages is None:
        from nemo_curator.stages.audio.audio_agent.diagnostics import diagnose_failure

        build_error = "; ".join(str(issue.get("message") or issue.get("code") or "") for issue in issues)
        return _safety.redact(
            {
                "status": "error",
                "recipe_id": rec.recipe_id,
                "issues": issues,
                "diagnosis": diagnose_failure(
                    build_error,
                    operation="run",
                    phase="stage_construction",
                ),
                "data_binding": binding.to_dict(),
            }
        )
    resume_sinks = _verb_outputs._atomic_source_resume_sinks(stages) if checkpoint_path is not None else []
    if resume_sinks:
        return _safety.redact(
            {
                "status": "refused",
                "reason_code": "source_resume_atomic_output_unsupported",
                "reason": (
                    "source checkpoint resume is unsupported with atomically staged outputs: "
                    "completed sources can outlive discarded output after failure. "
                    "Use a fresh run without checkpoint_path; "
                    "metadata checkpoint continuation is independent of source resume."
                ),
                "stages": resume_sinks,
                "recipe_id": rec.recipe_id,
                "config_hash": rec.config_hash,
                "data_binding": binding.to_dict(),
            }
        )
    pretrain_finalizer, pretrain_error = _verb_pretrain._pretrain_finalizer(stages)
    if pretrain_error:
        return _safety.redact(
            {
                "status": "refused",
                "reason": pretrain_error,
                "recipe_id": rec.recipe_id,
                "config_hash": rec.config_hash,
                "data_binding": binding.to_dict(),
            }
        )

    caller_executor = executor
    execution_target = (
        "custom_executor"
        if caller_executor is not None
        else _verb_inputs._ray_execution_target(os.environ.get("RAY_ADDRESS"))
    )
    preflight_env = profiler.probe_env()
    from nemo_curator.stages.audio.audio_agent.diagnostics import environment_preflight

    environment_decision = environment_preflight(
        stages,
        preflight_env,
        operation="run",
        execution_target=execution_target,
    )
    if not environment_decision.get("can_execute", False):
        return _safety.redact(
            {
                "status": "refused",
                "reason_code": "environment_action_required",
                "reason": environment_decision.get("summary"),
                "environment_decision": environment_decision,
                "recipe_id": rec.recipe_id,
                "config_hash": rec.config_hash,
                "data_binding": binding.to_dict(),
            }
        )
    owned_ray_address: str | None = None
    ray_address = os.environ.get("RAY_ADDRESS") if caller_executor is None else None
    if bootstrap_ray and caller_executor is None:
        try:
            from nemo_curator.stages.audio.audio_agent._ray import owns_cluster

            owned_before = owns_cluster()
            ray_address = _verb_runtime._bootstrap_ray()
            if not owned_before and owns_cluster(ray_address):
                owned_ray_address = ray_address
        except Exception as exc:  # noqa: BLE001 - structured preflight failure
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
                        operation="run",
                        phase="ray_bootstrap",
                        execution_target=execution_target,
                    ),
                    "recipe_id": rec.recipe_id,
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
        env = env_obj.to_dict()
        calibration, calibration_note = _verb_runtime._calibration_for_run(calibration, rec.config_hash)
        rplan = _verb_runtime._plan_resources(
            stages,
            env_obj,
            data_profile,
            calibration=calibration,
        )
        if calibration_note:
            rplan.notes.append(calibration_note)
        _verb_runtime._adapt_resource_plan_for_target(
            rplan,
            execution_target=execution_target,
            operation="run",
        )
    except Exception as exc:  # noqa: BLE001 - public verb returns structured failures
        ray_cleanup = _verb_runtime._shutdown_owned_ray(owned_ray_address)
        from nemo_curator.stages.audio.audio_agent.diagnostics import diagnose_failure

        error_text = f"{type(exc).__name__}: {exc}"
        return _safety.redact(
            {
                "status": "error",
                "reason": ("resource planning failed: " + error_text),
                "diagnosis": diagnose_failure(
                    error_text,
                    stages=stages,
                    env=preflight_env,
                    operation="run",
                    phase="resource_planning",
                    execution_target=execution_target,
                ),
                "recipe_id": rec.recipe_id,
                "config_hash": rec.config_hash,
                "data_binding": binding.to_dict(),
                **({"ray_bootstrap_cleanup": ray_cleanup} if ray_cleanup is not None else {}),
            }
        )
    rec.with_machine_plan(rplan.to_dict(), machine_fingerprint=rplan.machine_fingerprint)
    if not rplan.feasible:
        ray_cleanup = _verb_runtime._shutdown_owned_ray(owned_ray_address)
        from nemo_curator.stages.audio.audio_agent.diagnostics import diagnose_failure

        infeasible = "; ".join(rplan.escalations) or "resource plan infeasible"
        return _safety.redact(
            {
                "status": "refused",
                "reason": "resource plan is infeasible on the selected execution target",
                "recipe_id": rec.recipe_id,
                "config_hash": rec.config_hash,
                "escalations": rplan.escalations,
                "machine_plan": rplan.to_dict(),
                "diagnosis": diagnose_failure(
                    infeasible,
                    stages=stages,
                    env=env_obj,
                    operation="run",
                    phase="resource_planning",
                    execution_target=execution_target,
                ),
                "data_binding": binding.to_dict(),
                **({"ray_bootstrap_cleanup": ray_cleanup} if ray_cleanup is not None else {}),
            }
        )

    failures: list[dict[str, Any]] = []
    runtime_diagnosis: dict[str, Any] | None = None
    results: list[Any] | None = None
    used_mode: str | None = None
    pretrain_output_rows: int | None = None
    staged_file_outputs: list[_verb_outputs._StagedFileOutput] = []
    started_at = _verb_publish._utc_now()
    t0 = time.perf_counter()
    try:
        staged_file_outputs = _verb_outputs._prepare_staged_file_outputs(stages, checkpoint_path=checkpoint_path)
        if pretrain_finalizer is not None:
            pretrain_finalizer.prepare()
        results, used_mode = _verb_runtime._run_pipeline_autofallback(
            stages,
            rplan.mode,
            caller_executor,
            checkpoint_path=checkpoint_path,
            before_retry=_verb_runtime._automatic_retry_reset_hook(
                stages,
                staged_outputs=staged_file_outputs if checkpoint_path is not None else None,
                extra_reset=(pretrain_finalizer.prepare if pretrain_finalizer is not None else None),
            ),
        )
        if pretrain_finalizer is not None:
            finalized_rows = pretrain_finalizer.finalize()
            if isinstance(finalized_rows, int) and not isinstance(finalized_rows, bool):
                pretrain_output_rows = max(0, finalized_rows)
        _verb_outputs._promote_staged_file_outputs(staged_file_outputs)
    except Exception as e:  # noqa: BLE001 - classify + report, do not crash the caller
        # No finalize on the failure path: merging unconditionally would overwrite a prior
        # completed output bundle with empty files when this attempt produced no shards.
        from nemo_curator.stages.audio.audio_agent.diagnostics import diagnose_failure

        runtime_diagnosis = diagnose_failure(
            f"{type(e).__name__}: {e}",
            stages=stages,
            env=env_obj,
            operation="run",
            phase="pipeline_execution",
            execution_target=execution_target,
        )
        failures.append(dict(runtime_diagnosis.get("failure") or {}))
        _verb_outputs._discard_staged_file_outputs(staged_file_outputs)
    finally:
        _verb_outputs._restore_staged_file_outputs(staged_file_outputs)
    elapsed = time.perf_counter() - t0
    ended_at = _verb_publish._utc_now()
    if used_mode is not None and used_mode != rplan.mode:
        actual_plan = rplan.to_dict()
        actual_plan["planned_mode"] = rplan.mode
        actual_plan["mode"] = used_mode
        actual_plan.setdefault("notes", []).append(f"executor fell back from {rplan.mode} to {used_mode}")
        rec.with_machine_plan(
            actual_plan,
            machine_fingerprint=rplan.machine_fingerprint,
        )

    # Download-capable sources have no bytes to identify before their first
    # successful run. Re-resolve read-only after execution so that first run can
    # still publish artifacts under the dataset that was actually materialized.
    if not failures and dp_obj is None and binding.generated:
        refreshed = _verb_inputs._dataset_binding(rec, data)
        refreshed_profile = _verb_inputs._profile_binding(refreshed)
        if refreshed_profile is not None:
            binding = refreshed
            dp_obj = refreshed_profile
            data_profile = dp_obj.to_dict()
            if logical_identity is None:
                data_fp = dp_obj.fingerprint()
                dataset_key = dp_obj.dataset_key()
                fingerprint_tier = dp_obj.fingerprint_tier
                recorded_data = binding.primary_path or data

    output_paths = _verb_evidence._recipe_outputs(rec, output_dir)
    report_obj = _report_data.build_run_report(
        recipe=rec,
        result_tasks=results,
        data_profile=data_profile,
        env_profile=env,
        output_paths=output_paths,
        elapsed_sec=elapsed,
        failures=failures,
        examples=_verb_evidence._examples(results, limit=5) if results else [],
        next_action="review retained/rejected; adjust thresholds and re-run if needed"
        if not failures
        else "triage the failure_reasons and re-validate",
    )
    if pretrain_output_rows is not None:
        report_obj.accepted = pretrain_output_rows
    source_recipe = Recipe.from_dict(_continuation_context.logical_recipe) if logical_identity else rec
    if logical_identity:
        report_obj.input_count = logical_identity["source_items"]
    terminal_outputs, cardinality_proven = _verb_evidence._terminal_evidence_outputs(
        rec,
        list(getattr(report_obj, "output_paths", []) or []),
    )
    # Read the output back on EVERY run, not only one that declared a success contract. The
    # counts below are what tell a caller its 4-row manifest has 3 blank rows, and gating them
    # on acceptance criteria left the runs nobody was checking as the only ones reporting
    # nothing. The scan streams and keeps counts, so it costs a pass over the manifest the run
    # just wrote -- and it is now that pass rather than a second one: _acceptance_result is
    # handed this result instead of repeating the read.
    _preview, output_scan = _verb_evidence._scan_terminal_output(terminal_outputs, limit=0)
    report_obj.source_items = int(getattr(report_obj, "input_count", 0))
    report_obj.output_rows = int(getattr(report_obj, "accepted", 0))
    report_obj.output_rows_written = _report_data.rows_written_in(output_scan)
    report_obj.sparse_fields = _report_data.sparse_fields_in(output_scan)
    report_obj.cardinality_proven = cardinality_proven
    report_obj.rejected = (
        _report_data.source_rejection_count(
            source_recipe, report_obj.source_items, report_obj.output_rows, stages=None if logical_identity else stages
        )
        if cardinality_proven
        else None
    )

    from nemo_curator.stages.audio.audio_agent import run_store

    run_id = run_store.new_run_id(rec.config_hash)
    # Declared before the derivation below so that one can report its own failure through the
    # same channel the publication step already uses, rather than returning empty in silence.
    provenance: dict[str, Any] = {
        "run_record_persisted": False,
        "artifacts_published": 0,
        "warnings": [],
    }
    persistence_warnings: list[str] = provenance["warnings"]
    roles, keys = _verb_publish._produced_roles_keys(stages, data_profile, warnings=persistence_warnings)
    acceptance_result = _verb_evidence._acceptance_result(
        rec,
        report_obj,
        roles,
        keys,
        list(getattr(report_obj, "output_paths", []) or []),
        output_scan=output_scan,
        source_recipe=source_recipe,
        finalized_output=(pretrain_finalizer.manifest_path, pretrain_output_rows)
        if pretrain_finalizer is not None and pretrain_output_rows is not None
        else None,
    )
    published: list[dict[str, Any]] = []
    if not failures:
        # Only a run that completed may publish: a crashed run's partial output must never
        # acquire the _COMPLETE marker that makes it look reusable.
        published = _verb_publish._publish_artifacts(
            rec,
            stages,
            dataset_key=dataset_key,
            fingerprint_tier=fingerprint_tier,
            per_stage=getattr(report_obj, "per_stage_metrics", {}) or {},
            run_id=run_id,
            input_count=int(getattr(report_obj, "input_count", 0)),
            data_profile=data_profile,
            # A continuation's profile describes the artifact it physically read, not the corpus
            # its logical dataset key names, so recording that as the corpus coverage would
            # describe one dataset with another's file list. Withholding it costs a later delta
            # (which says so by name) instead of computing one against the wrong inventory.
            inventory=(None if logical_identity else _verb_publish._consumed_inventory(rec, dp_obj)),
            started_at=started_at,
            ended_at=ended_at,
            elapsed_sec=elapsed,
            step_identity=(list(logical_identity["step_identity"]) if logical_identity else None),
            persistence_warnings=persistence_warnings,
        )
    provenance["artifacts_published"] = len(published)
    lineage = {**(reuse or {}), "published": published}
    _verb_publish._record_run(
        rec,
        run_id=run_id,
        data=recorded_data,
        data_fp=data_fp,
        dataset_key=dataset_key,
        fingerprint_tier=fingerprint_tier,
        report=report_obj,
        failed=bool(failures),
        goal=goal,
        elapsed=elapsed,
        env=env,
        produced_roles=roles,
        produced_keys=keys,
        acceptance_result=acceptance_result,
        reuse=lineage,
        logical_steps=(list(logical_identity["logical_steps"]) if logical_identity else None),
        persistence_status=provenance,
    )
    result = {
        "status": "completed" if not failures else "failed",
        "run_id": run_id,
        "report": report_obj.to_dict(),
        "acceptance": acceptance_result,
        "machine_plan": rec.machine_plan,
        "environment_decision": environment_decision,
        "reuse": lineage,
        "provenance": provenance,
        "data_binding": {
            **binding.to_dict(),
            **(
                {
                    "logical_dataset_key": dataset_key,
                    "logical_data_source": recorded_data,
                }
                if logical_identity
                else {}
            ),
        },
    }
    if runtime_diagnosis is not None:
        result["diagnosis"] = runtime_diagnosis
    warnings = list(persistence_warnings)
    if output_dir:
        warnings.append(
            "output_dir is a legacy no-op and was not reported as an output; configure output paths on recipe stages."
        )
    if warnings:
        result["warnings"] = warnings
    ray_cleanup = _verb_runtime._shutdown_owned_ray(owned_ray_address)
    if ray_cleanup is not None:
        result["ray_bootstrap_cleanup"] = ray_cleanup
        if not ray_cleanup:
            result.setdefault("warnings", []).append("the locally bootstrapped Ray head could not be stopped safely")
    return _verb_evidence._redact_result(result, rec)
