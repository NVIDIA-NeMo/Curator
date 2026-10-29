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

"""Recipe validation and execution approval checks."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit

from nemo_curator.stages.audio.audio_agent import _safety, _verb_inputs, profiler
from nemo_curator.stages.audio.audio_agent.contracts import Issue, Verdict
from nemo_curator.stages.audio.audio_agent.recipe import (
    OUTPUT_LOCATION_PARAMS,
    REUSABLE_CHECKPOINT_PROVENANCE,
    Recipe,
    build_stages,
)


def _derive_initial(data_profile: dict[str, Any] | None) -> tuple[set[str], set[str]]:
    """Seed initial roles + literal keys from the input data profile."""
    from nemo_curator.stages.audio._agent._roles import role_for_value

    keys: set[str] = {"audio_filepath"}
    if data_profile and data_profile.get("manifest_keys"):
        keys |= set(data_profile["manifest_keys"])
    roles = {role_for_value(k) for k in keys} | {"audio_filepath"}
    roles.discard("unknown")
    return roles, keys


def validate(  # noqa: C901, PLR0912, PLR0913, PLR0915
    recipe: Recipe | dict[str, Any],
    *,
    data: str | None = None,
    initial_keys: list[str] | None = None,
    initial_roles: list[str] | None = None,
    initial_tensor_keys: list[str] | None = None,
    expected_outputs: list[str] | None = None,
    acceptance_criteria: list[dict[str, Any]] | None = None,
    request_type: str | None = None,
) -> dict[str, Any]:
    """Validate a recipe: does it compose, and can it run in this environment?

    Well-formedness is checked here (real stages, constructible params); the rest
    runs through the pluggable check registry (``audio_agent.checks``): data-flow
    (role/key/residency/serialization), card constraints, environment gates,
    unproducible roles, task-type, output-completeness, and request-type sanity.
    ``expected_outputs`` (semantic roles) enables the output-completeness check;
    ``acceptance_criteria`` (1A.1) additionally compile their output/metric fields
    into that check and drive request-type sanity via ``request_type``. They are
    a cross-check only: the same success contract must be embedded in the recipe
    so smoke/run and the confirmation hash cannot lose it.

    Mechanical progression requires returned ``runnable`` (no error-severity
    problem anywhere) or ``status == "pass"`` -- NOT ``ok``, which is data-flow
    only. Before smoke or confirmation, the host must also complete the returned
    ``semantic_review`` and reach an intent-level pass.

    ``output_targets`` states what is already at each location the recipe writes to, so an
    occupied output can be REPORTED to the user rather than guessed at or quietly cleared.

    ``semantic_review`` is a separate, advisory evidence packet for the host LLM.
    It co-locates configured field lineage, cardinality seams, and card prose but
    never claims that the recipe expresses the user's intent.

    The first supported source stage's parameters are execution truth. ``data``
    and ``Recipe.inputs`` are optional assertions and never rewrite that stage.
    ``initial_tensor_keys`` disambiguates genuinely resident input tensors from
    identically named JSON manifest columns; manifest-backed validation defaults
    it to empty because a JSON row cannot carry a framework tensor.
    """
    from nemo_curator.stages.audio.audio_agent.acceptance import expected_roles_from_criteria, parse_criteria
    from nemo_curator.stages.audio.audio_agent.checks import CheckContext, run_checks

    rec = _verb_inputs._as_recipe(recipe)
    verdict = Verdict()

    if not rec.stages:
        verdict.issues.append(Issue("empty_recipe", "error", "recipe has no stages"))
        return verdict.to_dict()

    pviol = _safety.path_violations([data, *_safety.recipe_path_params(rec)])
    if pviol:
        verdict.issues.append(
            Issue(
                "path_outside_workspace",
                "error",
                "; ".join(pviol),
                fix="move the source/output under AUDIO_AGENT_WORKSPACE or change that lock",
            )
        )
        return verdict.to_dict()
    binding = _verb_inputs._dataset_binding(rec, data)
    verdict.data_binding = binding.to_dict()
    binding_issue = _verb_inputs._binding_issue(binding, data)
    if binding_issue is not None:
        verdict.issues.append(binding_issue)
    dp_obj = _verb_inputs._profile_binding(binding)
    data_profile = dp_obj.to_dict() if dp_obj else None
    if dp_obj is not None and _verb_inputs._profile_error(dp_obj):
        verdict.issues.append(
            Issue(
                "data_source_unreadable",
                "error",
                _verb_inputs._profile_error(dp_obj),
                stage_index=binding.source_index,
                stage=binding.source_ref,
                fix="repair or replace the manifest before validation or execution",
            )
        )
    reusable_dataset_key = (
        dp_obj.dataset_key() if dp_obj is not None and not _verb_inputs._profile_error(dp_obj) else None
    )
    for reason in _checkpoint_output_refusals(
        rec,
        reusable_dataset_key=reusable_dataset_key,
    ):
        verdict.issues.append(
            Issue(
                "checkpoint_output_occupied",
                "error",
                reason,
                fix="choose a new, distinct, versioned local JSONL path for ManifestCheckpointStage",
            )
        )
    env = profiler.probe_env()
    execution_target = _verb_inputs._ray_execution_target(os.environ.get("RAY_ADDRESS"))

    stages, build_issues = build_stages(rec)
    verdict.issues.extend(_issue_from_dict(i) for i in build_issues)
    if stages is None:
        from nemo_curator.stages.audio.audio_agent.diagnostics import diagnose_failure

        verdict.diagnosis = diagnose_failure(
            "; ".join(str(issue.get("message") or issue.get("code") or "") for issue in build_issues),
            env=env,
            operation="validate",
            phase="stage_construction",
            execution_target=execution_target,
        )
        return verdict.to_dict()

    roles0, keys0 = _derive_initial(data_profile)
    if initial_roles is not None:
        roles0 = set(initial_roles)
    if initial_keys is not None:
        keys0 = set(initial_keys)
    if initial_tensor_keys is not None:
        tensor_keys0: set[str] | None = set(initial_tensor_keys)
    elif data_profile is not None:
        # A profiled source is JSON/file backed. Its schema may contain a nullable
        # ``waveform`` column, but key presence alone does not make that value a
        # resident framework tensor.
        tensor_keys0 = set()
    else:
        # No source evidence: preserve the foundation's conservative inference
        # for callers validating an already-resident in-memory task.
        tensor_keys0 = None

    recipe_criteria = parse_criteria(rec.acceptance_criteria)
    if acceptance_criteria is None:
        criteria = recipe_criteria
    else:
        criteria = parse_criteria(acceptance_criteria)
        if criteria != recipe_criteria:
            verdict.issues.append(
                Issue(
                    "acceptance_contract_not_embedded",
                    "error",
                    "the criteria passed to validate differ from the recipe's "
                    "acceptance_criteria, so a later smoke/run would not be bound "
                    "to the success contract that was validated",
                    fix=("copy the complete acceptance_criteria list into recipe.yaml and validate that same recipe"),
                )
            )
    expected = set(expected_outputs or []) | set(expected_roles_from_criteria(criteria))

    ctx = CheckContext(
        recipe=rec,
        stages=stages,
        data_profile=data_profile,
        env=env,
        initial_roles=roles0,
        initial_keys=keys0,
        available_gpus=float(env.gpu_count) if env.has_gpu else 0.0,
        initial_tensor_keys=tensor_keys0,
        expected_outputs=sorted(expected),
        acceptance_criteria=criteria,
        request_type=request_type,
        execution_target=execution_target,
    )
    result = run_checks(ctx)
    verdict.ok = bool(result.ok)
    verdict.keys_ok = bool(result.keys_ok)
    verdict.produced_roles = result.produced_roles
    verdict.produced_keys = result.produced_keys
    verdict.issues.extend(result.issues)
    verdict.card_violations.extend(result.card_violations)
    verdict.gate_flags.extend(result.gate_flags)
    verdict.unproducible_roles = result.unproducible_roles

    # Mechanical validation cannot decide whether a valid field/filter/model
    # expresses open-ended user intent.  Assemble the exact configured lineage
    # and its card prose for the mandatory host-LLM critic, without letting an
    # advisory packet failure take down the deterministic validator.
    semantic_response: dict[str, Any] | None = None
    try:
        from nemo_curator.stages.audio.audio_agent.semantic_review import (
            build_semantic_review,
            semantic_response_contract,
        )

        semantic_response = semantic_response_contract()
        verdict.semantic_review = build_semantic_review(
            stages,
            initial_keys=keys0,
            recipe=rec,
            data_profile=data_profile,
        )
    except Exception as exc:  # noqa: BLE001 - validation stays available; packet says it is incomplete
        verdict.semantic_review = {
            "status": "unavailable",
            "review_required": True,
            "advisory_only": True,
            "intent_interpretation_performed": False,
            "required_response": semantic_response,
            "contract_issues": [
                {
                    "code": "semantic_review_context_unavailable",
                    "message": _safety.redact_secret_text(f"{type(exc).__name__}: {exc}"),
                }
            ],
            "checklist": [
                {
                    "id": "missing_review_context",
                    "required": True,
                    "instruction": (
                        "Do not infer intent from a mechanically runnable verdict; "
                        "retrieve the configured stage cards and lineage manually."
                    ),
                }
            ],
        }

    from nemo_curator.stages.audio.audio_agent.diagnostics import (
        environment_preflight,
        verdict_issues,
    )

    environment_decision = environment_preflight(
        stages,
        env,
        operation="validate",
        execution_target=execution_target,
    )
    verdict.environment_decision = environment_decision
    for env_issue in verdict_issues(environment_decision):
        matches = [issue for issue in verdict.gate_flags if issue.code == env_issue.code]
        if matches:
            # Keep the existing per-stage location, but promote a proven,
            # recipe-relevant execution blocker to the shared safe severity.
            for issue in matches:
                issue.severity = "error"
                issue.fix = env_issue.fix
                issue.escalate_to = "user"
        else:
            verdict.gate_flags.append(env_issue)
    # Soft authoring preferences never participate in mechanical runnability.
    # Their advisory builder fails closed: incomplete equivalence evidence means
    # an empty list, never a guessed alternative or a validation error.
    authoring_valid = not any(
        issue.severity == "error" for pool in (verdict.issues, verdict.card_violations) for issue in pool
    )
    if authoring_valid:
        try:
            from nemo_curator.stages.audio.audio_agent.planning_advisory import (
                build_planning_advisories,
            )

            verdict.planning_advisories = build_planning_advisories(
                rec,
                stages,
                initial_keys=keys0,
                data_profile=data_profile,
            )
        except Exception:  # noqa: BLE001 - optional advice cannot break validate
            verdict.planning_advisories = []
    verdict.output_targets = _output_targets(rec)
    return verdict.to_dict()


def _output_targets(rec: Recipe) -> list[dict[str, Any]]:
    """What is already sitting at each location this recipe writes to.

    Facts only, no predictions: whether it exists, whether it is a file or a directory, and how
    much is in it. An agent that cannot see this has to guess, and guessing led somewhere bad --
    reading the writer's append-mode open, concluding reruns would double, and deleting the
    user's file before the confirm gate. The path was in fact replaced cleanly by the run.

    Reporting an occupied path is the whole point; deciding what to do about it belongs to the
    user, and clearing it belongs to nobody (the pipeline replaces its own manifest output).
    """
    from fsspec.core import url_to_fs

    from nemo_curator.stages.audio.audio_agent import _verb_evidence

    targets: list[dict[str, Any]] = []
    for path in _verb_evidence._recipe_outputs(rec, None):
        options = next(
            (stage.params.get("storage_options") or {} for stage in rec.stages if path in stage.params.values()),
            {},
        )
        entry: dict[str, Any] = {"path": path, "exists": None}
        try:
            fs, resolved = url_to_fs(os.path.expanduser(path), **options)
            entry["exists"] = fs.exists(resolved)
            if entry["exists"]:
                if fs.isdir(resolved):
                    entry["kind"] = "directory"
                    entry["files"] = len(fs.find(resolved, withdirs=False))
                else:
                    entry["kind"] = "file"
                    entry["bytes"] = fs.info(resolved).get("size")
                    if resolved.endswith((".jsonl", ".json")):
                        with fs.open(resolved, "rt", encoding="utf-8") as handle:
                            entry["rows"] = sum(1 for line in handle if line.strip())
                entry["note"] = "already present; the run writes here. Report this -- do not clear it yourself."
        except Exception as exc:  # noqa: BLE001 - an inaccessible target is unknown, never proven absent
            entry["probe_error"] = _safety.redact_secret_text(f"{type(exc).__name__}: {exc}")
            entry["note"] = "output occupancy could not be fully inspected; verify before approving replacement."
        targets.append(entry)
    return targets


def _checkpoint_output_refusals(  # noqa: C901 - one branch per checkpoint path invariant
    rec: Recipe,
    *,
    reusable_dataset_key: str | None = None,
) -> list[str]:
    """Fail-closed execution checks unique to the dedicated reusable checkpoint.

    Ordinary writers intentionally replace their output. A checkpoint is different: an existing
    path may be the only reusable copy of an earlier run, so silently truncating it destroys the
    evidence this feature exists to retain. Planning catches this first; run repeats it to close
    the race between approval and execution.

    Validation is read-only and may inspect an occupied checkpoint only when
    ``reusable_dataset_key`` proves that the exact recipe prefix has a complete,
    content-bound artifact for this dataset. Execution deliberately omits that
    evidence and therefore still refuses every occupied destination.
    """

    outputs: dict[str, list[str]] = {}
    checkpoint_paths: list[tuple[int, str, str]] = []
    for index, stage in enumerate(rec.stages):
        for key in OUTPUT_LOCATION_PARAMS:
            raw = stage.params.get(key)
            if not isinstance(raw, str) or not raw:
                continue
            parsed = urlsplit(raw)
            canonical = os.path.realpath(os.path.expanduser(raw)) if not parsed.scheme else raw
            outputs.setdefault(canonical, []).append(f"{stage.ref}[{index}].{key}")
            if stage.ref == "ManifestCheckpointStage" and key == "output_path":
                checkpoint_paths.append((index, raw, canonical))
    reasons: list[str] = []
    for index, raw, canonical in checkpoint_paths:
        parsed = urlsplit(raw)
        if parsed.scheme:
            reasons.append(f"{raw!r}: dedicated checkpoint reuse requires a plain local path, not a URI")
            continue
        users = outputs.get(canonical, [])
        if len(users) > 1:
            reasons.append(f"{raw!r}: checkpoint output collides with {', '.join(users)}")
        if os.path.exists(canonical):
            reusable_reason = _configured_reusable_checkpoint_reason(
                rec,
                checkpoint_index=index,
                canonical_path=canonical,
                dataset_key=reusable_dataset_key,
            )
            if reusable_reason:
                reasons.append(f"{raw!r}: {reusable_reason}")
        elif os.path.exists(f"{canonical}._COMPLETE"):
            reasons.append(
                f"{raw!r}: a stale completion marker exists without its checkpoint; "
                "choose a new versioned path or have the user remove the stale marker"
            )
    return reasons


def _configured_reusable_checkpoint_reason(  # noqa: PLR0911 - one refusal per artifact proof
    rec: Recipe,
    *,
    checkpoint_index: int,
    canonical_path: str,
    dataset_key: str | None,
) -> str:
    """Why an occupied checkpoint is not proven safe for read-only validation."""
    if not dataset_key:
        return (
            "checkpoint output already exists and no matching dataset identity "
            "is available; choose a new versioned path instead of overwriting retained work"
        )

    from nemo_curator.stages.audio.audio_agent import artifacts

    plans = artifacts.plan_steps(rec, dataset_key)
    if checkpoint_index >= len(plans):
        return "checkpoint prefix identity could not be derived"
    plan = plans[checkpoint_index]
    if plan.stage_ref != "ManifestCheckpointStage":
        return "checkpoint prefix identity resolved to a different stage"
    planned_path = os.path.realpath(os.path.expanduser(plan.uri))
    if planned_path != canonical_path:
        return "checkpoint artifact path does not match the configured recipe prefix"
    artifact, invalid = artifacts.lookup(plan.step_key, dataset_key=dataset_key)
    if artifact is None:
        return "checkpoint output exists but no matching artifact record proves it complete"
    artifact_path = os.path.realpath(os.path.expanduser(artifact.uri))
    if artifact_path != canonical_path:
        return "matching checkpoint record points at a different output path"
    if invalid:
        return "checkpoint artifact is not reusable: " + "; ".join(invalid)
    return ""


def _is_reusable_checkpoint_plan(rec: Recipe) -> bool:
    """Whether this exact recipe opted into the planner's stricter run gate."""
    return any(
        stage.ref == "ManifestCheckpointStage"
        and stage.params.get("planning_provenance") == REUSABLE_CHECKPOINT_PROVENANCE
        for stage in rec.stages
    )


def _approval_error(rec: Recipe, *, confirm: bool | str, smoke_token: str | None) -> dict[str, Any] | None:
    """Check the caller's evidence against the exact approved recipe before deriving execution."""
    checkpoint_plan = _is_reusable_checkpoint_plan(rec)
    confirm_with = (
        f"pass confirm={rec.config_hash!r} to proceed"
        if checkpoint_plan
        else f"pass confirm={rec.config_hash!r} (or confirm=True) to proceed"
    )
    if confirm is not True and not isinstance(confirm, str):
        return {
            "reason": "full run requires explicit confirmation (0 silent full-scale runs)",
            "confirm_with": confirm_with,
        }
    if checkpoint_plan and confirm is True:
        return {
            "reason": "checkpoint-planned recipes require exact-hash approval; bare confirm=True is not authoritative",
            "confirm_with": confirm_with,
        }
    if isinstance(confirm, str) and confirm != rec.config_hash:
        return {
            "reason": "plan-execution integrity check failed: confirmed hash does not match the recipe",
            "confirmed": confirm,
        }
    if (checkpoint_plan or _safety.require_smoke()) and not _safety.verify_smoke_token(smoke_token, rec.config_hash):
        return {
            "reason": (
                "checkpoint-planned recipes require authoritative smoke evidence for this exact config_hash: "
                "run smoke on this recipe and pass its 'smoke_token'"
                if checkpoint_plan
                else "run requires smoke evidence (AUDIO_AGENT_REQUIRE_SMOKE is set): "
                "run smoke on this recipe and pass its 'smoke_token'"
            )
        }
    return None


def _checkpoint_decision_refusal(rec: Recipe) -> dict[str, Any] | None:
    """Block smoke/run until a recommended checkpoint is accepted or declined."""
    from nemo_curator.stages.audio.audio_agent import reusable_pipeline

    requirement = reusable_pipeline.checkpoint_decision_requirement(rec)
    if requirement is None:
        return None
    return {
        "status": "refused",
        "recipe_id": rec.recipe_id,
        **requirement,
    }


def _issue_from_dict(d: dict[str, Any]) -> Issue:
    return Issue(
        code=d.get("code", "error"),
        severity=d.get("severity", "error"),
        message=d.get("message", ""),
        stage_index=d.get("stage_index"),
        stage=d.get("stage"),
        fix=d.get("fix"),
    )


def _carry_approval(
    approved: Recipe,
    derived: Recipe,
    *,
    confirm: bool | str,
    smoke_token: str | None,
) -> tuple[bool | str, str | None, str]:
    """Carry an approval given for ``approved`` onto the ``derived`` recipe that actually runs.

    Returns ``(confirm, smoke_token, refusal)``; a non-empty refusal means stop.

    Continuation and delta both execute a recipe THIS module derives -- a reader on the reused
    artifact plus the tail, or the prefix narrowed to the changed files. Its ``config_hash`` is
    one no human has ever seen, and holding the caller's evidence against it made both gates
    unsatisfiable rather than strict: ``continue --confirm <hash>`` (the form ``AGENTS.md``
    documents) was always refused, and under ``AUDIO_AGENT_REQUIRE_SMOKE`` no ``delta_run``
    could succeed at all, because the surface exposes no way to smoke a prefix recipe.

    The gate exists to bind INTENT. What the user read and approved is ``approved``; ``derived``
    is a deterministic function of it plus an artifact already verified against the registry, so
    the honest check is the human's evidence against the human's recipe -- after which the
    derived recipe may carry its own. This does not widen the gate: an unconfirmed call is left
    for ``run`` to refuse exactly as before, and a wrong hash or a missing/forged token still
    stops here.
    """
    error = _approval_error(approved, confirm=confirm, smoke_token=smoke_token)
    if error is not None:
        return confirm, smoke_token, error["reason"]
    carried = derived.config_hash if isinstance(confirm, str) else confirm
    return carried, _safety.smoke_token(derived.config_hash), ""
