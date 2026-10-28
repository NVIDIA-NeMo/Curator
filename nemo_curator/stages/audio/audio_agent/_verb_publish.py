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

"""Artifact publication and run recording."""

from __future__ import annotations

import contextlib
import os
import time
from typing import Any

from nemo_curator.stages.audio.audio_agent import _reuse_summary, _safety
from nemo_curator.stages.audio.audio_agent import report as _report_data
from nemo_curator.stages.audio.audio_agent.recipe import (
    GENERATED_ASSET_PARAMS,
    Recipe,
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _produced_roles_keys(
    stages: list[Any],
    data_profile: dict[str, Any] | None,
    *,
    warnings: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Cumulative semantic roles + literal keys the built pipeline produces.

    Same derivation the resume-safety guard uses, so an artifact advertises exactly the roles
    a later suffix will be re-validated against.

    A failure still returns ``([], [])`` -- provenance detail is never worth failing a completed
    run over -- but it now SAYS so through ``warnings``. Empty and unknown are indistinguishable
    downstream: ``acceptance._completeness`` reads an empty declared set as "no producer
    evidence" and reports the vaguer of two true-ish answers, so a catalogue-wide contract
    failure would quietly cost every acceptance message its precision with nothing anywhere
    recording that the derivation broke.

    Note this is NOT the artifact's ``produced_roles``: ``_publish_artifacts`` derives those
    itself, seeded from ``_derive_initial`` and suppressing failures per stage, so one
    unreadable contract cannot empty them. What this feeds is the run record and the acceptance
    evidence.
    """
    from nemo_curator.stages.audio.audio_agent import _verb_validation

    try:
        from nemo_curator.stages.audio import agent as foundation
        from nemo_curator.stages.audio._agent._conformance import produced_roles

        roles, keys = _verb_validation._derive_initial(data_profile)
        for st in stages:
            c = foundation.build_contract(st)
            roles |= produced_roles(c)
            keys |= set(c.writes.data_keys) | set(c.writes.segment_data_keys)
        roles.discard("unknown")
        return sorted(roles), sorted(keys)
    except Exception as exc:  # noqa: BLE001 - provenance detail, never worth failing a completed run
        if warnings is not None:
            warnings.append(
                "produced roles/keys could not be derived, so the run record and the acceptance "
                f"evidence report none rather than unknown: {type(exc).__name__}: {exc}"
            )
        return [], []


def _stage_cost(per_stage: dict[str, Any], stage: Any) -> tuple[float, float]:  # noqa: ANN401
    """``(duration_sec, gpu_seconds)`` measured for one stage, 0 when unmeasured.

    Real numbers only -- an unmeasured stage reports 0 rather than a share of the total, so a
    "time saved" estimate is never inflated by guesswork.
    """
    duration = _report_data.stage_duration_sec(per_stage, str(getattr(stage, "name", "") or ""))
    gpus = float(getattr(getattr(stage, "resources", None), "gpus", 0) or 0)
    return round(duration, 3), round(duration * gpus, 3)


def _consumed_inventory(rec: Recipe, dp: Any) -> dict[str, str] | None:  # noqa: ANN401 - DataProfile
    """The files this run actually read, which is what its artifacts cover.

    Normally every file the profiler found. A run whose source was narrowed to a named subset
    (``include_files``, how a delta processes only what changed) consumed only those, and
    recording the whole corpus as its coverage would claim results it never computed.
    """
    if dp is None or not dp.inventory:
        return None
    named = rec.stages[0].params.get("include_files") if rec.stages else None
    if not isinstance(named, list):
        return dict(dp.inventory)
    root = dp.inventory_root or ""
    wanted = {os.path.relpath(os.path.abspath(os.path.expanduser(str(p))), root) for p in named}
    return {rel: token for rel, token in dp.inventory.items() if rel in wanted}


def _publish_artifacts(  # noqa: PLR0913, C901, PLR0912, PLR0915 - publishing gathers the run's whole context
    rec: Recipe,
    stages: list[Any],
    *,
    dataset_key: str,
    fingerprint_tier: str,
    per_stage: dict[str, Any],
    run_id: str,
    input_count: int,
    data_profile: dict[str, Any] | None,
    inventory: dict[str, str] | None = None,
    started_at: str,
    ended_at: str,
    elapsed_sec: float = 0.0,
    step_identity: list[tuple[str, str, int]] | None = None,
    persistence_warnings: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Register every step that actually persisted output as a reusable artifact.

    A step with no output-location param writes nothing, so it can never be a resume point and
    gets no artifact -- reuse always resumes from disk. ``step_identity`` overrides the
    ``(step_key, input_key, stage_index)`` of each step positionally, which is how a
    materialized continuation registers its tail under the identity of the pipeline the user
    actually asked for. Best-effort throughout: a bookkeeping problem must not turn a
    successful run into a failure.
    """
    from nemo_curator.stages.audio.audio_agent import _verb_evidence, _verb_validation

    # Unknown input is not a reusable namespace.  Publishing under ``""`` would
    # let a later unknown input look identical merely because neither was
    # profiled.
    if not dataset_key:
        if persistence_warnings is not None:
            persistence_warnings.append("artifacts were not published because source data identity is unavailable")
        return []

    try:
        from nemo_curator.stages.audio import agent as foundation
        from nemo_curator.stages.audio._agent._conformance import produced_roles as _roles_of
        from nemo_curator.stages.audio.audio_agent import artifacts as art_mod
    except Exception as exc:  # noqa: BLE001
        if persistence_warnings is not None:
            persistence_warnings.append(f"artifact registry could not be loaded: {type(exc).__name__}: {exc}")
        return []

    out: list[dict[str, Any]] = []
    try:
        plans = art_mod.plan_steps(rec, dataset_key)
    except Exception as exc:  # noqa: BLE001
        if persistence_warnings is not None:
            persistence_warnings.append(f"artifact publication plan could not be built: {type(exc).__name__}: {exc}")
        return []

    # Provenance, resolved once per run rather than per artifact. Both degrade to "" and are
    # never consulted by reuse, so a failure here costs traceability, not correctness.
    from nemo_curator.stages.audio.audio_agent import run_store as _run_store

    origin_recipe_uri = _run_store.save_origin_recipe(rec.config_hash, rec.to_dict()) or ""
    origin_workspace = _run_store.workspace_id()

    roles, keys = _verb_validation._derive_initial(data_profile)
    rows_in = input_count
    cumulative = 0.0
    # Persisted results inherit the trust and generated assets of their entire prefix.
    dependencies: dict[str, str] = {}
    generated_audio_keys: set[str] = set()
    prefix_deterministic = True
    prefix_ttl_sec = 0
    if step_identity:
        inherited = art_mod.load(step_identity[0][0])
        if inherited is not None:
            prefix_deterministic = inherited.deterministic
            prefix_ttl_sec = inherited.ttl_sec
            dependencies.update(inherited.dependencies)
    for i, (plan, st) in enumerate(zip(plans, stages, strict=False)):
        prefix_deterministic = prefix_deterministic and plan.deterministic
        if plan.ttl_sec:
            prefix_ttl_sec = min(prefix_ttl_sec, plan.ttl_sec) if prefix_ttl_sec else plan.ttl_sec
        writes_assets = getattr(st, "write_to_disk", True) and getattr(st, "write_rttm", True)
        if writes_assets:
            for name in GENERATED_ASSET_PARAMS:
                uri = getattr(st, name, None) or rec.stages[i].params.get(name)
                if isinstance(uri, str) and uri:
                    dependencies[uri] = art_mod.content_digest(uri) or ""
        from nemo_curator.stages.audio.audio_agent._capabilities import generated_asset_paths

        for uri in generated_asset_paths(rec.stages[i].ref, rec.stages[i].params):
            dependencies[uri] = art_mod.content_digest(uri) or ""
        with contextlib.suppress(Exception):
            c = foundation.build_contract(st)
            roles |= _roles_of(c)
            keys |= set(c.writes.data_keys) | set(c.writes.segment_data_keys)
            if c.gates.writes_to_disk:
                generated_audio_keys.update(
                    key
                    for key in [*c.writes.data_keys, *c.writes.segment_data_keys]
                    if c.key_roles.get(key) == "audio_filepath"
                )
        duration, gpu_seconds = _stage_cost(per_stage, st)
        cumulative += duration  # every step so far, persisting or not: that is the true saving
        if not plan.persists():
            continue
        if plan.reuse_block:
            if persistence_warnings is not None:
                persistence_warnings.append(f"{plan.stage_ref} was not published for reuse: {plan.reuse_block}")
            continue
        if "://" in plan.uri and not plan.uri.startswith("file://"):
            if persistence_warnings is not None:
                persistence_warnings.append(
                    f"artifact publication/reuse is unsupported for non-local output backend {plan.uri!r}"
                )
            continue
        if not os.path.exists(os.path.expanduser(plan.uri)):
            if persistence_warnings is not None:
                persistence_warnings.append(
                    f"declared persisted output {plan.uri!r} was not found; "
                    f"{plan.stage_ref} was not published for reuse"
                )
            continue
        if generated_audio_keys and (art_mod.classify_output(plan.uri) or plan.kind) == "manifest":
            try:
                dependencies.update(_verb_evidence._manifest_asset_dependencies(plan.uri, generated_audio_keys))
            except (OSError, ValueError, UnicodeError) as exc:
                if persistence_warnings is not None:
                    persistence_warnings.append(
                        f"generated audio dependencies could not be verified for {plan.uri!r}: {exc}; "
                        "artifact was not published for reuse"
                    )
                continue
        # Reusing the LAST step skips the run outright, so its true cost is the wall clock,
        # not the sum of stage timers (which misses setup, scheduling and teardown).
        through_here = max(cumulative, elapsed_sec) if plan.index == len(plans) - 1 else cumulative
        step_key, input_key, stage_index = (
            step_identity[i]
            if step_identity and i < len(step_identity)
            else (plan.step_key, plan.input_key, plan.index)
        )
        is_checkpoint = plan.stage_ref == "ManifestCheckpointStage"
        # From the stage itself, NOT from ``semantic_params``: retention and owner are
        # lifecycle policy and were deliberately dropped from the reuse identity, so a
        # checkpoint keeps one address however long it is kept. They still have to reach the
        # record. Retention governs these checkpoint bytes; upstream freshness alone is
        # inherited by downstream artifacts, whose outputs have an independent lifetime.
        checkpoint_retention = int(getattr(st, "retention_sec", 0) or 0)
        checkpoint_policy = (
            {
                "owner": str(getattr(st, "owner", "") or "user"),
                "retention_sec": checkpoint_retention,
                "expires_at": (
                    time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ",
                        time.gmtime(time.time() + checkpoint_retention),
                    )
                    if checkpoint_retention
                    else None
                ),
                "automatic_deletion": False,
            }
            if is_checkpoint
            else {}
        )
        art = art_mod.Artifact(
            step_key=step_key,
            input_key=input_key,
            stage_ref=plan.stage_ref,
            stage_index=stage_index,
            semantic_params=plan.semantic_params,
            contract_hash=rec.contract_hash,
            uri=plan.uri,
            # Re-classified now rather than trusting the plan-time guess: at plan time a directory
            # that does not exist yet cannot be identified, and the kind decides which source stage
            # may re-read this artifact later.
            kind=art_mod.classify_output(plan.uri) or plan.kind,
            rows_in=rows_in,
            produced_roles=sorted(roles - {"unknown"}),
            produced_keys=sorted(keys),
            duration_sec=duration,
            cumulative_sec=round(through_here, 3),
            gpu_seconds=gpu_seconds,
            device="gpu" if gpu_seconds else "cpu",
            started_at=started_at,
            ended_at=ended_at,
            dataset_key=dataset_key,
            fingerprint_tier=fingerprint_tier,
            covers_files=len(inventory or {}),
            impl_version=plan.impl_version,
            code_version=art_mod.code_version(),
            model_version=plan.model_version,
            deterministic=prefix_deterministic,
            ttl_sec=prefix_ttl_sec,
            dependencies=dict(dependencies),
            metrics={"checkpoint_policy": checkpoint_policy} if checkpoint_policy else {},
            run_id=run_id,
            origin_config_hash=rec.config_hash or "",
            origin_recipe_uri=origin_recipe_uri,
            workspace_id=origin_workspace,
        )
        try:
            art_mod.publish(art)
            if inventory:
                # After publish, so a coverage file never outlives the artifact it describes.
                art_mod.save_coverage(art.step_key, inventory)
        except Exception as exc:  # noqa: BLE001 - compute completion stays separate
            if persistence_warnings is not None:
                persistence_warnings.append(
                    f"artifact publication failed for {plan.uri!r}: {type(exc).__name__}: {exc}"
                )
            continue
        rows_in = art.rows_out or rows_in
        out.append({"step_key": art.step_key, "stage": art.stage_ref, "uri": art.uri, "rows": art.rows_out})
    return out


def _record_run(  # noqa: PLR0913 - a provenance record intentionally gathers many fields
    rec: Recipe,
    *,
    run_id: str,
    data: str | None,
    data_fp: str | None,
    dataset_key: str,
    fingerprint_tier: str,
    report: Any,  # noqa: ANN401
    failed: bool,
    goal: dict[str, Any] | None = None,
    elapsed: float = 0.0,
    env: dict[str, Any] | None = None,
    produced_roles: list[str] | None = None,
    produced_keys: list[str] | None = None,
    acceptance_result: dict[str, Any] | None = None,
    reuse: dict[str, Any] | None = None,
    logical_steps: list[str] | None = None,
    persistence_status: dict[str, Any] | None = None,
) -> str:
    """Persist a local RunRecord (provenance for tracing + continuation). Best-effort."""
    from nemo_curator.stages.audio.audio_agent import _verb_evidence, run_store
    from nemo_curator.stages.audio.audio_agent.contracts import RunRecord

    env = env or {}
    pipeline_summary = ""
    if not failed:
        with contextlib.suppress(Exception):  # summary is best-effort; blank falls back at read time
            pipeline_summary = _reuse_summary.summarize_pipeline(rec)
    record = RunRecord(
        run_id=run_id,
        # Redact secret-valued params (e.g. hf_token) so they never land in the on-disk record.
        recipe=_safety.redact(rec.to_dict(), redact_transcripts=False),
        config_hash=rec.config_hash,
        semantic_hash=rec.semantic_hash,
        contract_hash=rec.contract_hash,
        parent_run_id=rec.parent_run_id,
        goal=dict(goal or {}),
        pipeline_summary=pipeline_summary,
        data_source=data,
        data_fingerprint=data_fp,
        dataset_key=dataset_key,
        fingerprint_tier=fingerprint_tier,
        acceptance_criteria=list(rec.acceptance_criteria),
        acceptance_result=(
            acceptance_result
            if acceptance_result is not None
            else _verb_evidence._acceptance_result(
                rec,
                report,
                produced_roles,
                produced_keys,
                list(getattr(report, "output_paths", []) or []),
            )
        ),
        status="failed" if failed else "completed",
        accepted=int(getattr(report, "accepted", 0)),
        input_count=int(getattr(report, "input_count", 0)),
        output_paths=list(getattr(report, "output_paths", []) or []),
        elapsed_sec=round(elapsed, 3),
        per_stage_metrics=dict(getattr(report, "per_stage_metrics", {}) or {}),
        env_summary={
            k: env.get(k) for k in ("has_gpu", "gpu_count", "gpu_names", "python_version", "cuda_runtime_version")
        },
        curator_version=str(env.get("curator_version") or ""),
        knowledge_version=rec.knowledge_version,
        # The chain of the pipeline the USER asked for. After a continuation ``rec`` is the
        # rewritten recipe, whose own keys describe a pipeline nobody requested and share no
        # prefix with the request -- which made every continued run unrecognisable to anything
        # matching on this field, including the disclosure of already-done-but-unsaved work.
        steps=logical_steps or _step_keys(rec, dataset_key),
        reuse=dict(reuse or {}),
        created_at=_utc_now(),
    )
    try:
        path = run_store.save(record)
    except Exception as exc:  # noqa: BLE001 - compute may succeed even if provenance does not
        if persistence_status is not None:
            persistence_status["run_record_persisted"] = False
            persistence_status.setdefault("warnings", []).append(
                f"run record persistence failed: {type(exc).__name__}: {exc}"
            )
        return record.run_id
    # Keep what actually ran, when the record above could not. Only a recipe whose params were
    # masked gets a copy, and only so "do that again on the new files" can adopt this exact
    # pipeline later -- a masked param changes reuse identity, so the record's copy would match
    # none of this run's own artifacts.
    with contextlib.suppress(Exception):  # provenance is best-effort; a failure here costs adoption
        run_store.save_exact_recipe(record.run_id, rec.to_dict())
    if persistence_status is not None:
        persistence_status["run_record_persisted"] = True
        persistence_status["run_record_path"] = path
    return record.run_id


def _step_keys(rec: Recipe, dataset_key: str) -> list[str]:
    try:
        from nemo_curator.stages.audio.audio_agent import artifacts as art_mod

        return art_mod.step_keys(rec, dataset_key)
    except Exception:  # noqa: BLE001
        return []


def _yield_input_count(rec: Recipe, input_count: int | None, output_rows: int) -> int | None:
    """Provide a percentage denominator only when output rows represent source items.

    Absolute yield still counts output rows. Fan-out and aggregation require
    source provenance before those rows can establish source-relative retention.
    """
    if input_count is None or _report_data.source_rejection_count(rec, input_count, output_rows) is None:
        return None
    return input_count
