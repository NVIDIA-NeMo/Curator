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

"""Continuation planning and approved execution."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any

from nemo_curator.stages.audio.audio_agent import (
    _safety,
    _verb_evidence,
    _verb_inputs,
    _verb_publish,
    _verb_report,
    _verb_validation,
)
from nemo_curator.stages.audio.audio_agent.recipe import (
    Recipe,
)
from nemo_curator.stages.audio.audio_agent.report import _row_count


@dataclass(frozen=True)
class _ContinuationRunContext:
    """The minimum claim a continued run may present at the execution boundary.

    None of the identities that are ultimately published are accepted directly.
    The verifier re-derives the logical chain and the materialized suffix from
    this recipe, the current source fingerprint, and one complete artifact.
    """

    logical_recipe: dict[str, Any]
    dataset_key: str
    reuse_step_key: str


def _continuation_context_from_plan(
    logical_recipe: Recipe,
    plan: dict[str, Any],
) -> _ContinuationRunContext:
    """Snapshot the logical request for verification by :func:`run`."""
    point = plan.get("reuse_point") or {}
    return _ContinuationRunContext(
        logical_recipe=logical_recipe.to_dict(),
        dataset_key=str(plan.get("dataset_key") or ""),
        reuse_step_key=str(point.get("step_key") or ""),
    )


def _verify_continuation_context(  # noqa: PLR0911, C901, PLR0912 - each refusal names one broken proof
    physical_recipe: Recipe,
    physical_binding: Any,  # noqa: ANN401 - DatasetBinding without eager import
    context: _ContinuationRunContext | None,
) -> tuple[dict[str, Any] | None, str]:
    """Prove a continued run, then derive every identity override internally.

    A caller cannot provide step tuples or a run-record chain. The reused step
    must belong to the logical recipe on the stated current dataset; its complete
    artifact must be the physical source; and materializing that exact boundary
    must reproduce the recipe about to execute.
    """
    if context is None:
        return None, ""
    if not isinstance(context, _ContinuationRunContext):
        return None, "continued-run context has an invalid type"

    from nemo_curator.stages.audio.audio_agent import artifacts as art_mod
    from nemo_curator.stages.audio.audio_agent import continuation as continuation_mod
    from nemo_curator.stages.audio.audio_agent.input_identity import canonical_source

    dataset_key = str(context.dataset_key or "")
    reuse_step_key = str(context.reuse_step_key or "")
    if not dataset_key or not reuse_step_key:
        return None, "continued-run context is incomplete"

    try:
        logical_recipe = Recipe.from_dict(context.logical_recipe).freeze()
        logical_binding = _verb_inputs._dataset_binding(logical_recipe, None)
        logical_profile = _verb_inputs._profile_binding(logical_binding)
    except (TypeError, ValueError) as exc:
        return None, f"continued-run logical recipe is invalid: {exc}"
    if logical_binding.status != "resolved" or logical_profile is None:
        return None, "continued-run logical source cannot be verified"
    if logical_profile.dataset_key() != dataset_key:
        return None, "continued-run logical source no longer matches the claimed dataset"

    try:
        logical_plans = art_mod.plan_steps(logical_recipe, dataset_key)
    except Exception as exc:  # noqa: BLE001 - turn an identity failure into a refusal
        return None, f"continued-run logical steps could not be derived: {exc}"
    matches = [plan for plan in logical_plans if plan.step_key == reuse_step_key]
    if len(matches) != 1:
        return None, "continued-run artifact step does not belong to the logical recipe"
    reused_plan = matches[0]

    artifact, reasons = art_mod.lookup(reuse_step_key, dataset_key=dataset_key)
    if artifact is None or reasons:
        why = "; ".join(reasons or ["artifact record is missing"])
        return None, f"continued-run lineage is no longer valid: {why}"
    expected_identity = {
        "step_key": reused_plan.step_key,
        "dataset_key": dataset_key,
        "input_key": reused_plan.input_key,
        "stage_index": reused_plan.index,
        "stage_ref": reused_plan.stage_ref,
        "semantic_params": reused_plan.semantic_params,
        "model_version": reused_plan.model_version,
    }
    try:
        actual_identity = {
            "step_key": str(getattr(artifact, "step_key", "") or ""),
            "dataset_key": str(getattr(artifact, "dataset_key", "") or ""),
            "input_key": str(getattr(artifact, "input_key", "") or ""),
            "stage_index": int(getattr(artifact, "stage_index", -1)),
            "stage_ref": str(getattr(artifact, "stage_ref", "") or ""),
            "semantic_params": dict(getattr(artifact, "semantic_params", {}) or {}),
            "model_version": str(getattr(artifact, "model_version", "") or ""),
        }
    except (TypeError, ValueError):
        return None, "continued-run artifact metadata is malformed"
    if actual_identity != expected_identity:
        return None, "continued-run artifact metadata does not match its logical step"

    if not physical_binding.primary_path:
        return None, "continued-run physical source could not be resolved"
    try:
        if canonical_source(physical_binding.primary_path) != canonical_source(artifact.uri):
            return None, "continued-run recipe does not read the verified artifact"
    except (TypeError, ValueError) as exc:
        return None, f"continued-run artifact path is invalid: {exc}"

    expected_recipe, materialize_error = continuation_mod.materialize(
        logical_recipe,
        uri=artifact.uri,
        kind=artifact.kind,
        prefix=reused_plan.index + 1,
    )
    if expected_recipe is None:
        return None, f"continued-run boundary cannot be materialized: {materialize_error}"
    if physical_recipe.compute_hash() != expected_recipe.compute_hash():
        return None, "continued-run physical recipe is not the verified logical suffix"

    return {
        "dataset_key": dataset_key,
        "fingerprint_tier": str(getattr(artifact, "fingerprint_tier", "") or ""),
        "data_source": logical_binding.primary_path,
        "source_items": logical_profile.num_files,
        "step_identity": [(plan.step_key, plan.input_key, plan.index) for plan in logical_plans[reused_plan.index :]],
        "logical_steps": [plan.step_key for plan in logical_plans],
    }, ""


def plan_continuation(  # noqa: PLR0913 - one verb covering plan + the three-way choice
    recipe: Recipe | dict[str, Any],
    parent_run_id: str | None = None,
    *,
    data: str | None = None,
    execute: bool = False,
    choice: str | None = None,
    confirm: bool | str = False,
    output_dir: str | None = None,
    checkpoint_path: str | None = None,
    bootstrap_ray: bool = False,
    smoke_token: str | None = None,
    calibration: dict[str, Any] | None = None,
    goal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Plan — and optionally execute — a follow-up run that reuses prior work.

    Two engines feed one answer. A ``parent_run_id`` is diffed stage-by-stage against its
    :class:`RunRecord` (the honest, exact-append case), and the artifact registry is probed
    with this recipe's step keys, which additionally catches reuse the diff cannot see: a
    middle-of-pipeline edit, or work done by a *different* recipe that happens to share a
    prefix. The deeper of the two wins.

    With ``execute`` the plan stops being advice. ``choice`` picks the branch — ``as_is``
    (serve the existing output and re-verify the contract), ``extend`` (materialize a recipe
    that starts from the artifact and run only what is new), or ``fresh`` — defaulting to
    whatever the plan concluded. Execution still goes through the normal confirm gate.

    ``data`` is an optional assertion about the configured first source stage,
    not an input override. Omit it when the recipe is already unambiguous.
    """
    from nemo_curator.stages.audio.audio_agent import continuation, run_store
    from nemo_curator.stages.audio.audio_agent import reuse as _reuse

    rec = _verb_inputs._as_recipe(recipe).freeze()
    pviol = _safety.path_violations([data, output_dir, checkpoint_path, *_safety.recipe_path_params(rec)])
    if pviol:
        return {
            "status": "refused",
            "reason": "path(s) resolve outside the allowed workspace",
            "violations": pviol,
        }
    binding = _verb_inputs._dataset_binding(rec, data)
    dp = _verb_inputs._profile_binding(binding)
    profile_error = _verb_inputs._profile_error(dp) if dp is not None else ""
    binding_blocked = _verb_inputs._binding_blocks_execution(binding, data) or bool(profile_error)
    dataset_key = dp.dataset_key() if dp and not binding_blocked else ""
    scan = _reuse.scan(rec, dataset_key=dataset_key)

    plan = _parent_plan(rec, parent_run_id, dp=dp, dataset_key=dataset_key)
    plan = _merge_plans(rec, plan, scan, dataset_key=dataset_key)
    plan["candidates"] = scan["candidates"]
    plan["estimated_saving_sec"] = scan["estimated_saving_sec"]
    plan["prompt_user"] = scan["prompt_user"]
    # Why the gate is asking, when it is not the size of the saving: work whose cost was never
    # recorded. Without this the card shows "saves 0.2 s" beside a question and looks broken.
    plan["unpriced_stages"] = scan.get("unpriced_stages", [])
    plan["recommended"] = scan.get("recommended", "fresh")
    plan["choices"] = scan.get("choices", [])
    plan["reuse_rationale"] = scan.get("rationale", "")
    plan["dataset_key"] = dataset_key
    plan["data_binding"] = binding.to_dict()
    if profile_error:
        plan["source_error"] = profile_error
        plan["reuse_rationale"] = f"prior work was not considered: {profile_error}"
    # Work that ran before but persisted nothing is invisible to BOTH engines: the parent diff
    # sees a changed recipe, the scan finds no artifact. Carry the disclosure onto the plan, or
    # the gate presents a recomputation as new work and the user pays for it twice unknowingly.
    if scan.get("prior_unsaved"):
        plan["prior_unsaved"] = scan["prior_unsaved"]
        plan["offer"] = scan["offer"]

    if not execute:
        return _safety.redact(plan)
    if binding_blocked:
        reason = profile_error or binding.reason
        return _safety.redact(
            {
                "status": "refused",
                "reason": f"continuation cannot execute with an unbound source: {reason}",
                "plan": plan,
                "data_binding": binding.to_dict(),
            }
        )
    return _execute_plan(
        rec,
        plan,
        choice=choice or _chosen_by_default(plan),
        data=data,
        confirm=confirm,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        bootstrap_ray=bootstrap_ray,
        smoke_token=smoke_token,
        calibration=calibration,
        goal=goal,
        parent=run_store.load(parent_run_id) if parent_run_id else None,
        continuation_mod=continuation,
    )


def _parent_plan(rec: Recipe, parent_run_id: str | None, *, dp: Any, dataset_key: str) -> dict[str, Any]:  # noqa: ANN401
    """The classic parent-diff plan, or a neutral 'nothing to diff against'."""
    from nemo_curator.stages.audio.audio_agent import continuation, run_store

    if not parent_run_id:
        return {
            "mode": "full_rerun",
            "reason": "no parent run given; reuse comes from the artifact scan alone",
            "run_stages": [s.ref for s in rec.stages],
        }
    parent = run_store.load(parent_run_id)
    if parent is None:
        return {
            "mode": "full_rerun",
            "reason": f"no parent run record {parent_run_id!r}; run fresh",
            "run_stages": [s.ref for s in rec.stages],
        }
    if dp is None or not dataset_key:
        return {
            "mode": "full_rerun",
            "parent_run_id": parent.run_id,
            "reason": "source data identity is unavailable; the parent cannot be claimed to be on the same data",
            "run_stages": [s.ref for s in rec.stages],
        }
    return continuation.plan_continuation(
        rec, parent, data_fingerprint=dp.fingerprint() if dp else None, dataset_key=dataset_key or None
    )


def _merge_plans(rec: Recipe, plan: dict[str, Any], scan: dict[str, Any], *, dataset_key: str) -> dict[str, Any]:
    """Take whichever engine reuses more VERIFIED work, and say which one found it.

    The parent diff and the step-key scan answer slightly different questions, so they can
    disagree, and depth breaks the tie. But depth alone used to decide it, and only the scan
    carried an artifact -- so a tie handed the plan to the engine that had proven nothing, and the
    executor then refused to extend for want of a resume URI. That was the ordinary case: a parent
    run whose artifact WAS published gives both engines the same depth.

    So the parent diff's claim is now resolved against the same registry, and verified depth beats
    unverified depth. When neither resolves, the plan says so instead of failing later.
    """
    from nemo_curator.stages.audio.audio_agent import reuse as _reuse

    scan_depth = len(scan.get("reuse_stages") or []) if scan["decision"] != "fresh" else 0
    plan_depth = len(plan.get("reuse_stages") or []) if plan["mode"] in ("already_done", "incremental") else 0
    if plan_depth >= scan_depth and plan_depth:
        point, why_not = _reuse.verified_point(rec, plan_depth, dataset_key=dataset_key)
        if point:
            plan["reuse_point"] = point
            plan.setdefault("reuse_from", [point["uri"]])
            plan.setdefault("source", "parent_diff")
            return plan
        if not scan_depth:
            plan["reuse_point_unavailable"] = why_not
            plan.setdefault("source", "parent_diff")
            return plan
        # The scan is shallower but backed by a real artifact, so it is the safer resume point.
        plan["superseded_parent_diff"] = f"deeper parent-diff reuse has no reusable artifact: {'; '.join(why_not)}"
    elif scan_depth <= plan_depth:
        plan.setdefault("source", "none")
        return plan
    point = scan["reuse_point"] or {}
    merged = {
        "mode": "already_done" if scan["decision"] == "already_done" else "incremental",
        "source": "artifact_scan",
        "parent_run_id": plan.get("parent_run_id"),
        "reuse_stages": scan["reuse_stages"],
        "run_stages": scan["run_stages"],
        "reuse_from": [point.get("uri")] if point.get("uri") else [],
        "reuse_point": point,
        "rationale": scan["rationale"],
    }
    superseded = plan.get("superseded_parent_diff") or plan.get("reason")
    if superseded:
        merged["superseded_parent_diff"] = superseded
    return merged


def _default_choice(mode: str) -> str:
    return {"already_done": "as_is", "incremental": "extend"}.get(mode, "fresh")


def _chosen_by_default(plan: dict[str, Any]) -> str:
    """What runs when the caller states no choice -- the plan's own recommendation if it made one.

    The scan already downgrades to ``fresh`` whenever a candidate's trust is anything less than
    high (a sampled dataset key that cannot see an in-place edit, a stage whose output is not
    reproducible). That recommendation only ever reached the approval card: execution went to the
    mode's default and reused the low-trust output anyway, so the warning and the behaviour
    disagreed. Recommending caution and then not taking it is worse than not warning at all.
    """
    recommended = str(plan.get("recommended") or "")
    offered = {str(c.get("id")) for c in plan.get("choices") or [] if isinstance(c, dict)}
    if recommended and (not offered or recommended in offered):
        return recommended
    return _default_choice(plan["mode"])


def _execute_plan(  # noqa: PLR0913, PLR0911 - executing a plan needs the plan's whole context
    rec: Recipe,
    plan: dict[str, Any],
    *,
    choice: str,
    data: str | None,
    confirm: bool | str,
    output_dir: str | None,
    bootstrap_ray: bool,
    goal: dict[str, Any] | None,
    parent: Any,  # noqa: ANN401
    continuation_mod: Any,  # noqa: ANN401
    checkpoint_path: str | None = None,
    smoke_token: str | None = None,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Carry out the user's three-way choice."""
    from nemo_curator.stages.audio.audio_agent import _verb_run

    lineage = {"choice": choice, "mode": plan["mode"], "reuse_source": plan.get("source")}
    if choice == "fresh":
        return _verb_run.run(
            rec,
            confirm=confirm,
            data=data,
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            bootstrap_ray=bootstrap_ray,
            smoke_token=smoke_token,
            calibration=calibration,
            goal=goal,
            reuse={**lineage, "reused": []},
        )

    point = plan.get("reuse_point") or {}
    if choice == "as_is":
        return _serve_as_is(rec, plan, parent=parent, lineage=lineage)

    if plan["mode"] != "incremental" or not point.get("uri"):
        why = "; ".join(plan.get("reuse_point_unavailable") or []) or "the plan names no reusable output"
        return {
            "status": "refused",
            "reason": f"there is nothing to extend from ({why}); choose 'fresh'",
            "plan": plan,
        }

    materialized, err = continuation_mod.materialize(
        rec, uri=point["uri"], kind=point.get("kind", "unknown"), prefix=int(point.get("stage_index", -1)) + 1
    )
    if materialized is None:
        return {"status": "refused", "reason": err, "plan": plan}

    # Re-derive the boundary from the serialized artifact and its new source
    # stage. Cumulative in-memory metadata in the registry is not proof that a
    # manifest or bare audio directory serialized those roles/keys.
    verdict = _verb_validation.validate(materialized, data=None)
    if not verdict.get("runnable", False):
        return _safety.redact(
            {
                "status": "refused",
                "reason": "the continued pipeline does not validate when resumed from the reused output",
                "recipe": materialized.to_dict(),
                "verdict": verdict,
                "plan": plan,
            },
            redact_transcripts=False,
        )
    inner_confirm, inner_token, refusal = _verb_validation._carry_approval(
        rec, materialized, confirm=confirm, smoke_token=smoke_token
    )
    if refusal:
        return _safety.redact(
            {
                "status": "refused",
                "reason": refusal,
                "confirm_with": f"pass confirm={rec.config_hash!r} (or confirm=True) to proceed",
                "recipe": materialized.to_dict(),
                "plan": plan,
            },
            redact_transcripts=False,
        )
    result = _verb_run.run(
        materialized,
        confirm=inner_confirm,
        data=None,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        bootstrap_ray=bootstrap_ray,
        smoke_token=inner_token,
        calibration=calibration,
        goal=goal,
        reuse={**lineage, "reused": plan.get("reuse_stages", []), "reused_from": point.get("uri")},
        _continuation_context=_continuation_context_from_plan(rec, plan),
    )
    # An unconfirmed call is still refused by ``run`` (unchanged), but its advice names the
    # DERIVED hash -- which the user never saw and cannot meaningfully approve. Name theirs.
    if result.get("status") == "refused" and result.get("confirm_with"):
        result["confirm_with"] = f"pass confirm={rec.config_hash!r} (or confirm=True) to proceed"
    result["recipe"] = _safety.redact(
        materialized.to_dict(),
        redact_transcripts=False,
    )
    result["plan"] = _safety.redact(plan, redact_transcripts=False)
    return _safety.redact(result)


def _declared_output(rec: Recipe) -> str:
    """The last output location the recipe names, or ``""`` -- where the user asked for it."""
    outs = _verb_evidence._recipe_outputs(rec, None)
    return outs[-1] if outs else ""


def _deliver_to_declared_path(  # noqa: C901, PLR0911 - fail-closed delivery has one guard per unsafe state
    rec: Recipe,
    uri: str,
    *,
    expected_digest: str | None = None,
) -> tuple[str, bool | None]:
    """Put the reused output where the recipe asked for it.

    Returns ``(path, delivered)``: ``None`` when no copy was needed, ``True`` when the artifact
    was materialized at the declared path, ``False`` when it could not be.

    Serving the stored URI when the recipe declares a different ``output_path`` answers a
    question the user did not ask -- they get a path they never named, and the file they DID
    name is silently absent. Copying is cheap next to recomputing, so reuse honors the request.
    """
    from nemo_curator.stages.audio.audio_agent import _verb_delta

    want = _declared_output(rec)
    if not want or os.path.abspath(os.path.expanduser(want)) == os.path.abspath(os.path.expanduser(uri)):
        return uri, None
    src, dst = os.path.expanduser(uri), os.path.expanduser(want)
    from nemo_curator.stages.audio.audio_agent.artifacts import content_digest

    source_digest = content_digest(src)
    if source_digest is None or (expected_digest is not None and source_digest != expected_digest):
        return uri, False

    # Reuse is evidence about the source artifact's exact bytes. Merging those bytes into a
    # pre-existing destination would produce a third artifact that was never verified. An
    # already-identical destination needs no write; any other existing target remains the
    # user's and is refused rather than overwritten.
    if os.path.lexists(dst):
        if content_digest(dst) == source_digest:
            return want, None
        return uri, False

    staged = ""
    staged_identity: tuple[int, int] | None = None
    try:
        parent = os.path.abspath(os.path.dirname(dst) or ".")
        os.makedirs(parent, exist_ok=True)
        if os.path.isdir(src):
            staged = tempfile.mkdtemp(prefix=f".{os.path.basename(dst)}.audio-agent-", dir=parent)
            staged_identity = _verb_delta._owned_path_identity(staged)
            shutil.copytree(src, staged, dirs_exist_ok=True)
            if content_digest(staged) != source_digest:
                return uri, False
            # Atomic when ``dst`` is absent. If another writer created a non-empty directory
            # after the check above, rename refuses instead of combining the two trees.
            os.rename(staged, dst)
            staged = ""
            staged_identity = None
        elif os.path.isfile(src):
            fd, staged = tempfile.mkstemp(prefix=f".{os.path.basename(dst)}.audio-agent-", dir=parent)
            os.close(fd)
            staged_identity = _verb_delta._owned_path_identity(staged)
            shutil.copy2(src, staged)
            if content_digest(staged) != source_digest:
                return uri, False
            # A sibling hard link publishes the complete file without a replace race: link(2)
            # fails if another writer created ``dst`` after the existence check.
            os.link(staged, dst)
        else:
            return uri, False
    except OSError:
        return uri, False
    finally:
        if staged and staged_identity is not None:
            _verb_delta._remove_owned_temporary(staged, staged_identity)
    return want, True


def _has_content(uri: str) -> bool:
    """True if something is actually there -- a non-empty file, or a directory with entries."""
    path = os.path.expanduser(uri)
    if os.path.isdir(path):
        return bool(os.listdir(path))
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _as_is_evidence(
    plan: dict[str, Any],
    parent: Any,  # noqa: ANN401
    uri: str,
) -> tuple[str, list[str], int | None, str | None]:
    """How well the bytes about to be served are proven to be a complete result.

    Two engines can propose an ``as_is`` path and only one of them carries an artifact, so this is
    where both are held to a bar. The parent-diff path used to be served unconditionally from a
    recorded output path, which meant a record naming a file that was never written came back as
    ``status: reused`` with acceptance computed over nothing at all.

    An artifact is the strong evidence: a ``_COMPLETE`` marker, a matching dataset and code
    version. Without one -- a run from before artifacts, or a pruned record -- fall back to what
    can still be checked rather than refusing outright, and name the weaker basis in the result.
    """
    from nemo_curator.stages.audio.audio_agent import artifacts as art_mod

    point = plan.get("reuse_point") or {}
    if point.get("step_key"):
        art, reasons = art_mod.lookup(point["step_key"], dataset_key=plan.get("dataset_key") or None)
        if art is not None and not reasons:
            marker = art_mod.read_marker(art.uri) or {}
            artifact_rows = getattr(art, "rows_out", None)
            marker_rows = marker.get("rows")
            expected_rows = (
                artifact_rows
                if (
                    isinstance(artifact_rows, int)
                    and not isinstance(artifact_rows, bool)
                    and artifact_rows >= 0
                    and marker_rows == artifact_rows
                )
                else None
            )
            expected_digest = str(getattr(art, "content_digest", "") or "") or None
            return "artifact", [], expected_rows, expected_digest
        return "", reasons or ["the recorded artifact is no longer reusable"], None, None
    status = str(getattr(parent, "status", "") or "")
    if status and status != "completed":
        reason = f"the run that produced it ended {status!r}, so its output is not a complete result"
        return "", [reason], None, None
    if not _has_content(uri):
        return "", [f"{uri!r} does not exist or is empty, so there is nothing to serve"], None, None
    return "run_record", [], None, None


def _serve_as_is(rec: Recipe, plan: dict[str, Any], *, parent: Any, lineage: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN401, C901
    """Serve an existing output — and RE-VERIFY it against the criteria asked for now.

    Reused data is still judged by today's bar: a stricter contract must be re-checked, never
    inherited from the run that produced the bytes.
    """
    point = plan.get("reuse_point") or {}
    uri = point.get("uri") or next(iter(plan.get("reuse_from") or []), None)
    if not uri:
        detail = str(plan.get("reuse_rationale") or "").strip()
        reason = "no completed output can be served"
        if detail:
            reason += f": {detail}"
        return {"status": "refused", "reason": f"{reason}; choose 'fresh'", "plan": plan}
    pviol = _safety.path_violations([uri, _declared_output(rec)])
    if pviol:
        return {
            "status": "refused",
            "reason": "reused input/output resolves outside the allowed workspace",
            "violations": pviol,
            "plan": plan,
        }
    # Checked BEFORE delivery: a refusal must not leave a copy behind.
    trust, why_not, expected_output_rows, expected_digest = _as_is_evidence(plan, parent, uri)
    if not trust:
        return {
            "status": "refused",
            "reason": f"the previous output cannot be served: {'; '.join(why_not)}; choose 'fresh'",
            "plan": plan,
        }
    uri, delivered_to = _deliver_to_declared_path(rec, uri, expected_digest=expected_digest)
    if delivered_to is False:
        return {
            "status": "refused",
            "reason": (
                f"the recipe asks for output at {_declared_output(rec)!r} but the reusable output is at "
                f"{uri!r}, and it could not be copied there; choose 'fresh' or point the recipe at that path"
            ),
            "plan": plan,
        }
    per_item, output_scan = (
        _verb_evidence._scan_terminal_output([uri], limit=0)
        if _verb_evidence._needs_terminal_evidence(rec)
        else ([], {})
    )
    source_run = parent
    source_run_id = getattr(source_run, "run_id", None)
    if source_run is None:
        source_run_id = point.get("run_id") or ((plan.get("candidates") or [{}])[0].get("run_id"))
        if source_run_id:
            from nemo_curator.stages.audio.audio_agent import run_store

            source_run = run_store.load(str(source_run_id))
    source_input_count = getattr(source_run, "input_count", None)
    if not isinstance(source_input_count, int) or isinstance(source_input_count, bool) or source_input_count < 0:
        source_input_count = None
    acceptance = _verb_report.verify(
        list(rec.acceptance_criteria),
        {
            "produced_roles": list(point.get("produced_roles") or []),
            "produced_keys": list(point.get("produced_keys") or []),
            # The artifact is a real file, so judge today's bar against its rows -- reused data
            # gets the same data-level scrutiny a fresh run gets, not a weaker label check.
            "per_item": per_item,
            "output_scan": output_scan,
            "metrics": _verb_evidence._aggregate_metrics(output_scan),
            # A valid artifact binds this count in both its registry record and
            # completion marker. A legacy run record names a path but does not
            # independently prove its serialized inventory, so it stays unknown.
            "expected_output_rows": expected_output_rows,
            "retained": int(point.get("rows") or 0),
            # Relative yield is defined against the producing run's source
            # inventory.  Substituting output rows here manufactures 100%
            # retention whenever the run record is unavailable.
            "input_count": _verb_publish._yield_input_count(rec, source_input_count, int(point.get("rows") or 0)),
        },
    )
    checked = len(rec.acceptance_criteria)
    note = "served from a previously completed run; no compute was done. "
    note += (
        f"The {checked} success criterion(a) were re-checked."
        if checked
        else "No success criteria were supplied, so nothing was re-checked."
    )
    if delivered_to:
        note += f" The reused output was copied to the path the recipe asked for ({uri})."
    if trust == "run_record":
        note += (
            " No artifact record backs this output, so completeness rests on the producing run"
            " having finished rather than on a completion marker."
        )
    return _safety.redact(
        {
            "status": "reused",
            "output": uri,
            "rows": point.get("rows") if point.get("rows") is not None else _row_count(uri),
            "evidence": trust,
            "reuse": {**lineage, "reused": plan.get("reuse_stages", []), "reused_from": uri},
            "acceptance": acceptance,
            "source_run_id": source_run_id,
            "note": note,
            "plan": plan,
        }
    )
