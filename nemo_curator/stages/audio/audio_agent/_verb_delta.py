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

"""Delta planning, execution, and checkpoint merging."""

from __future__ import annotations

import contextlib
import os
import shutil
import stat
import tempfile
import time
import types
import uuid
from typing import Any

from nemo_curator.stages.audio.audio_agent import (
    _safety,
    _verb_evidence,
    _verb_inputs,
    _verb_outputs,
    _verb_publish,
    _verb_validation,
)
from nemo_curator.stages.audio.audio_agent.recipe import (
    Recipe,
)


def delta_run(  # noqa: PLR0913, PLR0911 - the same execution knobs as run(), which it delegates to
    recipe: Recipe | dict[str, Any] | None = None,
    *,
    from_run: str | None = None,
    data: str | None = None,
    confirm: bool | str = False,
    executor: Any = None,  # noqa: ANN401
    bootstrap_ray: bool = False,
    smoke_token: str | None = None,
    calibration: dict[str, Any] | None = None,
    goal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run only the files that changed since a prior run, and merge them into its result.

    Without ``confirm`` this is a card: which files are new, changed or gone, which manifests
    would be rewritten, how many rows survive, and what it saves. With ``confirm`` it executes
    -- as an ordinary run of the user's own stages over the changed files (the source is
    narrowed by ``include_files``; no stage is told anything about being incremental), then
    merges each manifest and republishes it under the step key the full pipeline has for the
    enlarged corpus, so the next ordinary reuse probe finds it.

    ``from_run`` adopts a prior run's own recipe instead of taking one, which is what "do the
    same thing again on the new files" needs: a delta resumes from artifacts keyed on the
    pipeline that produced them, and a recipe rebuilt from the same request twice is not
    bit-identical -- a threshold written out where it was defaulted before is enough to miss
    every key. Adopting removes the retyping from the loop. The card shows what was adopted and
    the confirm gate is computed from it, so nothing runs that was not first shown.

    Every question that has to be true first is answered by ``delta.plan`` and refused by name:
    which files moved, how deep per-file work stays independent of the other files, whether the
    stages' declarations survive their own recorded row counts, and which prior row came from
    which file. A refusal here is a full run, never a partial result presented as a whole one.
    """
    from nemo_curator.stages.audio.audio_agent import delta as _delta

    rec, adopted, refusal = _delta_recipe(recipe, from_run)
    if refusal is not None:
        return _safety.redact(refusal)
    pviol = _safety.path_violations([data, *_safety.recipe_path_params(rec)])
    if pviol:
        return {
            "status": "refused",
            "reason": "path(s) resolve outside the allowed workspace",
            "violations": pviol,
        }
    binding = _verb_inputs._dataset_binding(rec, data)
    dp = _verb_inputs._profile_binding(binding)
    if dp is not None and _verb_inputs._profile_error(dp):
        return _safety.redact(_verb_inputs._profile_refusal(binding, dp))
    if _verb_inputs._binding_blocks_execution(binding, data):
        return _safety.redact(_verb_inputs._binding_refusal(binding))
    if dp is None:
        return {
            "status": "no_delta",
            "reason": "the source could not be profiled, so which files changed is unknown",
            "data_binding": binding.to_dict(),
        }

    decision = _delta.plan(
        rec,
        dataset_key=dp.dataset_key(),
        inventory=dp.inventory or None,
        inventory_root=dp.inventory_root,
    )
    card = {
        "status": "no_delta" if decision.status != "ready" else "ready",
        "recipe_id": rec.recipe_id,
        "config_hash": rec.config_hash,
        "delta": decision.to_dict(),
        "data_binding": binding.to_dict(),
    }
    if adopted is not None:
        card["adopted_from"] = adopted
        # The adopted pipeline, in the card, because the user is confirming a recipe they never
        # wrote. Naming it is the difference between an informed yes and a blind one.
        card["recipe"] = _safety.redact(rec.to_dict(), redact_transcripts=False)
    if decision.status != "ready":
        card["next"] = _no_delta_next(adopted, dataset_key=dp.dataset_key())
        return _safety.redact(card)
    approval_error = _verb_validation._approval_error(rec, confirm=confirm, smoke_token=smoke_token)
    if approval_error is not None:
        return _safety.redact({**card, "status": "refused", **approval_error})
    return _safety.redact(
        _execute_delta(
            rec,
            decision,
            dp=dp,
            data=data,
            card=card,
            executor=executor,
            bootstrap_ray=bootstrap_ray,
            smoke_token=smoke_token,
            calibration=calibration,
            goal=goal,
        )
    )


def _no_delta_next(adopted: dict[str, Any] | None, *, dataset_key: str) -> str:
    """What to do instead, when a delta is unavailable.

    "Run the pipeline normally" is the right answer for a corpus that moved in a way a delta
    cannot follow, and the wrong one for a corpus that did not move at all: adopting a prior run
    over an unchanged folder has no changed files by definition, and its result already covers
    this input. Telling that user to run again is telling them to pay twice for the answer they
    came back to reuse.
    """
    if adopted and adopted.get("dataset_key") and adopted["dataset_key"] == dataset_key:
        return (
            "nothing in this folder has changed since that run, so there are no files for a delta "
            "to process; its result already covers this input -- reuse-scan the adopted recipe and "
            "serve it with continue --choice as_is rather than running anything"
        )
    return "run the pipeline normally; a delta is not available for this input"


def _delta_recipe(
    recipe: Recipe | dict[str, Any] | None,
    from_run: str | None,
) -> tuple[Recipe, dict[str, Any] | None, dict[str, Any] | None]:
    """The recipe a delta will run: the caller's, or a prior run's adopted verbatim.

    Returns ``(recipe, adopted_provenance, refusal)``; a non-None refusal means stop. Passing
    both is refused rather than resolved by precedence: silently ignoring a recipe the caller
    handed over is how a run comes to execute stages nobody chose.
    """
    if recipe is not None and from_run:
        return (
            Recipe(),
            None,
            {
                "status": "refused",
                "reason": (
                    "from_run adopts that run's own recipe, so a second one cannot also apply; "
                    "pass a recipe or from_run, not both"
                ),
            },
        )
    if from_run:
        return _adopt_recipe(from_run)
    if recipe is None:
        return (
            Recipe(),
            None,
            {
                "status": "refused",
                "reason": "a delta needs a recipe: pass one, or from_run to adopt a prior run's",
            },
        )
    return _verb_inputs._as_recipe(recipe).freeze(), None, None


def _adopt_recipe(run_id: str) -> tuple[Recipe, dict[str, Any] | None, dict[str, Any] | None]:
    """A completed run's recipe, exactly as it ran, or a refusal naming what stopped it.

    The reuse identity is verified rather than assumed: re-freezing the stored recipe must
    reproduce the ``semantic_hash`` that run recorded, because that identity is what its
    artifacts are keyed on. A mismatch caused by a masked credential is refused -- the pipeline
    genuinely cannot be reproduced from history, and running a near-copy would match nothing
    while looking like it should. A mismatch with no secret in play is reported and allowed: the
    hash format can move between builds, and the delta engine probes the artifacts themselves.
    """
    from nemo_curator.stages.audio.audio_agent import run_store
    from nemo_curator.stages.audio.audio_agent._safety import is_secret_key

    record = run_store.load(run_id)
    if record is None:
        return Recipe(), None, {"status": "refused", "reason": f"no run record {run_id!r}"}
    if record.status != "completed":
        return (
            Recipe(),
            None,
            {
                "status": "refused",
                "reason": (
                    f"run {run_id!r} has status {record.status!r}; only a completed run has a result "
                    "for a delta to extend"
                ),
            },
        )
    raw = run_store.load_exact_recipe(run_id) or record.recipe
    if not isinstance(raw, dict) or not raw.get("stages"):
        return (
            Recipe(),
            None,
            {
                "status": "refused",
                "reason": f"run {run_id!r} recorded no recipe to adopt",
            },
        )
    try:
        rec = Recipe.from_dict(raw).freeze()
    except ValueError as exc:
        return (
            Recipe(),
            None,
            {
                "status": "refused",
                "reason": f"run {run_id!r}'s recorded recipe cannot be loaded by this build: {exc}",
            },
        )
    reproduced = not record.semantic_hash or rec.semantic_hash == record.semantic_hash
    masked = [f"{s.ref}.{param}" for s in rec.stages for param in s.params if is_secret_key(param)]
    if not reproduced and masked:
        return (
            Recipe(),
            None,
            {
                "status": "refused",
                "reason": (
                    f"run {run_id!r} cannot be adopted: its {', '.join(masked)} was masked in history, "
                    "and a pipeline missing that value matches none of that run's own results"
                ),
                "next": "pass the recipe with the credential supplied, and the delta works as usual",
            },
        )
    adopted: dict[str, Any] = {
        "run_id": record.run_id,
        "created_at": record.created_at,
        "goal": record.goal or None,
        "pipeline": [s.ref for s in rec.stages],
        "data_source": record.data_source,
        "dataset_key": record.dataset_key,
        "fingerprint_tier": record.fingerprint_tier,
        "identity_reproduced": reproduced,
    }
    if not reproduced:
        adopted["note"] = (
            "this build computes a different pipeline identity than the one recorded, so the "
            "delta depends on that run's artifacts still being probeable"
        )
    return rec, adopted, None


def _owned_path_identity(path: str) -> tuple[int, int]:
    """Filesystem identity used to prove a temporary path is still the one we created."""
    info = os.stat(path, follow_symlinks=False)
    return info.st_dev, info.st_ino


def _remove_owned_temporary(path: str, identity: tuple[int, int]) -> bool:
    """Remove ``path`` only while it is still the exact temporary entry we created.

    A unique name prevents ordinary collisions, but it does not make a later cleanup safe by
    itself: another process can rename that entry and put something else at the same path. The
    device/inode check turns that race into a leaked scratch path rather than deleting the
    replacement. Symlinks and special files are never followed.
    """
    try:
        info = os.stat(path, follow_symlinks=False)
        if (info.st_dev, info.st_ino) != identity:
            return False
        if stat.S_ISDIR(info.st_mode):
            shutil.rmtree(path)
        elif stat.S_ISREG(info.st_mode):
            os.unlink(path)
        else:
            return False
    except OSError:
        return False
    return True


def _create_delta_sandbox(uri: str) -> tuple[str, tuple[int, int]]:
    """Create one invocation-owned delta directory beside the output."""
    parent = os.path.abspath(os.path.dirname(os.path.expanduser(uri)) or ".")
    os.makedirs(parent, exist_ok=True)
    sandbox = tempfile.mkdtemp(prefix=".audio_agent_delta-", dir=parent)
    try:
        return sandbox, _owned_path_identity(sandbox)
    except OSError:
        with contextlib.suppress(OSError):
            shutil.rmtree(sandbox)
        raise


def _execute_delta(  # noqa: PLR0913 - forwards run()'s knobs and the plan it executes
    rec: Recipe,
    decision: Any,  # noqa: ANN401 - delta.Delta without an eager import
    *,
    dp: Any,  # noqa: ANN401 - DataProfile
    data: str | None,
    card: dict[str, Any],
    executor: Any,  # noqa: ANN401
    bootstrap_ray: bool,
    smoke_token: str | None,
    calibration: dict[str, Any] | None,
    goal: dict[str, Any] | None,
) -> dict[str, Any]:
    """Run changed files, promote merged manifests with rollback, then republish."""
    # Verify the caller's evidence before creating even an ephemeral output-side directory.
    # ``_carry_approval`` repeats this at the derived-recipe boundary; this first check keeps a
    # refused operation free of filesystem writes.
    if _safety.require_smoke() and not _safety.verify_smoke_token(smoke_token, rec.config_hash):
        return {
            **card,
            "status": "refused",
            "reason": (
                "run requires smoke evidence (AUDIO_AGENT_REQUIRE_SMOKE is set): run smoke on this "
                "recipe and pass its 'smoke_token'"
            ),
        }
    try:
        sandbox, sandbox_identity = _create_delta_sandbox(str(decision.uri))
    except OSError as exc:
        return {
            **card,
            "status": "error",
            "reason": f"the delta's working directory could not be created: {exc}",
        }
    try:
        return _execute_delta_in_sandbox(
            rec,
            decision,
            dp=dp,
            data=data,
            card=card,
            executor=executor,
            bootstrap_ray=bootstrap_ray,
            smoke_token=smoke_token,
            calibration=calibration,
            goal=goal,
            sandbox=sandbox,
        )
    finally:
        _remove_owned_temporary(sandbox, sandbox_identity)


def _execute_delta_in_sandbox(  # noqa: PLR0913 - split out so scratch ownership has one boundary
    rec: Recipe,
    decision: Any,  # noqa: ANN401 - delta.Delta without an eager import
    *,
    dp: Any,  # noqa: ANN401 - DataProfile
    data: str | None,
    card: dict[str, Any],
    executor: Any,  # noqa: ANN401
    bootstrap_ray: bool,
    smoke_token: str | None,
    calibration: dict[str, Any] | None,
    goal: dict[str, Any] | None,
    sandbox: str,
) -> dict[str, Any]:
    """Execute one delta inside a unique directory owned by the caller."""
    from nemo_curator.stages.audio.audio_agent import _verb_run
    from nemo_curator.stages.audio.audio_agent import delta as _delta

    prefix_rec, redirect, err = _delta.prefix_recipe(
        rec,
        prefix=decision.prefix,
        files=decision.files,
        sandbox=sandbox,
        sinks_=list(decision.sinks),
        inventory_key=getattr(dp, "inventory_key", "") or "",
    )
    if prefix_rec is None:
        return {**card, "status": "no_delta", "reason": err}
    # ``delta_run`` already held the caller's ``confirm`` to this recipe, so the approval is
    # established; what remains is to re-anchor it onto the prefix recipe that executes. A
    # refusal leaves no durable output; the invocation-owned sandbox is cleaned by the caller.
    inner_confirm, inner_token, refusal = _verb_validation._carry_approval(
        rec, prefix_rec, confirm=rec.config_hash, smoke_token=smoke_token
    )
    if refusal:
        return {**card, "status": "refused", "reason": refusal}
    started = time.perf_counter()
    inner: dict[str, Any] = {"status": "completed", "note": "nothing to run: every changed file was removed"}
    if decision.files:
        inner = _verb_run.run(
            prefix_rec,
            confirm=inner_confirm,
            data=data,
            executor=executor,
            bootstrap_ray=bootstrap_ray,
            smoke_token=inner_token,
            calibration=calibration,
            goal=goal,
        )
        if inner.get("status") != "completed":
            return {**card, "status": "failed", "reason": "the delta's run did not complete", "run": inner}
    elapsed = round(time.perf_counter() - started, 3)

    # Upsert every processed file so repeating an already successful delta cannot duplicate rows.
    stale = set(decision.change.touched) | set(decision.change.removed)
    merges: list[dict[str, Any]] = []
    outputs: list[_verb_outputs._StagedFileOutput] = []
    token = uuid.uuid4().hex
    try:
        for sink in decision.sinks:
            output = _verb_outputs._StagedFileOutput(
                stage=None,
                param=sink.param,
                final_path=sink.uri,
                staged_path=f"{sink.uri}.audio-agent-delta-{token}-{sink.index}.tmp",
            )
            outputs.append(output)
            kept, added, why = _delta.merge(
                sink,
                produced=redirect[sink.uri] if decision.files else None,
                stale=stale,
                key=sink.key,
                root=dp.inventory_root,
                prior_files=set(dp.inventory or {}) | set(decision.change.removed),
                produced_files=set(decision.change.touched),
                output_path=output.staged_path,
            )
            if why:
                return {**card, "status": "failed", "reason": why, "merged": []}
            merges.append({**sink.summary(), "rows_kept": kept, "rows_added": added})
        _verb_outputs._promote_staged_file_outputs(outputs)
    except Exception as exc:  # noqa: BLE001 - promotion rolls back before surfacing the failure
        return {**card, "status": "failed", "reason": f"delta output publication failed: {exc}", "merged": []}
    finally:
        _verb_outputs._discard_staged_file_outputs(outputs)

    published, problems = _delta.republish(
        rec,
        decision,
        dataset_key=dp.dataset_key(),
        fingerprint_tier=dp.fingerprint_tier,
        inventory=dict(dp.inventory),
        run_id=str(inner.get("run_id") or ""),
        added_sec=elapsed,
    )
    remaining = len(rec.stages) - decision.prefix
    result = {
        **card,
        # A prefix delta brings the checkpoint up to date, not the recipe's final output: the
        # stages past it still have to see every row. Reporting that as "completed" would present
        # a partial result as a whole one -- the one thing this verb promises never to do -- and a
        # host that reads the status and stops would hand the user yesterday's deliverable. The
        # status names the work that is left instead, and stays a success for the shell.
        "status": "completed" if remaining <= 0 else "tail_required",
        "ran_files": list(decision.files),
        "merged": merges,
        "published": published,
        "warnings": problems,
        "elapsed_sec": elapsed,
        "run": {k: inner.get(k) for k in ("status", "run_id", "output_paths") if k in inner},
        "next": (
            "the merged output covers every file; nothing further is needed"
            if remaining <= 0
            else f"the deliverable is NOT current yet: run the remaining {remaining} stage(s) with "
            "plan_continuation(execute=True, choice='extend') (CLI: continue --execute "
            "--choice extend --confirm), which now finds the merged manifest by an ordinary reuse probe"
        ),
    }
    if remaining > 0:
        result["tail"] = {
            "stages": remaining,
            "from_stage_index": decision.prefix,
            "stale_outputs": _stale_tail_outputs(rec, dp.dataset_key(), prefix=decision.prefix),
            "reason": decision.notes[0] if decision.notes else "the delta covers only a prefix of the recipe",
            # No acceptance verdict here on purpose: the criteria describe the recipe's final
            # output, which the tail has not produced yet. The continuation evaluates them.
            "acceptance": "judged after the remaining stage(s) run",
        }
    else:
        verdict = _delta_acceptance(rec, merged=merges, input_count=len(dp.inventory or {}))
        if verdict:
            result["acceptance"] = verdict
    return result


def _delta_acceptance(rec: Recipe, *, merged: list[dict[str, Any]], input_count: int) -> dict[str, Any]:
    """Judge the merged deliverable against the recipe's own success contract.

    A delta rewrites the user's output in place, so the bar they confirmed has to be re-checked
    against what is on disk NOW. Without this, ``status: completed`` says the curation finished
    while a ``must`` criterion it was gated on may be violated by the rows just merged in -- and
    the CLI exits 0, so a script ships it.

    Only called when the delta covered the whole recipe. For a prefix delta the final output does
    not exist yet, and a verdict computed over the checkpoint would be about a different artifact.
    """
    if not rec.acceptance_criteria or not merged:
        return {}
    deepest = merged[-1]
    rows = int(deepest.get("rows_kept") or 0) + int(deepest.get("rows_added") or 0)
    roles: list[str] = []
    keys: list[str] = []
    with contextlib.suppress(Exception):
        from nemo_curator.stages.audio.audio_agent.recipe import build_stages

        built, _ = build_stages(rec)
        roles, keys = _verb_publish._produced_roles_keys(built or [], None)
    return _verb_evidence._acceptance_result(
        rec,
        types.SimpleNamespace(accepted=rows, input_count=input_count),
        roles,
        keys,
        [str(m.get("uri") or "") for m in merged if m.get("uri")],
    )


def _stale_tail_outputs(rec: Recipe, dataset_key: str, *, prefix: int) -> list[str]:
    """Where the tail stages write, and therefore what still describes the corpus before the delta.

    Named rather than left to the reader: "run the remaining stages" does not tell anyone which
    files on disk are currently a lie, and those are the ones a user is about to ship.
    """
    from nemo_curator.stages.audio.audio_agent import artifacts as art_mod

    with contextlib.suppress(Exception):
        return [p.uri for p in art_mod.plan_steps(rec, dataset_key)[prefix:] if p.persists() and p.uri]
    return []
