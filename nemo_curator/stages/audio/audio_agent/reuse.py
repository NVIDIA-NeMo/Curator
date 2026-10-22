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

"""Reuse scan — find prior work for a recipe, and hand the user the choice.

Probes the artifact registry with the recipe's Merkle step keys and reports the longest
prefix that is provably safe to reuse, together with an approval card a human can actually
judge: what the earlier run was FOR, what it ran, on what data, where the output is, when,
how it scored, and how much time reusing it saves.

Two rules shape the UX, and both matter:

* **Never silent.** Reuse is always disclosed. Anything above a trivial saving asks first,
  because "you got yesterday's answer" is not a detail to bury in a log line.
* **Never nagging.** No candidate means no prompt at all. A saving *measured* under
  :data:`AUTO_REUSE_SEC` is simply taken and disclosed. A low-trust candidate is shown with
  its weakness spelled out and the *fresh* option pre-selected.

The word "measured" carries weight there. Silence about a stage's cost is not evidence that it
was cheap, and reading it that way is how an unmeasured hour of transcription qualified as
trivial. Nothing is auto-taken on an assumption; see :func:`_unpriced`.

See ``REUSE_ARCHITECTURE.md``."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from nemo_curator.stages.audio.audio_agent import _reuse_policy
from nemo_curator.stages.audio.audio_agent._reuse_history import (
    enrich_folder_run_cards,
    prior_on_path,
    run_overview,
    runs_on_path,
)
from nemo_curator.stages.audio.audio_agent._reuse_policy import AUTO_REUSE_SEC
from nemo_curator.stages.audio.audio_agent._reuse_summary import (
    SUMMARIZE_DIRECTIVE,
    goal_text,
    match_corpus_for_record,
    prompt_summary_match,
    resolve_pipeline_summary,
    summarize_pipeline,
)

if TYPE_CHECKING:
    from nemo_curator.stages.audio.audio_agent.artifacts import Artifact, StepPlan
    from nemo_curator.stages.audio.audio_agent.recipe import Recipe


def scan(recipe: Recipe, *, dataset_key: str, limit: int = 5) -> dict[str, Any]:
    """Find the longest safely reusable prefix of ``recipe`` and build the approval card.

    Returns ``decision`` (``already_done`` / ``incremental`` / ``fresh``, which
    ``verbs._attach_delta`` upgrades to ``delta`` on a miss a changed-file delta covers),
    ranked ``candidates``, an estimated saving, and whether to prompt.
    """
    from nemo_curator.stages.audio.audio_agent import artifacts as art_mod

    # An empty key means the caller did not identify the source dataset.  It is
    # not a legitimate identity shared by every unknown input: probing with it
    # can otherwise match an older empty-key artifact and turn "we do not know"
    # into "this exact data already ran".
    if not dataset_key:
        return {
            "decision": "fresh",
            "dataset_key": "",
            "reuse_point": None,
            "steps": [],
            "candidates": [],
            "estimated_saving_sec": 0.0,
            "prompt_user": False,
            "recommended": "fresh",
            "prior_on_other_data": None,
            "prior_unsaved": None,
            "offer": None,
            "rationale": "source data identity is unavailable; prior work cannot be matched safely",
        }

    plans = art_mod.plan_steps(recipe, dataset_key)
    probes: list[dict[str, Any]] = []
    reusable: list[tuple[StepPlan, Artifact]] = []
    for plan in plans:
        art, reasons = art_mod.lookup(plan.step_key, dataset_key=dataset_key)
        probes.append(
            {
                "stage_index": plan.index,
                "stage": plan.stage_ref,
                "step_key": plan.step_key,
                "found": art is not None,
                "reusable": bool(art is not None and not reasons),
                "blocked_by": reasons if art is not None else [],
            }
        )
        if art is not None and not reasons:
            reusable.append((plan, art))

    if not reusable:
        elsewhere = _prior_on_other_data(recipe, dataset_key)
        unsaved = _unsaved_prior_prefix(plans, dataset_key)
        return {
            "decision": "fresh",
            "dataset_key": dataset_key,
            "reuse_point": None,
            "steps": probes,
            "candidates": [],
            "estimated_saving_sec": 0.0,
            "prompt_user": False,  # nothing to offer -> no question
            "recommended": "fresh",
            "prior_on_other_data": elsewhere,
            "prior_unsaved": unsaved,
            "offer": _persist_offer(recipe, unsaved),
            "rationale": _fresh_rationale(probes, elsewhere, unsaved),
        }

    plan, artifact = reusable[-1]  # the deepest valid step; earlier keys are folded into it
    prefix = plan.index + 1
    n = len(plans)
    lost = _boundary_block(recipe, prefix) if prefix < n else None
    if lost:
        return {
            "decision": "fresh",
            "dataset_key": dataset_key,
            "reuse_point": None,
            "steps": probes,
            "candidates": [],
            "estimated_saving_sec": 0.0,
            "prompt_user": False,
            "recommended": "fresh",
            "rationale": (
                f"prior output exists through {plan.stage_ref}, but the remaining stage(s) need in-memory state "
                f"that a persisted artifact cannot carry ({lost}); resuming would silently drop it"
            ),
        }

    saving, measured = _saving(plans[:prefix])
    unpriced = _unpriced(plans[:prefix])
    decision = "already_done" if prefix == n else "incremental"
    candidates = _candidates(plans[:prefix], dataset_key=dataset_key, limit=limit)
    return {
        "decision": decision,
        "dataset_key": dataset_key,
        "reuse_point": reuse_point(plan, artifact),
        "reuse_stages": [p.stage_ref for p in plans[:prefix]],
        "run_stages": [p.stage_ref for p in plans[prefix:]],
        "steps": probes,
        "candidates": candidates,
        "estimated_saving_sec": round(saving, 1),
        "saving_is_lower_bound": not measured,
        "unpriced_stages": unpriced,
        "prompt_user": saving >= _reuse_policy.AUTO_REUSE_SEC or bool(unpriced),
        "recommended": _recommended(decision, candidates),
        "choices": _choices(decision),
        "rationale": (
            f"{prefix} of {n} stage(s) already produced output for this dataset key; "
            f"reuse it and run the remaining {n - prefix}"
            if decision == "incremental"
            else "this computation already ran for the matching dataset key; the output is ready to serve"
        ),
    }


def _fresh_rationale(
    probes: list[dict[str, Any]],
    elsewhere: dict[str, Any] | None = None,
    unsaved: dict[str, Any] | None = None,
) -> str:
    """Say WHY nothing was reused -- 'found it but the data changed' is very different
    from 'never ran this before', and only one of them is worth telling the user about."""
    blocked = [p for p in probes if p["found"] and p["blocked_by"]]
    if blocked:
        first = blocked[0]
        return f"prior work exists for {first['stage']} but is not reusable: {'; '.join(first['blocked_by'])}"
    if unsaved:
        return str(unsaved["note"])
    if elsewhere:
        when = f" on {elsewhere['created_at']}" if elsewhere.get("created_at") else ""
        if elsewhere.get("saved") is False:
            return (
                "this pipeline ran before on source data that has since changed "
                f"({elsewhere['dataset_key']}{when}), and it persisted nothing that a later run could "
                f"resume from -- add-checkpoint says where a manifest through {elsewhere['stage']} would go, "
                "which is also what a changed-file delta would then need"
            )
        return (
            "this pipeline has run before, but the detected source identity changed since then "
            f"(prior output through {elsewhere['stage']} came from {elsewhere['dataset_key']}{when}); "
            "nothing from it can be reused"
        )
    return "no prior artifact matches this pipeline on this data"


def _unsaved_prior_prefix(plans: list[StepPlan], dataset_key: str) -> dict[str, Any] | None:
    """Stages that already ran on THIS data in an earlier run but left nothing on disk.

    Only a stage with an output-location parameter publishes an artifact, so a pipeline whose
    middle stages compute in memory has nothing to resume from even when its step keys match an
    earlier run exactly. Without this the scan reports "no prior artifact matches this pipeline
    on this data" -- true, and misleading: the work *was* done, it simply was not saved.
    Recomputing it is the right behaviour. Being quiet about it is not, because the user cannot
    then tell the difference between "this is new" and "we are paying for this twice".

    Compares against the earlier run's *recorded* Merkle chain (``RunRecord.steps``) rather than
    re-deriving keys from its recipe, so this cannot drift from what that run actually executed.

    Asked of ``run_index`` rather than scanned by hand, which is both cheaper and wider: the scan
    parsed every record on disk, took the newest ``_MAX_PRIOR_RUNS`` of them, and only then looked
    for this dataset -- so on a busy store the run being described could sit one place past the
    cut and be reported as never having happened. The query filters on the dataset first and caps
    after, and falls back to the same JSON records when the index is unavailable.
    """
    from nemo_curator.stages.audio.audio_agent import run_index, run_store

    mine = [p.step_key for p in plans]
    # An empty dataset key means the caller gave us no data to identify. Two unknowns are not
    # the same dataset, and claiming prior work on that basis would be a guess dressed as a fact.
    if not mine or not dataset_key:
        return None
    best: dict[str, Any] | None = None
    for summary in run_index.find_runs(dataset_key=dataset_key, limit=_reuse_policy._MAX_PRIOR_RUNS):
        # Only a completed run proves the work was really done; a failed one proves nothing.
        if summary.get("status") != "completed":
            continue
        rec = run_store.load(str(summary.get("run_id") or ""))
        shared = _shared_prefix_len(mine, list(getattr(rec, "steps", None) or []))
        if shared and (best is None or shared > best["count"]):
            best = _unsaved_entry(rec, plans[:shared])
    return best


def _unsaved_entry(rec: Any, prefix: list[StepPlan]) -> dict[str, Any]:  # noqa: ANN401 - RunRecord
    """Describe a recomputed prefix, saying only what is true of THIS prefix.

    Why the resume point is missing has two genuinely different answers, and the earlier version
    of this asserted the first one unconditionally -- producing the note "none of them writes a
    file" about a prefix ending in ``ManifestWriterStage``, alongside advice to add a writer
    after the writer. An unverified claim, in the exact category the success contract exists to
    prevent, from the code meant to enforce it.
    """
    resume_point = prefix[-1]
    stages = [p.stage_ref for p in prefix]
    seconds = _prefix_seconds(rec, prefix)
    cost = f" (about {seconds}s last time)" if seconds else ""
    head = (
        f"{len(prefix)} stage(s) already ran for this dataset key in an earlier run "
        f"({', '.join(stages)}) and will be recomputed{cost}: "
    )
    if resume_point.persists():
        # It DID write somewhere. Nothing to resume from means the record is gone, not that the
        # work was ephemeral -- so recommending a writer would be advice for the wrong problem.
        why = (
            f"{resume_point.stage_ref} writes its output to {resume_point.uri!r}, but no valid "
            f"artifact record remains for it (pruned, or the run never published one)"
        )
    else:
        why = f"{resume_point.stage_ref} writes no file, so nothing was persisted to resume from"
    return {
        "count": len(prefix),
        "stages": stages,
        "run_id": getattr(rec, "run_id", None),
        "created_at": getattr(rec, "created_at", ""),
        "recompute_sec": seconds,
        "resume_point_persists": resume_point.persists(),
        # Worded once, here, so the scan rationale and the continuation gate say the same thing
        # rather than drifting into two descriptions of one fact.
        "note": head + why,
    }


def _shared_prefix_len(mine: list[str], theirs: list[str]) -> int:
    """How many leading step keys two runs have in common.

    A shared key means identical data, stages, settings, code and model versions up to that
    point -- the Merkle chain makes a match at position i a proof about everything before it.
    """
    n = 0
    for a, b in zip(mine, theirs, strict=False):
        if a != b:
            break
        n += 1
    return n


def reuse_point(plan: StepPlan, artifact: Artifact) -> dict[str, Any]:
    """The resume descriptor an executor needs: where the prior output is and what it contains.

    Built here for BOTH reuse engines. The parent-diff path used to hand over a bare list of the
    parent's output paths instead, which carried no artifact and so no validation -- and the
    executor, needing a ``uri``, refused to extend from it.
    """
    return {
        "stage_index": plan.index,
        "stage": plan.stage_ref,
        "step_key": plan.step_key,
        "run_id": artifact.run_id,
        "uri": artifact.uri,
        "kind": artifact.kind,
        "rows_in": artifact.rows_in,
        "rows": artifact.rows_out,
        "produced_roles": list(artifact.produced_roles),
        "produced_keys": list(artifact.produced_keys),
    }


def verified_point(recipe: Recipe, depth: int, *, dataset_key: str) -> tuple[dict[str, Any] | None, list[str]]:
    """The resume point for a prefix of ``depth`` stages, only if a VALID artifact backs it.

    Reuse depth claimed by one engine, proven against the registry the other engine uses, so both
    paths clear the same bar: existence, completeness, matching dataset, code version, determinism.
    """
    from nemo_curator.stages.audio.audio_agent import artifacts as art_mod

    if depth <= 0:
        return None, ["no reused stages to resume from"]
    if not dataset_key:
        return None, ["the source data was not identified, so prior work cannot be matched to it"]
    plans = art_mod.plan_steps(recipe, dataset_key)
    if depth > len(plans):
        return None, ["the claimed reuse is deeper than this pipeline"]
    step = plans[depth - 1]
    art, reasons = art_mod.lookup(step.step_key, dataset_key=dataset_key)
    if art is None or reasons:
        return None, reasons or ["no prior artifact for this step"]
    return reuse_point(step, art), []


def _runtime_name(stage_ref: str) -> str:
    """The name a stage reports metrics under, which is its own ``name`` field, not its class.

    ``ManifestWriterStage`` measures itself as ``manifest_writer`` and ``ASRStage`` as
    ``ASR_inference``, so no transformation of the class name finds them. Ask the class.
    """

    from nemo_curator.stages.audio.audio_agent._resolve import resolve_stage_class

    with contextlib.suppress(Exception):  # an unresolvable ref is the caller's problem, not ours
        return str(getattr(resolve_stage_class(stage_ref), "name", "") or stage_ref)
    return stage_ref


def _prefix_seconds(rec: Any, plans: list[StepPlan]) -> float | None:  # noqa: ANN401 - RunRecord | None
    """Seconds the earlier run spent on these stages, or ``None`` if it cannot be attributed.

    When any stage cannot be found we return ``None`` instead of a partial sum: a number presented
    as the cost of five stages that actually covers three is the kind of false precision this whole
    contract exists to prevent. Reading is delegated so this and the publish-time cost agree.
    """
    from nemo_curator.stages.audio.audio_agent.report import stage_duration_sec

    metrics = getattr(rec, "per_stage_metrics", None) or {}
    total = 0.0
    for plan in plans:
        name = _runtime_name(plan.stage_ref)
        if name not in metrics:
            return None
        total += stage_duration_sec(metrics, name)
    return round(total, 1)


def _persist_offer(recipe: Recipe, unsaved: dict[str, Any] | None) -> dict[str, Any] | None:
    """How to make this prefix reusable next time, using machinery that already exists.

    Deliberately a suggestion for the gate rather than something the agent does on its own. The
    alternative -- quietly caching intermediate state nobody asked for -- buys back seconds at
    the cost of an eviction policy, a staleness window, and a new way to serve wrong data
    silently. A writer the user agreed to is a visible file in a path they chose, and it already
    publishes an artifact through the normal path.

    Where that writer goes is asked of :mod:`checkpoint`, which simulates it, rather than being
    read off the end of the recomputed prefix. Those are different positions whenever the
    pipeline is still holding audio in memory there -- in the ALM recipe the prefix ends at the
    ASR stage and a manifest written after it crashes on the resident waveform, so the obvious
    advice was advice to break the run.
    """
    # A prefix that already ends in a writer needs no writer. Its output went to disk and the
    # missing piece is the artifact record, so this advice would not apply.
    if not unsaved or unsaved.get("resume_point_persists"):
        return None
    from nemo_curator.stages.audio.audio_agent import checkpoint

    spot, why = checkpoint.advise(recipe)
    if spot is not None:
        return spot.as_dict()
    # One action for every negative, with the distinction ("not worth it" / "you already have
    # one" / "the audio is still in memory") carried in prose. A code per case would be a
    # taxonomy to keep in sync with sentences that already say it.
    return {"action": "no_checkpoint", "why": why} if why else None


def _prior_on_other_data(recipe: Recipe, dataset_key: str) -> dict[str, Any] | None:
    """The same pipeline, previously run on a DIFFERENT source dataset.

    The dataset key is the root of the Merkle chain, so a changed dataset changes every step
    key and the ordinary probe finds nothing at all. Recomputing the chain against the datasets
    already in the registry is what turns a useless "never ran this before" into "you ran this,
    but your data moved on". Bounded by :data:`_MAX_OTHER_DATASETS`, and only ever reached on
    the miss path.
    """
    from nemo_curator.stages.audio.audio_agent import artifacts as art_mod

    for other in _known_dataset_keys():
        if not other or other == dataset_key:
            continue
        hits = [p for p in art_mod.plan_steps(recipe, other) if art_mod.load(p.step_key) is not None]
        if hits:
            art = art_mod.load(hits[-1].step_key)
            return {
                "dataset_key": other,
                "stage": hits[-1].stage_ref,
                "created_at": getattr(art, "created_at", ""),
            }
    return _ran_unsaved_elsewhere(recipe, dataset_key)


def _ran_unsaved_elsewhere(recipe: Recipe, dataset_key: str) -> dict[str, Any] | None:
    """A completed run of this pipeline on other data that left nothing on disk.

    Reached only when the artifact probe found nothing anywhere, and it looks where that probe
    structurally cannot: :func:`_known_dataset_keys` lists datasets that HAVE artifacts, so a
    pipeline computing entirely in memory is invisible to it. Saying "no prior artifact matches"
    about a pipeline that ran yesterday is true and reads as "this is new", which sends the user
    to the wrong problem -- the work is being paid for twice for want of somewhere to put it.
    """
    from nemo_curator.stages.audio.audio_agent import artifacts as art_mod
    from nemo_curator.stages.audio.audio_agent import run_store

    for summary in run_store.list_runs()[: _reuse_policy._MAX_PRIOR_RUNS]:
        other = str(summary.get("dataset_key") or "")
        if not other or other == dataset_key or summary.get("status") != "completed":
            continue
        rec = run_store.load(str(summary.get("run_id") or ""))
        chain = art_mod.plan_steps(recipe, other)
        shared = _shared_prefix_len([p.step_key for p in chain], list(getattr(rec, "steps", None) or []))
        if shared:
            return {
                "dataset_key": other,
                "stage": chain[shared - 1].stage_ref,
                "created_at": getattr(rec, "created_at", ""),
                "saved": False,
                "run_id": getattr(rec, "run_id", None),
            }
    return None


def _known_dataset_keys() -> list[str]:
    """Distinct source datasets seen before (index first, JSON records as the fallback).

    Underscored but NOT private: imported by ``delta``, which needs the same list to explain a
    miss ("prior work exists, but for another dataset") rather than just reporting nothing.
    """
    from nemo_curator.stages.audio.audio_agent import run_index

    keys = run_index.dataset_keys(limit=_reuse_policy._MAX_OTHER_DATASETS)
    if keys:
        return keys
    from nemo_curator.stages.audio.audio_agent import artifacts as art_mod

    seen: list[str] = []
    for art in art_mod.list_artifacts():
        if art.dataset_key and art.dataset_key not in seen:
            seen.append(art.dataset_key)
        if len(seen) >= _reuse_policy._MAX_OTHER_DATASETS:
            break
    return seen


def _boundary_block(recipe: Recipe, prefix: int) -> str | None:
    from nemo_curator.stages.audio.audio_agent.continuation import _resume_breaks_on_disk_boundary

    return _resume_breaks_on_disk_boundary(recipe, prefix)


def _unpriced(plans: list[StepPlan]) -> list[str]:
    """Stages here that could have cost real time and whose time nobody recorded.

    The gap this closes: :func:`_saving` counts an unmeasured step as zero seconds, so a prefix
    nobody timed scored the same as a genuinely quick one and slid under the auto-take threshold
    written for milliseconds -- an hour of transcription served without a question.

    "Unmeasured" alone is the wrong trigger, and trying it that way turned the gate into a
    permanent nag: most stages hold their output in memory, never persist, and so never have a
    duration, which is entirely normal and says nothing about cost. Two things narrow it to the
    cases that matter. A deepest artifact carrying ``cumulative_sec`` has already priced the
    whole prefix including the steps that persisted nothing, so nothing is unknown. And of what
    remains, only work the card calls expensive counts -- reading a manifest is cheap whether or
    not anyone timed it, and pretending otherwise spends the user's attention on nothing.
    """
    from nemo_curator.stages.audio.audio_agent import artifacts as art_mod

    deepest = art_mod.load(plans[-1].step_key) if plans else None
    if float(getattr(deepest, "cumulative_sec", 0.0) or 0.0) > 0:
        return []
    out: list[str] = []
    for plan in plans:
        art = art_mod.load(plan.step_key)
        if art is not None and art.duration_sec:
            continue
        if art_mod.stage_is_costly(plan.stage_ref):
            out.append(plan.stage_ref)
    return out


def _saving(plans: list[StepPlan]) -> tuple[float, bool]:
    """``(seconds saved, every_step_was_measured)`` for reusing this prefix.

    Two lower bounds, and the larger wins: the sum of per-artifact durations, and the deepest
    artifact's ``cumulative_sec`` (which also covers the expensive steps that persisted nothing
    and so have no artifact of their own — without it, a pipeline that only writes at the end
    would report its writer's milliseconds and serve an hour-old result without asking). Only
    real measurements count: an unmeasured step contributes 0 and flips the flag, so the number
    is reported as a lower bound rather than padded with guesswork.
    """
    from nemo_curator.stages.audio.audio_agent import artifacts as art_mod

    total = 0.0
    measured = True
    for plan in plans:
        art = art_mod.load(plan.step_key)
        if art is None or not art.duration_sec:
            measured = False
            continue
        total += float(art.duration_sec)

    deepest = art_mod.load(plans[-1].step_key) if plans else None
    cumulative = float(getattr(deepest, "cumulative_sec", 0.0) or 0.0)
    return max(total, cumulative), measured


def _candidates(plans: list[StepPlan], *, dataset_key: str, limit: int) -> list[dict[str, Any]]:
    """Approval cards for the reusable artifacts, deepest (most work saved) first."""
    from nemo_curator.stages.audio.audio_agent import artifacts as art_mod
    from nemo_curator.stages.audio.audio_agent import run_store

    out: list[dict[str, Any]] = []
    for plan in reversed(plans):
        art = art_mod.load(plan.step_key)
        if art is None or art_mod.invalid_reasons(art, dataset_key=dataset_key):
            continue
        out.append(_card(art, run_store.load(art.run_id) if art.run_id else None))
        if len(out) >= limit:
            break
    return out


def _card(art: Artifact, run: Any) -> dict[str, Any]:  # noqa: ANN401 - RunRecord | None
    """One reuse candidate, described so a human can decide without reading JSON."""
    recipe = (getattr(run, "recipe", None) or {}) if run else {}
    stages = [s.get("ref") for s in (recipe.get("stages") or []) if isinstance(s, dict)]
    trust, weaknesses = _trust(art)
    return {
        "step_key": art.step_key,
        "objective": _objective(run),
        "pipeline": stages or [art.stage_ref],
        "through_stage": art.stage_ref,
        "key_params": _key_params(art.semantic_params),
        "input": getattr(run, "data_source", None),
        "output": art.uri,
        "output_kind": art.kind,
        "rows": art.rows_out,
        "executed_at": art.created_at or getattr(run, "created_at", ""),
        "duration_sec": art.duration_sec,
        "metrics": _metrics(art, run),
        "estimated_saving_sec": art.cumulative_sec or art.duration_sec,
        "trust": trust,
        "weaknesses": weaknesses,
        "run_id": art.run_id,
    }


def _objective(run: Any) -> str:  # noqa: ANN401
    from nemo_curator.stages.audio.audio_agent import _reuse_summary

    goal = (getattr(run, "goal", None) or {}) if run else {}
    for key in ("task", "objective", "request", "summary"):
        if goal.get(key):
            return str(goal[key])
    return str(goal) if goal else _reuse_summary._NO_OBJECTIVE


def _key_params(params: dict[str, Any]) -> dict[str, Any]:
    """The few params that most identify a run, preferring thresholds and model choices."""
    interesting = [
        k for k in params if any(t in k for t in ("model", "threshold", "min_", "max_", "target", "rate", "type"))
    ]
    chosen = (interesting or list(params))[: _reuse_policy._CARD_PARAMS]
    return {k: params[k] for k in chosen}


def _metrics(art: Artifact, run: Any) -> dict[str, Any]:  # noqa: ANN401
    out: dict[str, Any] = dict(art.metrics or {})
    if run is not None:
        accepted = getattr(run, "accepted", 0)
        total = getattr(run, "input_count", 0)
        if total:
            out["retained"] = f"{accepted}/{total}"
        result = getattr(run, "acceptance_result", None) or {}
        if result.get("overall"):
            out["acceptance"] = result["overall"]
    return out


def _trust(art: Artifact) -> tuple[str, list[str]]:
    """``("high"|"low", why_it_is_low)``. Low trust pre-selects a fresh run.

    The weaknesses are the artifact's own cautions, worded once beside the check that decides
    them, plus the one thing that is a weakness only here: a freshness window that has not
    expired yet still means a re-fetch could differ.
    """
    from nemo_curator.stages.audio.audio_agent import artifacts as art_mod

    weaknesses = art_mod.caution_reasons(art)
    if art.ttl_sec:
        weaknesses.append("output has a freshness window (re-fetching could differ)")
    return ("low" if weaknesses else "high"), weaknesses


def _recommended(decision: str, candidates: list[dict[str, Any]]) -> str:
    """Default to fresh whenever trust is anything less than high."""
    if any(c.get("trust") != "high" for c in candidates):
        return "fresh"
    return "as_is" if decision == "already_done" else "extend"


def _choices(decision: str) -> list[dict[str, str]]:
    extend = {
        "id": "extend",
        "label": "Extend it",
        "effect": "reuse the finished stages and run only what is new",
    }
    as_is = {
        "id": "as_is",
        "label": "Use it as-is",
        "effect": "serve the existing output and re-check it against the current success criteria",
    }
    fresh = {"id": "fresh", "label": "Run fresh", "effect": "ignore prior work and recompute everything"}
    return [as_is, extend, fresh] if decision == "already_done" else [extend, as_is, fresh]


__all__ = [
    "AUTO_REUSE_SEC",
    "SUMMARIZE_DIRECTIVE",
    "enrich_folder_run_cards",
    "goal_text",
    "match_corpus_for_record",
    "prior_on_path",
    "prompt_summary_match",
    "resolve_pipeline_summary",
    "reuse_point",
    "run_overview",
    "runs_on_path",
    "scan",
    "summarize_pipeline",
    "verified_point",
]
