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

See ``REUSE_ARCHITECTURE.md``.
"""

from __future__ import annotations

import contextlib
import re
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from nemo_curator.audio_agent.artifacts import Artifact, StepPlan
    from nemo_curator.audio_agent.recipe import Recipe

# Below this, reusing is not worth a question -- take it and say so in the report.
AUTO_REUSE_SEC = 30.0
# How many important params to show per stage on the card (enough to recognise the run).
_CARD_PARAMS = 4
# How many previously-seen datasets to re-key against when explaining a miss.
_MAX_OTHER_DATASETS = 20
# How far back through run history to look when explaining a miss.
_MAX_PRIOR_RUNS = 50
# How many prior runs on the same folder to describe in full. A choice between three
# recognisable runs is one a person can make; a list of fifteen is one they skip.
_MAX_PRIOR_CARDS = 3
# Tokens shorter than this add noise to prompt↔summary matching (a, to, of, …).
_MIN_MATCH_TOKEN_LEN = 2
# Path-like params identify WHERE, not WHAT the pipeline did, and two runs of the same recipe
# on the same folder differ only in their scratch dirs — that is noise when comparing intent.
_SUMMARY_PATH_SUFFIXES = ("_dir", "_path", "_uri", "_file", "_folder")
_SUMMARY_PATH_KEYS = frozenset(
    {
        "output_path",
        "data_dir",
        "raw_data_dir",
        "resampled_audio_dir",
        "audio_dir",
        "manifest_filepath",
    }
)
# Written where a run has no recorded goal. Matched against as a literal so its filler words
# ("run", "that", "for") never count as intent overlap.
_NO_OBJECTIVE = "(objective not recorded for that run)"

SUMMARIZE_DIRECTIVE = (
    "pipeline_summary is the complete stage list with every behavioural param -- it is written "
    "for comparison, not for reading aloud. Never paste it verbatim. Retell it in one plain line: "
    "what the run produced, plus only the settings that differ between the priors you are showing "
    "or that the user asked about (thresholds, rates, channels, filters). Drop plumbing params."
)

_HOST_DIRECTIVE_FOLDER_RUNS = (
    "Before inventing a recipe, compare the user's current request to each prior's "
    "prompt (recorded goal) and pipeline_summary. Prefer a prior whose capabilities "
    "cover the full request over one that only covers a subset. Show the top 2–3 with "  # noqa: RUF001
    "stats; on pick use delta-run --from-run <run_id>. Do not invent a competing recipe first. " + SUMMARIZE_DIRECTIVE
)


def scan(recipe: Recipe, *, dataset_key: str, limit: int = 5) -> dict[str, Any]:
    """Find the longest safely reusable prefix of ``recipe`` and build the approval card.

    Returns ``decision`` (``already_done`` / ``incremental`` / ``fresh``, which
    ``verbs._attach_delta`` upgrades to ``delta`` on a miss a changed-file delta covers),
    ranked ``candidates``, an estimated saving, and whether to prompt.
    """
    from nemo_curator.audio_agent import artifacts as art_mod

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
        "prompt_user": saving >= AUTO_REUSE_SEC or bool(unpriced),
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
    from nemo_curator.audio_agent import run_index, run_store

    mine = [p.step_key for p in plans]
    # An empty dataset key means the caller gave us no data to identify. Two unknowns are not
    # the same dataset, and claiming prior work on that basis would be a guess dressed as a fact.
    if not mine or not dataset_key:
        return None
    best: dict[str, Any] | None = None
    for summary in run_index.find_runs(dataset_key=dataset_key, limit=_MAX_PRIOR_RUNS):
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
    from nemo_curator.audio_agent import artifacts as art_mod

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
    import contextlib

    from nemo_curator.audio_agent._resolve import resolve_stage_class

    with contextlib.suppress(Exception):  # an unresolvable ref is the caller's problem, not ours
        return str(getattr(resolve_stage_class(stage_ref), "name", "") or stage_ref)
    return stage_ref


def _prefix_seconds(rec: Any, plans: list[StepPlan]) -> float | None:  # noqa: ANN401 - RunRecord | None
    """Seconds the earlier run spent on these stages, or ``None`` if it cannot be attributed.

    When any stage cannot be found we return ``None`` instead of a partial sum: a number presented
    as the cost of five stages that actually covers three is the kind of false precision this whole
    contract exists to prevent. Reading is delegated so this and the publish-time cost agree.
    """
    from nemo_curator.audio_agent.report import stage_duration_sec

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
    from nemo_curator.audio_agent import checkpoint

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
    from nemo_curator.audio_agent import artifacts as art_mod

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
    from nemo_curator.audio_agent import artifacts as art_mod
    from nemo_curator.audio_agent import run_store

    for summary in run_store.list_runs()[:_MAX_PRIOR_RUNS]:
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


def runs_on_path(
    source_path: str,
    *,
    since: str | None = None,
    limit: int = _MAX_PRIOR_RUNS,
    completed_only: bool = False,
) -> list[dict[str, Any]]:
    """Index rows for runs that read the SAME folder, newest first.

    One definition of "the same folder", shared by :func:`prior_on_path` and the ``runs`` verb,
    and it is a canonical-path comparison rather than the string equality SQL would do:
    ``data_source`` is recorded as the caller passed it -- a relative path, a trailing slash, a
    symlink -- so an exact compare misses the same folder named two ways. The scan is bounded and
    already newest-first, and a folder is asked about once per request, not once per file.
    """
    from nemo_curator.audio_agent import run_index
    from nemo_curator.audio_agent.input_identity import canonical_source

    if not source_path:
        return []
    try:
        canonical = canonical_source(source_path)
    except (TypeError, ValueError):
        return []

    want = int(limit)
    # Scan wider than the caller's cap: the cap counts runs on THIS folder, and taking only the
    # newest ``want`` records overall would let a folder curated before a handful of unrelated
    # runs report as never curated at all -- the exact blind spot this exists to close.
    scan = max(want, _MAX_PRIOR_RUNS) if want >= 0 else -1
    out: list[dict[str, Any]] = []
    for row in run_index.find_runs(since=since, limit=scan):
        if completed_only and str(row.get("status") or "") != "completed":
            continue
        if not _same_folder(row.get("data_source"), canonical):
            continue
        out.append(row)
        if 0 <= want <= len(out):
            break
    return out


def prior_on_path(
    recipe: Recipe,
    *,
    source_path: str,
    current_inventory: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Prior runs that read the SAME source folder, matched by path -- not by recipe or data.

    The step-key matchers (:func:`scan`, :func:`_prior_on_other_data`, ``delta``) all key on the
    Merkle chain, which a changed source stage or a changed corpus moves wholesale, so a folder
    curated twenty minutes ago with a slightly different pipeline is invisible to every one of
    them. This one is anchored on the folder a run READ, which does not move, and reports it for
    a human to judge: when it ran, what it did (as a diff against the current plan), and what has
    changed in the folder since. It never reuses anything; the recommendation is advice.

    Returns the closest match's card at the top level -- one prior run is the ordinary case, and
    a host reading ``note`` should not have to know about ranking -- plus ``count`` and the
    ranked ``matches``. Closest means: the same pipeline first, then fewest structural
    differences, then fewest changed params, then most recent. Cards are built for at most
    :data:`_MAX_PRIOR_CARDS` of them, because a choice between three recognisable runs is one a
    person can make and a choice between fifteen is not.

    Reached only on the miss path, and skipped when the source is not one local folder -- a
    generated or multi-manifest source has no single path to anchor on.
    """
    from nemo_curator.audio_agent import run_store
    from nemo_curator.audio_agent.recipe import Recipe as _Recipe

    semantic = recipe.compute_semantic_hash()
    ranked: list[_PathMatch] = []
    for row in runs_on_path(source_path, limit=_MAX_PRIOR_RUNS, completed_only=True):
        record = run_store.load(str(row.get("run_id") or ""))
        if record is None:
            continue
        try:
            prior_recipe = _Recipe.from_dict(record.recipe) if isinstance(record.recipe, dict) else None
        except ValueError:  # a record in a shape this build cannot parse is history, not an error
            prior_recipe = None
        ranked.append(
            _PathMatch(
                record=record,
                recipe=prior_recipe,
                diff=_recipe_diff(prior_recipe, recipe) if prior_recipe is not None else {},
                same_recipe=bool(record.semantic_hash) and record.semantic_hash == semantic,
            )
        )
    if not ranked:
        return None

    # A stable sort on closeness over a list that arrives newest-first leaves recency as the
    # tiebreak, without a mixed-direction sort key.
    ranked.sort(key=_closeness)
    matches = [_prior_on_path_card(match, current_inventory=current_inventory) for match in ranked[:_MAX_PRIOR_CARDS]]
    best = matches[0]
    return {
        **best,
        "count": len(ranked),
        "matches": matches,
        "note": _all_matches_note(best["note"], len(ranked)),
        "next": _prior_next_steps(best["run_id"], source_path),
    }


class _PathMatch(NamedTuple):
    """One prior run on this folder, with the comparison against the current plan already made.

    The diff is computed before ranking (it is what ranking sorts on) and carried rather than
    recomputed, so a card and its rank cannot describe the run differently.
    """

    record: Any
    recipe: Recipe | None
    diff: dict[str, Any]
    same_recipe: bool


def _closeness(match: _PathMatch) -> tuple[int, int, int]:
    """Sort key: the run whose pipeline is nearest the current plan comes first.

    Nearest matters more than newest because the question the notice answers is "can this work
    be reused", and the same pipeline can be, while last night's different one cannot.
    """
    if match.recipe is None:
        return (2, 0, 0)  # its recipe could not be read, so nothing can be said about closeness
    diff = match.diff
    return (
        0 if match.same_recipe or diff.get("identical") else 1,
        len(diff.get("added_stages") or []) + len(diff.get("removed_stages") or []),
        len(diff.get("changed_params") or []),
    )


def _same_folder(stored: Any, canonical: str) -> bool:  # noqa: ANN401 - a stored path or None
    """Whether a recorded ``data_source`` names the same folder as ``canonical``."""
    from nemo_curator.audio_agent.input_identity import canonical_source

    if not isinstance(stored, str) or not stored:
        return False
    try:
        return canonical_source(stored) == canonical
    except (TypeError, ValueError):
        return False


def _prior_on_path_card(
    match: _PathMatch,
    *,
    current_inventory: dict[str, str] | None,
) -> dict[str, Any]:
    """Assemble the human-facing account of one prior run on this folder."""
    record = match.record
    data_delta = _data_delta_since(record, current_inventory)
    return {
        "run_id": record.run_id,
        "created_at": record.created_at,
        "goal": record.goal or None,
        "prompt": _objective(record),
        "pipeline_summary": resolve_pipeline_summary(record),
        "same_recipe": match.same_recipe,
        "prior_stages": [s.ref for s in (match.recipe.stages if match.recipe else [])],
        "prior_output_paths": list(getattr(record, "output_paths", []) or [])[:8],
        "prior_input_count": int(getattr(record, "input_count", 0) or 0),
        "recipe_diff": match.diff,
        "data_delta": data_delta,
        "recommendation": _prior_recommendation(match.same_recipe, match.diff, data_delta),
        "note": _prior_note(record, match.same_recipe, match.diff, data_delta),
    }


def _all_matches_note(best_note: str, count: int) -> str:
    """The closest match's sentence, plus the fact that there are others to choose from."""
    if count <= 1:
        return best_note
    others = count - 1
    return f"{best_note} ({others} other run(s) also read this folder; see 'matches'.)"


def _prior_next_steps(run_id: str, source_path: str) -> dict[str, str]:
    """The two commands that turn this notice into an answer, spelled out with the real ids.

    Named here rather than left to the host to compose, because the failure mode this whole
    notice exists for was a host that had the facts and did not act on them. ``inspect`` is what
    to show when the user asks what that run did; ``adopt`` re-runs its exact pipeline over only
    what has changed since -- the recipe is loaded from the record, so it cannot drift.
    """
    module = "python -m nemo_curator.audio_agent"
    return {
        "inspect": f"{module} runs --run-id {run_id}",
        "adopt": f"{module} delta-run --from-run {run_id} --data {source_path}",
    }


def run_overview(record: Any) -> dict[str, Any]:  # noqa: ANN401 - RunRecord without an eager import
    """A compact account of one stored run: what it did, to what, and how it came out.

    The ``runs --run-id`` payload is the whole record -- every param of every stage, the full
    step-key chain, per-stage metrics -- which is what tracing needs and not what a person needs
    after being told "you curated this folder before". Their next question is "what did that
    do?", and answering it from the record's own fields is the step between a notice and an
    informed choice. Nothing here is derived from anything but the record, so it cannot claim
    more than that run actually reported.
    """
    from nemo_curator.audio_agent.recipe import Recipe as _Recipe

    raw = record.recipe if isinstance(record.recipe, dict) else {}
    try:
        recipe = _Recipe.from_dict(raw)
    except ValueError:
        recipe = None
    stages = (
        [s.ref for s in recipe.stages]
        if recipe is not None
        else [str(s.get("ref")) for s in (raw.get("stages") or []) if isinstance(s, dict)]
    )
    return {
        "run_id": record.run_id,
        "created_at": record.created_at,
        "status": record.status,
        "objective": _objective(record),
        "prompt": _objective(record),
        "pipeline_summary": resolve_pipeline_summary(record),
        "pipeline": stages,
        "key_params": _pipeline_key_params(recipe),
        "data": {
            "source": getattr(record, "data_source", None),
            "dataset_key": getattr(record, "dataset_key", None),
            "fingerprint_tier": getattr(record, "fingerprint_tier", ""),
            "input_count": int(getattr(record, "input_count", 0) or 0),
        },
        "outputs": list(getattr(record, "output_paths", None) or []),
        "stats": _run_stats(record),
        "acceptance": _acceptance_view(record),
        "reuse": dict(getattr(record, "reuse", None) or {}),
    }


def summarize_pipeline(recipe: Recipe | None) -> str:
    """What the run did: every stage with the params that decide its behaviour.

    Written onto a successful run so a later session can compare a new request to "what that
    run did" without loading the full recipe. Complete and uncapped on purpose -- truncating
    here is guesswork about which threshold matters, and a clipped line makes two runs that
    differ only in that threshold look identical. Condensing for display is the host's job;
    ``host_directive`` on the payload says so. Locations are dropped (see the constants above).
    """
    if recipe is None:
        return ""
    parts: list[str] = []
    for stage in recipe.stages:
        params = _summary_params(stage.semantic_params())
        if params:
            parts.append(f"{stage.ref}({', '.join(f'{k}={v}' for k, v in params)})")
        else:
            parts.append(stage.ref)
    return " -> ".join(parts)


def _summary_params(params: dict[str, Any]) -> list[tuple[str, Any]]:
    """Behavioural params in recipe order: everything bar locations and no-op settings."""
    return [(k, v) for k, v in params.items() if not _is_location_param(k, v) and not _is_no_constraint(k, v)]


def _is_no_constraint(key: str, value: Any) -> bool:  # noqa: ANN401
    """A cap that caps nothing says nothing about the run -- ``max_samples=-1`` is not a choice."""
    if value is None:
        return True
    key_l = key.lower()
    unbounded_limit = key_l.endswith(("samples", "rows", "limit", "count"))
    return bool(unbounded_limit and isinstance(value, int) and value <= 0)


def _is_location_param(key: str, value: Any) -> bool:  # noqa: ANN401
    key_l = key.lower()
    if key_l in _SUMMARY_PATH_KEYS:
        return True
    if any(key_l.endswith(suffix) for suffix in _SUMMARY_PATH_SUFFIXES):
        return True
    return isinstance(value, str) and (value.startswith(("/", "~")))


def _match_aliases(key: str, value: Any) -> list[str]:  # noqa: ANN401
    """Words a person would use for a numeric param, added to the match corpus only.

    People ask for "16 kHz mono", recipes store ``target_sample_rate=16000`` and
    ``target_channels=1``. The aliases live here rather than in the displayed summary so
    matching does not push machine spellings like ``16000/16kHz`` in front of the reader.
    """
    key_l = key.lower()
    if "sample_rate" in key_l and isinstance(value, (int, float)) and value >= 1000:  # noqa: PLR2004
        rate = int(value)
        khz = rate // 1000
        return [str(rate), str(khz), f"{khz}k", f"{khz}khz"]
    if "channel" in key_l and isinstance(value, int):
        return {1: ["mono"], 2: ["stereo"]}.get(value, [])
    return []


def resolve_pipeline_summary(record: Any, recipe: Recipe | None = None) -> str:  # noqa: ANN401
    """Stored summary when present; otherwise derive from the record's recipe (older runs)."""
    stored = str(getattr(record, "pipeline_summary", "") or "").strip()
    if stored:
        return stored
    if recipe is not None:
        return summarize_pipeline(recipe)
    raw = getattr(record, "recipe", None)
    if not isinstance(raw, dict):
        return ""
    from nemo_curator.audio_agent.recipe import Recipe as _Recipe

    try:
        return summarize_pipeline(_Recipe.from_dict(raw))
    except ValueError:
        return ""


def goal_text(goal: dict[str, Any] | str | None) -> str:
    """Normalize a current or stored goal into a single comparable string."""
    if goal is None:
        return ""
    if isinstance(goal, str):
        return goal.strip()
    if isinstance(goal, dict):
        for key in ("task", "objective", "request", "summary"):
            if goal.get(key):
                return str(goal[key]).strip()
        return str(goal).strip() if goal else ""
    return str(goal).strip()


def prompt_summary_match(
    current_goal: dict[str, Any] | str | None,
    prior_prompt: str,
    pipeline_summary: str,
    *,
    match_corpus: str | None = None,
) -> dict[str, Any]:
    """How much of the current request is covered by the prior's prompt + pipeline.

    ``pipeline_summary`` is the short display string; ``match_corpus`` (when provided) is the
    uncapped stage+param text used for scoring so a truncated one-liner cannot hide a filter
    that actually ran. Score is coverage of current-request tokens. Empty current goal → 0.
    """
    current = _match_tokens(goal_text(current_goal))
    if match_corpus is None:
        prompt = "" if prior_prompt.strip() == _NO_OBJECTIVE else prior_prompt
        match_corpus = f"{prompt} {pipeline_summary}"
    prior = _match_tokens(match_corpus)
    if not current:
        return {"score": 0.0, "matched": [], "unmatched": [], "basis": "prior_prompt+pipeline_summary"}
    matched = sorted(current & prior)
    unmatched = sorted(current - prior)
    return {
        "score": round(len(matched) / len(current), 3),
        "matched": matched,
        "unmatched": unmatched,
        "basis": "prior_prompt+pipeline_summary",
    }


def match_corpus_for_record(record: Any, recipe: Recipe | None = None) -> str:  # noqa: ANN401
    """Uncapped prompt + stage refs + identifying params for ranking (not for display)."""
    # A run with no recorded goal must score on its pipeline alone. Left in, the placeholder's
    # filler ("run", "that", "for") counts as intent overlap with almost any request.
    prompt = _objective(record)
    if prompt.strip() == _NO_OBJECTIVE:
        prompt = ""
    if recipe is None:
        raw = getattr(record, "recipe", None)
        if isinstance(raw, dict):
            from nemo_curator.audio_agent.recipe import Recipe as _Recipe

            with contextlib.suppress(ValueError):
                recipe = _Recipe.from_dict(raw)
    parts = [prompt, resolve_pipeline_summary(record, recipe)]
    if recipe is not None:
        for stage in recipe.stages:
            parts.append(stage.ref)
            for key, value in _summary_params(stage.semantic_params()):
                parts.append(f"{key}={value}")
                parts.extend(_match_aliases(key, value))
    return " ".join(p for p in parts if p)


def _match_tokens(text: str) -> set[str]:
    """Alphanumeric tokens plus CamelCase splits so ``UTMOSFilterStage`` yields ``filter``."""
    lower = text.lower()
    tokens = {t for t in re.findall(r"[a-z0-9]+", lower) if len(t) >= _MIN_MATCH_TOKEN_LEN}
    for piece in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", text):
        p = piece.lower()
        if len(p) >= _MIN_MATCH_TOKEN_LEN:
            tokens.add(p)
    return tokens


def enrich_folder_run_cards(
    rows: list[dict[str, Any]],
    *,
    goal: dict[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Attach prompt, pipeline_summary and optional match score to thin index rows.

    Used by ``runs --data <folder> [--goal ...]`` so a host can compare the current
    request to prior work *before* inventing a recipe. Ranking (when ``goal`` is set)
    is by match score, then recency — never by stage edit-distance to a draft recipe.
    """
    from nemo_curator.audio_agent import run_store
    from nemo_curator.audio_agent.recipe import Recipe as _Recipe

    cards: list[dict[str, Any]] = []
    for row in rows:
        rid = str(row.get("run_id") or "")
        record = run_store.load(rid) if rid else None
        if record is None:
            cards.append({**row, "card_incomplete": True})
            continue
        overview = run_overview(record)
        prompt = overview["prompt"]
        summary = overview["pipeline_summary"]
        recipe = None
        if isinstance(record.recipe, dict):
            with contextlib.suppress(ValueError):
                recipe = _Recipe.from_dict(record.recipe)
        card = {
            **row,
            "goal": record.goal or None,
            "prompt": prompt,
            "pipeline_summary": summary,
            "pipeline": overview["pipeline"],
            "key_params": overview["key_params"],
            "input_count": overview["data"]["input_count"],
            "accepted": overview["stats"]["accepted"],
            "elapsed_sec": overview["stats"]["elapsed_sec"],
            "outputs": list(overview["outputs"] or [])[:8],
            "acceptance": overview["acceptance"],
            "next": _prior_next_steps(rid, str(getattr(record, "data_source", "") or "")),
        }
        if goal is not None:
            card["match"] = prompt_summary_match(
                goal,
                prompt,
                summary,
                match_corpus=match_corpus_for_record(record, recipe),
            )
        cards.append(card)

    out: dict[str, Any] = {"runs": cards, "host_directive": _HOST_DIRECTIVE_FOLDER_RUNS}
    if goal is not None:
        # Among priors that match the request equally well, prefer the one that already
        # processed more of the folder: adopting it leaves a smaller delta to recompute.
        # ``input_count`` is the cheap proxy -- the exact overlap costs a delta probe per run.
        cards.sort(
            key=lambda c: (
                float((c.get("match") or {}).get("score") or 0.0),
                int(c.get("input_count") or 0),
                -len((c.get("match") or {}).get("unmatched") or []),
                str(c.get("created_at") or ""),
            ),
            reverse=True,
        )
        out["runs"] = cards
        out["goal"] = goal_text(goal)
        out["ranked_by"] = "current_prompt vs prior_prompt + pipeline_summary, then prior coverage"
    return out


def _pipeline_key_params(recipe: Recipe | None) -> list[dict[str, Any]]:
    """Per stage, the few params that identify what it was configured to do.

    A list rather than a mapping by stage name: a pipeline can hold the same stage twice (two
    writers, two resamplers), and keying by ref would silently drop one of them and show the
    other's settings as if they were both.
    """
    if recipe is None:
        return []
    out: list[dict[str, Any]] = []
    for stage in recipe.stages:
        params = _key_params(stage.semantic_params())
        if params:
            out.append({"stage": stage.ref, "params": params})
    return out


def _run_stats(record: Any) -> dict[str, Any]:  # noqa: ANN401 - RunRecord
    """What the run cost and how much of the input survived it, as it was measured."""
    from nemo_curator.audio_agent.report import stage_duration_sec

    metrics = dict(getattr(record, "per_stage_metrics", None) or {})
    timed = {name: stage_duration_sec(metrics, name) for name in metrics}
    slowest = sorted((s for s in timed.items() if s[1]), key=lambda pair: pair[1], reverse=True)
    return {
        "elapsed_sec": float(getattr(record, "elapsed_sec", 0.0) or 0.0),
        "input_count": int(getattr(record, "input_count", 0) or 0),
        "accepted": int(getattr(record, "accepted", 0) or 0),
        # Kept at the precision the metrics were written with: rounding a 20 ms stage to one
        # decimal prints "0.0 seconds" beside a name, which reads as a measurement failure.
        "slowest_stages": [{"stage": name, "seconds": round(sec, 3)} for name, sec in slowest[:_CARD_PARAMS]],
    }


def _acceptance_view(record: Any) -> dict[str, Any]:  # noqa: ANN401 - RunRecord
    """The run's verdict against its own success bar, with each criterion named.

    Reported as ``not_recorded`` rather than as a pass when the run declared no contract: a run
    that was never checked is not a run that succeeded, and this view is read to decide whether
    to trust its output.
    """
    result = dict(getattr(record, "acceptance_result", None) or {})
    if not result:
        return {"overall": "not_recorded", "criteria": []}
    return {
        "overall": result.get("overall") or "not_recorded",
        "verdict": result.get("verdict") or "",
        "criteria": [
            {
                "id": c.get("id"),
                "status": c.get("status"),
                "severity": c.get("severity") or "must",
                "evidence": c.get("evidence") or c.get("note") or "",
            }
            for c in (result.get("criteria") or [])
            if isinstance(c, dict)
        ],
    }


def _recipe_diff(prior: Recipe, current: Recipe) -> dict[str, Any]:
    """A structural diff of two recipes, in the vocabulary a user reasons about.

    Aligns the two stage sequences by ref (``difflib`` over the ref lists), then, for stages that
    line up, compares SEMANTIC params -- the ones that move reuse identity -- so an output
    directory that differs by design does not read as a behavioural change. Output-location
    differences are reported apart, as information rather than as drift, because they are exactly
    what a second run of the same intent is expected to change.
    """
    import difflib

    from nemo_curator.audio_agent.recipe import OUTPUT_LOCATION_PARAMS

    prior_refs = [s.ref for s in prior.stages]
    current_refs = [s.ref for s in current.stages]
    added: list[str] = []
    removed: list[str] = []
    changed: list[dict[str, Any]] = []
    outputs_differ = False

    matcher = difflib.SequenceMatcher(a=prior_refs, b=current_refs, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            removed.extend(prior_refs[i1:i2])
        if tag in ("replace", "insert"):
            added.extend(current_refs[j1:j2])
        if tag == "equal":
            for offset in range(i2 - i1):
                p, c = prior.stages[i1 + offset], current.stages[j1 + offset]
                for param, before, after in _param_changes(p.semantic_params(), c.semantic_params()):
                    changed.append({"stage": c.ref, "param": param, "from": before, "to": after})
                if _output_params(p, OUTPUT_LOCATION_PARAMS) != _output_params(c, OUTPUT_LOCATION_PARAMS):
                    outputs_differ = True

    return {
        "added_stages": added,
        "removed_stages": removed,
        "changed_params": changed,
        "outputs_differ": outputs_differ,
        "identical": not (added or removed or changed),
        "phrase": _recipe_diff_phrase(added, removed, changed),
    }


def _param_changes(before: dict[str, Any], after: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """Semantic params whose value differs, present-on-one-side included.

    A secret-named param has its VALUES masked here, not later: the diff files them under
    ``from``/``to``, keys a value-blind ``redact`` pass would walk straight over, so an
    ``hf_token`` that changed between runs would otherwise be shown in full. Only the fact that
    it changed survives, which is all the diff needs to convey.
    """
    from nemo_curator.audio_agent._safety import is_secret_key

    out: list[tuple[str, Any, Any]] = []
    for key in sorted(set(before) | set(after)):
        b, a = before.get(key), after.get(key)
        if b == a:
            continue
        if is_secret_key(key):
            out.append(
                (key, "<redacted-secret>" if b is not None else None, "<redacted-secret>" if a is not None else None)
            )
        else:
            out.append((key, b, a))
    return out


def _output_params(stage: Any, output_keys: frozenset[str]) -> dict[str, Any]:  # noqa: ANN401 - StageRef
    return {k: v for k, v in stage.params.items() if k in output_keys}


def _recipe_diff_phrase(added: list[str], removed: list[str], changed: list[dict[str, Any]]) -> str:
    """One line a person can read: what this plan does that the prior one did not.

    Leads with the structural moves (a dropped or added stage) and genuine value changes -- both
    sides set, like ``threshold 0.5->0.8`` -- because those are decisions. A param that went from
    unset to a value is usually a default written out, not a behaviour change, so it is summarised
    as a trailing count rather than spelled out and left to crowd the real differences.
    """
    parts: list[str] = []
    if removed:
        parts.append("dropped " + ", ".join(removed))
    if added:
        parts.append("added " + ", ".join(added))
    value_changes = [c for c in changed if c["from"] is not None and c["to"] is not None]
    presence_changes = [c for c in changed if c["from"] is None or c["to"] is None]
    for c in value_changes[:_CARD_PARAMS]:
        parts.append(f"{c['stage']}.{c['param']} {c['from']!r}->{c['to']!r}")
    if presence_changes:
        parts.append(f"{len(presence_changes)} param default(s) made explicit")
    return "; ".join(parts) if parts else "the same pipeline"


def _data_delta_since(
    record: Any,  # noqa: ANN401 - RunRecord
    current_inventory: dict[str, str] | None,
) -> dict[str, Any] | None:
    """What changed in the folder between the prior run and now.

    Prefers a real file-level comparison: the prior run's deepest saved coverage inventory is the
    set of source files it processed, and :func:`delta.classify` reports added / modified /
    removed against the current inventory. Both are relative to the same folder -- the match is
    path-anchored -- so their relative paths line up. When no inventory was recorded on either
    side, it falls back to a labelled count comparison, which is coarse but honest.
    """
    from nemo_curator.audio_agent import delta as _delta

    prior_inventory = _deepest_coverage(record)
    if prior_inventory is not None and current_inventory is not None:
        change = _delta.classify(prior_inventory, current_inventory)
        if change is not None:
            payload = change.summary()
            payload["basis"] = "inventory"
            payload["phrase"] = change.phrase()
            return payload

    prior_count = int(getattr(record, "input_count", 0) or 0)
    now_count = len(current_inventory) if current_inventory is not None else None
    if now_count is None or not prior_count:
        return None
    delta_n = now_count - prior_count
    return {
        "basis": "counts_only",
        "prior_files": prior_count,
        "current_files": now_count,
        "phrase": (
            f"the folder held {prior_count} file(s) then and {now_count} now "
            f"({delta_n:+d}); which files changed was not recorded"
        ),
    }


def _deepest_coverage(record: Any) -> dict[str, str] | None:  # noqa: ANN401 - RunRecord
    """The source inventory the prior run's deepest artifact covered, if any was saved."""
    from nemo_curator.audio_agent import artifacts as art_mod

    for step_key in reversed(list(getattr(record, "steps", None) or [])):
        inventory = art_mod.load_coverage(str(step_key))
        if inventory:
            return inventory
    return None


def _changed_data(data_delta: dict[str, Any] | None) -> bool:
    """Whether the corpus itself moved since that run, on whichever basis was available."""
    if not data_delta:
        return False
    if data_delta.get("basis") == "inventory":
        return bool(data_delta.get("added") or data_delta.get("modified") or data_delta.get("removed"))
    return int(data_delta.get("prior_files") or 0) != int(data_delta.get("current_files") or 0)


def _delta_would_work(data_delta: dict[str, Any] | None) -> bool:
    """Whether a changed-file delta could actually run, not merely that files changed.

    A delta needs the prior run's per-file inventory to subtract from; a count comparison says
    the corpus moved but not which files, so ``delta`` there is advice that ends in a refusal.
    """
    return bool(data_delta) and data_delta.get("basis") == "inventory" and _changed_data(data_delta)


def _prior_recommendation(same_recipe: bool, diff: dict[str, Any], data_delta: dict[str, Any] | None) -> str:
    """The advice this notice carries -- never an action it takes.

    ``delta`` when the pipeline is the same and only the corpus moved (the changed-file path is
    the cheap correct answer, and ``_attach_delta`` will already have offered it). ``align`` when
    the pipeline differs, because matching the prior stages is what would make that work reusable.
    ``fresh`` otherwise -- there is prior work here, but nothing this run can borrow from it.
    """
    if same_recipe or (diff and diff.get("identical")):
        return "delta" if _delta_would_work(data_delta) else "fresh"
    return "align"


def _prior_note(
    record: Any,  # noqa: ANN401 - RunRecord
    same_recipe: bool,
    diff: dict[str, Any],
    data_delta: dict[str, Any] | None,
) -> str:
    """A single sentence stating the fact, so a host that reads nothing else still discloses it."""
    when = f" on {record.created_at}" if getattr(record, "created_at", "") else ""
    data_phrase = f" {data_delta['phrase']}." if data_delta and data_delta.get("phrase") else ""
    if same_recipe or (diff and diff.get("identical")):
        # Same pipeline, same files, and still a miss: the work happened, and what is missing is
        # a reusable record of it. Saying "only the data differs" here -- as this once did
        # unconditionally -- describes a change that did not happen.
        closing = (
            "Only the data differs -- a changed-file delta can reuse the rest."
            if _changed_data(data_delta)
            else (
                "Nothing reusable remains from it (its output was pruned, overwritten, or never "
                "persisted), so this would recompute work that already ran."
            )
        )
        return f"This folder was curated{when} by the same pipeline.{data_phrase} {closing}".strip()
    pipeline_phrase = diff.get("phrase") if diff else ""
    return (
        f"This folder was curated{when} by a different pipeline ({pipeline_phrase})."
        f"{data_phrase} Adopting that run's recipe would let its work be reused; otherwise this runs fresh."
    ).strip()


def _known_dataset_keys() -> list[str]:
    """Distinct source datasets seen before (index first, JSON records as the fallback).

    Underscored but NOT private: imported by ``delta``, which needs the same list to explain a
    miss ("prior work exists, but for another dataset") rather than just reporting nothing.
    """
    from nemo_curator.audio_agent import run_index

    keys = run_index.dataset_keys(limit=_MAX_OTHER_DATASETS)
    if keys:
        return keys
    from nemo_curator.audio_agent import artifacts as art_mod

    seen: list[str] = []
    for art in art_mod.list_artifacts():
        if art.dataset_key and art.dataset_key not in seen:
            seen.append(art.dataset_key)
        if len(seen) >= _MAX_OTHER_DATASETS:
            break
    return seen


def _boundary_block(recipe: Recipe, prefix: int) -> str | None:
    from nemo_curator.audio_agent.continuation import _resume_breaks_on_disk_boundary

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
    from nemo_curator.audio_agent import artifacts as art_mod

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
    from nemo_curator.audio_agent import artifacts as art_mod

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
    from nemo_curator.audio_agent import artifacts as art_mod
    from nemo_curator.audio_agent import run_store

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
    goal = (getattr(run, "goal", None) or {}) if run else {}
    for key in ("task", "objective", "request", "summary"):
        if goal.get(key):
            return str(goal[key])
    return str(goal) if goal else _NO_OBJECTIVE


def _key_params(params: dict[str, Any]) -> dict[str, Any]:
    """The few params that most identify a run, preferring thresholds and model choices."""
    interesting = [
        k for k in params if any(t in k for t in ("model", "threshold", "min_", "max_", "target", "rate", "type"))
    ]
    chosen = (interesting or list(params))[:_CARD_PARAMS]
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
    from nemo_curator.audio_agent import artifacts as art_mod

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
