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

"""Prior-run discovery and comparison on the same data path."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, NamedTuple

from nemo_curator.stages.audio.audio_agent import _reuse_policy

if TYPE_CHECKING:
    from nemo_curator.stages.audio.audio_agent.recipe import Recipe

from nemo_curator.stages.audio.audio_agent import _reuse_summary


def runs_on_path(
    source_path: str,
    *,
    since: str | None = None,
    limit: int = _reuse_policy._MAX_PRIOR_RUNS,
    completed_only: bool = False,
) -> list[dict[str, Any]]:
    """Index rows for runs that read the SAME folder, newest first.

    One definition of "the same folder", shared by :func:`prior_on_path` and the ``runs`` verb,
    and it is a canonical-path comparison rather than the string equality SQL would do:
    ``data_source`` is recorded as the caller passed it -- a relative path, a trailing slash, a
    symlink -- so an exact compare misses the same folder named two ways. The scan is bounded and
    already newest-first, and a folder is asked about once per request, not once per file.
    """
    from nemo_curator.stages.audio.audio_agent import run_index
    from nemo_curator.stages.audio.audio_agent.input_identity import canonical_source

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
    scan = max(want, _reuse_policy._MAX_PRIOR_RUNS) if want >= 0 else -1
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
    from nemo_curator.stages.audio.audio_agent import run_store
    from nemo_curator.stages.audio.audio_agent.recipe import Recipe as _Recipe

    semantic = recipe.compute_semantic_hash()
    ranked: list[_PathMatch] = []
    for row in runs_on_path(source_path, limit=_reuse_policy._MAX_PRIOR_RUNS, completed_only=True):
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
    matches = [
        _prior_on_path_card(match, current_inventory=current_inventory)
        for match in ranked[: _reuse_policy._MAX_PRIOR_CARDS]
    ]
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
    from nemo_curator.stages.audio.audio_agent.input_identity import canonical_source

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
    from nemo_curator.stages.audio.audio_agent import reuse

    record = match.record
    data_delta = _data_delta_since(record, current_inventory)
    return {
        "run_id": record.run_id,
        "created_at": record.created_at,
        "goal": record.goal or None,
        "prompt": reuse._objective(record),
        "pipeline_summary": _reuse_summary.resolve_pipeline_summary(record),
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
    module = "python -m nemo_curator.stages.audio.audio_agent"
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
    from nemo_curator.stages.audio.audio_agent import reuse
    from nemo_curator.stages.audio.audio_agent.recipe import Recipe as _Recipe

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
        "objective": reuse._objective(record),
        "prompt": reuse._objective(record),
        "pipeline_summary": _reuse_summary.resolve_pipeline_summary(record),
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
    from nemo_curator.stages.audio.audio_agent import run_store
    from nemo_curator.stages.audio.audio_agent.recipe import Recipe as _Recipe

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
            card["match"] = _reuse_summary.prompt_summary_match(
                goal,
                prompt,
                summary,
                match_corpus=_reuse_summary.match_corpus_for_record(record, recipe),
            )
        cards.append(card)

    out: dict[str, Any] = {"runs": cards, "host_directive": _reuse_summary._HOST_DIRECTIVE_FOLDER_RUNS}
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
        out["goal"] = _reuse_summary.goal_text(goal)
        out["ranked_by"] = "current_prompt vs prior_prompt + pipeline_summary, then prior coverage"
    return out


def _pipeline_key_params(recipe: Recipe | None) -> list[dict[str, Any]]:
    """Per stage, the few params that identify what it was configured to do.

    A list rather than a mapping by stage name: a pipeline can hold the same stage twice (two
    writers, two resamplers), and keying by ref would silently drop one of them and show the
    other's settings as if they were both.
    """
    from nemo_curator.stages.audio.audio_agent import reuse

    if recipe is None:
        return []
    out: list[dict[str, Any]] = []
    for stage in recipe.stages:
        params = reuse._key_params(stage.semantic_params())
        if params:
            out.append({"stage": stage.ref, "params": params})
    return out


def _run_stats(record: Any) -> dict[str, Any]:  # noqa: ANN401 - RunRecord
    """What the run cost and how much of the input survived it, as it was measured."""
    from nemo_curator.stages.audio.audio_agent.report import stage_duration_sec

    metrics = dict(getattr(record, "per_stage_metrics", None) or {})
    timed = {name: stage_duration_sec(metrics, name) for name in metrics}
    slowest = sorted((s for s in timed.items() if s[1]), key=lambda pair: pair[1], reverse=True)
    return {
        "elapsed_sec": float(getattr(record, "elapsed_sec", 0.0) or 0.0),
        "input_count": int(getattr(record, "input_count", 0) or 0),
        "accepted": int(getattr(record, "accepted", 0) or 0),
        # Kept at the precision the metrics were written with: rounding a 20 ms stage to one
        # decimal prints "0.0 seconds" beside a name, which reads as a measurement failure.
        "slowest_stages": [
            {"stage": name, "seconds": round(sec, 3)} for name, sec in slowest[: _reuse_policy._CARD_PARAMS]
        ],
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

    from nemo_curator.stages.audio.audio_agent.recipe import OUTPUT_LOCATION_PARAMS

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
    from nemo_curator.stages.audio.audio_agent._safety import is_secret_key

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
    for c in value_changes[: _reuse_policy._CARD_PARAMS]:
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
    from nemo_curator.stages.audio.audio_agent import delta as _delta

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
    from nemo_curator.stages.audio.audio_agent import artifacts as art_mod

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
