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

"""Run history, managed checkpoints, and reuse scanning."""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING, Any

from nemo_curator.stages.audio.audio_agent import _reuse_history, _reuse_summary, _safety, _verb_inputs, profiler
from nemo_curator.stages.audio.audio_agent.contracts import DATASET_KEY_TIERS
from nemo_curator.stages.audio.audio_agent.index import get_index

if TYPE_CHECKING:
    from nemo_curator.stages.audio.audio_agent.recipe import (
        Recipe,
    )


def runs(  # noqa: PLR0913
    run_id: str | None = None,
    *,
    data: str | None = None,
    stage: str | None = None,
    since: str | None = None,
    limit: int = 50,
    goal: dict[str, Any] | str | None = None,
) -> dict[str, Any]:
    """List local run records, or load one by ``run_id`` (provenance for tracing).

    Local history only — NOT shared memory / cross-user learning. Use it to trace a
    prior run or to feed ``plan_continuation`` for a follow-up request. ``data`` /
    ``stage`` / ``since`` query the index for what has already been done to a corpus.
    ``data`` takes either the dataset's path or a ``dataset_key`` copied from earlier
    reuse output — profiling a key as if it were a path would silently match nothing.

    A single record is returned with an ``overview``: the pipeline, its identifying params, what
    it read, what it produced and how it scored, so "what did that run do?" is answerable without
    reading every param of every stage. Given a folder PATH, the listing also includes runs that
    read that folder when its contents were different — the same-corpus key filter alone reports
    a folder curated last week as never curated at all.

    Pass ``goal`` (the user's current request) with a folder path to rank those priors by how
    much of the request is covered by each prior's recorded prompt plus its
    ``pipeline_summary`` — before inventing a new recipe. Ranking is never by stage
    edit-distance to a draft pipeline.
    """
    from nemo_curator.stages.audio.audio_agent import run_index, run_store

    if run_id:
        rec = run_store.load(run_id)
        if rec is None:
            return {"error": f"no run record {run_id!r}"}

        return _safety.redact(
            {
                **rec.to_dict(),
                "overview": _reuse_history.run_overview(rec),
                "host_directive": _reuse_summary.SUMMARIZE_DIRECTIVE,
            }
        )
    if data or stage or since:
        names_a_path = bool(data) and not str(data).startswith(tuple(f"{tier}:" for tier in DATASET_KEY_TIERS))
        pviol = _safety.path_violations([data] if names_a_path else [])
        if pviol:
            return {
                "status": "refused",
                "reason": "path(s) resolve outside the allowed workspace",
                "violations": pviol,
            }
        dataset_key = _dataset_key_arg(data) if data else None
        payload = {
            "dataset_key": dataset_key,
            "runs": run_index.find_runs(dataset_key=dataset_key, since=since, limit=limit),
            "artifacts": run_index.find_artifacts(
                dataset_key=dataset_key,
                stage_ref=stage,
                since=since,
                limit=limit,
            ),
        }
        if names_a_path:
            _attach_same_folder_runs(payload, str(data), since=since, limit=limit)

            enriched = _reuse_history.enrich_folder_run_cards(list(payload["runs"]), goal=goal)
            payload["runs"] = enriched["runs"]
            payload["host_directive"] = enriched["host_directive"]
            if "ranked_by" in enriched:
                payload["ranked_by"] = enriched["ranked_by"]
                payload["goal"] = enriched["goal"]
        return _safety.redact(payload)
    # ``limit`` applies here too. The filtered branch above has always honoured it while this one
    # returned every record ever written, so the argument silently meant different things
    # depending on whether a filter happened to be supplied.
    return {"runs": run_store.list_runs()[: int(limit)] if int(limit) >= 0 else run_store.list_runs()}


def _attach_same_folder_runs(
    payload: dict[str, Any],
    data: str,
    *,
    since: str | None,
    limit: int,
) -> None:
    """Add runs that read this same FOLDER but under a different corpus state or pipeline.

    ``dataset_key`` identifies a corpus, not a location, so filtering by it alone answers "what
    has been done to these exact bytes" -- and one added file makes the answer "nothing", about a
    folder that was curated an hour ago. The union is what the caller asked for when they passed a
    path. Which rows came from which axis stays visible in ``same_folder_only`` so an exact-corpus
    reader is not misled, and each row carries its own ``dataset_key`` either way.
    """

    rows = _reuse_history.runs_on_path(data, since=since, limit=limit)
    if not rows:
        return
    known = {str(r.get("run_id") or "") for r in payload["runs"]}
    added = [r for r in rows if str(r.get("run_id") or "") not in known]
    if not added:
        return
    merged = sorted(
        [*payload["runs"], *added],
        key=lambda row: str(row.get("created_at") or ""),
        reverse=True,
    )
    payload["runs"] = merged[: int(limit)] if int(limit) >= 0 else merged
    payload["same_folder_only"] = [str(r.get("run_id") or "") for r in added]
    payload["note"] = (
        f"{len(added)} of these read the same folder when its contents or pipeline differed, so "
        "they carry a different dataset_key; they are named in 'same_folder_only'"
    )


def _recipe_dataset_key(rec: Recipe, data: str | None) -> str:
    """The recipe's dataset key from ``data`` or its own source, ``""`` when unresolvable.

    Returns ``""`` instead of raising so an unreadable source degrades to a structured
    refusal from the caller rather than a traceback: every caller already treats a missing
    key as "cannot prove identity", which is the honest answer when profiling failed.
    """
    try:
        if data:
            return _dataset_key_arg(data)
        profile = _verb_inputs._profile_binding(_verb_inputs._dataset_binding(rec, None))
        if profile is not None and not _verb_inputs._profile_error(profile):
            return profile.dataset_key()
    except Exception:  # noqa: BLE001 - an unkeyable source is a refusal, not a crash
        return ""
    return ""


def _dataset_key_arg(data: str) -> str:
    """A dataset identity from either a path or a key already printed by an earlier scan."""
    from nemo_curator.stages.audio.audio_agent.contracts import DATASET_KEY_TIERS

    if data.startswith(tuple(f"{t}:" for t in DATASET_KEY_TIERS)):
        return data
    return profiler.profile_data(data).dataset_key()


def checkpoints(*, gc: bool = False) -> dict[str, Any]:
    """List the managed checkpoint cache; with ``gc`` delete what nothing can reuse.

    Addressing checkpoints instead of asking the user to name them moved the cleanup duty
    from them to us: nobody deletes a file they never chose. This is that duty.

    Every entry is one of three things. ``reusable`` -- a record exists and
    :func:`~nemo_curator.stages.audio.audio_agent.artifacts.invalid_reasons` clears it, so a matching run
    would resume from it. ``orphan`` -- bytes with no artifact record, so no step key can
    address them and no run will ever find them. ``stale`` -- a record exists but refuses
    reuse, with the reasons listed.

    ``gc`` removes orphans and entries past their freshness or checkpoint-retention deadline. A
    checkpoint stale because its stage's ``impl_version`` moved is kept: that is a property
    of the current checkout, and reverting the code makes it valid again. Deleting the bytes
    is reversible only by re-running the model, so the bar is "no future run can use this",
    not "no run can use it today".

    Only ever touches files inside :func:`~nemo_curator.stages.audio.audio_agent.run_store.checkpoints_dir`.
    A checkpoint the user pointed somewhere of their own is theirs, and is never a candidate.
    """
    from nemo_curator.stages.audio.audio_agent import artifacts as art_mod
    from nemo_curator.stages.audio.audio_agent.run_store import checkpoints_dir

    root = os.path.realpath(checkpoints_dir())
    entries: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    reclaimed = 0
    for name in sorted(os.listdir(root)):
        if not name.endswith(".jsonl"):
            continue  # the ``._COMPLETE`` marker travels with its file, never on its own
        path = os.path.join(root, name)
        # Refuse to follow a symlink out of the cache: this function deletes.
        if os.path.realpath(path) != path or not os.path.isfile(path):
            continue
        step_key = name[: -len(".jsonl")]
        artifact = art_mod.load(step_key)
        reasons = art_mod.invalid_reasons(artifact) if artifact is not None else []
        expired = bool(artifact and artifact.ttl_sec and art_mod._age_sec(artifact) > artifact.ttl_sec)
        if artifact is not None:
            # An unverifiable policy blocks reuse, but must never authorize deletion.
            with contextlib.suppress(TypeError, ValueError):
                expired = expired or art_mod.checkpoint_expired(artifact)
        size = os.path.getsize(path)
        entry = {
            "step_key": step_key,
            "status": "orphan" if artifact is None else ("stale" if reasons else "reusable"),
            "uri": path,
            "bytes": size,
            "rows": artifact.rows_out if artifact else 0,
            # What is actually IN it, which is what someone deciding whether to keep it asks.
            "produced_keys": list(artifact.produced_keys) if artifact else [],
            "stage": artifact.stage_ref if artifact else "",
            "created_at": artifact.created_at if artifact else "",
            "age_sec": round(art_mod._age_sec(artifact), 1) if artifact else None,
            "ttl_sec": artifact.ttl_sec if artifact else 0,
            "reasons": reasons,
            "collectable": artifact is None or expired,
        }
        if not (gc and entry["collectable"]):
            entries.append(entry)
            continue
        # Resolved before the unlink: ``marker_path`` asks whether the target is a directory,
        # which stops being answerable once the file is gone.
        marker = art_mod.marker_path(path)
        try:
            os.remove(path)
        except OSError:
            entries.append(entry)  # still on disk, so still reported
            continue
        with contextlib.suppress(OSError):  # a missing marker is the normal case
            os.remove(marker)
        reclaimed += size
        removed.append({"step_key": step_key, "bytes": size, "why": "expired" if expired else "orphan"})
    if gc:
        next_step = f"removed {len(removed)} checkpoint(s)" if removed else "nothing to collect"
    elif any(e["collectable"] for e in entries):
        next_step = "call checkpoints(gc=True) to delete the collectable entries"
    else:
        next_step = "nothing to collect"
    return {
        "dir": root,
        "checkpoints": entries,
        "total_bytes": sum(e["bytes"] for e in entries),
        "removed": removed,
        "reclaimed_bytes": reclaimed,
        "next": next_step,
    }


def reindex() -> dict[str, Any]:
    """Rebuild the run/artifact index from the JSON records, and drop the knowledge cache.

    The index is a cache; the JSON records are the source of truth. Run this after moving,
    pruning, or hand-editing ``.audio_agent_runs/``.

    The knowledge index is a second cache, held for the life of the process because the YAML is
    static (``index.get_index`` is ``lru_cache``d). That is right for a run, and a trap for the
    person editing a card: their change was invisible until a restart, from the one verb whose
    name promises otherwise. Clearing it here costs a re-read of the card directory on the next
    call and makes ``reindex`` mean what it says.
    """
    from nemo_curator.stages.audio.audio_agent import run_index

    get_index.cache_clear()

    return run_index.reindex()


def reuse_scan(recipe: Recipe | dict[str, Any], *, data: str | None = None, limit: int = 5) -> dict[str, Any]:
    """Find prior work this recipe could reuse, and describe it for a human decision.

    Probes the artifact registry with the recipe's step keys and returns the longest safely
    reusable prefix plus ranked approval cards (objective, pipeline, input/output, key params,
    date, metrics, estimated saving). Read-only: it never reuses anything by itself.

    ``prompt_user`` is the anti-nag signal — false when there is no candidate at all, or when
    the saving was measured and is too small to be worth a question (take it and disclose it).
    An unmeasured prefix containing work the cards call expensive is named in ``unpriced_stages``
    and always asks: not knowing what something cost is not the same as it having been cheap.

    ``data`` is an optional assertion about the configured first source stage,
    not an input override. Omit it when the recipe is already unambiguous.
    """
    from nemo_curator.stages.audio.audio_agent import reuse as _reuse

    rec = _verb_inputs._as_recipe(recipe).freeze()
    pviol = _safety.path_violations([data, *_safety.recipe_path_params(rec)])
    if pviol:
        return {
            "status": "refused",
            "reason": "path(s) resolve outside the allowed workspace",
            "violations": pviol,
        }
    binding = _verb_inputs._dataset_binding(rec, data)
    dp = _verb_inputs._profile_binding(binding)
    profile_error = _verb_inputs._profile_error(dp) if dp is not None else ""
    dataset_key = (
        dp.dataset_key()
        if dp and not profile_error and not _verb_inputs._binding_blocks_execution(binding, data)
        else ""
    )
    result = _reuse.scan(rec, dataset_key=dataset_key, limit=limit)
    result["data_binding"] = binding.to_dict()
    if profile_error:
        result["rationale"] = f"prior work was not considered: {profile_error}"
        result["data_profile"] = dp.to_dict()
    if _verb_inputs._binding_blocks_execution(binding, data):
        result["rationale"] = f"prior work was not considered: {binding.reason}"
    if result.get("decision") == "fresh" and dataset_key and dp is not None:
        _attach_delta(result, rec, dp)
    # Both misses, and both blind to a folder curated under another pipeline: ``fresh`` found no
    # matching key, and ``delta`` matched THIS pipeline's own earlier corpus and so still says
    # nothing about the others. Neither names the prior run's id, which is what the user needs to
    # ask about it.
    if result.get("decision") in ("fresh", "delta"):
        _attach_prior_on_path(result, rec, binding, dp)
    return _safety.redact(result)


def _attach_prior_on_path(
    result: dict[str, Any],
    rec: Recipe,
    binding: Any,  # noqa: ANN401 - DatasetBinding
    dp: Any,  # noqa: ANN401 - DataProfile | None
) -> None:
    """Disclose prior runs that read THIS folder, even when their keys and recipe do not match.

    The step-key matchers answer "can I reuse bytes"; this answers the question a person asks
    first -- "have I done this here before". It is advisory: it never changes ``decision`` or
    reuses anything, only adds ``prior_on_same_path`` for the host to surface. Skipped for a
    generated or pathless source, which has no folder to have been curated before.
    """

    source_path = getattr(binding, "primary_path", None)
    if not source_path or getattr(binding, "generated", False):
        return
    prior = _reuse_history.prior_on_path(
        rec,
        source_path=source_path,
        current_inventory=(dp.inventory or None) if dp is not None else None,
    )
    if prior is None:
        return
    result["prior_on_same_path"] = prior
    # Make the host STOP on it. Beside ``decision: fresh`` / ``prompt_user: false`` a notice
    # reads as nothing to do -- observed in a real session, where the field was correct and the
    # host still said "nothing to reuse" because it summarised the decision. ``prompt_user`` is
    # the one signal the never-nag rule cannot skip. This reuses nothing and does not change
    # ``decision``; it obliges the host to surface the fact and let the user choose.
    result["prompt_user"] = True
    # "fresh" reads as "never done here". When the folder HAS been curated, say so in the line a
    # host is most likely to read, without overriding the decision the key probe correctly made.
    # A ``delta`` rationale already discloses prior work on this pipeline, so it is left alone.
    if result.get("decision") == "fresh":
        result["rationale"] = f"{result.get('rationale', '')}; {prior['note']}".lstrip("; ")


def _attach_delta(result: dict[str, Any], rec: Recipe, dp: Any) -> None:  # noqa: ANN401 - DataProfile
    """Add the changed-file option to a card whose key missed, when there is one.

    A miss is where this belongs: a key names the whole corpus, so adding one file misses every
    key even though almost all of the work behind it still holds. Left alone the card says
    "nothing matches" and the honest reading of that is "recompute everything" -- which is how a
    one-file addition comes to cost a full run.

    A refusal is attached too when the change itself was understood. "Prior work exists but
    these stages cannot be split per file" is a different fact from "this has never run", it is
    the reason the full run is unavoidable, and it names what would have to change for it not
    to be -- usually a checkpoint, or a stage that has not declared per-row independence.
    """
    from nemo_curator.stages.audio.audio_agent import delta as _delta

    decision = _delta.plan(
        rec,
        dataset_key=dp.dataset_key(),
        inventory=dp.inventory or None,
        inventory_root=dp.inventory_root,
    )
    if decision.status != "ready" and decision.change is None and not result.get("prior_on_other_data"):
        # Nothing ran before, so there is nothing a delta could have narrowed and no reason to
        # mention one. When a prior run IS known, the same silence would hide why this costs
        # full price -- usually an inventory that was never recorded, which is fixable.
        return
    result["delta"] = decision.to_dict()
    if decision.status != "ready":
        result["rationale"] = (
            f"{result.get('rationale', '')}; running only the changed file(s) is not available here: {decision.reason}"
        ).lstrip("; ")
        return
    change = decision.change
    result.update(
        # The key really did miss. But "fresh" is the one word a host reads as "recompute
        # everything", and one did exactly that on a single-file addition -- recurating three
        # files and reporting a checkpoint was missing -- while `recommended: delta` sat two
        # fields away unread. `decision` names the cheapest correct action; the miss it rests
        # on stays visible in `key_matched` and in the rationale below, so nothing is hidden.
        decision="delta",
        key_matched=False,
        recommended="delta",
        prompt_user=True,
        estimated_saving_sec=decision.estimated_saving_sec,
        saving_is_lower_bound=True,
        choices=[
            {
                "id": "delta",
                "label": f"Run the {len(decision.files)} changed file(s) only",
                "effect": (
                    f"run {decision.prefix} stage(s) over the changed file(s) and merge the rows into "
                    f"the prior result, keeping {decision.keeps} row(s) and replacing {decision.drops}"
                ),
                "verb": "delta_run",
            },
            {"id": "fresh", "label": "Run fresh", "effect": "recompute every file from the start"},
        ],
        rationale=(
            f"no artifact matches this exact corpus, but {change.phrase()} since a prior run of this same "
            f"pipeline: the other {len(change.unchanged)} file(s)' results through {decision.stage_ref} still "
            f"hold, so delta_run processes only the changed file(s) and merges them in"
        ),
    )
