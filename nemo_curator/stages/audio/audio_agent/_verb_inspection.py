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

"""Stage discovery, configured contracts, and planning context."""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING, Any

from nemo_curator.stages.audio.audio_agent import _safety, _verb_inputs, index, profiler
from nemo_curator.stages.audio.audio_agent import context as _context
from nemo_curator.stages.audio.audio_agent.recipe import build_stages

if TYPE_CHECKING:
    from nemo_curator.stages.audio.audio_agent.recipe import (
        Recipe,
    )


def discover() -> dict[str, Any]:
    """List every agent-ready audio stage with its category + one-liner.

    Also reports stage modules that failed to import (``unavailable``), so a shorter
    catalog is never mistaken for a smaller library: on a supported CPU-only install the
    ASR/diarization stages are absent, and the host must be able to say "unavailable here,
    install the GPU extra" instead of "this cannot be done". The key is omitted entirely
    when everything imported, so a healthy environment's output is unchanged.
    """
    from nemo_curator.stages.audio._agent._catalog import unavailable_modules

    idx = index.get_index()
    stages = [
        {"stage": name, "category": idx.category_of(name), "summary": idx.one_liner(name), "tags": idx.tags_of(name)}
        for name in idx.stage_names()
    ]
    out: dict[str, Any] = {"count": len(stages), "stages": stages}
    unavailable = unavailable_modules()
    if unavailable:
        out["unavailable"] = unavailable
    return out


def describe(name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the contract (+ card, if any) for one stage, resolved against ``params``.

    Pass the params the recipe will use. Reads and writes follow from them --
    ``SplitLongAudioStage(segments_key="diar_segments")`` reads ``diar_segments`` -- so a
    contract asked for without them describes the defaults, not the pipeline. Params a stage
    does not accept, or required ones left out, produce a labelled fallback naming what to
    supply rather than a contract that appears to require nothing.
    """
    from nemo_curator.stages.audio.audio_agent._resolve import resolved_contract_for

    out: dict[str, Any] = {"stage": name, "category": index.get_index().category_of(name)}
    try:
        resolved = resolved_contract_for(name, params)
    except KeyError:
        return {"stage": name, "error": f"{name!r} is not a registered agent-ready audio stage"}
    except Exception as e:  # noqa: BLE001 - e.g. an optional dependency failing to import
        out["contract_error"] = f"{type(e).__name__}: {e}"
    else:
        out["contract"] = resolved.contract.to_dict()
        detail = resolved.unresolved_detail()
        if detail is not None:
            out["contract_unresolved"] = detail
        inner = _composite_detail(resolved.instance)
        if inner is not None:
            out["expands_to"] = inner
        varies = _contract_variants(name, resolved, params)
        if varies:
            out["contract_varies_with"] = varies
    card = index.get_index().card(name)
    if card:
        out["card"] = card
    return out


def _contract_variants(
    name: str,
    resolved: Any,  # noqa: ANN401 - ResolvedContract
    params: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Which other settings of an enumerable param would change what this stage reads or writes.

    A resolved contract answers for ONE configuration, and the caller cannot tell from it whether
    that was the only possibility. ``input_residency`` is the case that matters: at its ``file``
    default a stage reads ``audio_filepath``, at ``waveform`` it reads ``waveform`` +
    ``sample_rate``, and at ``auto`` it takes either. A caller holding an in-task waveform and
    reading only the default contract concludes the stage cannot consume it, and either inserts a
    needless write-to-disk or abandons the stage.

    Derived by resolving each declared choice and keeping only those whose reads or writes
    actually differ, so this reports facts rather than restating the parameter list -- and stays
    empty for the 37 stages with nothing enumerable to say. Every alternative costs one more
    constructor call, which is why this can be exhaustive: 53 across the whole catalog.
    """
    instance = resolved.instance
    if instance is None:
        return []
    from nemo_curator.stages.audio.audio_agent._resolve import resolved_contract_for

    baseline = _io_view(resolved.contract)
    out: list[dict[str, Any]] = []
    for spec in resolved.contract.params:
        if not spec.choices:
            continue
        current = getattr(instance, spec.name, spec.default)
        differing: list[dict[str, Any]] = []
        for choice in spec.choices:
            if choice == current:
                continue
            trial = dict(params or {})
            trial[spec.name] = choice
            candidate = resolved_contract_for(name, trial)
            if not candidate.resolved:
                continue
            view = _io_view(candidate.contract)
            if view != baseline:
                differing.append({"value": choice, **view})
        if differing:
            out.append({"param": spec.name, "current": current, "changes_it_to": differing})
    return out


def _composite_detail(instance: Any) -> dict[str, Any] | None:  # noqa: ANN401 - any configured stage
    """What a composite requires and produces, from the stages it actually expands into.

    A composite describes only itself: ``SplitASRAlignJoinStage.describe()`` declares no reads
    and no writes, so its resolved contract is as empty as the unresolved one and for a different
    reason. The work -- and the requirements -- live in ``decompose()``.

    ``requires_upstream`` is the actionable part: each inner read that no earlier inner stage
    produces, which is what the caller has to arrange before the composite will start. For a
    ``SplitASRAlignJoinStage`` that is ``segments``, the requirement that a pipeline producing
    ``diar_segments`` failed to meet after two model downloads and a GPU diarization pass.

    Uses the expansion primitive that pipeline validation and resource planning already share, so
    a composite cannot report one shape here and a different one when it runs.
    """
    if instance is None:
        return None
    from nemo_curator.stages.audio._agent._composite import expand_composites

    expansion = expand_composites([instance])
    leaves = [item for item in expansion.stages if item.stage is not instance]
    if not leaves:
        # Two different answers, and only one of them means "we could not tell". A stage the
        # executor will refuse outright contributes no leaf either, and reporting that as
        # silence -- or as mere opacity -- is the same "empty is not an answer" failure this
        # whole verb exists to end: the caller would read no requirements and conclude there
        # are none, for a stage that cannot run at all.
        unrunnable = expansion.unrunnable.get(0)
        if unrunnable:
            return {"unrunnable": unrunnable}
        reason = expansion.opaque.get(0)
        return {"opaque": reason} if reason else None

    from nemo_curator.stages.audio._agent._agent_registry import build_contract

    produced: set[str] = set()
    required: list[str] = []
    alternatives: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    for item in leaves:
        try:
            contract = build_contract(item.stage)
        except Exception as e:  # noqa: BLE001 - one unreadable child does not hide its siblings
            stages.append({"stage": type(item.stage).__name__, "error": f"{type(e).__name__}: {e}"})
            continue
        entry: dict[str, Any] = {"stage": type(item.stage).__name__, **_io_view(contract)}
        stages.append(entry)
        required.extend(k for k in contract.reads.data_keys if k not in produced and k not in required)
        unmet = _unmet_alternatives(contract, produced)
        if unmet:
            alternatives.append({"stage": entry["stage"], "one_of": unmet})
        produced.update(contract.writes.data_keys)

    out: dict[str, Any] = {"stages": stages, "requires_upstream": required, "produces": sorted(produced)}
    if alternatives:
        # Kept separate from ``requires_upstream`` rather than flattened into it. A stage reading
        # audio accepts a file path OR an in-task waveform; listing both as required would demand
        # a form the caller does not need, and picking one for them would hide the other -- the
        # same "an empty field is not an answer" failure in a different costume.
        out["requires_one_of"] = alternatives
    return out


def _io_view(contract: Any) -> dict[str, Any]:  # noqa: ANN401 - StageContract
    """The keys a contract reads and writes, including alternatives it would accept."""
    view: dict[str, Any] = {
        "reads": list(contract.reads.data_keys),
        "writes": list(contract.writes.data_keys),
    }
    one_of = [list(spec.data_keys) for spec in contract.reads_one_of if spec.data_keys]
    if one_of:
        view["reads_one_of"] = one_of
    if contract.optional_reads.data_keys:
        view["optional_reads"] = list(contract.optional_reads.data_keys)
    return view


def _unmet_alternatives(contract: Any, produced: set[str]) -> list[list[str]]:  # noqa: ANN401
    """Alternative read-sets when none of them is already satisfied, else nothing.

    Many stages declare no flat ``reads`` at all and put every requirement here --
    ``BandwidthEstimationStage`` reads ``audio_filepath`` plus one of ``segments``/``duration``
    through this field alone. Consulting only ``reads.data_keys`` would report such a stage as
    needing nothing from upstream.
    """
    options = [list(spec.data_keys) for spec in contract.reads_one_of if spec.data_keys]
    if not options or any(set(option) <= produced for option in options):
        return []
    return options


def producers(role: str) -> dict[str, Any]:
    """Which stages write ``role`` -- by semantic role, or by literal key name.

    Answers the question that otherwise sends a caller into the source tree: a stage requires
    ``segments`` and nothing says who makes one. Matching accepts either vocabulary because a
    caller reading a contract has a key name in hand and a caller reading a request has a role,
    and making them guess which to ask with is how the question gets abandoned.

    Built from contracts resolved at DEFAULT params, so it answers "which stage can produce
    this" rather than "which stage will, as configured". A stage whose output key is renamed
    through a param still appears under its default key; ``describe`` with the real params is
    the authority on any specific recipe.
    """
    from nemo_curator.stages.audio._agent._roles import role_for_value
    from nemo_curator.stages.audio.audio_agent._resolve import resolved_contract_for

    wanted = str(role)
    proven: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    not_searched: list[str] = []

    for entry in discover().get("stages", []):
        stage = str(entry.get("stage") or "")
        category = entry.get("category")
        try:
            resolved = resolved_contract_for(stage)
        except Exception as e:  # noqa: BLE001 - an unimportable stage is reported, not fatal
            not_searched.append(f"{stage} ({type(e).__name__})")
            continue
        if resolved.resolved:
            hit = _writes_role(resolved.contract, wanted, role_for_value)
            if hit:
                proven.append({"stage": stage, "category": category, **hit})
            continue
        # Unresolvable at defaults, so its writes are unknown -- but its params still DECLARE
        # the roles of the keys it names, and that declaration is data rather than a guess.
        # Kept apart from ``producers`` because a declared role does not distinguish a key the
        # stage reads from one it writes: enough to point the caller at ``describe``, not enough
        # to answer for it. Placeholder values were the alternative and they fabricate keys --
        # probing PreserveByValueStage that way has it "writing" a key called `placeholder`.
        if wanted in set(resolved.contract.key_roles.values()):
            candidate: dict[str, Any] = {
                "stage": stage,
                "category": category,
                "matched": "declared_role",
                "confirm_with": f"describe({stage!r}, params={{...}})",
                "reason": resolved.unresolved_reason,
            }
            if resolved.required_params:
                candidate["needs_params"] = list(resolved.required_params)
            candidates.append(candidate)
        else:
            not_searched.append(stage)

    out: dict[str, Any] = {"role": wanted, "producers": proven}
    if candidates:
        out["candidates"] = candidates
    if not_searched:
        # "Nothing produces this" and "some stages could not be asked" are different answers and
        # only one of them means stop looking, so an incomplete search never presents as complete.
        out["not_searched"] = sorted(not_searched)
    return out


def _writes_role(
    contract: Any,  # noqa: ANN401 - StageContract, imported lazily by callers
    wanted: str,
    role_for_value: Any,  # noqa: ANN401
) -> dict[str, Any] | None:
    """How ``contract`` produces ``wanted``, or ``None`` if it does not."""
    writes = getattr(contract, "writes", None)
    keys = list(getattr(writes, "data_keys", []) or [])
    segment_keys = list(getattr(writes, "segment_data_keys", []) or [])
    key_roles = dict(getattr(contract, "key_roles", {}) or {})

    if wanted in keys:
        return {"writes_key": wanted, "matched": "key"}
    if wanted in segment_keys:
        return {"writes_segment_key": wanted, "matched": "segment_key"}
    for key in keys + segment_keys:
        if (key_roles.get(key) or role_for_value(key)) == wanted:
            return {"writes_key": key, "matched": "role"}
    return None


def catalog_tree() -> dict[str, Any]:
    """L0: the full category tree the host prunes over before drilling in."""
    return {"categories": index.get_index().category_tree()}


def cards(category: str | None = None, names: list[str] | None = None) -> dict[str, Any]:
    """L1 (one-liners for a category) or L2 (full cards for named finalists)."""
    idx = index.get_index()
    if names:
        return {"cards": idx.full_cards(names)}
    if category:
        return {"category": category, "stages": idx.card_oneliners(category)}
    msg = "cards() requires either a category (L1) or a list of names (L2)"
    raise ValueError(msg)


def context(
    goal: dict[str, Any] | None = None,
    *,
    data: str | None = None,
    stages: list[str] | None = None,
    roles: list[str] | None = None,
    planning_preference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a compact PlanningContext for the host router/planner.

    Unlike recipe-driven verbs, ``data`` is the direct path to profile because
    no executable recipe source exists yet.
    """
    pviol = _safety.path_violations([data])
    if pviol:
        return {
            "status": "refused",
            "reason": "path(s) resolve outside the allowed workspace",
            "violations": pviol,
        }
    result = _context.assemble(
        goal,
        data=data,
        selected_stages=stages,
        roles=roles,
        planning_preference=planning_preference,
    ).to_dict()
    profile_errors = (result.get("data_profile") or {}).get("source_errors")
    if profile_errors:
        return _safety.redact(
            {
                "status": "refused",
                "reason": "; ".join(profile_errors),
                "data_profile": result["data_profile"],
            }
        )
    return result


def resolve(
    stage: str,
    *,
    label: str | None = None,
    use_case: str | None = None,
    explicit: dict[str, Any] | None = None,
    data: str | None = None,
) -> dict[str, Any]:
    """Resolve an outcome to concrete stage config via the card (1A.2).

    Path A maps a user-facing outcome — a quality ``label`` ("studio"), a named
    ``use_case`` preset, or an ``explicit`` ``{param: value}`` — to concrete
    params (or, for an annotator, a ``PreserveByValueStage`` filter), plus an
    auditable ``strategy`` trail. Keeps internal thresholds out of user questions
    and never invents a number.

    Path B (passing ``data``) adds the values the DATASET fixes rather than the user —
    currently the rate the audio actually is. It configures the stage for this recipe and
    never changes a stage default, so hand-written pipelines and tutorials are unaffected.
    Ambiguity comes back as an ``ask``: mixed sample rates are a question for the user
    rather than something to guess at.

    Supplying ``data`` IS the request for Path B. It used to need a separate
    ``data_driven=True`` alongside it, from when data context was optional; every other
    data-taking verb now resolves and profiles its source on its own, and the second flag
    only ever meant that a caller who passed one without the other silently got nothing.

    It does NOT infer which column holds the audio. ``audio_filepath`` is the NeMo manifest
    key, and supplying a manifest in that format is the caller's responsibility; a source
    that does not carry it is refused by the ``source_schema`` check with the columns it
    does have, rather than guessed at. Guessing would mean guessing wrong on some dataset
    and silently curating the wrong field.
    """
    from nemo_curator.stages.audio.audio_agent import config_strategy

    result = config_strategy.resolve(stage, label=label, use_case=use_case, explicit=explicit)
    if not data:
        return result
    # Held to the same workspace lock as every other verb that is handed a dataset path.
    # This one profiles ``data`` off the filesystem too, and was the only such verb without
    # the check -- unreachable while no adapter passed ``data``, and a hole the moment one did.
    pviol = _safety.path_violations([data])
    if pviol:
        return {
            "status": "refused",
            "reason": "path(s) resolve outside the allowed workspace",
            "violations": pviol,
        }
    profile: dict[str, Any] | None = None
    with contextlib.suppress(Exception):  # a source we cannot profile simply adds nothing
        profile = profiler.profile_data(data).to_dict()
    # Path A wins on conflict: an outcome the user asked for outranks an inference. Telling
    # Path B what is already decided, rather than merging over it afterwards, keeps the
    # ``strategy`` trail honest -- it used to record the data-informed value and its rationale
    # for a param the merge then discarded.
    derived = config_strategy.resolve_from_data(
        stage, profile, already_set=set(result["params"]), configured=result["params"]
    )
    result["params"] = {**derived["params"], **result["params"]}
    result["strategy"] = list(result["strategy"]) + list(derived["strategy"])
    asks = list(result["asks"]) + list(derived["asks"])
    if derived["params"] or derived["asks"]:
        # Path B answered (or asked something concrete), so Path A's "you gave me no
        # outcome" prompt is no longer true and would only add noise.
        asks = [a for a in asks if a != config_strategy.NO_OUTCOME_ASK]
    result["asks"] = asks
    return result


def diagnose(  # noqa: PLR0913
    error: str,
    *,
    recipe: Recipe | dict[str, Any] | None = None,
    operation: str = "run",
    phase: str = "runtime",
    attempted_actions: list[str] | None = None,
    execution_target: str | None = None,
) -> dict[str, Any]:
    """Analyze a captured failure for the host LLM without applying a fix.

    The result combines the sanitized failure taxonomy, a fresh machine probe,
    recipe-specific environment applicability, ranked grounded choices, and an
    explicit user-decision prompt. Unknown failures stay unknown and receive only
    diagnostic next steps.
    """
    stages: list[Any] = []
    recipe_issues: list[dict[str, Any]] = []
    if recipe is not None:
        stages_or_none, recipe_issues = build_stages(_verb_inputs._as_recipe(recipe))
        stages = list(stages_or_none or [])
    target = (
        execution_target
        if execution_target in {"local", "external_ray", "custom_executor"}
        else _verb_inputs._ray_execution_target(os.environ.get("RAY_ADDRESS"))
    )
    from nemo_curator.stages.audio.audio_agent.diagnostics import diagnose_failure

    result = diagnose_failure(
        error,
        stages=stages,
        env=profiler.probe_env(),
        operation=operation,
        phase=phase,
        execution_target=target,
        attempted_actions=attempted_actions,
    )
    if recipe_issues:
        result["recipe_issues"] = recipe_issues
    return _safety.redact(result)
