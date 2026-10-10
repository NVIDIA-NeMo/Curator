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

"""Pluggable validation-check registry for agent-composed audio recipes.

``verbs.validate`` builds a :class:`CheckContext`, runs every registered check,
and merges the results into a :class:`~nemo_curator.stages.audio.audio_agent.contracts.Verdict`.
Each check is ``fn(ctx) -> CheckResult``; adding a check is a ``@register`` + a
function, with no change to callers. This is the extension point for the 1B
correctness taxonomy (task-type, key-flow, output-completeness) — the checks are
added here, and the verb surface stays stable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nemo_curator.stages.audio.audio_agent.contracts import Issue
from nemo_curator.stages.audio.audio_agent.index import get_index

if TYPE_CHECKING:
    from nemo_curator.stages.audio.audio_agent.recipe import Recipe

# Params whose value is compared against a card ``max_speakers`` constraint.
_MAX_SPEAKERS_KEYS = ("max_speakers", "num_speakers")


@dataclass
class CheckContext:
    """Everything a validation check needs, assembled once by ``validate``."""

    recipe: Recipe
    stages: list[Any]  # built stage instances (well-formedness already passed)
    data_profile: dict[str, Any] | None
    env: Any  # EnvProfile instance (has .has_gpu / .has_ffmpeg / .available_secrets)
    initial_roles: set[str]
    initial_keys: set[str]
    available_gpus: float
    initial_tensor_keys: set[str] | None = None
    expected_outputs: list[str] = field(default_factory=list)  # roles the user asked for
    acceptance_criteria: list[Any] = field(default_factory=list)  # parsed AcceptanceCriterion list (1A.1)
    request_type: str | None = None  # goal/request kind, for request-type sanity (1A.1)
    execution_target: str = "local"


@dataclass
class CheckResult:
    """A single check's contribution, merged into the Verdict by ``run_checks``."""

    issues: list[Issue] = field(default_factory=list)
    card_violations: list[Issue] = field(default_factory=list)
    gate_flags: list[Issue] = field(default_factory=list)
    unproducible_roles: list[str] = field(default_factory=list)
    produced_roles: list[str] = field(default_factory=list)
    produced_keys: list[str] = field(default_factory=list)
    ok: bool | None = None
    keys_ok: bool | None = None


Check = Callable[[CheckContext], CheckResult]
REGISTRY: list[tuple[str, Check]] = []


def register(name: str) -> Callable[[Check], Check]:
    """Register a check under ``name`` (order of registration is run order)."""

    def deco(fn: Check) -> Check:
        REGISTRY.append((name, fn))
        return fn

    return deco


def _to_int(v: Any) -> int | None:  # noqa: ANN401
    """Best-effort int coercion (None on failure), so a malformed card value or an
    LLM-supplied param (e.g. num_speakers='two') can't raise out of a check."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _to_float(v: Any) -> float | None:  # noqa: ANN401
    """Best-effort float coercion (None on failure); same intent as ``_to_int`` for
    duration/threshold values so a non-numeric card/profile entry skips the check
    instead of degrading the whole recipe to ``check_error``."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def run_checks(ctx: CheckContext) -> CheckResult:
    """Run every registered check and merge the results into one ``CheckResult``.

    Each check is isolated: if one raises, it is converted into a ``check_error``
    issue (and the recipe is marked not-ok) instead of propagating out of
    ``validate`` -- the grounding layer must never emit a traceback where it promised
    a JSON Verdict.
    """
    merged = CheckResult()
    for name, fn in REGISTRY:
        try:
            r = fn(ctx)
        except Exception as e:  # noqa: BLE001 - a single check must never crash the verb
            merged.issues.append(
                Issue(
                    "check_error",
                    "error",
                    f"check {name!r} could not run ({type(e).__name__}: {e}); recipe not fully validated",
                    fix="treat as not-runnable; fix the offending card/param, or report a bug",
                )
            )
            merged.ok = False
            continue
        merged.issues.extend(r.issues)
        merged.card_violations.extend(r.card_violations)
        merged.gate_flags.extend(r.gate_flags)
        merged.unproducible_roles.extend(r.unproducible_roles)
        if r.produced_roles:
            merged.produced_roles = r.produced_roles
        if r.produced_keys:
            merged.produced_keys = r.produced_keys
        if r.ok is not None:
            merged.ok = r.ok
        if r.keys_ok is not None:
            merged.keys_ok = r.keys_ok
    return merged


def _fix_for(code: str) -> str | None:
    return {
        "unsatisfied_reads": "insert an upstream stage that produces the missing role (see find_producers)",
        "dangling_key": "align the producer's *_key value with what this stage reads, or seed it from the source manifest",
        "ambiguous_default_key": (
            "set the *_key parameter explicitly to the producer you mean. Two upstream stages wrote "
            "same-kind keys, so a default is choosing between them silently -- e.g. a diarization merge "
            "left at segments_key='segments' merges into the VAD segments and produces a plausible, "
            "wrong answer with no error"
        ),
        "tensor_into_sink": (
            "a resident tensor/audio blob is reaching a sink that serializes task.data as-is; strip it "
            "before the sink using a method that preserves the sink's input task type -- e.g. read/score "
            "from file (input_residency=file) or stop carrying the waveform (keep_waveform_in_task=false). "
            "A sanitizer stage only helps if its OUTPUT task type matches the sink (e.g. AudioToDocumentStage "
            "emits a DocumentBatch, so it fits a DocumentBatch sink, not an AudioTask sink like ManifestWriterStage)."
        ),
        "gpu_unavailable": "run on a GPU host or set the stage to CPU resources",
        "composite": "decompose the composite (it hides its true I/O) before validating downstream",
        "task_type_mismatch": (
            "consecutive stages disagree on the task type they hand over, which fails on the "
            "first task rather than on some rows. Insert a converter between them, or reorder "
            "so the types line up. The common case is a DocumentBatch reaching an AudioTask-only "
            "sink: serialize it with DocumentBatchJsonlWriterStage instead of ManifestWriterStage, "
            "or move every AudioTask-only stage ahead of AudioToDocumentStage"
        ),
        "composite_unrunnable": (
            "the executor refuses this composite's SHAPE, so no configuration of it will run: "
            "Pipeline._decompose_stages substitutes children only when there is more than one, and "
            "rejects a child that is itself a decomposing composite. Replace the composite with the "
            "stages it decomposes into, written out at the top level of the recipe -- that is the "
            "same schedule, in a shape the executor accepts. This is an error rather than a warning "
            "because it is a certainty, not a gap in what validation could see"
        ),
        "key_removed_upstream": "reorder so the reader runs before the stage that removes the key, or re-produce the key",
        "unsatisfied_reads_in_composite": (
            "a stage INSIDE a composite reads a key nothing upstream produces. The composite's own "
            "contract is silent about this, so it will validate and then fail once that inner stage "
            "starts -- after any model downloads and GPU work ahead of it. The message names the "
            "composite parameter that forwards the key; set it to what the upstream stage actually "
            "wrote (e.g. segments_key='diar_segments' when a diarizer, not a VAD, produced the "
            "segments), or add a stage that produces the key under the name the inner stage reads"
        ),
    }.get(code)


def _escalate_for(code: str) -> str | None:
    # Both mean "this read might not be satisfied and validation alone cannot settle it" -> mark
    # the Verdict 'uncertain' and resolve with a smoke. The in-composite case is a warning rather
    # than an error deliberately: expansion earns the right to hard-fail a pipeline only once it
    # has been shown not to false-positive on pipelines known to work.
    return {
        "unsatisfied_reads_after_composite": "smoke",
        "unsatisfied_reads_in_composite": "smoke",
    }.get(code)


# --------------------------------------------------------------------------- #
# registered checks
# --------------------------------------------------------------------------- #
@register("data_flow")
def _check_data_flow(ctx: CheckContext) -> CheckResult:
    """Role/key composition, residency, and serializability (foundation).

    Environment gates are evaluated once by ``_check_gates`` below. Passing the
    GPU count into both checks produced duplicate ``gpu_unavailable`` warnings
    for the same stage.
    """
    from nemo_curator.stages.audio import agent as foundation

    report = foundation.validate_pipeline(
        ctx.stages,
        initial_roles=ctx.initial_roles,
        initial_keys=ctx.initial_keys,
        initial_tensor_keys=ctx.initial_tensor_keys,
        available_gpus=None,
    )
    issues = [
        Issue(
            pi.code,
            pi.severity,
            pi.message,
            stage_index=pi.stage_index,
            stage=pi.stage_name,
            fix=_fix_for(pi.code),
            escalate_to=_escalate_for(pi.code),
        )
        for pi in report.issues
    ]
    return CheckResult(
        issues=issues,
        ok=report.ok,
        keys_ok=report.keys_ok,
        produced_roles=sorted(report.produced_roles),
        produced_keys=sorted(report.produced_keys),
    )


def _rate_converted_to(stage: Any, built: Any) -> int | None:  # noqa: ANN401 - IR and built stage
    """The rate this stage CONVERTS its audio to, or ``None`` if it changes nothing.

    Only ``target_sample_rate`` counts as a conversion. ``MonoConversionStage`` takes an
    ``output_sample_rate``, but its card is explicit that this is a rate to VERIFY against and
    that the stage never resamples -- reading it as a conversion would tell the planner the
    audio had been converted when it had merely been checked (and non-matching rows dropped).

    The built stage is consulted so an omitted parameter still resolves to its real default
    (``ResampleAudioStage`` converts to 16 kHz whether or not the recipe says so).
    """
    v = stage.params.get("target_sample_rate")
    if v is None and built is not None:
        v = getattr(built, "target_sample_rate", None)
    return _to_int(v)


def _sample_rate_provenance(ctx: CheckContext, data_srs: set[int]) -> list[set[int]]:  # noqa: C901
    """Track file and resident rate evidence separately for each configured binding."""
    # A conversion only changes the keys it actually writes. In particular, the
    # default resampler leaves the source filepath untouched, and ASR's internal
    # input preparation does not change audio supplied to a later stage.
    file_rates: dict[str, set[int]] = {}
    resident_rates: dict[str, set[int]] = {}
    converted_resident_keys: set[str] = set()
    for key in ctx.initial_tensor_keys or ():
        resident_rates[key] = set(data_srs)
    if ctx.initial_tensor_keys is None and "waveform" in ctx.initial_roles:
        resident_rates["waveform"] = set(data_srs)
    # Built stages carry real parameter defaults, but a stage that failed to construct is
    # skipped in build_stages, so only trust the pairing when nothing was dropped.
    built = list(ctx.stages or []) if len(ctx.stages or []) == len(ctx.recipe.stages) else []
    rates: list[set[int]] = []
    for i, s in enumerate(ctx.recipe.stages):
        stage_obj = built[i] if built else None

        def param(name: str, default: Any = None, stage: Any = s, obj: Any = stage_obj) -> Any:  # noqa: ANN401
            return stage.params.get(name, getattr(obj, name, default))

        filepath_key = param("filepath_key", param("audio_filepath_key", "audio_filepath"))
        waveform_key = param("waveform_key", "waveform")
        residency = param("input_residency", "file")
        file_srs = file_rates.get(filepath_key, data_srs)
        waveform_srs = resident_rates.get(waveform_key)
        if residency == "waveform" or (residency == "auto" and waveform_key in converted_resident_keys):
            srs_here = set(waveform_srs or ())
        elif residency == "auto" and waveform_srs is not None:
            # Auto prefers a usable resident pair, but a file fallback can still be
            # taken on other rows. Both routes must meet the model constraint.
            srs_here = set(waveform_srs) | set(file_srs)
        else:
            srs_here = set(file_srs)
        rates.append(srs_here)
        if s.ref == "ResampleAudioStage":
            converted = _rate_converted_to(s, stage_obj)
            if converted is not None:
                if param("write_to_disk", True):
                    file_rates[param("resampled_audio_filepath_key", "resampled_audio_filepath")] = {converted}
                    if param("update_audio_filepath", False):
                        original_key = param("original_audio_filepath_key", "original_audio_filepath")
                        file_rates.setdefault(original_key, set(file_srs))
                        file_rates[filepath_key] = {converted}
                if param("keep_waveform_in_task", False):
                    resident_rates[waveform_key] = {converted}
                    converted_resident_keys.add(waveform_key)
                elif param("write_to_disk", True) and residency != "file":
                    resident_rates.pop(waveform_key, None)
                    converted_resident_keys.discard(waveform_key)
    return rates


@register("card_constraints")
def _check_card_constraints(ctx: CheckContext) -> CheckResult:
    """Model-card constraints (batch, sample-rate, duration, max-speakers)."""
    idx = get_index()
    out: list[Issue] = []
    data_profile = ctx.data_profile
    # sample-rate keys are strings post-serialization; coerce to int so a matching rate
    # (16000) doesn't false-warn against an int-typed card supported_sample_rates.
    data_srs = (
        {int(k) for k in (data_profile or {}).get("sample_rates", {}) if str(k).lstrip("-").isdigit()}
        if data_profile
        else set()
    )
    mean_dur = (_to_float((data_profile or {}).get("mean_duration_sec")) or 0.0) if data_profile else 0.0
    rates = _sample_rate_provenance(ctx, data_srs)
    for i, s in enumerate(ctx.recipe.stages):
        srs_here = rates[i]
        card = idx.card(s.ref)
        if not card:
            continue
        cons = card.get("constraints", {}) or {}
        bs = cons.get("batch_size")
        if isinstance(bs, dict) and "fixed" in bs and s.params.get("batch_size") not in (None, bs["fixed"]):
            out.append(
                Issue(
                    "card_batch_size",
                    "error",
                    f"{s.ref}: batch_size must be {bs['fixed']} ({bs.get('reason', 'model constraint')})",
                    stage_index=i,
                    stage=s.ref,
                    fix=f"set batch_size={bs['fixed']}",
                )
            )
        supported = cons.get("supported_sample_rates")
        supported_ints = {iv for iv in (_to_int(x) for x in (supported or [])) if iv is not None}
        if supported and srs_here and supported_ints and not srs_here.issubset(supported_ints):
            reached_via = (
                "" if srs_here == data_srs else f" (after an upstream conversion; source was {sorted(data_srs)})"
            )
            out.append(
                Issue(
                    "card_sample_rate",
                    "warning",
                    f"{s.ref}: input sample rates {sorted(srs_here)} not all in supported {supported}{reached_via}",
                    stage_index=i,
                    stage=s.ref,
                    fix="insert a resample stage upstream to the supported rate",
                )
            )
        sweet = cons.get("input_duration_sweetspot_sec")
        sweet_max = _to_float(sweet.get("max")) if isinstance(sweet, dict) else None
        if sweet_max is not None and mean_dur and mean_dur > sweet_max:
            out.append(
                Issue(
                    "card_duration",
                    "warning",
                    f"{s.ref}: mean input duration {mean_dur}s exceeds sweet-spot max {sweet['max']}s",
                    stage_index=i,
                    stage=s.ref,
                    fix="segment/split long audio upstream",
                )
            )
        for key in _MAX_SPEAKERS_KEYS:
            mx_int = _to_int(cons.get("max_speakers"))
            val_int = _to_int(s.params.get(key))
            if mx_int is not None and val_int is not None and val_int > mx_int:
                out.append(
                    Issue(
                        "card_max_speakers",
                        "error",
                        f"{s.ref}: {key}={s.params[key]} exceeds model max_speakers={mx_int}",
                        stage_index=i,
                        stage=s.ref,
                        fix=f"set {key}<={mx_int}",
                    )
                )
    return CheckResult(card_violations=out)


@register("gpu_reservation")
def _check_gpu_reservation(ctx: CheckContext) -> CheckResult:
    """Warn when a GPU-required stage reserves no GPU (``resources`` left at a CPU default).

    ``resources`` is a config knob: a stage whose card is ``bound: gpu`` and NOT
    ``gpu_optional`` must reserve a GPU (``resources.gpus`` or ``gpu_memory_gb``), else the
    executor runs it on CPU (very slow) and can over-parallelize into many model-loading
    actors. Card-driven, so a new GPU stage is covered with no code change. Composites are
    skipped -- they delegate resources to their decomposed inner stages.
    """
    from nemo_curator.stages.audio import agent as foundation
    from nemo_curator.stages.audio.audio_agent._device_capabilities import gpu_optional
    from nemo_curator.stages.base import CompositeStage

    idx = get_index()
    out: list[Issue] = []
    for i, s in enumerate(ctx.recipe.stages):
        card = idx.card(s.ref) or {}
        res_card = card.get("resource") or {}
        if res_card.get("bound") != "gpu":
            continue
        stage_obj = ctx.stages[i] if i < len(ctx.stages) else None
        if stage_obj is None or isinstance(stage_obj, CompositeStage):
            continue
        try:
            requires_gpu = bool(foundation.build_contract(stage_obj).gates.requires_gpu)
        except Exception:  # noqa: BLE001, S112 - an unreadable contract cannot justify a new warning
            continue
        configured_gpu_optional = gpu_optional(stage_obj, card)
        requires_gpu = requires_gpu or (res_card.get("bound") == "gpu" and configured_gpu_optional is False)
        if not requires_gpu or configured_gpu_optional is True:
            continue
        sres = getattr(stage_obj, "resources", None)
        gpus = float(getattr(sres, "gpus", 0.0) or 0.0)
        gpu_mem = float(getattr(sres, "gpu_memory_gb", 0.0) or 0.0)
        if gpus <= 0 and gpu_mem <= 0:
            out.append(
                Issue(
                    "gpu_reservation_missing",
                    "warning",
                    f"{s.ref}: the configured implementation requires a GPU but the stage reserves none "
                    "(resources.gpus=0)",
                    stage_index=i,
                    stage=s.ref,
                    fix=f"set resources=Resources(gpus=1) (VRAM ~ card gpu_mem_gb={res_card.get('gpu_mem_gb')})",
                )
            )
    return CheckResult(issues=out)


@register("gates")
def _check_gates(ctx: CheckContext) -> CheckResult:
    """Environment gates: ffmpeg / GPU / first-run download / runtime secrets."""
    from nemo_curator.stages.audio import agent as foundation

    env = ctx.env
    out: list[Issue] = []
    for idx, st in enumerate(ctx.stages):
        try:
            gates = foundation.build_contract(st).gates
        except Exception:  # noqa: BLE001, S112
            continue
        name = type(st).__name__
        target_is_local = ctx.execution_target == "local"
        if target_is_local and getattr(gates, "requires_ffmpeg", False) and not env.has_ffmpeg:
            out.append(
                Issue(
                    "ffmpeg_missing",
                    "error",
                    f"{name} needs ffmpeg but it is not on PATH",
                    stage_index=idx,
                    stage=name,
                    fix="install ffmpeg",
                )
            )
        if target_is_local and getattr(gates, "requires_gpu", False) and not env.has_gpu:
            # Mask-aware: a masked/unknown GPU is a re-verify decision, NOT a "no GPU"
            # warning. Only a definitively absent GPU (CPU-only torch build) is a real
            # gap. Codes match environment_preflight so the two paths never disagree.
            gpu_status = getattr(env, "gpu_status", "absent")
            if gpu_status == "possibly_masked":
                out.append(
                    Issue(
                        "gpu_possibly_masked",
                        "info",
                        f"{name} requires a GPU; none is reachable from this process but one is likely PRESENT and masked (sandbox/container) -- re-verify with full device access, do not conclude no GPU",
                        stage_index=idx,
                        stage=name,
                        fix="re-run with full device access (outside the sandbox/container)",
                    )
                )
            elif gpu_status == "unknown":
                out.append(
                    Issue(
                        "gpu_availability_unknown",
                        "info",
                        f"{name} requires a GPU but this environment supplied no GPU visibility facts -- re-verify with full device access",
                        stage_index=idx,
                        stage=name,
                        fix="re-verify the GPU with full device access",
                    )
                )
            else:
                out.append(
                    Issue(
                        "gpu_unavailable",
                        "warning",
                        f"{name} declares requires_gpu but this host has no usable GPU (CPU-only torch build)",
                        stage_index=idx,
                        stage=name,
                        fix="install the CUDA torch extra or run on a GPU host",
                    )
                )
        if getattr(gates, "requires_internet_first_run", False):
            out.append(
                Issue(
                    "internet_first_run", "info", f"{name} downloads a model on first run", stage_index=idx, stage=name
                )
            )
        for secret in getattr(gates, "runtime_secrets", []) or []:
            configured = bool(getattr(st, str(secret).lower(), None))
            if target_is_local and secret not in env.available_secrets and not configured:
                out.append(
                    Issue(
                        "missing_secret",
                        "warning",
                        f"{name} needs secret {secret!r} which is not set",
                        stage_index=idx,
                        stage=name,
                        fix=f"export {secret}",
                    )
                )
    return CheckResult(gate_flags=out)


@register("unproducible")
def _check_unproducible(ctx: CheckContext) -> CheckResult:
    """Roles the pipeline reads that no stage in the catalog can produce."""
    from nemo_curator.stages.audio import agent as foundation

    required: set[str] = set()
    for st in ctx.stages:
        try:
            c = foundation.build_contract(st)
        except Exception:  # noqa: BLE001, S112
            continue
        for key in [*c.reads.data_keys, *c.reads.segment_data_keys]:
            required.add(c.key_roles.get(key, "unknown"))
    return CheckResult(unproducible_roles=get_index().unproducible(sorted(required - {"unknown"})))


@register("output_completeness")
def _check_output_completeness(ctx: CheckContext) -> CheckResult:
    """Every requested output role must survive to the end of the recipe.

    Only active when the caller passes ``expected_outputs`` (semantic roles the
    user asked for). Catches the "asked for transcripts, no ASR stage" class.
    Phase 2 compiles ``expected_outputs`` from ``GoalSpec.acceptance_criteria``.
    """
    if not ctx.expected_outputs:
        return CheckResult()
    from nemo_curator.stages.audio import agent as foundation

    report = foundation.validate_pipeline(
        ctx.stages,
        initial_roles=ctx.initial_roles,
        initial_keys=ctx.initial_keys,
        initial_tensor_keys=ctx.initial_tensor_keys,
        available_gpus=None,
    )
    available_roles = set(report.produced_roles)
    available_keys = set(report.produced_keys)
    # A conditional writer is still a potential producer; acceptance verifies
    # whether its branch ran. Its field must survive all later transformations.
    from nemo_curator.stages.audio._agent._composite import expand_composites

    expanded = expand_composites(ctx.stages)
    execution_stages = [item.stage for item in expanded.stages] if expanded.fully_resolved else ctx.stages
    for index, stage in enumerate(execution_stages):
        try:
            contract = foundation.build_contract(stage)
        except Exception:  # noqa: BLE001, S112 - data_flow reports unreadable contracts
            continue
        for conditional in contract.conditional_writes:
            keys = set(conditional.writes.data_keys)
            segment_keys = set(conditional.writes.segment_data_keys)
            roles = {contract.key_roles.get(key, "unknown") for key in keys} - {"unknown"}
            segment_roles = {contract.key_roles.get(key, "unknown") for key in segment_keys} - {"unknown"}
            suffix = foundation.validate_pipeline(
                execution_stages[index + 1 :],
                initial_roles=roles,
                initial_keys=keys,
                initial_segment_roles=segment_roles,
                initial_segment_keys=segment_keys,
                initial_tensor_keys=set(),
                available_gpus=None,
            )
            available_keys |= (keys | segment_keys) & set(suffix.produced_keys)
            available_roles |= (roles | segment_roles) & set(suffix.produced_roles)
    out: list[Issue] = []
    for want in ctx.expected_outputs:
        # Satisfied by a produced semantic role OR a literal produced key. The key match
        # lets output-completeness distinguish specific metrics (e.g. wer vs a SIGMOS
        # sub-score) that all share the generic "score" role.
        if want not in available_roles and want not in available_keys:
            out.append(
                Issue(
                    "missing_output_producer",
                    "error",
                    f"requested output {want!r} is not available at the end of the recipe (no matching role or key)",
                    fix="add a stage that produces this output (see discover / find_producers), or drop the requirement",
                )
            )
    return CheckResult(issues=out)


@register("acceptance_evaluability")
def _check_acceptance_evaluability(ctx: CheckContext) -> CheckResult:
    """Surface evidence requirements before approving a mechanically valid plan."""
    from nemo_curator.stages.audio.audio_agent.acceptance import evaluability_issues
    from nemo_curator.stages.audio.audio_agent.verbs import _terminal_evidence_outputs

    _, row_coverage = _terminal_evidence_outputs(ctx.recipe, [])
    return CheckResult(issues=evaluability_issues(ctx.acceptance_criteria, row_coverage=row_coverage))


@register("request_type_sanity")
def _check_request_type_sanity(ctx: CheckContext) -> CheckResult:
    """Acceptance-set sanity (1A.1): a request implying an output must carry the
    matching criterion (filtering -> yield; transcription -> output_completeness).

    Output-completeness itself is enforced by the ``output_completeness`` check
    (``validate`` compiles criterion fields into ``expected_outputs``); this check
    only surfaces a *missing implied criterion* so success can't be declared while
    silently ignoring the point of the request. Warning + escalate-to-user (not a
    hard fail: the user may legitimately omit it, but it's flagged at the gate).
    Inactive unless a ``request_type`` or criteria were supplied.
    """
    if not ctx.request_type and not ctx.acceptance_criteria:
        return CheckResult()
    from nemo_curator.stages.audio.audio_agent.acceptance import missing_implied

    out = [
        Issue(
            "missing_implied_criterion",
            "warning",
            hint,
            fix="add an acceptance criterion of this type to define success for the request",
            escalate_to="user",
        )
        for _implied, hint in missing_implied(ctx.request_type, ctx.acceptance_criteria)
    ]
    return CheckResult(issues=out)


# Diarization needs continuous audio: a fragmenting VAD before a diarizer with no re-join
# hands it a torn signal (the `diarization-needs-continuous-audio` rule). Separating first,
# on continuous audio, is fine. Roles come from card metadata so a new stage needs no code
# change here -- diarizer = category 'diarize'; fragmenter = category 'segment' with a
# 'fanout' tag, minus diarizers; re-joiner has no card signal, so it stays an explicit set.
_DIARIZERS = frozenset({"InferenceSortformerStage", "PyAnnoteDiarizationStage", "SpeakerSeparationStage"})
_FRAGMENTERS = frozenset({"VADSegmentationStage", "WhisperXVADStage"})
_REJOINERS = frozenset({"SegmentConcatenationStage"})


def _is_diarizer(ref: str, idx: Any) -> bool:  # noqa: ANN401
    """A diarizer/separator: card category 'diarize' (extensible) or the explicit set."""
    if ref in _DIARIZERS:
        return True
    return (idx.card(ref) or {}).get("category") == "diarize"


def _is_fragmenter(ref: str, idx: Any) -> bool:  # noqa: ANN401
    """A VAD-style stage that fans continuous audio into per-segment tasks (breaking
    continuity): the explicit set, or any card with category 'segment' AND a 'fanout' tag
    that is not itself a diarizer -- so a NEW VAD-style stage is covered with no code change.
    """
    if ref in _FRAGMENTERS:
        return True
    card = idx.card(ref) or {}
    return card.get("category") == "segment" and "fanout" in (card.get("tags") or []) and not _is_diarizer(ref, idx)


def _is_rejoiner(ref: str, idx: Any) -> bool:  # noqa: ANN401, ARG001
    """A stage that stitches segments back into a continuous waveform. Structural: the only
    such stage today is SegmentConcatenationStage, so this is an explicit set kept behind a
    helper for symmetry with the derived diarizer/fragmenter checks (and a future card signal).
    """
    return ref in _REJOINERS


@register("diarization_continuity")
def _check_diarization_continuity(ctx: CheckContext) -> CheckResult:
    """Flag a diarizer/separator that runs after VAD fragmentation without a
    SegmentConcatenation re-join (it would see per-segment clips, not continuous
    audio). Diarizing/separating on the continuous audio before any VAD does not trip.
    """
    idx = get_index()
    refs = [s.ref for s in ctx.recipe.stages]
    out: list[Issue] = []
    for di, r in enumerate(refs):
        if not _is_diarizer(r, idx):
            continue
        frags_before = [fi for fi, fr in enumerate(refs[:di]) if _is_fragmenter(fr, idx)]
        if not frags_before:
            continue  # runs on continuous audio (no upstream fragmentation) -> fine
        fi = max(frags_before)  # nearest fragmenter before the diarizer
        if not any(_is_rejoiner(rr, idx) for rr in refs[fi + 1 : di]):
            out.append(
                Issue(
                    "diarization_needs_continuous_audio",
                    "error",
                    f"{r} runs after {refs[fi]} without re-joining segments; "
                    "diarization/separation needs a continuous waveform",
                    stage_index=di,
                    stage=r,
                    fix="insert SegmentConcatenationStage between the VAD and the diarizer, "
                    "or diarize/separate on the continuous audio before segmenting",
                )
            )
    return CheckResult(issues=out)


def _missing_source_audio_keys(stages: list[Any], columns: set[str]) -> set[str]:
    """Check configured file consumers against observed keys, not a reader's defaults."""
    from nemo_curator.stages.audio._agent._agent_registry import build_contract
    from nemo_curator.stages.audio._agent._composite import expand_composites
    from nemo_curator.stages.audio._agent._roles import role_for_value

    available = set(columns)
    expansion = expand_composites(stages[1:])
    if not expansion.fully_resolved:
        msg = "cannot resolve source audio consumers through an opaque composite"
        raise ValueError(msg)

    def missing_paths(spec: Any, roles: dict[str, str]) -> set[str]:  # noqa: ANN401 - configured IOSpec
        return {
            key
            for key in spec.data_keys
            if roles.get(key, role_for_value(key)) == "audio_filepath" and key not in available
        }

    for item in expansion.stages:
        contract = build_contract(item.stage)
        missing = missing_paths(contract.reads, contract.key_roles)
        groups = [contract.reads_one_of] if contract.reads_one_of else []
        groups.extend(
            branch.reads_one_of
            for branch in contract.conditional_reads
            if set(branch.requires_keys) <= available and not set(branch.forbids_keys) & available
        )
        for alternatives in groups:
            # An available resident/nested alternative needs no top-level file.
            if alternatives and not any(set(spec.data_keys) <= available for spec in alternatives):
                missing.update(key for spec in alternatives for key in missing_paths(spec, contract.key_roles))
        if missing:
            return missing
        if not contract.preserves_upstream_keys:
            available.clear()
        available.difference_update(contract.removes_keys)
        available.difference_update(contract.invalidates_keys)
        available.update(contract.writes.data_keys)
    return set()


@register("source_schema")
def _check_source_schema(ctx: CheckContext) -> CheckResult:
    """The profiled manifest must actually carry the audio path the stages will read.

    ``ManifestReaderStage`` declares ``writes=["audio_filepath"]`` but is schema-agnostic:
    it emits whatever columns each JSONL row happens to have. Validation therefore trusts
    the declaration over the data, so a manifest using another convention (Common Voice's
    ``path``/``sentence``) satisfies every downstream read and the recipe validates clean --
    then produces zero rows. This is the "validates green, yields nothing" class, and the
    profile is the evidence that settles it.

    The manifest format is the CALLER's contract to meet, not something to infer around.
    ``audio_filepath`` is the documented NeMo manifest key, and guessing which other column
    might hold the audio means guessing wrong on some dataset and silently curating the
    wrong field. So this refuses with a precise, actionable message instead: an unreadable
    source is a blocking error, and the fix is either to convert the manifest or to say
    explicitly which column holds the audio.

    Text/metadata-only recipes need no audio column. Configured consumers,
    including composite leaves and alternative resident inputs, determine
    whether the observed schema actually needs a file carrier.
    """
    profile = ctx.data_profile or {}
    if profile.get("kind") != "manifest":
        return CheckResult()
    columns = {str(c) for c in (profile.get("manifest_keys") or [])}
    if not columns:
        return CheckResult()  # nothing observed -> no evidence, so no claim
    missing = _missing_source_audio_keys(ctx.stages, columns)
    if not missing:
        return CheckResult()
    return CheckResult(
        issues=[
            Issue(
                "source_schema_mismatch",
                "error",
                f"the manifest's columns {sorted(columns)} cannot supply the configured "
                f"audio_filepath input key(s) {sorted(missing)} required by this recipe",
                fix=(
                    "convert the manifest to the NeMo format, where each row carries its audio under "
                    "'audio_filepath' -- or, if the column is deliberately named something else, say so "
                    "explicitly on the reading stage (audio_filepath_key='<column>')"
                ),
                escalate_to="user",
            )
        ]
    )
