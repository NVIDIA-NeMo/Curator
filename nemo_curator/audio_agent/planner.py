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

"""Deterministic resource planner (1C.1): pick execution mode + feasibility.

Given each stage's declared needs (capability-card ``resource`` facts when present,
else conservative defaults) and the machine (:class:`EnvProfile`), decide
**streaming vs batch** by a feasibility check over CPU / GPU-memory / host-RAM
(+ disk headroom), defaulting to streaming and falling back to batch. The result
is a :class:`ResourcePlan` attached to the recipe as a recomputable annotation
(layered save, 1.2).

Scope of 1C.1: mode selection + feasibility + escalation. Per-stage resource
*assignment* (1C.2) and measured calibration (the ``calibration`` module) refine
these numbers later; this module never reimplements Xenna's bin-packer.

Feasibility gates ONLY on exact facts that are knowable ahead of time and that Xenna
itself enforces -- never on a guess:
  (a) Ray reservation (scheduling) -- each stage reserves ``resources.gpus`` (the
      fraction Xenna pins per replica); the concurrent sum must fit the GPU COUNT,
      else Xenna aborts ("requires 1.5 but only 1 are available"). Exact -> a gate.
  (b) GPU presence -- a GPU-only stage needs a GPU on the box. Exact -> a gate.
  (c) CPU demand/reservation and host-RAM against known machine totals -> a gate.

VRAM fit (memory) is DELIBERATELY NOT a gate. Real device-VRAM need depends on
weights x activations x batch x precision -- the card ``gpu_mem_gb`` is a best_guess and
is frequently *unknown* on multi-NIC / cloud hosts (Ray advertises a non-loopback IP).
Xenna does not reserve VRAM (it reserves the GPU *fraction*), and the agent already
measures real VRAM in ``smoke`` on the actual GPU. So VRAM is surfaced as an ADVISORY
estimate/warning only: it never sets ``feasible=False`` and never forces batch. A real
over-subscription shows up in the bounded ``smoke`` (the ground truth) as an OOM the
failure classifier turns into "lower batch_size / smaller model", and the runtime
auto-fallback drops streaming -> batch. This keeps the planner from out-predicting the
scheduler and -- critically -- from blocking the very measurement step (``smoke``).

CPU feasibility tracks estimated demand separately from the exact Ray reservation in
``stage.resources.cpus``. Positive fixed ``num_workers()`` values multiply both per-worker
footprints. The Xenna CPU budget is ``floor(total_cpus * 0.95)`` in both modes.

    streaming feasible iff  Sum(cpu_demand) <= allocatable_cpus
                            AND Sum(cpu_reservation) <= allocatable_cpus
                            AND Sum(host_mem) <= ram*0.90
                            AND Sum(gpu_reservation) <= num_gpus
                            AND (a GPU exists when any stage is GPU-only)
    else batch (only the largest single stage must fit each *gated* dimension);
    if even that fails -> escalate. VRAM only ever adds an advisory note.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_curator.audio_agent.contracts import EnvProfile

CPU_ALLOC = 0.95  # cpu_allocation_percentage headroom
RAM_ALLOC = 0.90  # host-RAM headroom
_DEFAULT_GPU_MEM_GB = 2.0  # conservative VRAM estimate for a GPU stage without a card fact
_DEFAULT_HOST_MEM_GB = 1.0  # conservative host-RAM estimate per stage without a card fact


@dataclass
class StageNeed:
    """Per-stage resource needs (absolute, machine-independent)."""

    index: int
    name: str
    # Card/calibration estimate of CPU demand per worker. This is distinct from
    # ``cpu_reservation``, which is the exact value Ray schedules.
    cpus: float
    gpu_mem_gb: float
    host_mem_gb: float
    gpu_optional: bool = True
    # Exact per-worker Ray CPU reservation from ``stage.resources.cpus``.
    cpu_reservation: float = 0.0
    # Ray ``Resources.gpus`` the stage reserves for SCHEDULING (the fraction Xenna
    # pins per replica) -- independent of ``gpu_mem_gb`` (the VRAM it actually needs).
    gpu_reservation: float = 0.0
    # A positive explicit worker count means that many replicas are scheduled
    # concurrently, even in batch mode. ``None`` means executor-sized; planning
    # accounts for the minimum schedulable footprint of one replica.
    num_workers: int | None = None
    source: str = "default"  # measured (calibration raised a value) | card | default
    # Per-resource provenance keeps the aggregate ``source`` honest when only
    # some measurements exceed their conservative card/default floors.
    resource_sources: dict[str, str] = field(default_factory=dict)


@dataclass
class ResourcePlan:
    """The planner's output: mode + per-stage needs + feasibility + estimate."""

    mode: str = "streaming"  # "streaming" | "batch"
    feasible: bool = True
    per_stage: list[dict[str, Any]] = field(default_factory=list)
    estimate: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    escalations: list[str] = field(default_factory=list)
    machine_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "feasible": self.feasible,
            "per_stage": self.per_stage,
            "estimate": self.estimate,
            "notes": self.notes,
            "escalations": self.escalations,
            "machine_fingerprint": self.machine_fingerprint,
        }


def _gpu_fraction(gpu_mem_gb: float, machine_gpu_mem_gb: float) -> float:
    """VRAM need as a fraction of one GPU; 1.0 (a whole GPU) when the size is unknown."""
    if gpu_mem_gb <= 0:
        return 0.0
    if machine_gpu_mem_gb > 0:
        return gpu_mem_gb / machine_gpu_mem_gb
    return 1.0  # GPU needed but machine VRAM unknown -> assume a whole GPU (conservative)


def _worker_count(need: StageNeed) -> int:
    """Concurrent replicas the planner must reserve for one execution stage."""
    return need.num_workers if need.num_workers is not None else 1


def _fixed_num_workers(stage: Any) -> int | None:  # noqa: ANN401
    """Read a stage's fixed worker request without changing the stage."""
    try:
        value = stage.num_workers()
    except Exception:  # noqa: BLE001 - an unreadable hint falls back to executor sizing
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _stage_need(
    index: int,
    stage: Any,  # noqa: ANN401
    contract: Any,  # noqa: ANN401
    card: dict[str, Any] | None,
    calib: dict[str, Any] | None = None,
) -> StageNeed:
    """Derive a stage's needs from scheduling truth and conservative estimates.

    Until calibration is bound to the exact stage configuration and proves full
    configured-batch coverage, a bounded smoke may raise a card/default estimate
    but must never lower it. This prevents a sample-of-one measurement from
    understating full-run CPU/RAM/VRAM.
    """
    name = type(stage).__name__
    res = (card or {}).get("resource", {}) or {}
    calib = calib or {}
    requires_gpu = bool(getattr(getattr(contract, "gates", None), "requires_gpu", False))

    baseline_source = "card" if res else "default"

    def pick(key: str, card_default: Any) -> tuple[float, str]:  # noqa: ANN401
        # An explicit ``null`` says "nobody has established this yet", which is what a MISSING
        # key says too -- so both must reach the conservative default. Read as a value instead,
        # writing the placeholder was worse than omitting the line: ``gpu_mem_gb: null`` on a
        # GPU-required stage priced at 0.0 GB rather than the floor meant for unknowns.
        stated = res.get(key)
        baseline = float((card_default if stated is None else stated) or 0.0)
        measured = calib.get(key)
        if measured is not None and float(measured) > baseline:
            return float(measured), "measured"
        return baseline, baseline_source

    cpus, cpus_source = pick("cpus", 1.0)
    gpu_mem, gpu_mem_source = pick(
        "gpu_mem_gb",
        _DEFAULT_GPU_MEM_GB if requires_gpu else 0.0,
    )
    host_mem, host_mem_source = pick("host_mem_gb", _DEFAULT_HOST_MEM_GB)
    gpu_optional = bool(res.get("gpu_optional", not requires_gpu))
    resources = getattr(stage, "resources", None)
    # Ray CPU/GPU reservations are scheduling facts only the built stage knows;
    # they are NOT card/calibration numbers. Merely read the Resources object:
    # planning must not normalize or mutate stage defaults.
    cpu_reservation_raw = getattr(resources, "cpus", 1.0)
    cpu_reservation = float(1.0 if cpu_reservation_raw is None else cpu_reservation_raw)
    gpu_reservation = float(getattr(resources, "gpus", 0.0) or 0.0)
    num_workers = _fixed_num_workers(stage)
    resource_sources = {
        "cpus": cpus_source,
        "gpu_mem_gb": gpu_mem_source,
        "host_mem_gb": host_mem_source,
    }
    # A bounded measurement that is equal to or below its conservative floor did
    # not supply the selected value and must not be reported as its provenance.
    source = "measured" if "measured" in resource_sources.values() else baseline_source
    return StageNeed(
        index=index,
        name=name,
        cpus=cpus,
        gpu_mem_gb=float(gpu_mem or 0.0),
        host_mem_gb=host_mem,
        gpu_optional=gpu_optional,
        cpu_reservation=cpu_reservation,
        gpu_reservation=gpu_reservation,
        num_workers=num_workers,
        source=source,
        resource_sources=resource_sources,
    )


def _execution_needs(
    stages: list[Any],
    contracts: list[Any],
    idx: Any,  # noqa: ANN401
    calib_for: Any,  # noqa: ANN401
) -> list[StageNeed]:
    """Flatten composite stages into the stages the backend runs, and price those.

    A ``CompositeStage`` (e.g. ``SplitASRAlignJoinStage``) advertises only its own
    ``resources`` while hiding the inner stages it expands into at runtime. Summing the
    composite alone undercounts the concurrent Ray GPU reservation / VRAM / CPU and can
    wrongly pick streaming (which the executor then aborts). Expanding here makes mode
    selection match reality for ANY composite -- not one specific recipe.

    The traversal itself lives in :mod:`nemo_curator.stages.audio._agent._composite`, shared with
    pipeline validation: a composite that resolves to one set of stages for resource planning
    and a different set for correctness checking would be worse than either alone.
    """
    from nemo_curator.stages.audio._agent._composite import expand_composites

    expansion = expand_composites(stages)
    grouped = expansion.by_recipe_index()
    out: list[StageNeed] = []

    def need(index: int, stage: Any, contract: Any) -> StageNeed:  # noqa: ANN401
        return _stage_need(index, stage, contract, idx.card(type(stage).__name__), calib_for(stage))

    for i, st in enumerate(stages):
        contract = contracts[i] if i < len(contracts) else None
        if i in expansion.opaque or i in expansion.unrunnable:
            # A composite nobody could open still occupies the cluster: counting it as one leaf
            # understates its inner concurrency, but skipping it budgets nothing at all.
            # ``unrunnable`` lands here for the same reason -- the executor runs the composite
            # itself, so its own resources are what gets reserved. Omitting either would price a
            # stage at zero and let an infeasible plan look feasible.
            out.append(need(i, st, contract))
            continue
        for item in grouped.get(i, []):
            out.append(need(i, item.stage, contract if item.stage is st else None))
    return out


def _calibration_mapping(
    calibration: dict[str, Any] | None,
) -> tuple[Mapping[str, Any], Any]:
    """Return bare stage entries and an optional wrapper-level fingerprint.

    ``calibrate`` returns ``{"calibration": {stage: facts}}`` while the SDK has
    historically also accepted the inner mapping. Supporting both here keeps all
    callers behind the same validation boundary.
    """
    if not isinstance(calibration, Mapping):
        return {}, None
    if "calibration" not in calibration:
        return calibration, calibration.get("machine_fingerprint")
    nested = calibration.get("calibration")
    if not isinstance(nested, Mapping):
        return {}, calibration.get("machine_fingerprint")
    return nested, calibration.get("machine_fingerprint")


def _valid_calibration_number(value: Any) -> bool:  # noqa: ANN401
    """Whether a measured resource fact is finite and non-negative."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def cpu_fallback(stages: list[Any], env: Any, *, index: Any = None) -> list[str]:  # noqa: ANN401
    """Drop the Ray GPU reservation of GPU-OPTIONAL stages on a host with no GPU.

    ``audio_cpu`` is a supported install profile, and a stage whose card declares
    ``resource.gpu_optional: true`` runs on CPU. Such stages nevertheless declare a GPU
    reservation by default (``Resources(gpus=0.5)`` on UTMOS/SIGMOS), which a GPU-less
    scheduler can never satisfy -- so the plan was refused for a recipe that would in
    fact have run. Zero the reservation for exactly those stages (replacing entries with
    ``.with_()`` copies, which the caller then executes) and report what changed.

    Deliberately conservative: a stage is downgraded only when its card EXPLICITLY says
    the GPU is optional. A stage that requires a GPU -- or one with no card to vouch for
    it -- keeps its reservation and still escalates, so a genuinely GPU-bound recipe is
    never quietly turned into a CPU run.

    Not applied when the GPU is merely MASKED (present but unreachable from this
    process): there a device is assumed present and the reservation must stand.
    """
    from nemo_curator.audio_agent.index import get_index
    from nemo_curator.stages.resources import Resources

    if int(getattr(env, "gpu_count", 0) or 0) > 0 or bool(getattr(env, "gpu_possibly_masked", False)):
        return []
    idx = index or get_index()
    notes: list[str] = []
    for i, stage in enumerate(stages):
        resources = getattr(stage, "resources", None)
        reserved = float(getattr(resources, "gpus", 0.0) or 0.0)
        if reserved <= 0:
            continue
        card_resource = (idx.card(type(stage).__name__) or {}).get("resource") or {}
        if card_resource.get("gpu_optional") is not True:
            continue  # required, or unvouched: leave it to escalate honestly
        cpus = float(getattr(resources, "cpus", 1.0) or 1.0)
        stages[i] = stage.with_(resources=Resources(cpus=cpus))
        notes.append(
            f"{type(stage).__name__}: released a {reserved} GPU reservation and scheduled on CPU "
            "(card declares gpu_optional; no GPU on this host)"
        )
    return notes


def plan(  # noqa: C901, PLR0912, PLR0913, PLR0915
    stages: list[Any],
    contracts: list[Any],
    env: EnvProfile,
    data_profile: dict[str, Any] | None = None,
    *,
    index: Any = None,  # noqa: ANN401
    calibration: dict[str, Any] | None = None,
) -> ResourcePlan:
    """Choose execution mode + report feasibility for ``stages`` on ``env``.

    Default streaming; fall back to batch when the concurrent sum doesn't fit
    (CPU, host-RAM, Ray GPU reservations, or VRAM). If even the largest single
    stage doesn't fit (reservation > num_gpus, VRAM > machine VRAM, or a GPU-only
    stage has no GPU), mark ``feasible=False`` with escalations. Per-stage
    estimates conservatively take the maximum of measured ``calibration`` and
    card/default facts; the Ray CPU/GPU reservations are always read from the
    built stage's ``resources``.
    """
    from nemo_curator.audio_agent.index import get_index

    idx = index or get_index()
    machine_fingerprint = env.fingerprint()
    calibration_entries, wrapper_fingerprint = _calibration_mapping(calibration)
    calibration_notes: list[str] = []
    noted_calibration_issues: set[str] = set()

    def _calibration_note(message: str) -> None:
        if message not in noted_calibration_issues:
            noted_calibration_issues.add(message)
            calibration_notes.append(message)

    def _calib_for(st: Any) -> dict[str, Any] | None:  # noqa: ANN401
        # Match by class name (manual calibration dicts) OR stage.name (what a smoke's
        # per_stage_metrics / from_smoke key by, e.g. "UTMOSFilter" vs "UTMOSFilterStage").
        stage_name = type(st).__name__
        runtime_name = getattr(st, "name", "") or "\0"
        raw = calibration_entries.get(stage_name) or calibration_entries.get(runtime_name)
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            _calibration_note(f"ignored invalid calibration for {stage_name}: entry is not a mapping")
            return None

        explicit_fingerprint = raw.get("machine_fingerprint", wrapper_fingerprint)
        if explicit_fingerprint is not None:
            if not isinstance(explicit_fingerprint, str) or not explicit_fingerprint:
                _calibration_note(f"ignored calibration for {stage_name}: machine_fingerprint is invalid")
                return None
            if explicit_fingerprint != machine_fingerprint:
                _calibration_note(
                    f"ignored calibration for {stage_name}: machine_fingerprint does not match this machine"
                )
                return None

        clean: dict[str, Any] = {}
        for key in ("cpus", "gpu_mem_gb", "host_mem_gb"):
            if key not in raw or raw[key] is None:
                continue
            if not _valid_calibration_number(raw[key]):
                _calibration_note(
                    f"ignored invalid calibration value for {stage_name}.{key}: expected a finite non-negative number"
                )
                continue
            clean[key] = float(raw[key])
        if not clean:
            return None
        clean["source"] = raw.get("source", "measured")
        return clean

    needs = [
        _stage_need(i, st, contracts[i] if i < len(contracts) else None, idx.card(type(st).__name__), _calib_for(st))
        for i, st in enumerate(stages)
    ]
    # Feasibility math runs over the EXECUTION stages (composites flattened), so a
    # composite's hidden inner GPU stages are counted; the per-stage *report* below stays
    # at the recipe level. exec_needs == needs when there are no composites.
    exec_needs = _execution_needs(stages, contracts, idx, _calib_for)
    measured_count = sum(1 for n in exec_needs if n.source == "measured")
    if measured_count:
        rp_note = f"using measured calibration for {measured_count} stage(s) (counted after composite expansion)"
    else:
        rp_note = ""

    total_cpus = float(env.total_cpus or 1)
    # Xenna converts its fractional CPU allocation to an integral worker budget.
    # Use the same floor for streaming and batch so a batch fallback cannot claim
    # schedulability with CPUs Xenna intentionally withholds.
    allocatable_cpus = math.floor(total_cpus * CPU_ALLOC)
    num_gpus = int(env.gpu_count or 0)
    # A masked GPU (unreachable from this possibly-sandboxed process, but likely
    # PRESENT) must not be treated as absent: GPU presence then DEFERS to smoke/run
    # with full device access -- consistent with VRAM being advisory here (smoke is
    # the oracle). Only a host with no masking signal blocks a GPU-only stage.
    gpu_masked = bool(getattr(env, "gpu_possibly_masked", False))
    machine_gpu_mem = float(env.gpu_mem_gb or 0.0)
    total_ram = float(env.total_ram_gb or 0.0)

    # Explicit ``num_workers`` means those actors coexist in both Xenna modes.
    # Without it, account for the minimum one-replica schedulable footprint.
    sum_cpu_demand = sum(n.cpus * _worker_count(n) for n in exec_needs)
    sum_cpu_reservation = sum(n.cpu_reservation * _worker_count(n) for n in exec_needs)
    sum_ram = sum(n.host_mem_gb * _worker_count(n) for n in exec_needs)
    max_cpu_demand = max((n.cpus * _worker_count(n) for n in exec_needs), default=0.0)
    max_cpu_reservation = max(
        (n.cpu_reservation * _worker_count(n) for n in exec_needs),
        default=0.0,
    )
    max_ram = max((n.host_mem_gb * _worker_count(n) for n in exec_needs), default=0.0)
    # GPU is TWO INDEPENDENT constraints, tracked separately (both must hold):
    #  (a) Ray reservation (scheduling): stages reserve ``resources.gpus``; Xenna pins
    #      that fraction per replica, so the CONCURRENT sum must fit the GPU COUNT (else
    #      it aborts, e.g. "requires 1.5 but only 1 are available").
    #  (b) VRAM fit (memory): the CONCURRENT VRAM must fit one GPU's memory (when known).
    sum_reservation = sum(n.gpu_reservation * _worker_count(n) for n in exec_needs)
    max_reservation = max(
        (n.gpu_reservation * _worker_count(n) for n in exec_needs),
        default=0.0,
    )
    # A GPU-optional stage with no Ray GPU reservation is executing in CPU mode;
    # its card's possible GPU footprint is not a demand on this plan. GPU-only
    # stages remain a memory demand even if their reservation was misconfigured.
    gpu_memory_needs = [n for n in exec_needs if n.gpu_mem_gb > 0 and (n.gpu_reservation > 0 or not n.gpu_optional)]
    sum_gpu_mem = sum(n.gpu_mem_gb * _worker_count(n) for n in gpu_memory_needs)
    max_gpu_mem = max(
        (n.gpu_mem_gb * _worker_count(n) for n in gpu_memory_needs),
        default=0.0,
    )
    sum_gpu_fraction = sum(
        _gpu_fraction(n.gpu_mem_gb, machine_gpu_mem) * _worker_count(n) for n in gpu_memory_needs
    )  # informational

    rp = ResourcePlan(machine_fingerprint=machine_fingerprint)
    rp.notes.extend(str(note) for note in (env.notes or []) if str(note))
    rp.notes.extend(calibration_notes)
    if rp_note:
        rp.notes.append(rp_note)
    if len(exec_needs) != len(needs):
        rp.notes.append(
            f"expanded {len(needs)} recipe stage(s) into {len(exec_needs)} execution stage(s) "
            "(composites flattened) for resource planning"
        )

    cpu_demand_ok = sum_cpu_demand <= allocatable_cpus
    cpu_reservation_ok = sum_cpu_reservation <= allocatable_cpus
    cpu_ok = cpu_demand_ok and cpu_reservation_ok
    ram_ok = total_ram <= 0 or sum_ram <= total_ram * RAM_ALLOC  # unknown RAM -> don't block
    reservation_stream_ok = sum_reservation <= num_gpus  # (a) concurrent Ray reservations fit the GPU count
    gpu_presence_ok = (not gpu_memory_needs) or num_gpus > 0 or gpu_masked
    vram_known = machine_gpu_mem > 0

    # VRAM is NOT part of mode selection: it's a best-guess (often unknown), and forcing
    # batch on it pre-empts the scheduler. Default streaming on the exact dims; a real VRAM
    # over-subscription is caught by smoke + the runtime streaming->batch auto-fallback.
    if cpu_ok and ram_ok and gpu_presence_ok and reservation_stream_ok:
        rp.mode = "streaming"
    else:
        rp.mode = "batch"
        rp.notes.append(
            f"streaming does not fit (cpu_demand_ok={cpu_demand_ok}, "
            f"cpu_reservation_ok={cpu_reservation_ok} "
            f"[demand {round(sum_cpu_demand, 2)}, reservation {round(sum_cpu_reservation, 2)} "
            f"vs {allocatable_cpus} allocatable CPU(s)], ram_ok={ram_ok}, "
            f"gpu_available_ok={gpu_presence_ok}, "
            f"gpu_reservation_ok={reservation_stream_ok} [sum {round(sum_reservation, 2)} vs {num_gpus} GPU(s)]); "
            f"falling back to batch (sequential)"
        )

    # Batch feasibility: only the largest single stage must fit each dimension; else escalate.
    if rp.mode == "batch":
        if max_cpu_demand > allocatable_cpus:
            rp.feasible = False
            rp.escalations.append(
                f"a single stage demands {max_cpu_demand} CPUs > Xenna allocatable "
                f"{allocatable_cpus} of machine {total_cpus}"
            )
        if max_cpu_reservation > allocatable_cpus:
            rp.feasible = False
            rp.escalations.append(
                f"a single stage reserves {max_cpu_reservation} Ray CPU(s) > Xenna allocatable "
                f"{allocatable_cpus} of machine {total_cpus}"
            )
        if (
            num_gpus == 0
            and gpu_masked
            and (max_reservation > 0 or any(n.gpu_mem_gb > 0 and not n.gpu_optional for n in exec_needs))
        ):
            # Masked, not absent: defer to smoke/run with full device access rather
            # than refuse. A genuinely GPU-less run surfaces at smoke (the oracle).
            rp.notes.append(
                "GPU appears masked (unreachable from this process); assuming a GPU is present "
                "on the execution host -- smoke/run with full device access will confirm"
            )
        elif num_gpus == 0 and max_reservation > 0:
            rp.feasible = False
            rp.escalations.append(f"a stage reserves {max_reservation} Ray GPU(s) but no GPU is available")
        elif num_gpus == 0 and any(n.gpu_mem_gb > 0 and not n.gpu_optional for n in exec_needs):
            rp.feasible = False
            rp.escalations.append("a GPU-only stage requires a GPU but none is available")
        if num_gpus > 0 and max_reservation > num_gpus:
            rp.feasible = False
            rp.escalations.append(f"a single stage reserves {max_reservation} Ray GPU(s) > available {num_gpus}")
        # NOTE: VRAM is intentionally NOT escalated here -- it is advisory (see below).
        if total_ram > 0 and max_ram > total_ram:
            rp.feasible = False
            rp.escalations.append(f"a single stage needs {max_ram} GB RAM > machine {total_ram} GB")

    # VRAM advisory (never a gate): the estimate is best_guess and often unknown, and smoke
    # measures the real fit on the actual GPU. Warn but keep the plan feasible -- smoke is
    # the oracle and the runtime auto-fallback handles a real streaming OOM.
    if gpu_memory_needs:
        if not vram_known:
            rp.notes.append(
                f"GPU VRAM for this machine is unknown; the ~{round(sum_gpu_mem, 2)} GB estimate is "
                "unverified (best-guess) -- smoke will measure the real fit on the actual GPU"
            )
        elif sum_gpu_mem > machine_gpu_mem or max_gpu_mem > machine_gpu_mem:
            rp.notes.append(
                f"estimated VRAM (sum {round(sum_gpu_mem, 2)} GB, peak stage {round(max_gpu_mem, 2)} GB) "
                f"may exceed machine GPU memory {machine_gpu_mem} GB (best-guess) -- smoke will confirm; "
                "streaming auto-falls-back to batch and you can lower batch_size if it OOMs"
            )

    # Disk headroom (best-effort; per-file sizing is a data-driven refinement).
    if env.free_disk_gb is not None and env.free_disk_gb < 1.0:
        rp.notes.append(f"low free disk ({env.free_disk_gb} GB) - outputs may not fit")

    rp.per_stage = [
        {
            "stage_index": n.index,
            "stage": n.name,
            "cpus": n.cpus,
            "cpu_reservation": n.cpu_reservation,
            "gpu_reservation": n.gpu_reservation,
            "gpu_mem_gb": n.gpu_mem_gb,
            "host_mem_gb": n.host_mem_gb,
            "num_workers": n.num_workers,
            "source": n.source,
            "resource_sources": dict(n.resource_sources),
        }
        for n in needs
    ]
    rp.estimate = {
        "mode": rp.mode,
        "num_files": int((data_profile or {}).get("num_files", 0)),
        # ``sum_cpus`` remains the card/calibration demand for compatibility.
        "sum_cpus": round(sum_cpu_demand, 2),
        "sum_cpu_demand": round(sum_cpu_demand, 2),
        "sum_cpu_reservation": round(sum_cpu_reservation, 2),
        "allocatable_cpus": allocatable_cpus,
        "sum_gpu_reservation": round(sum_reservation, 2),
        "max_gpu_reservation": round(max_reservation, 2),
        "sum_gpu_mem_gb": round(sum_gpu_mem, 2),
        "sum_gpu_fraction": round(sum_gpu_fraction, 2),
        "gpu_mem_known": vram_known,
        "sum_host_mem_gb": round(sum_ram, 2),
        "machine": {"cpus": total_cpus, "gpus": num_gpus, "gpu_mem_gb": machine_gpu_mem, "ram_gb": total_ram},
        "execution_stages": [
            {
                "stage_index": n.index,
                "stage": n.name,
                "cpus": n.cpus,
                "cpu_reservation": n.cpu_reservation,
                "gpu_reservation": n.gpu_reservation,
                "gpu_mem_gb": n.gpu_mem_gb,
                "host_mem_gb": n.host_mem_gb,
                "num_workers": n.num_workers,
                "source": n.source,
                "resource_sources": dict(n.resource_sources),
            }
            for n in exec_needs
        ],
    }
    return rp
