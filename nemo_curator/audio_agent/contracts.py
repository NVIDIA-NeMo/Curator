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

"""Typed data objects of the planning arc.

These are the JSON-safe contracts the deterministic core emits and consumes so
the host LLM (and the eval harness) get structured grounding. The host-produced
``GoalSpec`` / ``Critique`` are defined by the skill, not here — the objects in
this module are the ones our verbs return.

    PlanningContext   what the router/planner is handed (category tree + facts)
    Verdict           the output of ``validate`` (roles / keys / cards / gates)
    SmokeReport       the output of ``smoke`` (bounded evidence)
    PlanResult        a frozen, validated plan (or an escalate/refuse decision)
    DataProfile       the profiler's read of the input data
    EnvProfile        the env probe's read of the machine
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, NoReturn

Severity = Literal["error", "warning", "info"]


def _fingerprint(payload: dict[str, Any]) -> str:
    """Stable short hash of a payload (for machine/data fingerprints, layered save)."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _clean(value: Any) -> Any:  # noqa: ANN401
    """Coerce dataclasses / sets / tuples to JSON-serializable primitives.

    Underscored but NOT private: imported by ``report`` for its own ``to_dict``. Every
    contract here serializes through it, so the set of containers it flattens is a shared
    assumption rather than a local one -- ``_safety.redact`` walks the same shapes.
    """
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, (set, frozenset)):
        return sorted(_clean(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    return value


# Prefixes of a dataset_key, strongest first. Named here so the profiler, the reuse scan
# and the CLI all recognize the same identities (see DataProfile.dataset_key).
DATASET_KEY_TIERS: tuple[str, ...] = ("stat", "shape")


@dataclass
class DataProfile:
    """The profiler's structured read of an input dataset (no learning/memory)."""

    source: str = ""
    kind: Literal["manifest", "folder", "unknown"] = "unknown"
    num_files: int = 0
    sample_rates: dict[int, int] = field(default_factory=dict)  # sr -> count
    channels: dict[int, int] = field(default_factory=dict)  # nchan -> count
    total_duration_sec: float = 0.0
    mean_duration_sec: float = 0.0
    codecs: dict[str, int] = field(default_factory=dict)
    has_transcripts: bool = False
    manifest_keys: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)
    # Fatal errors in the source definition itself (open/decode/JSON shape).
    # Unlike an unreadable referenced audio file, these mean execution cannot
    # be trusted to consume the complete authored dataset.
    source_errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Content-ish key for execution reuse (REUSE_ARCHITECTURE.md §3). Set by the profiler
    # only when it could stat every local source file.
    stat_digest: str = ""
    # A source-definition identity retained when metadata is incomplete. It can distinguish
    # manifests/folders better than the sampled shape, but remains a low-trust ``shape`` key.
    identity_digest: str = ""
    excluded_intermediates: int = 0  # stage-written files (e.g. split chunks) skipped in the scan
    # Per-file identity behind the ``stat`` key: ``{relpath: token}``. The digest answers "did
    # this dataset change", which is all reuse needed; the inventory answers "which files
    # changed", which is what a delta run needs to process only those (REUSE_ARCHITECTURE.md
    # §7). Populated only in the ``stat`` tier -- a partial inventory would report untouched
    # files as absent and delete their prior results, so incomplete means empty here.
    inventory: dict[str, str] = field(default_factory=dict)
    # The directory the inventory's relative paths are relative to.
    inventory_root: str = ""
    # For a manifest source, the row column those paths were read from; empty for a folder scan,
    # where the paths are the files themselves. A delta narrows a source by handing it these
    # paths, so it has to tell the source which column to match them against -- guessing would
    # silently select no rows and report the resulting no-op as a successful delta.
    inventory_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = _clean(asdict(self))
        # Deliberately not serialized: this rides along on every run record, report and plan
        # that carries a profile, and a corpus of any size would bury the fields a human reads.
        # It is persisted once, per artifact, by ``artifacts.save_coverage``.
        d.pop("inventory", None)
        d["dataset_key"] = self.dataset_key()
        d["fingerprint_tier"] = self.fingerprint_tier
        return d

    def fingerprint(self) -> str:
        """Stable id of the dataset's identifying shape (not its contents).

        Used to stamp data-derived recipe annotations (layered save): a change to
        this fingerprint means data-derived values (e.g. relative thresholds) must
        be recomputed rather than silently reused.
        """
        return _fingerprint(
            {
                "source": self.source,
                "kind": self.kind,
                "num_files": self.num_files,
                "sample_rates": {str(k): v for k, v in self.sample_rates.items()},
                "channels": {str(k): v for k, v in self.channels.items()},
                "mean_duration_sec": round(self.mean_duration_sec, 3),
                "has_transcripts": self.has_transcripts,
                "manifest_keys": sorted(self.manifest_keys),
            }
        )

    @property
    def fingerprint_tier(self) -> Literal["stat", "shape"]:
        """Which tier backs :meth:`dataset_key` -- and therefore how much to trust reuse.

        ``stat`` covers every local source file's size + mtime (and, for a manifest, its full
        content), so ordinary metadata-visible edits are caught. A mutation that deliberately
        restores both size and mtime still needs a future content-hash tier. ``shape`` is backed
        by an incomplete source identity when available, otherwise by the legacy sampled shape
        hash; either is low trust because metadata gaps can hide in-place edits.
        """
        return "stat" if self.stat_digest else "shape"

    def dataset_key(self) -> str:
        """Tiered source identity for reuse: ``"<tier>:<hash>"`` (strongest available)."""
        if self.stat_digest:
            return f"stat:{self.stat_digest}"
        return f"shape:{self.identity_digest or self.fingerprint()}"


@dataclass
class EnvProfile:
    """The env probe's structured read of the machine (deps / GPU / secrets)."""

    has_gpu: bool = False
    gpu_count: int = 0
    gpu_names: list[str] = field(default_factory=list)
    gpu_mem_gb: float = 0.0  # VRAM per GPU (GB); for the resource planner's GPU-fit math
    # Why ``has_gpu`` is false matters: no hardware, a hidden/unallocated device,
    # a broken driver, and a CPU-only torch build require different remedies.
    gpu_visibility: str = "unknown"
    nvidia_smi_status: str = "unknown"
    nvidia_smi_gpu_count: int = 0
    nvidia_device_nodes: int = 0
    cuda_visible_devices: str = "unset"  # presence only: unset / empty / masked / set
    torch_version: str = ""
    torch_cuda_built: bool = False
    cuda_probe_error: str = ""
    total_cpus: int = 0
    total_ram_gb: float = 0.0
    # ``None`` means the probe could not determine capacity; 0.0 means a genuinely
    # full filesystem. Keeping those states distinct prevents a full disk reading OK.
    free_disk_gb: float | None = None
    has_ffmpeg: bool = False
    installed_extras: list[str] = field(default_factory=list)
    missing_packages: list[str] = field(default_factory=list)
    available_secrets: list[str] = field(default_factory=list)
    curator_version: str = ""
    python_version: str = ""  # running interpreter, e.g. "3.13.1"
    python_supported: bool = True  # satisfies the project's requires-python (else a note is added)
    cuda_runtime_version: str = ""  # CUDA toolkit torch was built with, e.g. "12.9" (torch.version.cuda)
    cuda_driver_max_version: str = ""  # max CUDA the GPU driver supports, e.g. "12.6" (cuDriverGetVersion)
    cuda_compatible: bool = True  # driver's max CUDA >= torch's built CUDA (else JIT/CUDA-graph kernels may fail)
    # True when has_gpu is false but a GPU may actually be PRESENT and merely unreachable
    # from this process (a sandbox/container blocking /dev/nvidia*, or CUDA_VISIBLE_DEVICES
    # masking). A caller must NOT report "no GPU" as a hardware fact when this is set --
    # re-verify with full device access first.
    gpu_possibly_masked: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def gpu_status(self) -> str:
        """Canonical GPU class for gating -- the ONE source of truth shared by the
        deterministic gate check, the environment preflight, and the resource
        planner so they can never disagree about what "no GPU" means.

        - ``available``       : torch can use >=1 GPU in THIS process right now.
        - ``possibly_masked`` : torch cannot, but hardware/driver signals say a GPU is
          likely PRESENT and merely unreachable from this (sandboxed/containerized)
          process. NOT a hardware fact -- callers MUST re-verify with full device
          access, never hard-fail, and never tell the user "no GPU".
        - ``absent``          : definitively no usable GPU on this host. A CPU-only
          torch build cannot use a GPU regardless of hardware -- a real, actionable
          blocker (install the CUDA extra / move to a GPU host).
        - ``unknown``         : no visibility facts at all (torch missing, or a
          remote/adapter profile carrying no GPU info) -- treat as re-verify, not fail.

        Only a CPU-only torch build yields ``absent``; every other "torch can't see a
        device" case defers to re-verification, which is what stops a masked GPU from
        being reported as absent regardless of the user's machine or sandbox.
        """
        if self.has_gpu:
            return "available"
        if self.gpu_possibly_masked:
            return "possibly_masked"
        if self.gpu_visibility == "cpu_only_torch":
            return "absent"
        if self.gpu_visibility in {"torch_unavailable", "unknown", ""}:
            return "unknown"
        return "absent"

    def to_dict(self) -> dict[str, Any]:
        d = _clean(asdict(self))
        d["gpu_status"] = self.gpu_status
        return d

    def fingerprint(self) -> str:
        """Stable id of the machine's resource shape (layered save).

        A change means the machine plan (mode + per-stage resources) must be
        recomputed for the new hardware rather than reused from another machine.
        """
        return _fingerprint(
            {
                "gpu_count": self.gpu_count,
                "gpu_names": sorted(self.gpu_names),
                "gpu_mem_gb": round(self.gpu_mem_gb, 1),
                "gpu_visibility": self.gpu_visibility,
                "total_cpus": self.total_cpus,
                "total_ram_gb": round(self.total_ram_gb, 1),
                "has_gpu": self.has_gpu,
                "has_ffmpeg": self.has_ffmpeg,
                "installed_extras": sorted(self.installed_extras),
                "curator_version": self.curator_version,
            }
        )


@dataclass
class PlanningContext:
    """The compact, high-signal bundle handed to the host router/planner.

    Built deterministically by the Knowledge Index + Context Assembler; it never
    dumps source. ``category_tree`` (L0) lets the host prune before drilling into
    ``selected_stages`` (L2 full cards for finalists it picked).
    """

    goal: dict[str, Any] = field(default_factory=dict)
    category_tree: list[dict[str, Any]] = field(default_factory=list)
    selected_stages: list[dict[str, Any]] = field(default_factory=list)
    presets: dict[str, Any] = field(default_factory=dict)
    matched_blueprints: list[dict[str, Any]] = field(default_factory=list)
    matched_recipes: list[dict[str, Any]] = field(default_factory=list)
    patterns: list[dict[str, Any]] = field(default_factory=list)
    role_graph_slice: dict[str, Any] = field(default_factory=dict)
    data_profile: dict[str, Any] | None = None
    env_profile: dict[str, Any] | None = None
    # Machine-wide checks + grounded remediation options for the host LLM.
    # Recipe-specific applicability is resolved later by ``validate``/preflight.
    env_health: dict[str, Any] | None = None
    # Optional non-semantic workflow tie-breaker supplied by the host before
    # routing. It is never inferred from a folder or required by execution verbs.
    planning_preference: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class Issue:
    """A single problem surfaced by ``validate`` (mirrors the planner Verdict)."""

    code: str
    severity: Severity
    message: str
    stage_index: int | None = None
    stage: str | None = None
    fix: str | None = None
    # Set by a check that cannot decide (missing fact / ambiguous case) instead of
    # guessing pass or false-failing: hand off to "smoke" / "reviewer" / "user".
    escalate_to: Literal["smoke", "reviewer", "user"] | None = None

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class Verdict:
    """The output of ``validate``: is this recipe composable and runnable here?

    IMPORTANT — pick the right field to gate on:

    * ``ok`` / ``keys_ok`` are the *data-flow* necessary conditions ONLY (roles connect,
      then literal keys connect). They are **not** "safe to run": a recipe can be
      ``ok=True`` while a card constraint (e.g. ``task_type_mismatch``, batch > model max)
      or an environment gate makes it unrunnable.
    * ``runnable`` / ``status == "pass"`` are the deterministic mechanical
      prerequisite for execution. The host must also complete the advisory
      ``semantic_review`` and reach an intent-level ``pass`` before smoke or
      confirmation. Use ``ok``/``keys_ok`` only for data-flow diagnostics.

    ``card_violations`` and ``gate_flags`` carry the model-constraint and environment
    problems that ``ok`` deliberately ignores.
    """

    ok: bool = False  # data-flow only (roles connect); NOT safe-to-run -- gate on runnable/status
    keys_ok: bool = False
    issues: list[Issue] = field(default_factory=list)
    card_violations: list[Issue] = field(default_factory=list)
    gate_flags: list[Issue] = field(default_factory=list)
    unproducible_roles: list[str] = field(default_factory=list)
    produced_roles: list[str] = field(default_factory=list)
    produced_keys: list[str] = field(default_factory=list)
    # What is ALREADY at each location the recipe writes to. Stated as fact because an agent
    # that cannot see this guesses: in one test run it read the writer's append-mode open,
    # concluded reruns would accumulate, and deleted the user's file BEFORE the confirm gate.
    # The pipeline manages its own outputs, so the answer to an occupied path is to report it,
    # never to clear it.
    output_targets: list[dict[str, Any]] = field(default_factory=list)
    # Read-only proof of which configured source (if any) the profiler was bound to.
    # This is execution metadata, not a stage parameter, and never rewrites a recipe.
    data_binding: dict[str, Any] | None = None
    # Recipe-aware environment decision. This is additive to ``gate_flags``:
    # flags keep the stable validation contract, while this packet gives the host
    # evidence, viable alternatives, and the exact user decision it must request.
    environment_decision: dict[str, Any] | None = None
    diagnosis: dict[str, Any] | None = None
    # Recipe-aware grounding for the host LLM's semantic review.  Appended to
    # preserve existing positional construction. The deterministic core
    # assembles configured lineage/card facts but does not decide user intent.
    semantic_review: dict[str, Any] | None = None
    # Preference-specific, non-blocking authoring suggestions. These are kept
    # outside Issue pools so they cannot alter status or runnable.
    planning_advisories: list[dict[str, Any]] = field(default_factory=list)

    @property
    def runnable(self) -> bool:
        """True when there are no mechanically provable error-severity problems.

        This is a runnability result, not approval that the recipe expresses the
        user's intent.  The host must separately resolve ``semantic_review``.
        """
        pools = [self.issues, self.card_violations, self.gate_flags]
        return not any(i.severity == "error" for pool in pools for i in pool)

    @property
    def status(self) -> Literal["pass", "fail", "uncertain"]:
        """Mechanical tri-state: ``fail`` if any error; else ``uncertain`` if any check
        escalated (couldn't decide); else ``pass``. ``runnable`` remains the
        no-error boolean; ``status`` distinguishes "clean pass" from "needs a
        human/smoke/reviewer to resolve an unknown." It does not encode the host
        LLM's separate intent critique.
        """
        pools = [self.issues, self.card_violations, self.gate_flags]
        if any(i.severity == "error" for pool in pools for i in pool):
            return "fail"
        if any(i.escalate_to for pool in pools for i in pool):
            return "uncertain"
        return "pass"

    def summary(self) -> str:
        all_issues = [*self.issues, *self.card_violations, *self.gate_flags]
        errs = [i for i in all_issues if i.severity == "error"]
        warns = [i for i in all_issues if i.severity == "warning"]
        if not all_issues:
            return (
                "mechanically runnable (roles satisfied, keys connect, no "
                "card/gate problems); semantic intent review is still required"
            )
        if errs:
            heading = f"mechanical validation failed with {len(errs)} error(s) and {len(warns)} warning(s):"
        else:
            heading = f"mechanically runnable with {len(warns)} warning(s); semantic intent review is still required:"
        lines = [heading]
        for i in all_issues:
            loc = f" [{i.stage}]" if i.stage else ""
            fix = f" -> {i.fix}" if i.fix else ""
            lines.append(f"  [{i.severity}] {i.code}{loc}: {i.message}{fix}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "keys_ok": self.keys_ok,
            "runnable": self.runnable,
            "status": self.status,
            "issues": [i.to_dict() for i in self.issues],
            "card_violations": [i.to_dict() for i in self.card_violations],
            "gate_flags": [i.to_dict() for i in self.gate_flags],
            "unproducible_roles": sorted(self.unproducible_roles),
            "produced_roles": sorted(self.produced_roles),
            "produced_keys": sorted(self.produced_keys),
            "output_targets": list(self.output_targets),
            "data_binding": _clean(self.data_binding),
            "environment_decision": _clean(self.environment_decision),
            "validation_scope": "mechanical_runnability_not_intent_approval",
            "semantic_review": _clean(self.semantic_review),
            "planning_advisories": _clean(self.planning_advisories),
            "diagnosis": _clean(self.diagnosis),
            "summary": self.summary(),
        }


@dataclass
class SmokeReport:
    """The output of ``smoke``: bounded execution on a small sample."""

    ran: bool = False
    sample: int = 0
    input_count: int = 0
    retained: int = 0
    rejected: int = 0
    errors: list[str] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)
    per_stage_metrics: dict[str, Any] = field(default_factory=dict)
    goals_met: bool | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class PlanResult:
    """A frozen, validated plan handed to the confirm-gate + full run.

    Emitted by the (host-driven) Finalizer/Controller once the loops converge,
    or an ``escalate`` / ``refused`` decision.
    """

    status: Literal["finalized", "escalate", "refused"] = "escalate"
    recipe: dict[str, Any] | None = None
    rationale: str = ""
    confidence: float | None = None
    blocking_issues: list[Issue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


# --------------------------------------------------------------------------- #
# acceptance layer (1A.1): the "did we solve the user's problem?" contract
# --------------------------------------------------------------------------- #
CriterionType = Literal["quality_standard", "output_completeness", "yield", "distribution", "honesty", "semantic_fit"]
CriterionStatus = Literal["met", "not_met", "unverifiable", "unachievable"]
CriterionKind = Literal["absolute", "relative", "operational"]

_CRITERION_TYPES = frozenset(
    {"quality_standard", "output_completeness", "yield", "distribution", "honesty", "semantic_fit"}
)
_CRITERION_KINDS = frozenset({"absolute", "relative", "operational"})
_CRITERION_SEVERITIES = frozenset({"must", "nice"})
_UNACHIEVABLE_POLICIES = frozenset({"escalate", "relax_with_confirmation"})
_CHECK_FIELDS = frozenset({"scope", "field", "op", "value", "tolerance", "method"})
_CHECK_SCOPES = frozenset({"aggregate", "per_item", "per_retained_item"})
_CHECK_OPERATORS = frozenset({">=", "<=", "==", "!=", ">", "<", "~=", "non_empty"})
_CHECK_METHODS = frozenset({"deterministic", "reviewer_judgment"})
_CRITERION_FIELDS = frozenset(
    {
        "id",
        "type",
        "description",
        "kind",
        "check",
        "compiles_to",
        "source",
        "severity",
        "on_unachievable",
    }
)


def _is_criterion_number(value: Any) -> bool:  # noqa: ANN401
    """True only for finite JSON-style numbers (booleans are not thresholds)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _invalid_criterion(message: str) -> NoReturn:
    """Raise the public schema error used by every acceptance-contract entry point."""
    raise ValueError(message)


def _validated_acceptance_mapping(raw: Any) -> dict[str, Any]:  # noqa: ANN401, C901, PLR0912, PLR0915
    """Validate one host-authored success criterion without silently coercing it."""
    if isinstance(raw, AcceptanceCriterion):
        raw = asdict(raw)
    if not isinstance(raw, Mapping):
        msg = f"acceptance criterion must be a mapping, got {type(raw).__name__}"
        _invalid_criterion(msg)
    data = dict(raw)
    unknown = sorted((key for key in data if key not in _CRITERION_FIELDS), key=repr)
    if unknown:
        msg = f"acceptance criterion has unknown field(s): {unknown}"
        _invalid_criterion(msg)

    criterion_id = data.get("id")
    if not isinstance(criterion_id, str) or not criterion_id.strip():
        _invalid_criterion("acceptance criterion id must be a non-empty string")
    if criterion_id != criterion_id.strip():
        _invalid_criterion(f"acceptance criterion id {criterion_id!r} must not have surrounding whitespace")

    criterion_type = data.get("type")
    if criterion_type not in _CRITERION_TYPES:
        _invalid_criterion(
            f"acceptance criterion {criterion_id!r} has invalid type {criterion_type!r}; "
            f"expected one of {sorted(_CRITERION_TYPES)}"
        )

    description = data.get("description", "")
    if not isinstance(description, str):
        _invalid_criterion(f"acceptance criterion {criterion_id!r} description must be a string")

    kind = data.get("kind")
    if kind is not None and kind not in _CRITERION_KINDS:
        _invalid_criterion(
            f"acceptance criterion {criterion_id!r} has invalid kind {kind!r}; "
            f"expected one of {sorted(_CRITERION_KINDS)}"
        )

    severity = data.get("severity", "must")
    if severity not in _CRITERION_SEVERITIES:
        _invalid_criterion(
            f"acceptance criterion {criterion_id!r} has invalid severity {severity!r}; "
            f"expected one of {sorted(_CRITERION_SEVERITIES)}"
        )

    on_unachievable = data.get("on_unachievable", "escalate")
    if on_unachievable not in _UNACHIEVABLE_POLICIES:
        _invalid_criterion(
            f"acceptance criterion {criterion_id!r} has invalid on_unachievable "
            f"{on_unachievable!r}; expected one of {sorted(_UNACHIEVABLE_POLICIES)}"
        )

    compiles_to = data.get("compiles_to")
    if compiles_to is not None and (
        not isinstance(compiles_to, str) or not compiles_to.strip() or compiles_to != compiles_to.strip()
    ):
        _invalid_criterion(
            f"acceptance criterion {criterion_id!r} compiles_to must be a non-empty string "
            "without surrounding whitespace"
        )

    source = data.get("source", {})
    if source is None:
        source = {}
    elif not isinstance(source, Mapping):
        _invalid_criterion(f"acceptance criterion {criterion_id!r} source must be a mapping")

    check = data.get("check", {})
    if check is None:
        check = {}
    elif not isinstance(check, Mapping):
        _invalid_criterion(f"acceptance criterion {criterion_id!r} check must be a mapping")
    check = dict(check)
    unknown_check = sorted((key for key in check if key not in _CHECK_FIELDS), key=repr)
    if unknown_check:
        _invalid_criterion(f"acceptance criterion {criterion_id!r} has unknown check field(s): {unknown_check}")

    field_name = check.get("field")
    if field_name is not None and (
        not isinstance(field_name, str) or not field_name.strip() or field_name != field_name.strip()
    ):
        _invalid_criterion(
            f"acceptance criterion {criterion_id!r} check.field must be a non-empty string "
            "without surrounding whitespace"
        )

    scope = check.get("scope")
    if scope is not None and scope not in _CHECK_SCOPES:
        _invalid_criterion(
            f"acceptance criterion {criterion_id!r} has invalid check.scope {scope!r}; "
            f"expected one of {sorted(_CHECK_SCOPES)}"
        )
    # Historical contracts used ``per_item`` while the verifier's documented name is
    # ``per_retained_item``. Normalize only the parsed runtime object; Recipe keeps the
    # host-authored mapping unchanged, so config/contract hashes remain stable.
    if scope == "per_item":
        check["scope"] = "per_retained_item"
        scope = "per_retained_item"

    method = check.get("method")
    if method is not None and method not in _CHECK_METHODS:
        _invalid_criterion(
            f"acceptance criterion {criterion_id!r} has invalid check.method {method!r}; "
            f"expected one of {sorted(_CHECK_METHODS)}"
        )

    op = check.get("op")
    if op is not None and op not in _CHECK_OPERATORS:
        _invalid_criterion(
            f"acceptance criterion {criterion_id!r} has invalid check operator {op!r}; "
            f"expected one of {sorted(_CHECK_OPERATORS)}"
        )

    if "tolerance" in check:
        tolerance = check["tolerance"]
        if not _is_criterion_number(tolerance) or tolerance < 0:
            _invalid_criterion(f"acceptance criterion {criterion_id!r} check.tolerance must be a non-negative number")
        if op != "~=":
            _invalid_criterion(
                f"acceptance criterion {criterion_id!r} check.tolerance is only valid with check.op='~='"
            )

    if criterion_type == "output_completeness":
        usable_target = compiles_to if compiles_to != "producible_role" else None
        if not field_name and not usable_target:
            _invalid_criterion(
                f"acceptance criterion {criterion_id!r} output_completeness needs a target "
                "in check.field or compiles_to"
            )
        if op is not None and op != "non_empty":
            _invalid_criterion(
                f"acceptance criterion {criterion_id!r} output_completeness only supports check.op='non_empty'"
            )
        unsupported = sorted(set(check) - {"field", "op", "scope"})
        if unsupported:
            _invalid_criterion(
                f"acceptance criterion {criterion_id!r} output_completeness does not support "
                f"check field(s): {unsupported}"
            )
        if scope == "aggregate":
            _invalid_criterion(
                f"acceptance criterion {criterion_id!r} output_completeness scope must be "
                "'per_item' or 'per_retained_item'"
            )
    elif criterion_type in {"quality_standard", "distribution"}:
        if method != "reviewer_judgment":
            if not field_name:
                _invalid_criterion(f"acceptance criterion {criterion_id!r} needs check.field")
            if op is None or op == "non_empty":
                _invalid_criterion(f"acceptance criterion {criterion_id!r} needs a numeric check operator")
            if "value" not in check:
                _invalid_criterion(f"acceptance criterion {criterion_id!r} needs check.value")
            if not _is_criterion_number(check["value"]):
                _invalid_criterion(f"acceptance criterion {criterion_id!r} check.value must be numeric")
    elif criterion_type == "yield":
        if compiles_to is not None:
            _invalid_criterion(f"acceptance criterion {criterion_id!r} yield does not support compiles_to")
        if field_name is not None and field_name != "retained":
            _invalid_criterion(
                f"acceptance criterion {criterion_id!r} yield check.field must be 'retained' when supplied"
            )
        unsupported = sorted(set(check) - {"field", "op", "value", "tolerance"})
        if unsupported:
            _invalid_criterion(
                f"acceptance criterion {criterion_id!r} yield does not support check field(s): {unsupported}"
            )
        if op is None or op == "non_empty":
            _invalid_criterion(f"acceptance criterion {criterion_id!r} needs a numeric check operator")
        if "value" not in check:
            _invalid_criterion(f"acceptance criterion {criterion_id!r} needs check.value")
        if not _is_criterion_number(check["value"]):
            _invalid_criterion(f"acceptance criterion {criterion_id!r} check.value must be numeric")
    elif criterion_type in {"semantic_fit", "honesty"}:
        if compiles_to is not None:
            _invalid_criterion(
                f"acceptance criterion {criterion_id!r} type {criterion_type!r} does not support compiles_to"
            )
        unsupported = sorted(set(check) - {"field", "method"})
        if unsupported:
            _invalid_criterion(
                f"acceptance criterion {criterion_id!r} type {criterion_type!r} "
                f"does not support check field(s): {unsupported}"
            )
        if method == "deterministic":
            _invalid_criterion(
                f"acceptance criterion {criterion_id!r} type {criterion_type!r} "
                "must use method='reviewer_judgment' when a method is supplied"
            )

    return {
        "id": criterion_id,
        "type": criterion_type,
        "description": description,
        "kind": kind,
        "check": check,
        "compiles_to": compiles_to,
        "source": dict(source),
        "severity": severity,
        "on_unachievable": on_unachievable,
    }


@dataclass
class AcceptanceCriterion:
    """One checkable condition of success (1A §5.3).

    Host-derived from intent, confirmed at the gate, then verified against
    evidence. Generic and metric-agnostic: ``check.field`` is any metric/output
    key; the verifier contains no metric names. ``type`` routes it to the cheapest
    sufficient owner (deterministic vs reviewer).
    """

    id: str
    type: str  # CriterionType
    description: str = ""
    kind: CriterionKind | None = None
    check: dict[str, Any] = field(default_factory=dict)  # scope/field/op/value/tolerance/method
    compiles_to: str | None = None  # e.g. a producible-role name (output_completeness)
    source: dict[str, Any] = field(default_factory=dict)
    severity: Literal["must", "nice"] = "must"
    on_unachievable: Literal["escalate", "relax_with_confirmation"] = "escalate"

    @classmethod
    def from_dict(cls, d: Any) -> AcceptanceCriterion:  # noqa: ANN401
        return cls(**_validated_acceptance_mapping(d))

    @property
    def field_name(self) -> str | None:
        f = self.check.get("field")
        return str(f) if f else None

    @property
    def is_deterministic(self) -> bool:
        """True unless this criterion needs the reviewer (semantic / reviewer_judgment)."""
        return self.type != "semantic_fit" and self.check.get("method") != "reviewer_judgment"

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class CriterionResult:
    """One criterion's verification outcome (four honest states)."""

    id: str
    status: str  # CriterionStatus
    severity: str = "must"
    evidence: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class AcceptanceReport:
    """Per-criterion verification of the success contract (1A §8).

    ``overall`` is ``met`` iff every ``must`` criterion is ``met`` — the
    anti-goalpost-moving gate. ``not_met``/``unverifiable``/``unachievable`` are
    reported honestly, never silently relaxed.
    """

    overall: Literal["met", "not_met", "unverifiable"] = "not_met"
    criteria: list[CriterionResult] = field(default_factory=list)
    verdict: str = ""
    # Honesty meta-check (1A.3): goalpost-moving violations found by comparing the
    # criteria actually verified against the confirmed (frozen) contract. Non-empty
    # forces overall=not_met (a relaxed 'must' bar cannot be declared met).
    honesty: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "criteria": [c.to_dict() for c in self.criteria],
            "verdict": self.verdict,
            "honesty": list(self.honesty),
        }


@dataclass
class ConfigStrategyEntry:
    """How one parameter's value was chosen (1A §6.4) — the auditable record.

    ``recompute_on`` drives layered save: ``none`` (absolute/explicit) is portable
    and travels with the recipe; ``data_change`` (relative/data-derived) and
    ``machine_change`` (operational) are recomputed on re-run rather than reused.
    """

    param: str
    value: Any = None
    metric: str | None = None
    kind: str = "absolute"  # absolute | relative | operational
    mode: str = "knowledge_driven"  # knowledge_driven | data_informed
    source: dict[str, Any] = field(default_factory=dict)  # {from: user_explicit|card_anchor|card_preset|..., ref: ...}
    recompute_on: str = "none"  # none | data_change | machine_change
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class RunRecord:
    """Local, per-run provenance for tracing + incremental continuation.

    A durable trace of one run — the frozen recipe + ``config_hash``, the success
    contract, evidence counts, output paths, the data fingerprint, the step-key
    chain, and the parent link — so a follow-up request can reuse prior work and
    every result is traceable.

    Scope of the "no cross-session memory" non-goal: records support **deterministic
    memoization** (content-addressed "has this exact computation already been done?",
    see ``REUSE_ARCHITECTURE.md``) and provenance. They are **not** learned priors,
    not cross-user/shared memory, and are never fed back to influence *what* the
    agent plans—reuse only skips work whose computation and resolved dataset
    identities match at the recorded trust tier, and never without the user's
    approval.
    """

    run_id: str
    recipe: dict[str, Any] = field(default_factory=dict)  # frozen Recipe.to_dict()
    config_hash: str | None = None
    parent_run_id: str | None = None
    goal: dict[str, Any] = field(default_factory=dict)
    # Brief deterministic one-liner of stages + identifying params, written on success so a
    # later session can compare the user's new request to "what this run did" without loading
    # every param. Empty on older records -- readers derive it from ``recipe`` when missing.
    pipeline_summary: str = ""
    data_source: str | None = None
    data_fingerprint: str | None = None
    acceptance_criteria: list[dict[str, Any]] = field(default_factory=list)
    status: str = ""
    accepted: int = 0
    input_count: int = 0
    output_paths: list[str] = field(default_factory=list)
    created_at: str = ""
    notes: list[str] = field(default_factory=list)
    # --- reuse / cost provenance (REUSE_ARCHITECTURE.md) --------------------- #
    semantic_hash: str | None = None  # identity that ignores execution knobs + output location
    contract_hash: str | None = None  # acceptance_criteria only (re-verified, never recomputed)
    dataset_key: str | None = None  # tiered "<tier>:<hash>" source key
    fingerprint_tier: str = ""  # which tier produced dataset_key (stat | shape)
    steps: list[str] = field(default_factory=list)  # per-stage step_key chain (Merkle)
    elapsed_sec: float = 0.0
    per_stage_metrics: dict[str, Any] = field(default_factory=dict)
    acceptance_result: dict[str, Any] = field(default_factory=dict)  # verification OUTCOME, not the criteria
    env_summary: dict[str, Any] = field(default_factory=dict)
    curator_version: str = ""
    knowledge_version: str | None = None
    reuse: dict[str, Any] = field(default_factory=dict)  # what was reused vs run, for lineage

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunRecord:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))
