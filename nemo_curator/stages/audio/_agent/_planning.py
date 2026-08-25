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

"""Pipeline-level validation for agent-composed audio pipelines.

``validate_pipeline([stageA, stageB, ...])`` walks an ordered list of configured
stages and checks they actually compose — each stage's required inputs must be
produced by an upstream stage or be present in the initial task. It also surfaces
resource-gate problems (GPU needed but none available).

A read is satisfied by matching *either* the literal key it names or the semantic
*role* behind it. Both routes are needed: role matching tolerates a producer that
writes ``resampled_audio_filepath`` where the consumer reads ``audio_filepath``,
and key matching covers the reverse, where the names agree but the two sides file
that name under different roles. Requiring both would report breaks in pipelines
that run.

Composites are expanded (see :mod:`nemo_curator.stages.audio._agent._composite`) so the
stages inside them are checked too — the requirements of the stages that do the
work, rather than the empty contract the composite advertises. Reads that fail
inside a composite are warnings, not errors, until the expansion has proven it
does not false-positive. A composite that cannot be expanded, or whose children
include something with no contract at all, falls back to being treated as opaque:
it is reported, and reads after it are no longer judged by role.

Two levels of confidence, deliberately separated:

* ``report.ok`` certifies a *role-level necessary condition* — every required
  input role is available. This is rename-tolerant by design and is the gate.
* ``report.keys_ok`` adds the stronger *literal-key-identity* check: each
  role-satisfied read's actual key *value* is produced upstream (or seeded). A
  ``True`` ``ok`` with ``False`` ``keys_ok`` means the roles line up but a
  producer key was renamed away from what the consumer reads — the pipeline
  would validate yet yield zero rows at runtime. It is surfaced as a WARNING
  (not an error) so that legitimate reads of source-manifest columns are not
  false-rejected.

This is advisory and read-only — it never executes a stage. ``ok`` is a
necessary, not sufficient, condition for a pipeline to run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from nemo_curator.stages.audio._agent._agent_registry import build_contract
from nemo_curator.stages.audio._agent._composite import expand_composites
from nemo_curator.stages.audio._agent._conformance import produced_roles, reads_satisfied_by_role
from nemo_curator.stages.audio._agent._roles import role_for_value

if TYPE_CHECKING:
    from nemo_curator.stages.audio._agent._agent_ready import StageContract

Severity = Literal["error", "warning"]

# Roles a typical audio task carries at the start (a manifest row with a file path).
_DEFAULT_INITIAL_ROLES: frozenset[str] = frozenset({"audio_filepath"})
# Key VALUES a typical audio task carries at the start (the conventional path key).
_DEFAULT_INITIAL_KEYS: frozenset[str] = frozenset({"audio_filepath"})

# The role of the key a tensor producer parks its waveform under. Tracking the carrier
# rather than a bare "a tensor is resident" flag is what lets a stage that DROPS that key
# end the residency, instead of only a stage flagged ``sanitizes_output``.
_TENSOR_ROLE = "waveform"
# Stand-in for a producer that declares ``produces=["tensor"]`` without naming a
# waveform-roled key. Its residency is still tracked, but no key removal can match it, so
# only an explicit sanitizer clears it -- deliberately the pre-existing behaviour.
_UNNAMED_TENSOR = "<unnamed tensor>"


@dataclass(frozen=True)
class PipelineIssue:
    """A single problem found while validating a pipeline."""

    stage_index: int
    stage_name: str
    severity: Severity
    code: str
    message: str


@dataclass(frozen=True)
class PipelineReport:
    """Result of :func:`validate_pipeline`."""

    issues: list[PipelineIssue] = field(default_factory=list)
    produced_roles: set[str] = field(default_factory=set)  # roles available after the last stage
    produced_keys: set[str] = field(default_factory=set)  # key VALUES available after the last stage

    @property
    def ok(self) -> bool:
        """True when there are no error-severity issues (role-level composability).

        This is a *necessary* condition, not a guarantee the pipeline runs — see
        :attr:`keys_ok` for the stronger literal-key check.
        """
        return not any(i.severity == "error" for i in self.issues)

    @property
    def keys_ok(self) -> bool:
        """True when no ``dangling_key`` warnings — every role-satisfied read's
        actual key *value* is produced upstream or seeded. ``ok and keys_ok`` is
        the strong signal that the pipeline will actually flow data end-to-end.
        """
        return not any(i.code == "dangling_key" for i in self.issues)

    @property
    def errors(self) -> list[PipelineIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[PipelineIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def summary(self) -> str:
        if not self.issues:
            return (
                "pipeline mechanically composable (all reads satisfied by role); "
                "this does not certify intent or field meaning"
            )
        lines = [f"{len(self.errors)} error(s), {len(self.warnings)} warning(s):"]
        for i in self.issues:
            lines.append(f"  [{i.severity}] stage {i.stage_index} {i.stage_name}: {i.message}")
        return "\n".join(lines)


def _required_roles(contract: StageContract) -> set[str]:
    keys = [*contract.reads.data_keys, *contract.reads.segment_data_keys]
    return {contract.key_roles.get(k, "unknown") for k in keys}


def _requirement_str(contract: StageContract, available: set[str]) -> str:
    """Human-readable "what this stage needs" for an unsatisfied-reads message.

    Renders top-level ``reads`` (all required) and ``reads_one_of`` (any one), so a
    stage whose reads live entirely in ``reads_one_of`` (e.g. a residency-derived
    contract) no longer renders a misleading empty ``role(s) []``.
    """
    missing = _required_roles(contract) - (available | {"unknown"})
    reqs: list[str] = []
    if missing:
        reqs.append(f"role(s) {sorted(missing)}")
    if contract.reads_one_of:
        reqs.append(f"one of {[sorted(_roles_of(o, contract)) for o in contract.reads_one_of]}")
    return "; ".join(reqs) or f"role(s) {sorted(_required_roles(contract))}"


def _write_key_values(contract: StageContract) -> set[str]:
    """The literal key VALUES a stage writes (top-level + segment-level)."""
    return {*contract.writes.data_keys, *contract.writes.segment_data_keys}


def _key_family(key: str) -> str:
    """The trailing token of a key name -- ``diar_segments`` and ``segments`` share ``segments``.

    A crude but load-bearing notion of "these two keys hold the same KIND of thing". Producers
    qualify a shared noun with a prefix (``diar_``, ``vad_``, ``pred_``), so the bare noun and
    its qualified siblings are exactly the set a consumer might have meant.
    """
    return key.rsplit("_", 1)[-1]


def _ambiguous_default_reads(
    stage: Any,  # noqa: ANN401 - any built stage
    contract: StageContract,
    available_keys: set[str],
    key_producer: dict[str, str],
) -> list[tuple[str, str, list[tuple[str, str]]]]:
    """``(key, attribute, rivals)`` for read keys left at a default while a sibling key exists.

    The failure this catches is silence. ``MergeAlignmentDiarizationStage`` documents itself as
    merging into DIARIZATION segments, yet its ``segments_key`` defaults to ``"segments"`` --
    the key VAD writes. In a VAD+diarization pipeline both keys exist, so the read is satisfied
    and every other check passes: transcripts get merged into the wrong segments and the output
    is plausible, complete, and wrong.

    Deliberately narrow, because a warning nobody trusts is worse than none. It fires only when
    the key is still at its CLASS DEFAULT (an explicit setting is a decision, not an accident),
    the key IS available (an unavailable one is already reported as dangling), and some other
    available key of the same family was written by a DIFFERENT upstream stage -- so a real
    choice existed and was made by a default rather than by anyone.
    """
    fields = getattr(type(stage), "__dataclass_fields__", {})
    reads = {*contract.reads.data_keys, *contract.reads.segment_data_keys}
    found: list[tuple[str, str, list[tuple[str, str]]]] = []
    for attr, spec in fields.items():
        value = getattr(stage, attr, None)
        if not (isinstance(value, str) and value in reads and value in available_keys and value == spec.default):
            continue
        rivals = sorted(
            (k, key_producer[k])
            for k in available_keys
            if k != value
            # If the consumer reads both siblings (for example reference
            # ``text`` and ASR ``pred_text`` for WER), they are independent
            # operands rather than competing choices.
            and k not in reads
            and k in key_producer
            and _key_family(k) == _key_family(value)
            and key_producer[k] != key_producer.get(value)
        )
        if rivals:
            found.append((value, attr, rivals))
    return found


@dataclass(frozen=True)
class _Site:
    """The stage being checked right now, and how to name it to whoever wrote the recipe."""

    index: int
    """Index of the RECIPE stage, so an inner stage points at something the caller can edit."""

    name: str
    stage: Any
    composite: Any | None = None
    """The recipe-level composite this was expanded from, when it is not a stage in its own right."""


def _gate_issues(
    site: _Site,
    contract: StageContract,
    available_gpus: float | None,
    *,
    tensor_resident: bool,
) -> list[PipelineIssue]:
    """Environment/serialization gate problems for one stage.

    These reason about GPUs and serialization rather than roles, so they apply to every concrete
    stage even downstream of a composite that hides its writes.
    """
    out = []
    if available_gpus is not None and contract.gates.requires_gpu and available_gpus <= 0:
        out.append(
            PipelineIssue(
                site.index,
                site.name,
                "warning",
                "gpu_unavailable",
                "declares requires_gpu but available_gpus <= 0",
            )
        )
    # A resident tensor (e.g. a waveform) reaching a serialize-as-is sink (raw json.dumps)
    # crashes at runtime. A sanitizing stage upstream clears the flag before we get here.
    if contract.gates.requires_serializable_input and tensor_resident:
        out.append(
            PipelineIssue(
                site.index,
                site.name,
                "error",
                "tensor_into_sink",
                "a resident tensor/audio blob from an upstream stage reaches this "
                "serialize-as-JSON sink; it WILL fail at json.dumps — drop the tensor "
                "upstream (e.g. keep_segment_waveform_in_task=False) or route through "
                "a sanitizing stage before the sink",
            )
        )
    return out


def _ambiguity_issues(
    site: _Site,
    contract: StageContract,
    available_keys: set[str],
    key_producer: dict[str, str],
) -> list[PipelineIssue]:
    """``ambiguous_default_key`` warnings for this stage, naming who wrote each candidate."""
    out = []
    for key, attr, rivals in _ambiguous_default_reads(site.stage, contract, available_keys, key_producer):
        others = ", ".join(f"{k!r} from {p}" for k, p in rivals)
        out.append(
            PipelineIssue(
                site.index,
                site.name,
                "warning",
                "ambiguous_default_key",
                f"reads {key!r} (the default for {attr}), but upstream also produced {others}. "
                f"The default silently picks {key!r} "
                f"({key_producer.get(key, 'the source manifest')}); if you meant the other, "
                f"set {attr} explicitly.",
            )
        )
    return out


def _missing_read_keys(contract: StageContract, available_keys: set[str]) -> set[str]:
    """Read key VALUES this stage wants that nothing upstream produced or seeded."""
    reads = {*contract.reads.data_keys, *contract.reads.segment_data_keys}
    return {k for k in reads if k not in available_keys}


def _reads_satisfied_by_key(contract: StageContract, available_keys: set[str]) -> bool:
    """Whether every read is met by the LITERAL key it names.

    A stage reads ``task.data[self.segments_key]`` at runtime -- a key string, never a role. So
    a diarizer that writes ``diar_segments`` does satisfy a consumer configured to read
    ``diar_segments``, even though the producer registers that key under the role
    ``diar_segments`` while the consumer's contract calls the same slot ``segments``. Judging
    that pairing only by role reports a break in a pipeline that runs, and the caller's options
    are then to distrust the validator or to rename a key to appease it -- both worse than the
    check not existing.

    Role matching stays as the rename-tolerant fallback for the opposite case, where the key
    names differ but mean the same thing (a producer writing ``resampled_audio_filepath``
    satisfying a consumer reading ``audio_filepath``). A read is satisfied by either route.
    """
    if {*contract.reads.data_keys, *contract.reads.segment_data_keys} - available_keys:
        return False
    if not contract.reads_one_of:
        return True
    return any(not ({*spec.data_keys, *spec.segment_data_keys} - available_keys) for spec in contract.reads_one_of)


def _forwarding_param(inner: Any, composite: Any, missing: set[str]) -> str | None:  # noqa: ANN401
    """The composite parameter to set so an inner stage stops reading the wrong key.

    A caller who configured ``SplitASRAlignJoinStage`` has never heard of ``SplitLongAudioStage``
    and cannot configure it directly, so naming the inner stage alone leaves them stuck. The
    remedy is always a parameter the composite forwards down, and it is identifiable rather than
    guessable: the attribute exists on both classes and its current value IS the key that went
    missing. Returns ``None`` when no such parameter exists, in which case the inner stage's
    requirement genuinely cannot be reached from the recipe.
    """
    inner_fields = getattr(type(inner), "__dataclass_fields__", {})
    composite_fields = getattr(type(composite), "__dataclass_fields__", {})
    for attr in inner_fields:
        if attr not in composite_fields:
            continue
        value = getattr(inner, attr, None)
        # ``missing`` holds key names, so only a string can ever match -- and testing anything
        # else against a set hashes it, which raises TypeError on the list and dict parameters
        # real stages carry (``file_extensions``, ``storage_options``). That exception escapes
        # ``run_checks`` and kills the whole verb, so the recipe gets a traceback instead of a
        # verdict over a remedy hint that was never going to apply.
        if isinstance(value, str) and value in missing:
            return attr
    return None


def _unreadable_child(group: list[Any]) -> str | None:
    """Why this composite's expansion cannot be reasoned about, or ``None`` if it can.

    A composite is only as legible as its least legible child. Composites routinely contain
    plumbing that was never annotated for the agent -- ``ManifestReader`` expands through
    ``FilePartitioningStage``, which has no ``describe()`` at all -- and a child whose reads and
    writes are unknown leaves a hole in the role bookkeeping that makes every later stage's
    verdict unsound. Blaming the caller for that with a hard error would fail pipelines that run
    perfectly well, on the name of a stage they never wrote. So the composite reverts to being
    opaque, which is exactly how it was treated before it could be expanded at all.
    """
    for item in group:
        if not item.composite_ref:
            continue  # a top-level stage that cannot describe itself is the caller's own error
        try:
            build_contract(item.stage)
        except Exception as e:  # noqa: BLE001 - any failure to describe means the same thing here
            return f"{type(item.stage).__name__} does not describe its I/O ({type(e).__name__})"
    return None


def _describes_itself(stage: Any) -> bool:  # noqa: ANN401 - any child stage
    """Whether a contract can be built for this stage at all."""
    try:
        build_contract(stage)
    except Exception:  # noqa: BLE001 - any failure to describe means the same thing here
        return False
    return True


def _dangling_read_keys(contract: StageContract, available_keys: set[str]) -> set[str]:
    """Read key VALUES whose role is known but whose exact value was not
    produced upstream nor seeded — the renamed-producer dangle the role check misses.

    Covers primary ``reads`` plus a ``reads_one_of`` that offers a *single*
    alternative: one option is not a choice, so its keys are as mandatory as a
    primary read (this is how a residency-derived contract expresses
    ``input_residency="file"``). A genuine multi-way ``reads_one_of`` is skipped —
    the stage may legitimately take the other branch. Role-bearing keys only
    (``unknown``/internal bookkeeping keys are excluded — a separate
    value-identity check for those is tracked in the backlog).
    """
    reads = [*contract.reads.data_keys, *contract.reads.segment_data_keys]
    if len(contract.reads_one_of) == 1:
        only = contract.reads_one_of[0]
        reads += [*only.data_keys, *only.segment_data_keys]
    dangling: set[str] = set()
    for k in reads:
        role = contract.key_roles.get(k, "unknown")
        if role == "unknown":
            continue
        if k not in available_keys:
            dangling.add(k)
    return dangling


@dataclass
class _Walk:
    """What the pipeline carries from one stage to the next while being validated."""

    available: set[str]  # roles produced so far
    available_keys: set[str]  # literal key VALUES produced so far
    tensor_keys: set[str] = field(default_factory=set)
    removed_roles: set[str] = field(default_factory=set)
    key_producer: dict[str, str] = field(default_factory=dict)
    past_composite: bool = False  # an UNEXPANDABLE composite hid its writes; reads past it can't be judged


def _read_issues(walk: _Walk, site: _Site, contract: StageContract) -> list[PipelineIssue]:
    """Whether this stage's reads are met, and how loudly to say so if not.

    Severity is graded by how sure we are, because a wrong hard error is worse than a wrong
    warning: it stops the caller with no recourse and invites them to fake a value to get past
    the gate rather than fix anything. A read that fails on a stage the caller wrote is certain,
    so it is an error. A read that fails inside an expanded composite is reported as a warning
    for now -- the expansion is new, and it earns the right to block only once it has been shown
    not to false-positive on pipelines known to work.
    """
    if reads_satisfied_by_role(contract, walk.available) or _reads_satisfied_by_key(contract, walk.available_keys):
        if walk.past_composite:
            return []
        out: list[PipelineIssue] = []
        dangling = _dangling_read_keys(contract, walk.available_keys)
        if dangling:
            out.append(
                PipelineIssue(
                    site.index,
                    site.name,
                    "warning",
                    "dangling_key",
                    f"reads key(s) {sorted(dangling)} satisfied by role but not produced "
                    f"upstream under that key value nor seeded (renamed producer key?); "
                    f"available keys: {sorted(walk.available_keys)}",
                )
            )
        out.extend(_ambiguity_issues(site, contract, walk.available_keys, walk.key_producer))
        return out

    if site.composite is not None:
        composite_name = type(site.composite).__name__
        missing = _missing_read_keys(contract, walk.available_keys)
        param = _forwarding_param(site.stage, site.composite, missing)
        remedy = (
            f"set {param} on {composite_name} (it forwards the value to this inner stage)"
            if param
            else "produce the missing key upstream"
        )
        return [
            PipelineIssue(
                site.index,
                site.name,
                "warning",
                "unsatisfied_reads_in_composite",
                f"this stage runs inside {composite_name} and requires "
                f"{_requirement_str(contract, walk.available)}"
                + (f" (key(s) {sorted(missing)})" if missing else "")
                + f", not produced upstream; {remedy}. Available keys: {sorted(walk.available_keys)}",
            )
        ]

    if walk.past_composite:
        return [
            PipelineIssue(
                site.index,
                site.name,
                "warning",
                "unsatisfied_reads_after_composite",
                f"requires {_requirement_str(contract, walk.available)} "
                f"not visibly produced — but an upstream composite hides its writes; "
                f"decompose it to validate this read",
            )
        ]

    needed = _required_roles(contract) | {r for o in contract.reads_one_of for r in _roles_of(o, contract)}
    removed_hit = (needed & walk.removed_roles) - walk.available
    if removed_hit:
        return [
            PipelineIssue(
                site.index,
                site.name,
                "error",
                "key_removed_upstream",
                f"reads role(s) {sorted(removed_hit)} that an upstream stage removed "
                f"(removes_keys) and no stage re-produced; available so far: {sorted(walk.available)}",
            )
        ]
    return [
        PipelineIssue(
            site.index,
            site.name,
            "error",
            "unsatisfied_reads",
            f"requires {_requirement_str(contract, walk.available)} "
            f"not produced upstream; available so far: {sorted(walk.available)}",
        )
    ]


def _advance(walk: _Walk, contract: StageContract, name: str) -> None:
    """Fold one stage's writes, removals and tensor residency into the running state."""
    produced = produced_roles(contract)
    walk.available |= produced
    walk.removed_roles -= produced  # a re-produced role is no longer "removed"
    written = _write_key_values(contract)
    # Most recent writer wins -- that is who a downstream reader would actually get.
    walk.key_producer.update(dict.fromkeys(written, name))
    walk.available_keys |= written
    for rk in contract.removes_keys:
        walk.available_keys.discard(rk)
        # Dropping the carrier ends the tensor residency as surely as sanitizing does.
        walk.tensor_keys.discard(rk)
        role = contract.key_roles.get(rk, role_for_value(rk))
        if (
            role != "unknown"
            and role not in produced
            and not any(role_for_value(k) == role for k in walk.available_keys)
        ):
            walk.available.discard(role)
            walk.removed_roles.add(role)
    if "tensor" in contract.writes.produces:
        # The stage's OWN key_roles first, global names only as fallback. A custom
        # ``waveform_key`` still declares its role in the contract, but the global lookup
        # returned "unknown", so residency tracked ``_UNNAMED_TENSOR`` instead of the real
        # carrier -- and a downstream stage dropping that carrier still looked resident,
        # raising a spurious ``tensor_into_sink`` on a recipe that had cleaned up correctly.
        carriers = {k for k in written if contract.key_roles.get(k, role_for_value(k)) == _TENSOR_ROLE}
        walk.tensor_keys |= carriers or {_UNNAMED_TENSOR}
    if contract.gates.sanitizes_output:
        walk.tensor_keys.clear()


def validate_pipeline(  # noqa: C901
    stages: list[Any],
    *,
    initial_roles: set[str] | None = None,
    initial_keys: set[str] | None = None,
    available_gpus: float | None = None,
) -> PipelineReport:
    """Validate that an ordered list of configured stages composes.

    Args:
        stages: Configured stage instances in execution order.
        initial_roles: Semantic roles present in the input task. Defaults to
            ``{"audio_filepath"}`` (a manifest row). Pass an explicit set when the
            first stage is a source/reader or the input already carries waveforms.
        initial_keys: Literal key VALUES present in the input task (e.g. the
            columns of the source manifest: ``{"audio_filepath", "text"}``).
            Defaults to ``{"audio_filepath"}``. Seeding this lets the
            literal-key check (``keys_ok``) recognize reads satisfied by the
            input rather than by an upstream producer.
        available_gpus: If given, stages whose contract declares ``requires_gpu``
            while this is ``<= 0`` raise a warning.

    Returns:
        A :class:`PipelineReport`. ``report.ok`` is True when no errors were
        found (role-level); ``report.keys_ok`` additionally confirms literal-key
        identity (see the class docstring).
    """
    if initial_keys is not None:
        seed_keys = set(initial_keys)
    elif initial_roles is not None:
        # Both seeds describe ONE task, so they cannot default independently: "no roles" does
        # not also mean "the default columns". Seed only the roles that ARE their own key name --
        # roles and key values coincide for ``audio_filepath`` and diverge immediately after, so
        # seeding ``transcript`` as a literal column invents a key the task does not carry.
        seed_keys = {r for r in initial_roles if role_for_value(r) == r}
    else:
        seed_keys = set(_DEFAULT_INITIAL_KEYS)
    walk = _Walk(
        available=set(initial_roles) if initial_roles is not None else set(_DEFAULT_INITIAL_ROLES),
        available_keys=seed_keys,
    )
    expansion = expand_composites(stages)
    leaves = expansion.by_recipe_index()
    opaque = dict(expansion.opaque)
    # A composite with one illegible child is not a composite nobody could open. Discarding the
    # whole group left an eight-stage composite unchecked because one piece of plumbing lacks
    # describe() -- ``ManifestReader`` expands through ``FilePartitioningStage``, so the reader
    # starting most recipes contributed no keys at all. Legible siblings are kept; only the
    # unknown part is treated as unknown, via ``past_composite``.
    partly_opaque: dict[int, str] = {}
    for index, group in list(leaves.items()):
        reason = _unreadable_child(group)
        if reason:
            partly_opaque[index] = reason
            leaves[index] = [item for item in group if _describes_itself(item.stage)]
    issues: list[PipelineIssue] = []

    for index, recipe_stage in enumerate(stages):
        if index in expansion.unrunnable:
            issues.append(
                PipelineIssue(
                    index,
                    type(recipe_stage).__name__,
                    "error",
                    "composite_unrunnable",
                    f"the executor will refuse this stage: {expansion.unrunnable[index]}",
                )
            )
            walk.past_composite = True
            continue
        if index in opaque:
            issues.append(
                PipelineIssue(
                    index,
                    type(recipe_stage).__name__,
                    "warning",
                    "composite",
                    f"composite stage — its data flow could not be resolved ({opaque[index]}), "
                    f"so reads after it cannot be judged by role",
                )
            )
            walk.past_composite = True
            continue
        if index in partly_opaque:
            issues.append(
                PipelineIssue(
                    index,
                    type(recipe_stage).__name__,
                    "warning",
                    "composite",
                    f"composite stage — part of it is unreadable ({partly_opaque[index]}), "
                    f"so reads after it cannot be judged by role; its remaining stages are "
                    f"still checked",
                )
            )
            # Set BEFORE its own legible children are walked, not after. One child's writes
            # are unknown, and this composite's later children may be the very readers of
            # them -- judging those reads against a key set that is missing exactly the
            # unknown part is how a working pipeline gets failed on the name of a stage the
            # caller never wrote.
            walk.past_composite = True

        for item in leaves.get(index, []):
            stage = item.stage
            try:
                contract = build_contract(stage)
            except Exception as e:  # noqa: BLE001 - a stage that can't describe itself is an error
                issues.append(PipelineIssue(index, item.label, "error", "contract_error", f"describe() failed: {e}"))
                continue
            site = _Site(
                index=index,
                name=item.label if item.composite_ref else (contract.stage_id or type(stage).__name__),
                stage=stage,
                composite=recipe_stage if item.composite_ref else None,
            )

            if not contract.wrappable:
                # It calls itself a composite yet arrived here unexpanded, so it is not a
                # CompositeStage the expander could open. Its real I/O stays unknown and the
                # pre-expansion caution applies: warn, and judge nothing downstream by role.
                issues.append(
                    PipelineIssue(
                        site.index,
                        site.name,
                        "warning",
                        "composite",
                        "composite stage — decompose before validating its data flow",
                    )
                )
                walk.past_composite = True
                continue

            issues.extend(_read_issues(walk, site, contract))
            # Serialization / GPU gates reason about the environment rather than about roles, so
            # they run for every concrete stage even downstream of a composite nobody could expand.
            issues.extend(_gate_issues(site, contract, available_gpus, tensor_resident=bool(walk.tensor_keys)))
            _advance(walk, contract, site.name)

    return PipelineReport(issues=issues, produced_roles=walk.available, produced_keys=walk.available_keys)


def _roles_of(spec: Any, contract: StageContract) -> set[str]:  # noqa: ANN401
    keys = [*spec.data_keys, *spec.segment_data_keys]
    return {contract.key_roles.get(k, "unknown") for k in keys}
