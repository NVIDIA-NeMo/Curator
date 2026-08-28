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

"""Conformance harness for agent-ready audio stages.

``assert_agent_ready(stage, fixture_factory, ...)`` is the gate each stage's CL
must pass. It runs a set of *static* checks (contract well-formedness, semantic
roles, JSON serialization, by-role read satisfiability) that need no execution,
plus optional *dynamic* checks (run ``process``/``process_batch`` on a fixture
and verify declared writes appear, no undeclared top-level keys leak, cardinality
matches runtime, and ``accepts``/``produces`` hold).

The static checks alone catch the contract↔reality drift the prototype lacked
and can sweep every stage with no fixtures (see ``assert_contract_wellformed``).
"""

# ruff: noqa: S101 - this module IS the assertion harness; `assert` is its output format.
# It ships under nemo_curator/ (not tests/) because stage authors call it from their own
# suites, so the tests/** ignore does not reach it; raising would lose pytest's assertion
# rewriting that makes these failures readable.

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, get_args

from nemo_curator.stages.audio._agent._agent_ready import (
    AudioForm,
    Cardinality,
    ProducedForm,
    Role,
    StageContract,
    WriteValueOrigin,
    to_json_schema,
)
from nemo_curator.stages.audio._agent._agent_registry import build_contract, static_contract
from nemo_curator.stages.audio._agent._residency import accepts_for_residency
from nemo_curator.stages.audio._agent._roles import field_has_declared_role

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

_VALID_CARDINALITY: frozenset[str] = frozenset(get_args(Cardinality))
_VALID_ROLES: frozenset[str] = frozenset(get_args(Role))
_VALID_ACCEPTS: frozenset[str] = frozenset(get_args(AudioForm))
_VALID_PRODUCES: frozenset[str] = frozenset(get_args(ProducedForm))
_VALID_WRITE_VALUE_ORIGINS: frozenset[str] = frozenset(get_args(WriteValueOrigin))
# The param a delta run sets to feed a source only the files that changed.
_NARROWING_PARAM = "include_files"


# --------------------------------------------------------------------------- #
# By-role matching (the planner primitive)
# --------------------------------------------------------------------------- #
def _spec_roles(contract: StageContract, spec_keys: Iterable[str]) -> set[str]:
    return {contract.key_roles.get(k, "unknown") for k in spec_keys}


def produced_roles(producer: StageContract) -> set[str]:
    """Roles a producer emits (from its ``writes`` keys); excludes ``unknown``."""
    keys = [*producer.writes.data_keys, *producer.writes.segment_data_keys]
    return _spec_roles(producer, keys) - {"unknown"}


def reads_satisfied_by_role(consumer: StageContract, available_roles: set[str]) -> bool:
    """Can ``consumer`` run given upstream-produced ``available_roles``?

    Matches by semantic role, not key-string equality, so a producer that
    renamed its output key still satisfies a consumer that needs that role.
    ``unknown`` is permissive (never blocks).
    """
    avail = set(available_roles) | {"unknown"}
    reads_keys = [*consumer.reads.data_keys, *consumer.reads.segment_data_keys]
    if reads_keys and not _spec_roles(consumer, reads_keys).issubset(avail):
        return False
    if consumer.reads_one_of:
        return any(
            _spec_roles(consumer, [*opt.data_keys, *opt.segment_data_keys]).issubset(avail)
            for opt in consumer.reads_one_of
        )
    return True


# --------------------------------------------------------------------------- #
# Static checks (no execution)
# --------------------------------------------------------------------------- #
def _check_shape(c: StageContract, name: str) -> None:  # noqa: C901
    assert c.cardinality in _VALID_CARDINALITY, f"{name}: invalid cardinality {c.cardinality!r}"
    for opt in c.cardinality_options:
        # cardinality_options are short flag names (e.g. "fan_out","nested") OR full cardinalities
        assert isinstance(opt, str), f"{name}: bad cardinality option {opt!r}"
        assert opt, f"{name}: bad cardinality option {opt!r}"
    if c.iteration_key is not None:
        assert c.cardinality in {"1:1 nested-list", "1:N fan-out", "N:1"}, (
            f"{name}: iteration_key set but cardinality is {c.cardinality!r}"
        )
        # iteration_key must name something real: either a key the contract
        # reads/writes, or a role-resolvable key value (fan-out stages iterate a
        # list that is deliberately NOT re-emitted into children, so it may be
        # absent from writes — but it must still resolve to a semantic role).
        # Catches synthetic labels like the former 'speakers' that name nothing.
        contract_keys: set[str] = set()
        for spec in [c.reads, c.writes, *c.reads_one_of]:
            contract_keys.update(spec.data_keys)
            contract_keys.update(spec.segment_data_keys)
        assert c.iteration_key in contract_keys or c.iteration_key in c.key_roles, (
            f"{name}: iteration_key {c.iteration_key!r} is neither a contract read/write "
            f"key nor a role-resolvable key value — it names nothing an agent can find"
        )
    for spec, label in [(c.reads, "reads"), (c.writes, "writes"), *[(s, "reads_one_of") for s in c.reads_one_of]]:
        for a in spec.accepts:
            assert a in _VALID_ACCEPTS, f"{name}: {label}.accepts has invalid form {a!r}"
        for p in spec.produces:
            assert p in _VALID_PRODUCES, f"{name}: {label}.produces has invalid form {p!r}"
    for index, conditional in enumerate(c.conditional_writes):
        label = f"conditional_writes[{index}]"
        assert conditional.condition.strip(), f"{name}: {label}.condition must be non-empty"
        assert conditional.value_origin in _VALID_WRITE_VALUE_ORIGINS, (
            f"{name}: {label}.value_origin has invalid value {conditional.value_origin!r}"
        )
        assert conditional.writes.data_keys or conditional.writes.segment_data_keys or conditional.metadata_writes, (
            f"{name}: {label} must name at least one task, segment, or metadata key"
        )
        for a in conditional.writes.accepts:
            assert a in _VALID_ACCEPTS, f"{name}: {label}.writes.accepts has invalid form {a!r}"
        for p in conditional.writes.produces:
            assert p in _VALID_PRODUCES, f"{name}: {label}.writes.produces has invalid form {p!r}"
        assert len(conditional.writes.data_keys) == len(set(conditional.writes.data_keys)), (
            f"{name}: duplicate {label}.writes.data_keys"
        )
        assert len(conditional.writes.segment_data_keys) == len(set(conditional.writes.segment_data_keys)), (
            f"{name}: duplicate {label}.writes.segment_data_keys"
        )
        assert len(conditional.metadata_writes) == len(set(conditional.metadata_writes)), (
            f"{name}: duplicate {label}.metadata_writes"
        )
        assert all(isinstance(key, str) and key for key in conditional.metadata_writes), (
            f"{name}: {label}.metadata_writes must contain non-empty strings"
        )
    # no duplicate keys within a single spec list
    for spec, label in [(c.reads, "reads"), (c.writes, "writes")]:
        assert len(spec.data_keys) == len(set(spec.data_keys)), f"{name}: duplicate {label}.data_keys"


def _check_roles(stage_or_cls: Any, c: StageContract, name: str) -> None:  # noqa: ANN401
    for value, role in c.key_roles.items():
        assert role in _VALID_ROLES, f"{name}: key_roles[{value!r}] has invalid role {role!r}"
    for p in c.params:
        if p.role is not None:
            assert p.role in _VALID_ROLES, f"{name}: param {p.name!r} has invalid role {p.role!r}"
        # check #8: a *_key constructor field must have a KEY_ROLES entry or be
        # explicitly allowlisted as internal (catches a forgotten role mapping).
        if p.name.endswith("_key"):
            stage_cls = stage_or_cls if isinstance(stage_or_cls, type) else type(stage_or_cls)
            assert field_has_declared_role(p.name, stage_cls), (
                f"{name}: param {p.name!r} ends in '_key' but declares no role. Either give it "
                "a shared role (KEY_ROLES in _roles.py) if another stage consumes it, or -- for "
                "this stage's own bookkeeping -- declare it on the stage itself via "
                "KEY_ROLE_OVERRIDES or INTERNAL_KEY_FIELDS"
            )


def _check_per_row_independence(c: StageContract, name: str) -> None:
    """A narrowable source has to answer whether narrowing it is sound -- either way.

    ``False`` is a legitimate answer, not a violation: a source that cannot be narrowed soundly
    says so per instance and ``delta.region`` stops there. Requiring ``True`` would force such a
    source to lie or to drop the parameter. Silence is what is forbidden.

    The case that first motivated this -- ``CreateInitialManifestAudioFolderStage`` under a
    bounded ``max_samples``, which truncates the sorted listing -- now declares ``True`` anyway
    by an explicit product decision recorded at that declaration. The rule is unchanged: it was
    never "must be False when narrowing is lossy", only "must not be silent".

    A companion rule ("``True`` contradicts ``N:1``") was removed as wrong: cardinality counts
    TASKS, this is about row VALUES, and ``AudioToDocumentStage`` repacks tasks while leaving
    values untouched. ``delta._TRACEABLE`` refuses ``N:1`` before the gate is read anyway.
    """
    if any(p.name == _NARROWING_PARAM for p in c.params):
        assert c.gates.per_row_independent is not None, (
            f"{name}: accepts {_NARROWING_PARAM!r} but leaves gates.per_row_independent undeclared -- "
            "a source a delta run can narrow to a subset of files has to say whether the rows it "
            "emits depend on which other files were present. False is a legitimate answer (the "
            "delta then refuses to narrow it); silence is not"
        )


def _check_serialization(c: StageContract, name: str) -> None:
    try:
        json.dumps(c.to_dict())
    except (TypeError, ValueError) as e:  # pragma: no cover - defensive
        msg = f"{name}: contract.to_dict() is not JSON-serializable: {e}"
        raise AssertionError(msg) from e
    schema = to_json_schema(c.params)
    assert schema.get("type") == "object", f"{name}: bad json schema"
    assert "properties" in schema, f"{name}: bad json schema"


def _check_residency_accepts(stage: Any, c: StageContract, name: str) -> None:  # noqa: ANN401
    """A residency-configurable stage must advertise exactly the forms it consumes.

    Derives the expected audio forms from the instance's ``input_residency`` and
    asserts the contract's declared ``accepts`` (across ``reads`` + ``reads_one_of``)
    match — catching the hand-typed "lying accepts" drift (a ``file``-mode instance
    that still advertises ``waveform``). Skips when the contract carries no audio
    ``accepts`` (e.g. an instance-free static contract with unresolved reads).
    """
    residency = getattr(stage, "input_residency", None)
    if residency is None:
        return
    declared = set(c.reads.accepts) | {a for opt in c.reads_one_of for a in opt.accepts}
    if not declared:
        return
    expected = set(accepts_for_residency(residency))
    assert declared == expected, (
        f"{name}: declared accepts {sorted(declared)} != residency-derived {sorted(expected)} "
        f"for input_residency={residency!r} — derive accepts from input_residency (lying/drifted accepts)"
    )


def assert_contract_wellformed(stage_or_cls: Any) -> StageContract:  # noqa: ANN401
    """Static-only conformance: shape, roles, serialization. No execution.

    Accepts an instance (dynamic contract via ``build_contract``) or a class
    (instance-free ``static_contract``). Returns the contract so callers can
    reuse it. Safe to run across every stage with no fixtures.
    """
    if isinstance(stage_or_cls, type):
        c = static_contract(stage_or_cls)
        name = stage_or_cls.__name__
    else:
        c = build_contract(stage_or_cls)
        name = type(stage_or_cls).__name__
    _check_shape(c, name)
    _check_roles(stage_or_cls, c, name)
    _check_per_row_independence(c, name)
    _check_serialization(c, name)
    _check_residency_accepts(stage_or_cls, c, name)
    return c


# --------------------------------------------------------------------------- #
# Dynamic checks (execute the stage on a fixture)
# --------------------------------------------------------------------------- #
def _supports_batch(stage: Any) -> bool:  # noqa: ANN401
    fn = getattr(stage, "supports_batch_processing", None)
    try:
        return bool(fn()) if callable(fn) else False
    except Exception:  # noqa: BLE001
        return False


def _normalize_results(out: Any) -> list[Any]:  # noqa: ANN401
    if out is None:
        return []
    if isinstance(out, list):
        flat: list[Any] = []
        for item in out:
            if item is None:
                continue
            if isinstance(item, list):
                flat.extend(x for x in item if x is not None)
            else:
                flat.append(item)
        return flat
    return [out]


def _data_of(task: Any) -> dict[str, Any]:  # noqa: ANN401
    data = getattr(task, "data", None)
    return data if isinstance(data, dict) else {}


def _check_gpu_gate(stage: Any, c: StageContract, name: str) -> None:  # noqa: ANN401
    """A stage that reserves GPU resources must not report that it needs none.

    One-sided deliberately. The reverse -- declaring the gate while reserving nothing -- is
    legitimate: InferenceSortformerStage passes ``map_location="cuda"`` unconditionally, so it
    needs a GPU whatever its ``resources`` say. Only the false negative is dangerous, because
    it lets the planner put a GPU stage on a CPU-only worker.
    """
    resources = getattr(stage, "resources", None)
    if resources is not None and bool(getattr(resources, "requires_gpu", False)) and not c.gates.requires_gpu:
        msg = (
            f"{name}: reserves GPU resources (gpus={getattr(resources, 'gpus', 0)}, "
            f"gpu_memory_gb={getattr(resources, 'gpu_memory_gb', 0)}) but its contract reports "
            f"requires_gpu=False. build_contract derives this from resources, so reaching here "
            f"means the contract was built by hand or the derivation was bypassed."
        )
        raise AssertionError(msg)


def assert_agent_ready(  # noqa: C901, PLR0912, PLR0913 (complexity accepted: one linear checklist of independent conformance checks)
    stage: Any,  # noqa: ANN401
    fixture_factory: Callable[[], Any] | None = None,
    *,
    expected_cardinality: str | None = None,
    available_keys: Iterable[str] | None = None,
    segments_key: str | None = None,
    ignore_new_keys: Iterable[str] = (),
    run: bool = True,
    setup: bool = False,
) -> StageContract:
    """Assert a stage is agent-ready. Returns its (dynamic) contract.

    Always runs the static checks. When ``run`` and a ``fixture_factory`` are
    given, also executes the stage and verifies declared writes appear, no
    undeclared top-level keys leak, and cardinality matches the runtime shape.

    Args:
        stage: A constructed stage instance.
        fixture_factory: Returns a fresh input task (or batch) each call.
        expected_cardinality: If given, assert the contract declares it.
        available_keys: Upstream-available key values; asserts reads are
            satisfiable by role.
        segments_key: Resolved segments key, for checking segment-level writes.
        ignore_new_keys: Extra top-level keys allowed in output (framework
            bookkeeping) beyond declared writes.
        run: Execute the stage (default True).
        setup: Call ``stage.setup()`` before processing (default False; most
            lightweight stages need no setup, heavy ones are pre-set-up/stubbed
            by the caller).
    """
    c = build_contract(stage)
    name = type(stage).__name__
    _check_shape(c, name)
    _check_roles(stage, c, name)
    _check_per_row_independence(c, name)
    _check_serialization(c, name)
    _check_residency_accepts(stage, c, name)
    _check_gpu_gate(stage, c, name)

    if expected_cardinality is not None:
        assert c.cardinality == expected_cardinality, (
            f"{name}: cardinality {c.cardinality!r} != expected {expected_cardinality!r}"
        )
    if available_keys is not None:
        avail_roles = {c.key_roles.get(k, "unknown") for k in available_keys}
        # also resolve via literal table for keys not in this stage's key_roles
        from nemo_curator.stages.audio._agent._roles import role_for_value

        avail_roles |= {role_for_value(k) for k in available_keys}
        assert reads_satisfied_by_role(c, avail_roles), (
            f"{name}: reads {c.reads.data_keys}/{[s.data_keys for s in c.reads_one_of]} "
            f"not satisfied by available roles {avail_roles}"
        )

    if not run or fixture_factory is None:
        return c

    if setup and hasattr(stage, "setup"):
        stage.setup()

    task = fixture_factory()
    batch_input = isinstance(task, list)
    input_keys = set(_data_of(task[0] if batch_input else task))

    if c.batch_only or _supports_batch(stage):
        out = stage.process_batch(task if batch_input else [task])
    else:
        out = stage.process(task)
    results = _normalize_results(out)

    # (6) cardinality vs runtime shape
    if c.cardinality == "1:N fan-out":
        assert isinstance(out, list), f"{name}: fan-out must return a list"
    elif c.cardinality in {"1:1", "1:1 nested-list"} and results:
        assert len(results) == 1, f"{name}: {c.cardinality} produced {len(results)} tasks"
    elif c.cardinality == "filter":
        assert len(results) <= (len(task) if batch_input else 1), f"{name}: filter increased task count"

    # (3) declared writes appear; (4) no undeclared top-level keys (non-fanout)
    if c.cardinality in {"1:1", "1:1 nested-list", "filter"} and results:
        out_data = _data_of(results[0])
        for key in c.writes.data_keys:
            assert key in out_data, f"{name}: declared write {key!r} missing from task.data"
        for key in c.removes_keys:
            assert key not in out_data, f"{name}: declared removes_keys {key!r} but it is still present in task.data"
        declared = set(c.writes.data_keys) | set(ignore_new_keys) | input_keys
        undeclared = set(out_data) - declared
        assert not undeclared, f"{name}: undeclared new top-level keys {sorted(undeclared)} (add to writes.data_keys)"
        # segment-level writes
        seg_key = segments_key or c.iteration_key
        if c.writes.segment_data_keys and seg_key and isinstance(out_data.get(seg_key), list) and out_data[seg_key]:
            seg0 = out_data[seg_key][0]
            if isinstance(seg0, dict):
                for key in c.writes.segment_data_keys:
                    assert key in seg0, f"{name}: declared segment write {key!r} missing from segment dict"
    return c


def assert_residency_consumption(
    stage_factory: Callable[[str], Any],
    *,
    file_fixture: Callable[[], Any],
    waveform_fixture: Callable[[], Any],
    setup: bool = False,
) -> None:
    """Prove a residency-configurable stage actually consumes each residency it advertises.

    Runs the stage in ``file`` and ``waveform`` modes on matching fixtures and
    asserts it produced output — so a stage that declares an ``input_residency``
    choice its ``process()`` cannot actually consume fails CI. This is the
    *dynamic* complement to the *static* :func:`_check_residency_accepts` drift
    guard: the static check ties ``accepts`` to ``input_residency`` in the
    contract; this one proves the code honors it.

    Intended for 1:1 / annotate stages (valid input -> non-empty output). Fan-out
    stages that may legitimately return no items on a given fixture should assert
    consumption differently.

    Args:
        stage_factory: ``residency -> constructed stage`` (e.g. ``lambda r: MyStage(input_residency=r)``).
        file_fixture: returns a task carrying only a file path.
        waveform_fixture: returns a task carrying only an in-memory waveform + sample rate.
        setup: call ``stage.setup()`` before processing (default False).
    """
    for residency, fixture in (("file", file_fixture), ("waveform", waveform_fixture)):
        stage = stage_factory(residency)
        if setup and hasattr(stage, "setup"):
            stage.setup()
        task = fixture()
        if _supports_batch(stage) or getattr(stage, "BATCH_ONLY", False):
            out = stage.process_batch([task])
        else:
            out = stage.process(task)
        results = _normalize_results(out)
        assert results, (
            f"{type(stage).__name__}: produced no output for input_residency={residency!r} — "
            f"it advertises this residency but process() did not consume the matching input"
        )
