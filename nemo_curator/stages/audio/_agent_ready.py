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

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

AudioForm = Literal["file", "waveform"]
ProducedForm = Literal["tensor", "disk"]
Cardinality = Literal["1:1", "1:1 nested-list", "1:N fan-out", "N:1", "filter"]
Dispatch = Literal["process", "process_batch", "auto"]
# How a stage handles per-item failures at runtime. "unknown" means the stage
# has not declared a uniform policy (the default for most stages today).
ErrorPolicy = Literal["skip", "fail", "annotate", "unknown"]

# Semantic role vocabulary. Because key *names* are agent-configurable
# (config-knobs-only standardization: every read/written key is a ``*_key``
# constructor field that an agent may rename), two stages can only be chained
# reliably by matching the *role* a key plays, not its literal string. Roles are
# resolved from the invariant ``*_key`` field name (see ``_roles.py``), so they
# survive an agent renaming a key's value. ``"unknown"`` is the safe default and
# never blocks composition.
Role = Literal[
    "audio_filepath",
    "waveform",
    "sample_rate",
    "duration",
    "num_samples",
    "segments",
    "diar_segments",
    "vad_segments",
    "overlap_segments",
    "timestamps",
    "start",
    "end",
    "start_ms",
    "end_ms",
    "text",
    "pred_text",
    "reference_text",
    "words",
    "alignment",
    "speaker_id",
    "num_speakers",
    "score",
    "metrics",
    "prediction",
    "segment_num",
    "original_file",
    "item_id",
    "windows",
    "manifest_path",
    "output_path",
    "unknown",
]


@dataclass(frozen=True)
class ParamSpec:
    """A single constructor parameter an agent can set on a stage.

    Usually derived automatically from the stage's dataclass fields (or
    ``__init__`` signature) via
    :func:`nemo_curator.stages.audio._agent_registry.stage_params`, but a stage
    may also override/augment entries in ``StageContract.params``.
    """

    name: str
    type: str = "Any"
    default: Any = None
    required: bool = False
    choices: list[Any] | None = None  # populated for Literal[...] params
    description: str | None = None
    role: Role | None = None  # semantic role of the key this param configures


@dataclass(frozen=True)
class IOSpec:
    """Task data keys and audio forms read or written by a stage."""

    data_keys: list[str] = field(default_factory=list)
    segment_data_keys: list[str] = field(default_factory=list)
    accepts: list[AudioForm] = field(default_factory=list)
    produces: list[ProducedForm] = field(default_factory=list)


@dataclass(frozen=True)
class Gates:
    """Execution gates or side effects an agent should know before wrapping a stage."""

    writes_to_disk: bool = False
    requires_gpu: bool = False
    requires_internet_first_run: bool = False
    requires_ffmpeg: bool = False
    lifecycle_side_effects: bool = False
    runtime_secrets: list[str] = field(default_factory=list)
    # Serializability contract for sinks (distinguishes the two JSON sinks):
    #   requires_serializable_input — this stage serializes task.data as-is (e.g.
    #     raw json.dumps) and will fail on a resident tensor/audio blob.
    #   sanitizes_output — this stage strips tensors/audio blobs, so anything
    #     downstream of it is serialization-safe.
    requires_serializable_input: bool = False
    sanitizes_output: bool = False


@dataclass(frozen=True)
class SizeEnvelope:
    """Coarse size and memory hints for agent planning."""

    max_input_sec: float | None = None
    allowed_sample_rates: list[int] | None = None
    channels: Literal["mono", "stereo", "any"] = "any"
    memory_hint: str | None = None


@dataclass(frozen=True)
class StaticHints:
    """Instance-independent hints a stage may declare for discovery/planning.

    Lets a stage expose ``cardinality_options``, ``gates``, ``dispatch``,
    ``error_policy``, ``description`` and ``stage_id`` *without* being
    instantiated (see :meth:`AgentReady.describe_static`). Declared on a class
    via the ``AGENT_STATIC`` ClassVar; every field defaults so declaring it is
    fully optional and additive.
    """

    cardinality_options: list[str] = field(default_factory=list)
    gates: Gates = field(default_factory=Gates)
    dispatch: Dispatch = "auto"
    error_policy: ErrorPolicy = "unknown"
    description: str | None = None
    stage_id: str | None = None


@dataclass(frozen=True)
class StageContract:
    """Read-only discovery contract for an agent-ready processing stage."""

    reads: IOSpec = field(default_factory=IOSpec)
    writes: IOSpec = field(default_factory=IOSpec)
    reads_one_of: list[IOSpec] = field(default_factory=list)
    metadata_reads: list[str] = field(default_factory=list)
    metadata_writes: list[str] = field(default_factory=list)
    cardinality: Cardinality = "1:1"
    cardinality_options: list[str] = field(default_factory=list)
    iteration_key: str | None = None
    preserves_upstream_keys: bool = True
    wrappable: bool = True
    size_envelope: SizeEnvelope = field(default_factory=SizeEnvelope)
    gates: Gates = field(default_factory=Gates)
    # Agent-facing metadata (advisory; defaults keep older describe() calls valid).
    stage_id: str | None = None  # stable semantic id; defaults to the class name when None
    description: str | None = None  # one-line human summary for planners/UIs
    params: list[ParamSpec] = field(default_factory=list)  # usually auto-derived at discovery time
    dispatch: Dispatch = "auto"  # "auto" => infer from the stage at runtime
    error_policy: ErrorPolicy = "unknown"
    # Resolved-key-value -> semantic role. Populated at discovery time by
    # ``_agent_registry.build_contract``; empty when a contract is built by hand.
    key_roles: dict[str, Role] = field(default_factory=dict)
    # True when the stage only implements ``process_batch`` (``process`` raises).
    # Auto-derived at discovery time; agents must not call ``process`` on these.
    batch_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict of this contract (``json.dumps`` never raises)."""
        return {
            "reads": asdict(self.reads),
            "writes": asdict(self.writes),
            "reads_one_of": [asdict(spec) for spec in self.reads_one_of],
            "metadata_reads": list(self.metadata_reads),
            "metadata_writes": list(self.metadata_writes),
            "cardinality": self.cardinality,
            "cardinality_options": list(self.cardinality_options),
            "iteration_key": self.iteration_key,
            "preserves_upstream_keys": self.preserves_upstream_keys,
            "wrappable": self.wrappable,
            "size_envelope": asdict(self.size_envelope),
            "gates": asdict(self.gates),
            "stage_id": self.stage_id,
            "description": self.description,
            "params": [_paramspec_to_dict(p) for p in self.params],
            "dispatch": self.dispatch,
            "error_policy": self.error_policy,
            "key_roles": dict(self.key_roles),
            "batch_only": self.batch_only,
        }


def _jsonable_default(value: Any) -> Any:  # noqa: ANN401, PLR0911 (complexity accepted: one early return per JSON type family)
    """Coerce an arbitrary param default to a JSON-serializable value.

    Non-serializable objects (tensors, models, callables) become a short
    ``"<non-serializable: Type>"`` sentinel rather than raising.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [_jsonable_default(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable_default(v) for k, v in value.items()}
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        try:
            return {k: _jsonable_default(v) for k, v in asdict(value).items()}
        except Exception:  # noqa: BLE001
            return f"<non-serializable: {type(value).__name__}>"
    return f"<non-serializable: {type(value).__name__}>"


def _paramspec_to_dict(p: ParamSpec) -> dict[str, Any]:
    return {
        "name": p.name,
        "type": p.type,
        "default": _jsonable_default(p.default),
        "required": p.required,
        "choices": None if p.choices is None else [_jsonable_default(c) for c in p.choices],
        "description": p.description,
        "role": p.role,
    }


def _json_type(type_str: str | None) -> str | None:
    """Map a rendered ParamSpec.type string to a JSON-Schema type (or None to omit)."""
    if not type_str:
        return None
    t = type_str.replace(" ", "").replace("|None", "")
    if t.startswith("Optional["):
        t = t[len("Optional[") : -1] if t.endswith("]") else t
    scalar = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}
    if t in scalar:
        return scalar[t]
    lowered = t.lower()
    if lowered.startswith(("list", "sequence", "tuple")):
        return "array"
    if lowered.startswith(("dict", "mapping")):
        return "object"
    return None


def to_json_schema(params: list[ParamSpec]) -> dict[str, Any]:
    """Build a JSON-Schema ``object`` describing a stage's configurable params.

    Suitable as the argument schema for an agent's stage-configuration form.
    """
    properties: dict[str, Any] = {}
    required: list[str] = []
    for p in params:
        schema: dict[str, Any] = {}
        json_type = _json_type(p.type)
        if json_type is not None:
            schema["type"] = json_type
        if p.choices is not None:
            schema["enum"] = [_jsonable_default(c) for c in p.choices]
        if p.default is not None:
            schema["default"] = _jsonable_default(p.default)
        if p.description:
            schema["description"] = p.description
        if p.role:
            schema["x-role"] = p.role
        properties[p.name] = schema
        if p.required:
            required.append(p.name)
    out: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        out["required"] = required
    return out


class AgentReady:
    """Mixin for stages that expose a read-only agent discovery contract."""

    # Opt-in, instance-independent discovery hints. Annotated as ClassVar so
    # dataclass stages do NOT treat these as fields. All optional/additive.
    AGENT_STATIC: ClassVar[StaticHints | None] = None
    # Set True on stages whose ``process`` raises (only ``process_batch`` works).
    BATCH_ONLY: ClassVar[bool] = False
    # Rare per-field role overrides keyed by ``*_key`` field name. Consulted
    # before the shared ``_roles.KEY_ROLES`` table.
    KEY_ROLE_OVERRIDES: ClassVar[Mapping[str, Role]] = {}

    def describe(self) -> StageContract:
        raise NotImplementedError

    @classmethod
    def describe_static(cls) -> StageContract:
        """Instance-free contract for discovery/planning.

        Uses class defaults + ``AGENT_STATIC`` and never runs ``__init__`` side
        effects, so it is safe for stages with required constructor args. Prefer
        the instance-level :meth:`describe` (or
        ``_agent_registry.build_contract``) when resolved key *values* are
        needed.
        """
        from nemo_curator.stages.audio._agent_registry import static_contract

        return static_contract(cls)


# (resolve_contract was removed: dead code whose signature promised class
# acceptance the body rejected. Instances: use stage.describe() / build_contract;
# classes: use describe_static() / static_contract.)
