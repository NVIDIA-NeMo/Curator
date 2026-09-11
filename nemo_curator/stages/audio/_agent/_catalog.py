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

"""Discovery API for agent-ready audio stages.

Built on the framework's existing class-name registry
(:data:`nemo_curator.stages.base._STAGE_REGISTRY`, populated by ``StageMeta``);
no new registry is introduced. ``_ensure_audio_stages_imported`` triggers the
audio stage modules so their registration is populated before discovery.

    list_agent_ready_stages()         # -> ["MonoConversionStage", ...]
    describe_stage("UTMOSFilterStage")  # -> StageContract (static, instance-free)
    catalog_as_json()                 # -> JSON an agent/UI can consume
"""

from __future__ import annotations

import importlib
import json
import pkgutil
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from nemo_curator.stages.audio._agent._agent_ready import AgentReady, to_json_schema
from nemo_curator.stages.audio._agent._agent_registry import build_contract, static_contract
from nemo_curator.stages.audio._agent._conformance import produced_roles

if TYPE_CHECKING:
    from nemo_curator.stages.audio._agent._agent_ready import StageContract

_IMPORTED = False
# Modules discovery could not import, kept so the failure can be reported rather than
# silently shrinking the catalog.
_SKIPPED: list[dict[str, str]] = []


def _ensure_audio_stages_imported() -> None:
    """Import audio stage modules so ``StageMeta`` has registered their classes.

    Defensive: a submodule whose optional heavy dependency (whisperx, pyannote,
    nemo_text_processing, ...) is absent is skipped with a warning rather than
    breaking discovery. Idempotent.
    """
    global _IMPORTED  # noqa: PLW0603
    if _IMPORTED:
        return
    import nemo_curator.stages.audio as audio_pkg

    # onerror: a failing subpackage __init__ (non-ImportError too — OSError /
    # RuntimeError are realistic for heavy audio deps) must skip, not kill
    # discovery; the loop body below warns for the same module on import.
    for modinfo in pkgutil.walk_packages(
        audio_pkg.__path__, prefix=audio_pkg.__name__ + ".", onerror=lambda _name: None
    ):
        leaf = modinfo.name.rsplit(".", 1)[-1]
        if leaf.startswith("_"):  # private support modules carry no stages
            continue
        try:
            importlib.import_module(modinfo.name)
        except Exception as e:  # noqa: BLE001 - optional dep or import-time issue; skip
            warnings.warn(f"audio catalog: skipped {modinfo.name} ({type(e).__name__}: {e})", stacklevel=2)
            # Recorded, not just warned: a warning goes to stderr where the agent's JSON
            # consumer never sees it, and a silently shorter catalog looks like a smaller
            # library. Deduplicated because ``_IMPORTED`` is set only after the loop completes,
            # so anything escaping it makes the next call re-walk and append every failure
            # again -- which reads like a worsening install. See :func:`unavailable_modules`.
            if not any(entry["module"] == modinfo.name for entry in _SKIPPED):
                _SKIPPED.append({"module": modinfo.name, "error": f"{type(e).__name__}: {e}"})
    _IMPORTED = True


def unavailable_modules() -> list[dict[str, str]]:
    """Stage modules that could not be imported, so a caller can report what is MISSING.

    Discovery degrades to whatever imported successfully. Without this, a CPU-only
    (``audio_cpu``) install -- a supported profile -- simply has no ASR or diarization
    stages, and the agent concludes they do not exist rather than that they are unavailable
    *here*, which is the difference between "your library cannot do this" and "install the
    GPU extra".
    """
    _ensure_audio_stages_imported()
    return [dict(entry) for entry in _SKIPPED]


def _agent_ready_registry() -> dict[str, type]:
    from nemo_curator.stages.base import _STAGE_REGISTRY

    return {
        name: cls for name, cls in _STAGE_REGISTRY.items() if isinstance(cls, type) and issubclass(cls, AgentReady)
    }


def list_agent_ready_stages() -> list[str]:
    """Sorted class names of all registered agent-ready audio stages."""
    _ensure_audio_stages_imported()
    return sorted(_agent_ready_registry())


def get_agent_ready_stage_class(name: str) -> type:
    """Return the registered stage class for ``name`` (must be agent-ready)."""
    _ensure_audio_stages_imported()
    registry = _agent_ready_registry()
    if name not in registry:
        msg = f"{name!r} is not a registered agent-ready audio stage"
        raise KeyError(msg)
    return registry[name]


def describe_stage(name: str, stage: AgentReady | None = None) -> StageContract:
    """Return a stage's contract.

    With ``stage`` (an instance) -> dynamic contract with resolved key values.
    Otherwise -> instance-free ``static_contract`` (no resolved keys/cardinality).
    """
    if stage is not None:
        return build_contract(stage)
    return static_contract(get_agent_ready_stage_class(name))


def audio_stage_catalog(*, include_dynamic_defaults: bool = False) -> list[dict[str, Any]]:
    """Return the catalog as a list of ``{name, contract[, default_contract]}`` dicts.

    ``contract`` is the static (instance-free) contract. ``default_contract``
    (only with ``include_dynamic_defaults``) is the dynamic contract of a
    no-arg instance, attempted best-effort and ``None`` when the stage needs
    required constructor args.
    """
    entries: list[dict[str, Any]] = []
    for name in list_agent_ready_stages():
        cls = get_agent_ready_stage_class(name)
        static = static_contract(cls)
        entry: dict[str, Any] = {
            "name": name,
            "contract": static.to_dict(),
            # A JSON-Schema config form for the stage's params — directly usable
            # as an agent tool-argument schema (enum/default/x-role included).
            "params_schema": to_json_schema(static.params),
        }
        if include_dynamic_defaults:
            try:
                entry["default_contract"] = build_contract(cls()).to_dict()
            except Exception:  # noqa: BLE001 - required-arg stages have no no-arg default
                entry["default_contract"] = None
        entries.append(entry)
    return entries


def catalog_as_json(*, include_dynamic_defaults: bool = False, indent: int | None = None) -> str:
    """JSON-serialized :func:`audio_stage_catalog` (an agent/UI tool schema)."""
    return json.dumps(audio_stage_catalog(include_dynamic_defaults=include_dynamic_defaults), indent=indent)


# --------------------------------------------------------------------------- #
# Role -> producer/consumer index (composition + repair)
# --------------------------------------------------------------------------- #
def _consumed_roles(contract: StageContract) -> set[str]:
    """Semantic roles a stage requires (primary reads + every reads_one_of option)."""
    roles = {
        contract.key_roles.get(k, "unknown") for k in [*contract.reads.data_keys, *contract.reads.segment_data_keys]
    }
    for opt in contract.reads_one_of:
        roles |= {contract.key_roles.get(k, "unknown") for k in [*opt.data_keys, *opt.segment_data_keys]}
    return roles - {"unknown"}


def _dummy_for_param(type_str: str | None) -> Any:  # noqa: ANN401 - placeholder is deliberately any primitive shape
    """A harmless placeholder for a required constructor arg, so ``describe()`` can
    run for a required-arg stage. ``*_key`` fields have defaults (never required),
    so these dummies only fill non-semantic args (paths, model names) and never
    perturb the resolved key roles."""
    t = (type_str or "").lower()
    if "bool" in t:
        return False
    if "int" in t:
        return 0
    if "float" in t:
        return 0.0
    if t.startswith("list"):
        return []
    if t.startswith("dict"):
        return {}
    return "x"  # str / path / model-name / anything else


def _default_contract(cls: type) -> StageContract | None:
    """Best-effort dynamic contract. Tries progressively: a no-arg instance, then
    a probe filling required args with harmless dummies, then also filling
    ``None``-default args (to satisfy "one-of" ``__post_init__`` guards, e.g. ASR
    needs ``model_name`` OR ``asr_model``). Only ``describe()`` is called, and
    ``*_key`` fields keep their real defaults, so produced/consumed roles stay
    correct. ``None`` only if every probe fails (e.g. needs a live model object)."""
    try:
        return build_contract(cls())
    except Exception:  # noqa: BLE001, S110 - deliberate fall-through to dummy-filled probes
        pass
    from nemo_curator.stages.audio._agent._agent_registry import stage_params

    params = stage_params(cls)
    for also_fill_none in (False, True):
        kwargs = {
            p.name: _dummy_for_param(p.type) for p in params if p.required or (also_fill_none and p.default is None)
        }
        if not kwargs:
            continue
        try:
            return build_contract(cls(**kwargs))
        except Exception:  # noqa: BLE001, S112 - deliberately try the next, broader probe
            continue
    return None


def role_index() -> dict[str, Any]:
    """Map each semantic role to the stages that produce/consume it.

    Returns ``{"producers": {role: [stage, ...]}, "consumers": {...},
    "unresolved_stages": [...]}``. Built from each stage's no-arg dynamic
    contract when possible, else from a probe instance with required (and
    one-of ``None``-default) args filled by harmless dummies — only
    ``describe()`` runs and ``*_key`` defaults are preserved, so the roles stay
    correct. ``unresolved_stages`` lists only stages where every probe fails.

    This is what turns an ``unsatisfied_reads`` validation error into an
    actionable repair ("insert a stage that produces role X") and lets an agent
    detect an *unproducible* role (``find_producers`` returns ``[]``).
    """
    producers: dict[str, set[str]] = defaultdict(set)
    consumers: dict[str, set[str]] = defaultdict(set)
    unresolved: list[str] = []
    for name in list_agent_ready_stages():
        contract = _default_contract(get_agent_ready_stage_class(name))
        if contract is None:
            unresolved.append(name)
            continue
        for role in produced_roles(contract):
            producers[role].add(name)
        for role in _consumed_roles(contract):
            consumers[role].add(name)
    return {
        "producers": {r: sorted(v) for r, v in sorted(producers.items())},
        "consumers": {r: sorted(v) for r, v in sorted(consumers.items())},
        "unresolved_stages": sorted(unresolved),
    }


def find_producers(role: str) -> list[str]:
    """Stages that produce ``role``. An empty list means no stage produces it
    (the role is *unproducible* — an agent should not try to satisfy it)."""
    return role_index()["producers"].get(role, [])


def find_consumers(role: str) -> list[str]:
    """Stages that consume ``role``."""
    return role_index()["consumers"].get(role, [])
