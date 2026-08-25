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

"""Name -> class / target resolution over the Milestone-1 discovery catalog.

Every recipe ``ref`` is a registered agent-ready stage class name. We resolve it
through the existing foundation catalog so the recipe can only reference real,
importable stages (the anti-hallucination guarantee), and derive the Hydra
``_target_`` string for round-tripping to ``config.run``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_curator.stages.audio._agent._agent_ready import StageContract


def resolve_stage_class(ref: str) -> type:
    """Return the registered stage class for ``ref`` (raises ``KeyError`` if unknown)."""
    from nemo_curator.stages.audio._agent._catalog import get_agent_ready_stage_class

    return get_agent_ready_stage_class(ref)


def resolve_target(ref: str) -> str:
    """Return the fully qualified ``module.ClassName`` target for ``ref``."""
    cls = resolve_stage_class(ref)
    return f"{cls.__module__}.{cls.__qualname__}"


def static_contract_for(ref: str) -> StageContract:
    """Return the instance-free contract for ``ref``."""
    from nemo_curator.stages.audio._agent._agent_registry import static_contract

    return static_contract(resolve_stage_class(ref))


@dataclass(frozen=True)
class ResolvedContract:
    """A stage's contract, plus -- when it could not be resolved -- why, and what would.

    The instance-free contract reports no reads and no writes by construction;
    :func:`nemo_curator.stages.audio._agent._agent_registry.static_contract` says so in its own
    docstring. Handed to a caller unlabelled, that does not read as "unknown", it reads as
    "this stage requires nothing" -- which is how a pipeline feeding ``diar_segments`` into a
    stage requiring ``segments`` validated clean, downloaded two models, ran a GPU diarization
    pass, and only then refused to start. An unresolved contract must therefore say that it is
    unresolved and name what would settle it, so the caller asks again instead of concluding.
    """

    contract: StageContract
    unresolved_reason: str | None = None
    required_params: tuple[str, ...] = ()
    accepted_params: tuple[str, ...] = field(default=(), repr=False)
    instance: Any = field(default=None, repr=False, compare=False)
    """The configured stage, when one could be built.

    Carried because a composite's own contract is empty by design and the only way to learn what
    it needs is to expand *this* instance -- expanding the class would report the defaults, which
    is the difference between requiring ``segments`` and requiring ``diar_segments``.
    """

    @property
    def resolved(self) -> bool:
        return self.unresolved_reason is None

    def unresolved_detail(self) -> dict[str, Any] | None:
        """The caller-facing account of a fallback, or ``None`` when nothing fell back."""
        if self.resolved:
            return None
        detail: dict[str, Any] = {
            "reason": self.unresolved_reason,
            "reads_writes_are": "unknown, not empty",
        }
        if self.required_params:
            detail["required_params"] = list(self.required_params)
            detail["retry_with"] = dict.fromkeys(self.required_params, "<value>")
        if self.accepted_params:
            detail["accepted_params"] = list(self.accepted_params)
        return detail


def _fallback_reason(exc: Exception, missing: tuple[str, ...]) -> str:
    """Why instantiation failed, led by the actionable cause when there is one."""
    if missing:
        return f"needs parameters that were not supplied: {', '.join(missing)}"
    return f"could not be configured: {type(exc).__name__}: {exc}"


def resolved_contract_for(ref: str, params: dict[str, Any] | None = None) -> ResolvedContract:
    """The contract ``ref`` will honour when configured with ``params``.

    A stage's reads and writes come from its PARAMETERS -- ``SplitLongAudioStage(
    segments_key="diar_segments")`` reads ``diar_segments``, not ``segments`` -- so the only
    honest answer comes from an instance, which is what ``build_contract`` has always needed
    and ``describe`` never gave it.

    Instantiating to ask is nearly free of side effects: models load in ``setup()``, which this
    never calls, and execution knobs are dropped first so a caller can pass a recipe's params
    verbatim without the ``resources`` entry turning into a spurious "could not be configured".

    It is NOT true that no stage touches disk in ``__init__``, though this docstring said so and
    cited a regression test that did not exist. ``AudioDataFilterStage.__init__`` calls
    ``load_config(config_path)``, so a ``config_path`` supplied through the ``describe`` MCP tool
    is opened here -- an existence-and-parseability oracle for arbitrary paths, and one the
    workspace lock does not cover (``describe`` performs no path check, and ``config_path`` is a
    deliberately unlocked shared-dependency param). No file content reaches the response. The
    test in ``test_contract_resolution.py`` now pins the real boundary: that stage is listed as a
    known exception, so a NEW stage doing I/O here fails rather than joining it silently.

    Falls back to the instance-free contract rather than raising, because a stage with required
    arguments has to stay describable to a caller who does not yet know what to pass -- that
    caller is asking in order to find out. The fallback is labelled; see
    :meth:`ResolvedContract.unresolved_detail`.
    """
    from nemo_curator.audio_agent.recipe import EXECUTION_KNOB_PARAMS
    from nemo_curator.stages.audio._agent._agent_registry import build_contract, stage_params, static_contract

    cls = resolve_stage_class(ref)
    specs = stage_params(cls)
    accepted = tuple(spec.name for spec in specs)
    # Read once, not once per param: the condition of a comprehension is evaluated per item, so
    # this used to re-import the module and rebuild a frozenset from a frozenset for every key.
    given = {k: v for k, v in (params or {}).items() if k not in EXECUTION_KNOB_PARAMS}
    missing = tuple(spec.name for spec in specs if spec.required and spec.name not in given)

    try:
        instance = cls(**given)
    except Exception as exc:  # noqa: BLE001 - any constructor failure falls back, labelled
        return ResolvedContract(static_contract(cls), _fallback_reason(exc, missing), missing, accepted)
    try:
        return ResolvedContract(build_contract(instance), instance=instance)
    except Exception as exc:  # noqa: BLE001 - a stage that cannot describe itself is still listable
        reason = f"describe() failed on the configured stage: {type(exc).__name__}: {exc}"
        return ResolvedContract(static_contract(cls), reason, missing, accepted)
