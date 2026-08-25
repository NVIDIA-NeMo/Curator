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

"""Resolve a recipe's stages into the concrete stages a backend will actually run.

A :class:`~nemo_curator.stages.base.CompositeStage` describes itself and nothing else --
``SplitASRAlignJoinStage.describe()`` returns ``StageContract(wrappable=False)``, declaring no
reads and no writes -- while ``decompose()`` builds the three real stages underneath. Anything
that reasons about a recipe from contracts alone is therefore blind at exactly the stages that
do the work, and blind in a way that is invisible: the composite looks like a stage with no
requirements rather than a stage whose requirements are unknown.

That blindness has cost real runs. ``SplitLongAudioStage``, the first stage inside
``SplitASRAlignJoinStage``, requires a ``segments`` key. A pipeline that diarized into
``diar_segments`` validated clean, downloaded two models, ran diarization on the GPU, and only
then refused to start the splitter -- a failure the composite's own ``decompose()`` had spelled
out all along, including a comment about this precise mismatch.

Expanding the *configured* composite is what makes this exact rather than approximate: the
children carry the parameters the caller actually set, so ``SplitASRAlignJoinStage(
segments_key="diar_segments")`` yields ``SplitLongAudioStage(segments_key="diar_segments")`` and
the check reflects what will run rather than what the defaults would have run.

When a composite cannot be expanded -- it raises, returns nothing, returns something that is not
a stage, or returns another composite (which the executor itself refuses) -- no leaf is invented
for it. It is reported in :attr:`Expansion.opaque` so callers can fall back to whatever they did
before rather than reason from a fabricated stage list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nemo_curator.stages.base import CompositeStage, ProcessingStage


@dataclass(frozen=True)
class ExpandedStage:
    """One concrete stage the backend will run, and where in the recipe it came from."""

    recipe_index: int
    """Index of the recipe-level stage this came from -- what the caller wrote."""

    stage: Any
    """The configured, concrete (non-composite) stage instance."""

    path: tuple[int, ...] = ()
    """Child indices from the recipe stage down to this leaf. Empty for a top-level stage."""

    composite_ref: str | None = None
    """Class name of the recipe-level composite this was expanded from, if any."""

    @property
    def label(self) -> str:
        """How to name this stage to someone who only knows the recipe they wrote.

        A bare ``SplitLongAudioStage`` is not a stage the caller has ever heard of; they
        configured ``SplitASRAlignJoinStage``. Naming both ends keeps the report actionable.
        """
        name = type(self.stage).__name__
        return f"{self.composite_ref} -> {name}" if self.composite_ref else name


@dataclass(frozen=True)
class Expansion:
    """The concrete run order, plus the composites that refused to reveal theirs."""

    stages: list[ExpandedStage] = field(default_factory=list)
    opaque: dict[int, str] = field(default_factory=dict)
    """recipe_index -> why that composite could not be expanded."""

    unrunnable: dict[int, str] = field(default_factory=dict)
    """recipe_index -> why the executor will refuse this stage outright.

    Separate from :attr:`opaque` because the two demand opposite answers. Opaque means "we
    cannot tell" and degrades to a warning; this means "we can tell, and it will fail", which
    has to reach the caller as an error before they confirm a full-scale run.
    """

    @property
    def fully_resolved(self) -> bool:
        return not self.opaque and not self.unrunnable

    def by_recipe_index(self) -> dict[int, list[ExpandedStage]]:
        """Leaves grouped under the recipe stage that produced them, in run order.

        Callers walk the recipe so an opaque composite keeps its place in the sequence; the
        leaves it would have contributed are simply absent rather than reordered.
        """
        grouped: dict[int, list[ExpandedStage]] = {}
        for item in self.stages:
            grouped.setdefault(item.recipe_index, []).append(item)
        return grouped


def _nested_composite(child: Any) -> bool:  # noqa: ANN401 - any child stage
    """Whether a child is itself a composite the executor would refuse.

    Mirrors ``Pipeline._decompose_stages`` exactly: a ``CompositeStage`` whose ``decompose()``
    returns just itself is run as an ordinary stage, so only a genuinely decomposing one counts.
    """
    if not isinstance(child, CompositeStage):
        return False
    try:
        return len(child.decompose()) > 1
    except Exception:  # noqa: BLE001 - an undecomposable child is handled by the caller's checks
        return False


def _decompose(stage: Any) -> tuple[list[Any], str | None]:  # noqa: ANN401 - any composite
    """This composite's children, or the reason they cannot be trusted as a stage list."""
    try:
        children = list(stage.decompose_and_apply_with() or ())
    except Exception as exc:  # noqa: BLE001 - a composite that cannot plan-time decompose stays opaque
        return [], f"decompose() raised {type(exc).__name__}"
    if not children:
        return [], "decomposition produced no stages"
    alien = next((c for c in children if not isinstance(c, ProcessingStage)), None)
    if alien is not None:
        return [], f"decomposition returned {type(alien).__name__}, not a ProcessingStage"
    return children, None


def expand_composites(stages: list[Any]) -> Expansion:
    """Flatten composites into the stages a backend will run, preserving recipe order.

    Expansion is SINGLE-LEVEL, because that is the only shape the executor supports:
    ``Pipeline._decompose_stages`` expands each stage once and raises ``TypeError``
    ("Nested composition is not supported") if a child is itself a decomposing composite.
    Modelling deeper nesting here would be worse than not modelling it -- validation would
    approve a recipe the backend then refuses to run -- and no composite in the catalog is
    deeper than one level anyway.

    Uses ``decompose_and_apply_with()`` rather than ``decompose()`` so a composite configured
    through ``with_()`` contributes the same resource overrides the executor will see -- the
    call ``Pipeline._decompose_stages`` makes, so callers reason about the real schedule.

    A composite that cannot expand contributes no leaf and lands in :attr:`Expansion.opaque`.
    Inventing a leaf would be worse than admitting the gap: a caller that trusts a guessed stage
    list reports confident nonsense, whereas one that sees the gap can stay conservative.
    """
    out: list[ExpandedStage] = []
    opaque: dict[int, str] = {}
    unrunnable: dict[int, str] = {}

    for index, stage in enumerate(stages):
        if not isinstance(stage, CompositeStage):
            out.append(ExpandedStage(index, stage))
            continue
        children, reason = _decompose(stage)
        if reason is not None:
            opaque[index] = reason
            continue
        if len(children) == 1:
            # ``Pipeline._decompose_stages`` only substitutes children when there is more than
            # one, so a single-child decomposition leaves the COMPOSITE in the execution list --
            # and ``CompositeStage.process`` raises "should not be executed directly" on the
            # first task. Substituting the child here would model a pipeline the backend will
            # never run and hand the caller a clean verdict for a recipe that dies on contact.
            unrunnable[index] = (
                f"decomposes into a single stage ({type(children[0]).__name__}), which the "
                f"executor does not substitute -- it runs the composite itself and raises"
            )
            continue
        # After the length check, exactly as the executor orders it: the nested-composite
        # rejection lives inside its ``len(sub_stages) > 1`` branch and is never reached for a
        # single child. Asking first inverted the verdict for a composite that decomposes into
        # one decomposing composite -- reported as opaque, "we cannot tell", when the executor
        # can tell perfectly well that it will run the outer composite and raise.
        nested = next((c for c in children if _nested_composite(c)), None)
        if nested is not None:
            opaque[index] = (
                f"decomposition returned another composite ({type(nested).__name__}); "
                "nested composition is not supported"
            )
            continue
        ref = type(stage).__name__
        out.extend(ExpandedStage(index, child, (child_index,), ref) for child_index, child in enumerate(children))

    return Expansion(stages=out, opaque=opaque, unrunnable=unrunnable)
