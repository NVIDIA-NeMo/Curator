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

"""Where a mid-pipeline manifest would make the expensive work reusable.

Only a stage naming an output location publishes an artifact, and reuse resumes from disk --
so a pipeline whose GPU stages hand their results to the next stage in memory can never
resume past them. The ALM pipeline is the shape of the problem: resampling and splitting
persist because they were configured with directories, while diarization and ASR persist
nothing, leaving reuse able to skip only the cheap half.

The fix is one manifest written after the expensive stages. That is an ordinary
``ManifestWriterStage`` in the user's recipe writing to a path they chose, not a hidden
cache: caching intermediate state nobody asked for would buy back seconds at the price of an
eviction policy, a staleness window, and a new way to serve wrong data silently (see
``reuse._persist_offer``). This module only says *where* such a writer may go and why.

Both constraints are derived, never listed:

* A manifest serializes ``task.data`` as-is, so a resident waveform tensor crashes it.
  ``validate_pipeline`` already models tensor residency, key removal and ``sanitizes_output``
  and raises ``tensor_into_sink``, so legality is answered by inserting the writer and asking
  the same validator a real recipe is judged by.
* The stages below must survive being handed a manifest, which is exactly what
  ``continuation._resume_breaks_on_disk_boundary`` simulates -- the dropped waveform, and the
  ``task._metadata`` keys a manifest does not carry either.

A stage written next year therefore participates without anything here being updated: it
either satisfies both simulations or it moves the answer, and no position is ever assumed.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_curator.audio_agent.recipe import Recipe

_WRITER_REF = "ManifestWriterStage"
_WRITER_PARAM = "output_path"
# Only used to make a candidate recipe valid enough to validate; never advised to a user.
_PROBE_PATH = "/checkpoint-probe/checkpoint.jsonl"


@dataclass(frozen=True)
class Spot:
    """A legal, worthwhile place to write a checkpoint."""

    index: int
    after_stage: str
    skips: list[str]
    not_earlier: str = ""

    def as_dict(self) -> dict[str, Any]:
        out = {
            "action": "add_checkpoint",
            "after_stage": self.after_stage,
            "at_index": self.index,
            "skips_on_reuse": list(self.skips),
            "how": (
                f"add a {_WRITER_REF} after {self.after_stage} (recipe index {self.index}); "
                f"pass data to have its location derived"
            ),
            "effect": (
                "a later request that starts the same way resumes from that manifest instead of "
                f"recomputing {', '.join(self.skips)}"
                if self.skips
                else "a later request that starts the same way resumes from that manifest, though "
                "nothing expensive sits above it, so it saves little"
            ),
        }
        if self.not_earlier:
            out["why_not_earlier"] = self.not_earlier
        return out


def advise(recipe: Recipe) -> tuple[Spot | None, str]:
    """``(spot, "")`` where a checkpoint would pay for itself, else ``(None, why not)``.

    Placed at the first legal position *after the last expensive stage* rather than as deep as
    possible. Depth alone would put the writer immediately before the final one, duplicating it
    to save nothing; and the shallowest position that clears the expensive work leaves the most
    of the tail still editable, which is where a checkpoint earns its keep -- a tweak below the
    checkpoint reuses it, a tweak above it cannot.
    """
    stages = list(recipe.stages)
    if len(stages) < 2:  # noqa: PLR2004 - a checkpoint needs something on each side of it
        return None, "a checkpoint needs at least one stage on each side of it"
    baseline = _error_codes(recipe)
    if baseline is None:
        return None, "this recipe does not build, so no position can be checked"

    last_costly = _last_costly(stages)
    if last_costly is None:
        return None, "nothing before the final stage is expensive enough for a checkpoint to pay for itself"

    # A writer already standing at or past the expensive work is the checkpoint -- including the
    # expensive stage itself, when it was configured with an output location. The pipeline's own
    # final sink is not: resuming from it means the whole request was already done, which does
    # nothing for the case this exists to serve -- a changed tail, where the final artifact no
    # longer matches and the GPU stages are recomputed to reach it.
    for index in range(last_costly, len(stages) - 1):
        already = _resume_point(stages[index])
        if already:
            return None, already

    blocked = ""
    for index in range(last_costly + 1, len(stages)):
        why = _rejection(recipe, index, baseline)
        if not why:
            spot = Spot(
                index=index,
                after_stage=stages[index - 1].ref,
                skips=_skips(stages, index),
                not_earlier=blocked,
            )
            return spot, ""
        blocked = blocked or f"no earlier position works: after {stages[index - 1].ref}, {why}"
    return None, f"nowhere after {stages[last_costly].ref} can hold a manifest -- {blocked}"


def at(recipe: Recipe, *, index: int) -> tuple[Spot | None, str]:
    """A caller's own position, judged by the checks :func:`advise` applies to its own.

    A user may want the checkpoint above the stage they are about to tune rather than as deep
    as it will go, which is a preference and not an error -- so a position that saves nothing
    expensive is returned and says so, while one that cannot physically work is refused.
    """
    stages = list(recipe.stages)
    if not 0 < index < len(stages):
        return None, f"a checkpoint at index {index} would have no stage on one side of it"
    baseline = _error_codes(recipe)
    if baseline is None:
        return None, "this recipe does not build, so no position can be checked"
    why = _rejection(recipe, index, baseline)
    if why:
        return None, f"a checkpoint after {stages[index - 1].ref} does not work: {why}"
    return Spot(index=index, after_stage=stages[index - 1].ref, skips=_skips(stages, index)), ""


def _skips(stages: list[Any], index: int) -> list[str]:
    """The expensive stages a resume from ``index`` would not repeat."""
    return [s.ref for i, s in enumerate(stages) if i < index and _costly(s.ref)]


def insert(recipe: Recipe, *, index: int, output_path: str) -> tuple[Recipe | None, str]:
    """``(recipe, "")`` with a manifest checkpoint at ``index``, else ``(None, error)``.

    An ordinary recipe out: it goes through the same validate -> confirm -> run path as
    anything else, so a checkpoint buys no shortcut past the safety gates.
    """
    from nemo_curator.audio_agent.recipe import Recipe as RecipeType
    from nemo_curator.audio_agent.recipe import StageRef

    if not output_path:
        return None, "a checkpoint needs an output path to write to"
    if not 0 < index <= len(recipe.stages):
        return None, f"index {index} is outside the recipe ({len(recipe.stages)} stages)"
    stages = [StageRef(ref=s.ref, params=dict(s.params)) for s in recipe.stages]
    stages.insert(index, StageRef(ref=_WRITER_REF, params={_WRITER_PARAM: output_path}))
    checkpointed = RecipeType(
        stages=stages,
        inputs=dict(recipe.inputs),
        preset=recipe.preset,
        acceptance_criteria=list(recipe.acceptance_criteria),
        rationale=recipe.rationale,
        name=recipe.name,
        knowledge_version=recipe.knowledge_version,
        parent_run_id=recipe.parent_run_id,
        planning_preference=(
            dict(recipe.planning_preference) if isinstance(recipe.planning_preference, dict) else None
        ),
    )
    return checkpointed.freeze(), ""


def _rejection(recipe: Recipe, index: int, baseline: collections.Counter[str]) -> str:
    """Why a manifest cannot go at ``index``, or ``""`` when it can.

    Two questions, both answered by simulation: does writing here work, and can the rest of the
    run start from what was written.
    """
    from nemo_curator.audio_agent.continuation import _resume_breaks_on_disk_boundary

    candidate, err = insert(recipe, index=index, output_path=_PROBE_PATH)
    if candidate is None:
        return err or "a manifest cannot be placed here"
    codes = _error_codes(candidate)
    if codes is None:
        return "the recipe with a manifest here does not build"
    added = codes - baseline
    if "tensor_into_sink" in added:
        return "the pipeline is still carrying audio in memory and a manifest cannot hold it"
    if added:
        return "writing here breaks the pipeline (" + ", ".join(sorted(added)) + ")"
    # The writer becomes stage ``index``, so a resume reuses everything through it.
    broken = _resume_breaks_on_disk_boundary(candidate, index + 1)
    return f"the stages below would lose state a manifest cannot carry ({broken})" if broken else ""


def _error_codes(recipe: Recipe) -> collections.Counter[str] | None:
    """Validation error codes for a recipe, counted; ``None`` when it will not build.

    Counted rather than collected as ``(stage, code)`` pairs because inserting a second writer
    duplicates a stage *name*, and a pair keyed on the name cannot then tell the new error from
    the old one.
    """
    try:
        from nemo_curator.audio_agent.recipe import build_stages
        from nemo_curator.stages.audio import agent as foundation

        built, _issues = build_stages(recipe)
        if not built:
            return None
        report = foundation.validate_pipeline(built)
        return collections.Counter(i.code for i in report.issues if i.severity == "error")
    except Exception:  # noqa: BLE001 - an unanalysable recipe simply gets no advice
        return None


def _costly(stage_ref: str) -> bool:
    from nemo_curator.audio_agent.artifacts import stage_is_costly

    return stage_is_costly(stage_ref)


def _last_costly(stages: list[Any]) -> int | None:
    """Index of the deepest stage the cards call expensive, not counting the final one.

    The final stage is excluded so that a candidate position exists in front of it: a checkpoint
    is only ever placed *after* the deepest expensive stage, and counting the last one would
    leave nowhere to put it. That also keeps an uncarded final stage -- costly by default,
    because ``stage_is_costly`` fails towards asking -- from suppressing the answer.
    """
    for index in range(len(stages) - 2, -1, -1):
        if _costly(stages[index].ref):
            return index
    return None


def _resume_point(stage: Any) -> str:  # noqa: ANN401 - StageRef, kept structural
    """Why this stage's own output already serves as a checkpoint, or ``""`` when it does not.

    Writing a file is not the same as being resumable from. ``InferenceSortformerStage`` fills
    an ``rttm_out_dir`` and no source stage can re-read an RTTM directory as a pipeline input
    (``continuation._SOURCE_FOR_KIND``), so treating that as cover would answer "you already
    have a checkpoint" to a user whose diarization is recomputed every time.

    An output path that does not exist yet cannot be sampled, so a directory a stage is about
    to fill reads as ``"unknown"``. That still counts -- advising a second writer beside one the
    user already configured is the worse mistake -- but the sentence says what is conditional
    about it rather than promising a resume that depends on what lands there.
    """
    from nemo_curator.audio_agent.artifacts import output_uri

    uri, kind = output_uri(stage)
    if not uri:
        return ""
    if kind in {"manifest", "audio_dir"}:
        return f"{stage.ref} already writes a {kind.replace('_', ' ')} to {uri!r}, so a resume can start from there"
    if kind == "unknown":
        return (
            f"{stage.ref} already writes to {uri!r}, and a resume can start from there if that output "
            f"is a manifest or a directory of audio"
        )
    return ""
