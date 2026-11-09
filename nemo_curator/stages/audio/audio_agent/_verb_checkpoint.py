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

"""Explicit checkpoint planning and insertion."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from nemo_curator.stages.audio.audio_agent import _safety, _verb_delta, _verb_inputs, _verb_reuse

if TYPE_CHECKING:
    from nemo_curator.stages.audio.audio_agent.recipe import (
        Recipe,
    )


def add_checkpoint(
    recipe: Recipe | dict[str, Any],
    *,
    data: str | None = None,
    output_path: str | None = None,
    after: str | None = None,
) -> dict[str, Any]:
    """Where a mid-pipeline manifest would make the expensive work reusable, and the recipe for it.

    With ``data`` the checkpoint is addressed at its managed location and the recipe comes
    back carrying it, so nobody is asked to invent a path for a file the core can name
    itself. ``output_path`` overrides that for a user who wants the metadata somewhere of
    their own. With neither, this answers only where such a checkpoint may go and what a
    later run would then skip.

    The recipe returned is a recipe to look at, never written to disk and never run. Adding
    the checkpoint is a recipe change the user makes, so it goes back through ``validate`` ->
    confirm -> ``run`` like any other.

    Both halves of the position are simulated rather than assumed: a manifest cannot serialize a
    resident waveform, and the stages after it have to survive being handed a manifest. ``after``
    names a stage to place it behind instead, and is still checked against both.
    """
    from nemo_curator.stages.audio.audio_agent import checkpoint as _checkpoint

    rec = _verb_inputs._as_recipe(recipe).freeze()
    pviol = _safety.path_violations([data, output_path, *_safety.recipe_path_params(rec)])
    if pviol:
        return {
            "status": "refused",
            "reason": "path(s) resolve outside the allowed workspace",
            "violations": pviol,
        }
    spot, why = _checkpoint.advise(rec)
    if after:
        chosen = [i for i, s in enumerate(rec.stages) if s.ref == after]
        if not chosen:
            return {"status": "error", "reason": f"no stage named {after!r} in this recipe"}
        spot, why = _checkpoint.at(rec, index=chosen[-1] + 1)
    if spot is None:
        return {"status": "no_checkpoint", "reason": why, "advice": None}
    advice = spot.as_dict()
    # After the workspace check, so a path that violates the lock is never profiled.
    chosen_path, path_source = output_path, "explicit"
    if not chosen_path and data:
        chosen_path = _derived_checkpoint_path(rec, spot.index, _verb_reuse._recipe_dataset_key(rec, data))
        path_source = "derived"
    if not chosen_path:
        return {
            "status": "advice",
            "advice": advice,
            "next": "call again with data to address it automatically, or output_path to name it",
        }
    checkpointed, err = _checkpoint.insert(rec, index=spot.index, output_path=chosen_path)
    if checkpointed is None:
        return {"status": "error", "reason": err, "advice": advice}
    return _safety.redact(
        {
            "status": "ok",
            "advice": advice,
            "path_source": path_source,
            "output_path": chosen_path,
            "recipe": checkpointed.to_dict(),
            "next": "save this recipe, then validate -> smoke -> run it as usual",
        }
    )


def _derived_checkpoint_path(rec: Recipe, index: int, dataset_key: str) -> str:
    """The managed location for a writer at ``index``, or ``""`` when it cannot be derived.

    The step key is computed from a probe carrying a placeholder path, which is safe because
    ``output_path`` is an output-location param and so never reaches the key it is being used
    to compute. Returns ``""`` rather than raising: an underivable path falls back to asking
    for one, exactly as before this could be derived at all.
    """
    if not dataset_key:
        return ""
    try:
        from nemo_curator.stages.audio.audio_agent import checkpoint as _checkpoint
        from nemo_curator.stages.audio.audio_agent.artifacts import plan_steps
        from nemo_curator.stages.audio.audio_agent.run_store import checkpoint_path

        probe, _ = _checkpoint.insert(rec, index=index, output_path=_checkpoint._PROBE_PATH)
        if probe is None:
            return ""
        steps = plan_steps(probe, dataset_key)
        if not 0 <= index < len(steps):
            return ""
        return checkpoint_path(steps[index].step_key) or ""
    except Exception:  # noqa: BLE001 - an underivable path degrades to asking, never to a crash
        return ""


def plan_checkpoint(  # noqa: C901, PLR0911, PLR0912, PLR0913 - one refusal per safety proof
    recipe: Recipe | dict[str, Any] | None = None,
    *,
    from_run: str | None = None,
    data: str | None = None,
    output_path: str | None = None,
    decision_stage: str | None = None,
    decision_value: Any = None,  # noqa: ANN401 - a card-declared decision value
    decision_conditions: Any = None,  # noqa: ANN401 - complete card-declared compound surface
    choice: str | None = None,
    retention_sec: int = 0,
    owner: str = "user",
) -> dict[str, Any]:
    """Build complete, validated metadata-checkpoint candidates before authoritative smoke.

    This is additive to :func:`add_checkpoint`: the older advisor and its ordinary
    ``ManifestWriterStage`` recipe remain unchanged. This verb only handles a card-declared
    annotate/selector pair and uses ``ManifestCheckpointStage``.

    ``from_run`` is the safe feedback path: it adopts the exact successful recipe,
    verifies that its configured source (and optional ``data`` assertion) still has that
    run's strong dataset key, and lets the core change only the
    declared scalar selector value or complete compound selector conditions. A
    changed dataset is routed to the existing delta/fresh flow.
    """
    if recipe is not None and from_run:
        return {
            "status": "refused",
            "reason": "pass recipe for a first-run candidate or from_run for feedback, not both",
        }
    if choice not in {None, "checkpoint", "baseline"}:
        return {
            "status": "refused",
            "reason": "choice must be 'checkpoint', 'baseline', or omitted for inspection",
        }
    if choice == "baseline" and output_path:
        return {
            "status": "refused",
            "reason": "choice='baseline' cannot be combined with output_path",
        }
    early_pviol = _safety.path_violations([data, output_path])
    if early_pviol:
        return {
            "status": "refused",
            "reason": "path(s) resolve outside the allowed workspace",
            "violations": early_pviol,
        }
    adopted: dict[str, Any] | None = None
    if from_run:
        rec, adopted, refusal = _verb_delta._adopt_recipe(from_run)
        if refusal:
            return refusal
        rec.parent_run_id = from_run
    elif recipe is not None:
        rec = _verb_inputs._as_recipe(recipe).freeze()
    else:
        return {
            "status": "refused",
            "reason": "pass recipe for checkpoint planning or from_run to tune a completed run",
        }
    pviol = _safety.path_violations([data, output_path, *_safety.recipe_path_params(rec)])
    if pviol:
        return {
            "status": "refused",
            "reason": "path(s) resolve outside the allowed workspace",
            "violations": pviol,
        }
    # After the workspace check, so a path that violates the lock is never profiled. The key
    # decides both same-dataset feedback reuse and where a checkpoint is addressed.
    dataset_key = _verb_reuse._recipe_dataset_key(rec, data)
    dataset_state = "not_applicable"
    if adopted is not None:
        current_key = dataset_key
        prior_key = str(adopted.get("dataset_key") or "")
        if not prior_key or not current_key:
            return {
                "status": "dataset_identity_unproven",
                "route": "fresh",
                "reason": (
                    "same-dataset feedback reuse requires a recorded and currently resolvable "
                    "dataset key; use a fresh run when unchanged input cannot be proved"
                ),
                "prior_dataset_key": prior_key or None,
                "current_dataset_key": current_key or None,
                "adopted": adopted,
            }
        if current_key != prior_key:
            return {
                "status": "changed_dataset",
                "route": "delta_run",
                "reason": (
                    "the input dataset changed since this checkpoint run; threshold feedback "
                    "reuse is same-dataset only, so use the existing delta-run or fresh-run path"
                ),
                "prior_dataset_key": prior_key,
                "current_dataset_key": current_key,
                "adopted": adopted,
            }
        if not prior_key.startswith("stat:"):
            return {
                "status": "dataset_identity_unproven",
                "route": "fresh",
                "reason": (
                    "the matching dataset identity is shape-tier and cannot prove files were "
                    "unchanged; this feedback path requires a stat-tier identity"
                ),
                "prior_dataset_key": prior_key,
                "current_dataset_key": current_key,
                "adopted": adopted,
            }
        dataset_state = "verified_unchanged"
    if decision_value is not None:
        if not isinstance(decision_value, (bool, int, float, str)):
            return {
                "status": "refused",
                "reason": ("decision_value must be a JSON scalar (boolean, number, or string); null means no change"),
            }
        if isinstance(decision_value, float) and not math.isfinite(decision_value):
            return {
                "status": "refused",
                "reason": "decision_value must be a finite JSON number",
            }
    if decision_value is not None and decision_conditions is not None:
        return {
            "status": "refused",
            "reason": (
                "decision_value and decision_conditions are mutually exclusive; "
                "use decision_value for scalar decisions or decision_conditions "
                "for a complete compound decision"
            ),
        }
    if decision_conditions is not None and not isinstance(
        decision_conditions,
        (list, Mapping),
    ):
        return {
            "status": "refused",
            "reason": "decision_conditions must be a non-empty JSON list or mapping",
        }
    if not isinstance(retention_sec, int) or isinstance(retention_sec, bool) or retention_sec < 0:
        return {"status": "refused", "reason": "retention_sec must be a non-negative integer"}
    if owner not in {"user", "project"}:
        return {"status": "refused", "reason": "owner must be 'user' or 'project'"}

    from nemo_curator.stages.audio.audio_agent import reusable_pipeline

    result = reusable_pipeline.plan(
        rec,
        output_path=output_path,
        dataset_key=dataset_key,
        accept=choice == "checkpoint",
        decision_stage=decision_stage,
        decision_value=decision_value,
        decision_conditions=decision_conditions,
        retention_sec=retention_sec,
        owner=owner,
    )
    candidate_ids = reusable_pipeline.recommended_candidate_ids(result)
    if choice == "baseline":
        if not candidate_ids:
            return {
                "status": "refused",
                "reason": "there is no recommended checkpoint option to decline",
                "analysis": result,
            }
        baseline = reusable_pipeline.with_declined_checkpoint(rec, candidate_ids)
        result["status"] = "baseline_selected"
        result["checkpoint_decision_required"] = False
        result["checkpoint_decision"] = dict(baseline.checkpoint_decision or {})
        result["baseline"]["recipe"] = baseline.to_dict()
        result["host_directive"] = (
            "The user explicitly declined the recommended metadata checkpoint. "
            "Validate, critique, and smoke only the returned baseline recipe."
        )
    if adopted:
        result["adopted"] = adopted
    if decision_value is not None:
        result["feedback"] = {
            "decision_stage": decision_stage,
            "decision_value": decision_value,
            "dataset": dataset_state,
            "next": "reuse-scan the returned exact recipe; never execute it without a new smoke token",
        }
    elif decision_conditions is not None:
        result["feedback"] = {
            "decision_stage": decision_stage,
            "decision_conditions": decision_conditions,
            "dataset": dataset_state,
            "next": "reuse-scan the returned exact recipe; never execute it without a new smoke token",
        }
    return _safety.redact(result)
