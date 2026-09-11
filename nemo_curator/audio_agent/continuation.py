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

"""Incremental continuation planner — reuse prior work for a follow-up request.

Deterministic recipe diff against a parent :class:`RunRecord`. Reuse is only
claimed where it is provably safe:

* **already_done** — the new recipe is identical to the parent's: reuse the parent
  output as-is.
* **incremental** — the parent's stages are an exact PREFIX of the new recipe (the
  request only *appends*, e.g. "also add transcripts"): reuse the parent's final
  output and run only the appended suffix.
* **full_rerun** — any divergence *within* the shared range (a changed param, a
  removed/reordered stage) or a different source dataset: there is no persisted
  intermediate to reuse from, so honestly rerun. The divergence point is reported.

Reuse additionally requires the SAME source data (a matching ``data_fingerprint``),
so "add transcripts to this dataset" never silently reuses another dataset's output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_curator.audio_agent.contracts import RunRecord
    from nemo_curator.audio_agent.recipe import Recipe


# How to re-enter a pipeline from a persisted artifact: the source stage that reads each
# artifact kind, and the param naming the location.
_SOURCE_FOR_KIND: dict[str, tuple[str, str]] = {
    "manifest": ("ManifestReader", "manifest_path"),
    "audio_dir": ("CreateInitialManifestAudioFolderStage", "data_dir"),
}
_SOURCE_ASSERTION_KEYS = frozenset({"manifest_path", "input_manifest", "audio_dir", "raw_data_dir", "data_dir"})


def _stage_dicts(recipe: Recipe) -> list[dict[str, Any]]:
    return [s.to_dict() for s in recipe.stages]


def materialize(new_recipe: Recipe, *, uri: str, kind: str, prefix: int) -> tuple[Recipe | None, str]:
    """Rewrite ``new_recipe`` to start from a persisted artifact: ``(recipe, error)``.

    Drops the first ``prefix`` stages (the work being reused) and prepends a source stage
    that reads ``uri``. The result is an ordinary recipe -- it goes through the same
    ``validate`` -> confirm -> ``run`` path as anything else, so reuse buys no shortcut past
    the safety gates. Everything downstream is left exactly as the user configured it.
    """
    from nemo_curator.audio_agent.recipe import Recipe, StageRef

    if prefix <= 0 or prefix > len(new_recipe.stages):
        return None, f"reuse point {prefix} is outside the recipe ({len(new_recipe.stages)} stages)"
    source = _SOURCE_FOR_KIND.get(kind)
    if source is None:
        return None, (
            f"no source stage can re-read a {kind!r} artifact; reuse would need the pipeline to start "
            f"from a manifest or an audio directory"
        )
    ref, param = source
    suffix = [StageRef(ref=s.ref, params=dict(s.params)) for s in new_recipe.stages[prefix:]]
    if not suffix:
        return None, "nothing left to run after the reuse point (this is an 'already done' case)"
    # Recipe.inputs is assertion metadata, not parameter injection. A continued
    # recipe physically reads the verified artifact, so carrying the original
    # corpus assertion unchanged would make its own metadata contradict its
    # source stage. Keep unrelated/output metadata and bind the physical input;
    # the original corpus identity travels separately as verified lineage.
    execution_inputs = {key: value for key, value in new_recipe.inputs.items() if key not in _SOURCE_ASSERTION_KEYS}
    execution_inputs[param] = uri
    materialized = Recipe(
        stages=[StageRef(ref=ref, params={param: uri}), *suffix],
        inputs=execution_inputs,
        preset=new_recipe.preset,
        acceptance_criteria=list(new_recipe.acceptance_criteria),
        rationale=new_recipe.rationale,
        name=f"{new_recipe.name}_continued",
        knowledge_version=new_recipe.knowledge_version,
        parent_run_id=new_recipe.parent_run_id,
        planning_preference=(
            dict(new_recipe.planning_preference) if isinstance(new_recipe.planning_preference, dict) else None
        ),
    )
    return materialized.freeze(), ""


def _common_prefix_len(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> int:
    """Number of leading stages identical (ref + params) in both recipes."""
    n = 0
    for sa, sb in zip(a, b, strict=False):
        if sa != sb:
            break
        n += 1
    return n


def _resume_breaks_on_disk_boundary(new_recipe: Recipe, prefix: int) -> str | None:
    """What the appended suffix needs that the parent's *persisted* (on-disk) output cannot
    carry, or ``None`` when resuming from disk is safe.

    Incremental reuse resumes the suffix from the parent's persisted output -- a manifest on
    disk, which cannot carry an in-memory ``waveform`` tensor. This re-validates *only the
    suffix*, seeded with the roles the parent produced, WITH vs WITHOUT the non-serializable
    ``waveform`` role. Only reads that break *specifically* because the waveform is dropped at
    that boundary are reported: reads that fail for other reasons (or are satisfied by
    pass-through columns) fail/pass in both runs and cancel out, so a legitimate incremental
    is never downgraded, and a suffix that reloads audio from file (``audio_filepath``
    survives) is correctly allowed. Best-effort: any internal error -> ``None`` (keep reuse).

    A manifest holds ``task.data``, so ``task._metadata`` does not survive the boundary either
    -- and unlike a missing waveform, nothing downstream raises. ``SegmentConcatenationStage``
    parks its ``segment_mappings`` there and ``TimestampMapperStage`` reads them; the pretrain
    planners park per-original counters that ``PretrainMetricsAggregatorStage`` reads. Resuming
    across either pair would hand the reader an empty dict and finish successfully with
    silently wrong timestamps or counts, which is the failure mode a guard is for.

    Underscored but NOT private: imported by ``checkpoint``, ``delta``, ``reusable_pipeline``
    and ``reuse``, which all have to ask the same question before resuming a suffix from a
    manifest. Moving, renaming or narrowing it breaks four modules, and none of them will say
    so until the boundary they were guarding is already crossed.
    """
    try:
        from nemo_curator.audio_agent.recipe import build_stages
        from nemo_curator.stages.audio import agent as foundation
        from nemo_curator.stages.audio._agent._conformance import produced_roles
        from nemo_curator.stages.audio._agent._roles import role_for_value

        built, _ = build_stages(new_recipe)
        if not built or not 0 < prefix < len(built):
            return None
        parent_built, suffix_built = built[:prefix], built[prefix:]

        roles: set[str] = {"audio_filepath"}
        keys: set[str] = {"audio_filepath"}
        for st in parent_built:
            c = foundation.build_contract(st)
            roles |= produced_roles(c)
            keys |= set(c.writes.data_keys) | set(c.writes.segment_data_keys)

        def _errs(initial_roles: set[str], initial_keys: set[str]) -> set[tuple[str, str]]:
            rep = foundation.validate_pipeline(suffix_built, initial_roles=initial_roles, initial_keys=initial_keys)
            return {(i.stage_name, i.code) for i in rep.issues if i.severity == "error"}

        # Persisting to disk drops the waveform ITSELF, not merely the role it was filed under, so
        # the boundary has to be modelled on both. Withholding only the role leaves the carrier key
        # visible, and a suffix stage reading that key by name is then judged satisfied -- the
        # simulation quietly stops simulating anything and every waveform suffix looks resumable.
        waveform_keys = {k for k in keys if role_for_value(k) == "waveform"}
        new_breaks = _errs(roles - {"waveform"}, keys - waveform_keys) - _errs(roles, keys)
        reasons = []
        if new_breaks:
            reasons.append("waveform needed by " + ", ".join(sorted({name for name, _ in new_breaks})))
        dropped = _metadata_lost_across(parent_built, suffix_built)
        if dropped:
            reasons.append(dropped)
        task_id_consumers = sorted(
            {
                type(stage).__name__
                for stage in suffix_built
                if foundation.build_contract(stage).gates.requires_stable_task_id
            }
        )
        if task_id_consumers:
            reasons.append(
                "stable framework task.task_id does not survive a metadata manifest "
                "and is required for durable output identity by " + ", ".join(task_id_consumers)
            )
        return "; ".join(reasons) or None
    except Exception:  # noqa: BLE001 - resume-safety is best-effort; never block reuse on a guard error
        return None


def _metadata_lost_across(parent_built: list[Any], suffix_built: list[Any]) -> str:
    """``task._metadata`` keys the suffix reads that only the dropped prefix produced.

    A suffix stage that re-writes the key before reading it is unaffected, so production inside
    the suffix is tracked in order -- otherwise a self-contained pretrain tail, which plans and
    then aggregates its own counters, would be refused for needing something it makes itself.
    """
    from nemo_curator.stages.audio import agent as foundation

    produced: dict[str, str] = {}
    for stage in parent_built:
        for key in foundation.build_contract(stage).metadata_writes:
            produced.setdefault(key, type(stage).__name__)
    lost: list[str] = []
    available: set[str] = set()
    for stage in suffix_built:
        contract = foundation.build_contract(stage)
        for key in contract.metadata_reads:
            if key in produced and key not in available:
                lost.append(f"{key!r} (from {produced[key]}) needed by {type(stage).__name__}")
        available |= set(contract.metadata_writes)
    return "task metadata does not survive a manifest: " + "; ".join(lost) if lost else ""


def _source_changed(parent: RunRecord, *, data_fingerprint: str | None, dataset_key: str | None) -> bool:
    """True when the source data is provably different from the parent run's.

    Prefers the tiered ``dataset_key`` (size+mtime, so an in-place edit is caught) and falls
    back to the legacy shape fingerprint for records written before it existed. Unknown on
    either side is not "changed" -- the caller simply gets no same-data evidence.
    """
    parent_key = getattr(parent, "dataset_key", None)
    if dataset_key and parent_key:
        return dataset_key != parent_key
    return bool(data_fingerprint and parent.data_fingerprint and data_fingerprint != parent.data_fingerprint)


def plan_continuation(
    new_recipe: Recipe,
    parent: RunRecord,
    *,
    data_fingerprint: str | None = None,
    dataset_key: str | None = None,
) -> dict[str, Any]:
    """Compute the incremental execution plan for ``new_recipe`` given a ``parent`` run."""
    from nemo_curator.audio_agent import _safety

    parent_stages = list(parent.recipe.get("stages") or [])
    # ``verbs.run`` persists a *redacted* recipe, so redact the new stages with the same
    # policy before diffing -- otherwise a secret-named param compares masked-vs-real and
    # forces a needless full_rerun on every follow-up. (refs are never secret, so
    # run/reuse-stage lists are unaffected.)
    new_stages = [_safety.redact(s, redact_transcripts=False) for s in _stage_dicts(new_recipe)]
    parent_refs = [s.get("ref") for s in parent_stages]
    new_refs = [s.get("ref") for s in new_stages]

    # Same-data guard: reuse is invalid if the source dataset changed.
    if _source_changed(parent, data_fingerprint=data_fingerprint, dataset_key=dataset_key):
        return {
            "mode": "full_rerun",
            "parent_run_id": parent.run_id,
            "reason": "source dataset identity changed since the parent run; nothing can be reused",
            "run_stages": new_refs,
        }

    # Identical recipe: prefer the canonical config_hash (computed from the *unredacted*
    # recipe, so it is exact and redaction-proof); fall back to the stage-diff for older
    # records that predate config_hash. The data guard above already matched the source.
    parent_hash = getattr(parent, "config_hash", None)
    new_hash = getattr(new_recipe, "config_hash", None)
    if (new_hash and parent_hash and new_hash == parent_hash) or new_stages == parent_stages:
        return {
            "mode": "already_done",
            "parent_run_id": parent.run_id,
            "reuse_from": list(parent.output_paths),
            "reuse_stages": parent_refs,
            "run_stages": [],
            "rationale": (
                "identical recipe with a matching dataset identity; reuse the parent output as-is (nothing to run)"
            ),
        }

    prefix = _common_prefix_len(parent_stages, new_stages)
    if prefix == len(parent_stages) and len(new_stages) > len(parent_stages):
        suffix = new_stages[prefix:]
        # Resume-safety: the reuse point is the parent's *persisted* output (a manifest on
        # disk), which can't carry an in-memory waveform. If dropping the waveform at that
        # boundary breaks the appended suffix (it needs a resident waveform no suffix stage
        # reloads from file), resuming would silently fail -> honest full_rerun instead.
        lost = _resume_breaks_on_disk_boundary(new_recipe, prefix)
        if lost:
            return {
                "mode": "full_rerun",
                "parent_run_id": parent.run_id,
                "diverged_at": prefix,
                "reason": (
                    f"the appended stage(s) need in-memory state the parent's persisted output cannot "
                    f"carry ({lost}); resuming from disk would drop it, so rerun to regenerate it "
                    f"(for audio, persist it in the parent: keep_waveform_in_task=False + write_to_disk)"
                ),
                "run_stages": new_refs,
            }
        return {
            "mode": "incremental",
            "parent_run_id": parent.run_id,
            "reuse_stages": parent_refs,
            "run_stages": [s.get("ref") for s in suffix],
            "reuse_from": list(parent.output_paths),
            "rationale": (
                f"the new recipe extends the parent by {len(suffix)} stage(s); reuse the parent's output "
                f"as input and run only the appended stage(s)"
            ),
        }

    return {
        "mode": "full_rerun",
        "parent_run_id": parent.run_id,
        "diverged_at": prefix,
        "reason": (
            f"recipes diverge at stage index {prefix} "
            f"(changed/removed/reordered stage); no persisted intermediate exists to reuse from"
        ),
        "run_stages": new_refs,
    }
