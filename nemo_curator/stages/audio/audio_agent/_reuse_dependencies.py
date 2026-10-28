# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bound reuse to file inputs covered by the existing source/asset identities."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_curator.stages.audio._agent._agent_ready import IOSpec
    from nemo_curator.stages.audio.audio_agent.input_identity import DatasetBinding
    from nemo_curator.stages.audio.audio_agent.recipe import Recipe


def _source_alias(binding: DatasetBinding, key: str) -> bool:
    """Prove a selected manifest column names the same files as the profiled column."""
    from nemo_curator.stages.audio.audio_agent._safety import path_violations
    from nemo_curator.stages.audio.common.manifest_reader import _row_names_file

    path = binding.profile_source
    primary = binding.profile_kwargs.get("audio_filepath_key", "audio_filepath")
    if not path or not path.endswith((".json", ".jsonl")) or path_violations([path]):
        return False
    if binding.profile_kwargs.get("audio_path_resolution") not in (None, "as_is"):
        return False
    selected = binding.profile_kwargs.get("include_files")
    wanted = None if selected is None else {os.path.abspath(os.path.expanduser(p)) for p in selected}
    seen = False
    try:
        with open(path, encoding="utf-8") as stream:
            for raw in stream:
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if not isinstance(row, dict):
                    return False
                if wanted is not None and not _row_names_file(row, primary, wanted):
                    continue
                # Exact spellings suffice without interpreting cwd/remote URI
                # semantics or introducing another path-resolution convention.
                if not isinstance(row.get(primary), str) or row.get(key) != row[primary]:
                    return False
                seen = True
    except (OSError, UnicodeError, ValueError):
        return False
    return seen


def _selected_reads(alternatives: list[IOSpec], auto: bool, data: set[str], segments: set[str]) -> list[IOSpec]:
    """The standard auto route prefers a guaranteed complete resident pair."""
    if auto:
        resident = [
            spec
            for spec in alternatives
            if "waveform" in spec.accepts and set(spec.data_keys) <= data and set(spec.segment_data_keys) <= segments
        ]
        if resident:
            return resident
    return alternatives


def untracked_audio_input(recipe: Recipe) -> tuple[int, str] | None:  # noqa: C901, PLR0911, PLR0912, PLR0915 - scoped coverage
    """First consumer of an external file absent from the source fingerprint.

    Per-row independence does not prove that a row's *other* files are unchanged.
    Until those files have an identity protocol, fresh execution is supported but
    neither whole-result nor changed-file reuse may skip their consumer.
    """
    from nemo_curator.stages.audio import agent as foundation
    from nemo_curator.stages.audio._agent._composite import expand_composites
    from nemo_curator.stages.audio.audio_agent.input_identity import resolve_dataset_binding
    from nemo_curator.stages.audio.audio_agent.recipe import build_stages

    binding = resolve_dataset_binding(recipe)
    primary = binding.profile_kwargs.get("audio_filepath_key", "audio_filepath")
    covered = {primary}
    segments: set[str] = set()
    guaranteed = {primary}
    guaranteed_segments: set[str] = set()
    absent: set[str] = set()
    absent_segments: set[str] = set()
    built, errors = build_stages(recipe)
    if errors or len(built) != len(recipe.stages):
        from nemo_curator.stages.audio.audio_agent.recipe import Recipe

        for index, stage in enumerate(recipe.stages):
            _, issues = build_stages(Recipe(stages=[stage]))
            if issues:
                return index, "configured input dependencies could not be resolved"
        return 0, "configured input dependencies could not be resolved"
    expansion = expand_composites(built)
    unresolved = {**expansion.opaque, **expansion.unrunnable}
    for leaf in expansion.stages:
        if unresolved and leaf.recipe_index >= min(unresolved):
            return min(unresolved), "composite input dependencies could not be resolved"
        if leaf.recipe_index == 0:
            continue
        try:
            contract = foundation.build_contract(leaf.stage)
        except Exception:  # noqa: BLE001 - unreadable I/O is unproven, not an empty dependency set
            return leaf.recipe_index, "configured input dependencies could not be resolved"

        auto = getattr(leaf.stage, "input_residency", None) == "auto"
        reads = [
            contract.reads,
            contract.optional_reads,
            *_selected_reads(contract.reads_one_of, auto, guaranteed, guaranteed_segments),
        ]
        reads.extend(
            spec
            for branch in contract.conditional_reads
            if not set(branch.forbids_keys) & guaranteed and not set(branch.requires_keys) & absent
            for spec in _selected_reads(branch.reads_one_of, auto, guaranteed, guaranteed_segments)
        )
        for spec in reads:
            for keys, known, removed in (
                (spec.data_keys, covered, absent),
                (spec.segment_data_keys, segments, absent_segments),
            ):
                missing = [
                    k
                    for k in keys
                    if contract.key_roles.get(k) == "audio_filepath"
                    and k not in known
                    and not (spec is contract.optional_reads and k in removed)
                ]
                if known is covered:
                    missing = [key for key in missing if not _source_alias(binding, key)]
                if missing:
                    return leaf.recipe_index, (
                        f"{type(leaf.stage).__name__} reads audio file key(s) {sorted(missing)} "
                        "not covered by the source fingerprint or generated-asset provenance; run this suffix fresh"
                    )
        if not contract.preserves_upstream_keys:
            absent.update(guaranteed - set(contract.writes.data_keys))
            covered.intersection_update(contract.writes.data_keys)
            guaranteed.clear()
        if not contract.preserves_upstream_keys or not contract.preserves_upstream_segment_keys:
            absent_segments.update(guaranteed_segments - set(contract.writes.segment_data_keys))
            segments.intersection_update(contract.writes.segment_data_keys)
            guaranteed_segments.clear()
        for known in (covered, segments):
            # Invalidating an audio *role* does not change a retained path's
            # file identity (it may still be consulted for provenance).
            known.difference_update(contract.removes_keys)
        guaranteed.difference_update(
            [*contract.removes_keys, *contract.invalidates_keys, *contract.invalidates_prior_writes.data_keys]
        )
        guaranteed_segments.difference_update(
            [*contract.removes_keys, *contract.invalidates_keys, *contract.invalidates_prior_writes.segment_data_keys]
        )
        guaranteed.update(contract.writes.data_keys)
        guaranteed_segments.update(contract.writes.segment_data_keys)
        absent.update(contract.removes_keys)
        absent_segments.update(contract.removes_keys)
        for spec in [contract.writes, *(branch.writes for branch in contract.conditional_writes)]:
            absent.difference_update(spec.data_keys)
            absent_segments.difference_update(spec.segment_data_keys)
        # Only guaranteed, persisted writes establish new file carriers. Merely
        # copying an external path into a row does not give that file an identity.
        if contract.gates.writes_to_disk:
            covered.update(k for k in contract.writes.data_keys if contract.key_roles.get(k) == "audio_filepath")
            segments.update(
                k for k in contract.writes.segment_data_keys if contract.key_roles.get(k) == "audio_filepath"
            )
    if unresolved:
        first = min(unresolved)
        return first, "composite input dependencies could not be resolved"
    return None
