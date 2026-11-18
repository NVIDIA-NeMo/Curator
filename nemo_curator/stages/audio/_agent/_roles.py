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

"""Semantic-role mapping for agent-ready audio stages.

Under the config-knobs-only standardization, every ``task.data`` key a stage
reads or writes is an agent-configurable ``*_key`` constructor field. The key's
*value* can therefore be renamed per pipeline, but the *field name* (e.g.
``score_key``) is invariant — it is what the producer/consumer code is written
around. We map that invariant field name to a stable semantic
:data:`~nemo_curator.stages.audio._agent._agent_ready.Role`, so an agent can chain a
producer's output to a consumer's input by *role* even when key values differ.

``KEY_ROLES`` covers cross-stage composable keys. ``INTERNAL_KEY_FIELDS`` lists
``*_key`` constructor fields that are intentionally stage-internal (no
cross-stage role; chained by value-equality within a tightly coupled pair).
``LITERAL_KEY_ROLES`` maps hard-coded key *values* emitted by producers that do
not expose a ``*_key`` field (e.g. ``ManifestReaderStage`` writes ``"audio_filepath"``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nemo_curator.stages.audio._agent._agent_ready import Role

# field name (e.g. "score_key") -> semantic role
KEY_ROLES: dict[str, Role] = {
    # audio file path (all variants resolve to the same role)
    "audio_filepath_key": "audio_filepath",
    "filepath_key": "audio_filepath",
    "resampled_audio_filepath_key": "audio_filepath",
    "original_audio_filepath_key": "audio_filepath",
    "output_audio_filepath_key": "audio_filepath",
    "swift_audio_filepath_key": "audio_filepath",
    # in-memory audio
    "waveform_key": "waveform",
    "sample_rate_key": "sample_rate",
    "audio_sample_rate_key": "sample_rate",
    # duration / size
    "duration_key": "duration",
    "duration_ms_key": "duration",
    "total_duration_sec_key": "duration",
    "speaking_duration_key": "duration",
    "num_samples_key": "num_samples",
    # segment lists
    "segments_key": "segments",
    "diar_segments_key": "diar_segments",
    "vad_segments_key": "vad_segments",
    "overlap_segments_key": "overlap_segments",
    # per-segment timing
    "start_key": "start",
    "end_key": "end",
    "start_ms_key": "start_ms",
    "end_ms_key": "end_ms",
    "original_start_ms_key": "start_ms",
    "original_end_ms_key": "end_ms",
    "segment_num_key": "segment_num",
    "original_file_key": "original_file",
    "audio_item_id_key": "item_id",
    # text
    "text_key": "text",
    "hypothesis_text_key": "text",
    "pred_text_key": "pred_text",
    "reference_text_key": "reference_text",
    "words_key": "words",
    "alignment_key": "alignment",
    # speaker
    "speaker_key": "speaker_id",
    "speaker_id_key": "speaker_id",
    "num_speakers_key": "num_speakers",
    # quality / metric scores (incl. SIGMOS sub-scores and WER)
    "score_key": "score",
    "metrics_key": "metrics",
    "stats_key": "metrics",
    "prediction_key": "prediction",
    "wer_key": "score",
    "sig_key": "score",
    "ovrl_key": "score",
    "noise_key": "score",
    "disc_key": "score",
    "reverb_key": "score",
    "loud_key": "score",
    "col_key": "score",  # SIGMOS coloration sub-score — a peer of its 6 siblings
    # windows (ALM snippet planning)
    "windows_key": "windows",
    "filtered_windows_key": "windows",
}

# ``*_key`` constructor fields with no cross-stage role (chained by value within
# a tightly coupled stage pair, or fully generic / user-defined). Listed so the
# conformance check does not flag them as "forgot a KEY_ROLES entry".
INTERNAL_KEY_FIELDS: frozenset[str] = frozenset(
    {
        # SplitLongAudio <-> JoinSplitAudioMetadata bookkeeping (value-matched pair)
        "split_filepaths_key",
        "split_metadata_key",
        "split_offsets_key",
        "split_timestamps_key",
        "mappings_key",
        # generic / user-defined targets
        "input_value_key",  # PreserveByValueStage: compares an arbitrary user key
        "items_key",  # PreserveByValueConditionsStage: caller-chosen one-level list
        "output_key",  # ITN/Chinese: caller-chosen output key
        "original_key",  # preserved prior value
        "sort_key",
        "cache_key",
        "oldest_key",
        # bookkeeping counters / flags
        "num_segments_key",
        "is_mono_key",
        "truncation_events_key",
    }
)

# Hard-coded key *values* emitted by producers that lack a ``*_key`` field.
LITERAL_KEY_ROLES: dict[str, Role] = {
    "audio_filepath": "audio_filepath",
    "waveform": "waveform",
    "sample_rate": "sample_rate",
    "duration": "duration",
    "segments": "segments",
    "diar_segments": "diar_segments",
    "vad_segments": "vad_segments",
    "text": "text",
    "text_ref": "reference_text",  # ComputeWERStage's documented reference default
    "pred_text": "pred_text",
    "words": "words",
    "alignment": "alignment",
}


def role_for_field(field_name: str) -> Role:
    """Return the semantic role for a ``*_key`` constructor field name.

    Unmapped field names return ``"unknown"`` (composition falls back to
    value-equality and is never blocked).
    """
    return KEY_ROLES.get(field_name, "unknown")


def role_for_value(key_value: str, *, field_name: str | None = None) -> Role:
    """Resolve a role from a key's resolved value, preferring its field name.

    ``field_name`` (authoritative) is the producer/consumer's ``*_key`` field;
    when absent (hard-coded producer output) we fall back to a literal-default
    table keyed on the value itself.
    """
    if field_name is not None:
        role = KEY_ROLES.get(field_name)
        if role is not None:
            return role
    return LITERAL_KEY_ROLES.get(key_value, "unknown")


def field_has_declared_role(field_name: str, stage_cls: type | None = None) -> bool:
    """True if a ``*_key`` field has a role or is declared internal bookkeeping.

    Used by the conformance harness to catch a newly added ``*_key`` field that forgot a
    :data:`KEY_ROLES` entry.

    A stage may satisfy this itself, via ``KEY_ROLE_OVERRIDES`` (this field means an
    existing role) or ``INTERNAL_KEY_FIELDS`` (this field is my own bookkeeping and chains
    with nothing). That is what keeps adding a stage from meaning editing this module. The
    shared tables stay authoritative for keys that cross stages, because a role is the
    vocabulary two stages connect through -- a privately invented one would compose with
    nothing while still passing the check.
    """
    if field_name in KEY_ROLES or field_name in INTERNAL_KEY_FIELDS:
        return True
    if stage_cls is None:
        return False
    return field_name in role_overrides_for(stage_cls) or field_name in internal_key_fields_for(stage_cls)


def internal_key_fields_for(stage_cls: type) -> frozenset[str]:
    """A stage's own bookkeeping ``*_key`` fields, UNIONED across its bases.

    ``getattr`` alone returns only the most-derived declaration, so a subclass that declares
    one internal field of its own shadows every field its parent declared -- and the parent's
    fields, still inherited and still bookkeeping, start failing the conformance check as
    though someone had forgotten a role for them. The subclass author's fix is then to
    re-list fields they did not write, which is how a shared table gets copied downwards.
    """
    fields: set[str] = set()
    for base in getattr(stage_cls, "__mro__", (stage_cls,)):
        fields |= set(base.__dict__.get("INTERNAL_KEY_FIELDS") or ())
    return frozenset(fields)


def role_overrides_for(stage_cls: type) -> Mapping[str, Role]:
    """Per-stage ``KEY_ROLE_OVERRIDES`` (consulted before the shared table)."""
    return getattr(stage_cls, "KEY_ROLE_OVERRIDES", {}) or {}
