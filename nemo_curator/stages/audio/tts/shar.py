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

"""Lhotse SHAR materialization of annotated TTS Granary manifests.

Port of the TTS Granary ``lhotse_cuts`` steps (``create_shar_tasks`` + ``materialize_shars``) and of the final
``update_lhotse_cuts_from_manifests`` rewrite, producing in one pass the cut format EasyMagpie trains on
(``cuts_with_ipaNG_early_cut_off_filter``): ``supervision.text`` is the spoken-form ``GranaryV2.tn_raw`` text and
``supervision.custom`` carries quality annotations, the context pairing, ``ipa`` and the training flags.

Per input manifest the stage writes, for every ``(language, sampling_rate)`` present in it, aligned shards::

    <out>/<manifest stem>-<hash>/<lang>/<rate>/cuts/cuts.NNNNNN.jsonl.gz          target MonoCuts
    <out>/<manifest stem>-<hash>/<lang>/<rate>/target_audio/recording.NNNNNN.tar   <cut_id>.wav + .json
    <out>/<manifest stem>-<hash>/<lang>/<rate>/context_audio/recording.NNNNNN.tar  context audio under the target id

plus ``<out>/<manifest stem>-<hash>/shar_registry.json`` listing the shards (each with its own ``dir``), consumed by
codec extraction, validation and publishing.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from loguru import logger

from nemo_curator.stages.audio.tts.audio_io import (
    MISSING_ENHANCED_AUDIO,
    PNC_TEXT_FIELD,
    TAR_BYTES_FIELD,
    TAR_KEY_FIELD,
    TAR_MEMBER_FIELD,
    TAR_OFFSET_FIELD,
    TN_RAW_FIELD,
    TarAudioConfig,
    enhanced_audio_path,
    make_audio_config,
    resolve_tar_path,
)
from nemo_curator.stages.audio.tts.fields import MISSING, get_dotted
from nemo_curator.stages.audio.tts.shar_materialize import (
    REGISTRY_NAME,
    REGISTRY_VERSION,
    entries_digest,
    load_registry,
    materialize_shard,
    read_cut_ids,
    read_tar_ids,
    shard_is_aligned,
    shard_marker,
    shard_paths,
    write_json_atomic,
)
from nemo_curator.stages.audio.tts.shar_packing import pack_best_fit_decreasing, plan_shards, split_balanced
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask

__all__ = [
    "DEFAULT_CUSTOM_FIELDS",
    "REGISTRY_NAME",
    "LhotseSharBuildStage",
    "SharEntry",
    "SharFieldMap",
    "apply_training_filter",
    "build_shar_entries",
    "build_shars_for_manifest",
    "deduplicate_entries",
    "load_registry",
    "manifest_output_dir",
    "materialize_shard",
    "pack_best_fit_decreasing",
    "plan_shards",
    "read_cut_ids",
    "read_tar_ids",
    "shard_is_aligned",
    "split_balanced",
]

SPEAKER_TEMPLATE = "| Language:{language} Dataset:{dataset} Speaker:{speaker_id} |"
_SPEAKER_RE = re.compile(r"\| Language:\w+ Dataset:[\w\d\W]+ Speaker:[\w\d\W]+ \|")
_LANGUAGE_RE = re.compile(r"Language:(\w+)")
MISSING_FIELD_POLICIES = ("fail", "null")
CONTEXT_SPEAKER_MISMATCH = "context_speaker_mismatch"

# Ordered like the published v2610.granary cuts: (custom name, source). Sources are dotted segment paths, or
# ``sample:<path>`` (sample row), or ``const:<json>``. An annotation dict whose ``error`` is set counts as missing.
DEFAULT_CUSTOM_FIELDS: tuple[tuple[str, str], ...] = (
    ("utt_id", TAR_MEMBER_FIELD),
    ("sed_weighted_speech_rate", "sed.weighted_speech_rate"),
    ("sed_weighted_model_confidence", "sed.weighted_model_confidence"),
    ("sed_weighted_classifier_confidence", "sed.weighted_classifier_confidence"),
    ("diarized_speech_rate", "diarization.diarized_speech_rate"),
    ("MOS", "utmosv2.score"),
    ("bandwidth", "bandwidth.bandwidth"),
    ("sampling_rate", "sample:orig_sample_rate"),
    ("tar_member_offset", TAR_OFFSET_FIELD),
    ("tar_member_bytes", TAR_BYTES_FIELD),
    ("cluster_similarity", "speaker_cluster.cluster_similarity"),
    ("cluster_margin", "speaker_cluster.cluster_margin"),
    ("context_recording_id", "context.context_recording_id"),
    ("context_audio_duration", "context.context_audio_duration"),
    ("context_audio_text", "context.context_audio_text"),
    ("context_audio_speaker_similarity", "context.context_audio_speaker_similarity"),
    ("context_audio_text_overlap", "context.context_audio_text_overlap"),
    ("context_tar_member_offset", "context.context_audio_offset"),
    ("context_tar_member_bytes", "context.context_audio_bytes"),
)
# LibriLight (reference ``create_shar_tasks`` override): sample-level speaker, fixed 16 kHz, no cluster scores.
LIBRILIGHT_CUSTOM_FIELDS: tuple[tuple[str, str], ...] = tuple(
    ("sampling_rate", "const:16000") if name == "sampling_rate" else (name, path)
    for name, path in DEFAULT_CUSTOM_FIELDS
    if name not in ("cluster_similarity", "cluster_margin")
)
FLAG_FIELDS = ("valid_tn_raw_transcript", "valid_ipa", "early_cut_off", "valid_context_audio_text")
_TRANSPORT_FIELDS = (
    "sampling_rate",
    "tar_member_offset",
    "tar_member_bytes",
    "context_recording_id",
    "context_audio_duration",
    "context_tar_member_offset",
    "context_tar_member_bytes",
)


@dataclass(frozen=True)
class SharFieldMap:
    """Where the SHAR builder reads each value from (segment dotted paths; ``sample:`` reads the sample row)."""

    tar_key: str = TAR_KEY_FIELD
    tar_member: str = TAR_MEMBER_FIELD
    text: str = TN_RAW_FIELD
    text_fallback: str = PNC_TEXT_FIELD
    speaker_id: str = "speaker_cluster.speaker_id"
    context: str = "context"
    ipa: str = "ipa.ipa"
    flags: str = "tts_flags"
    sample_id: str = "orig_sample_id"
    language: str = "source_lang"
    custom_fields: tuple[tuple[str, str], ...] = DEFAULT_CUSTOM_FIELDS

    @classmethod
    def for_preset(cls, preset: str = "default", **overrides: Any) -> SharFieldMap:  # noqa: ANN401
        """``default`` (YTC / YODAS / VoxPopuli: TitaNet cluster speakers) or ``librilight`` (sample speaker)."""
        presets = {
            "default": cls(),
            "librilight": cls(
                speaker_id="sample:speaker_id", sample_id="chapter_id", custom_fields=LIBRILIGHT_CUSTOM_FIELDS
            ),
        }
        if preset not in presets:
            msg = f"Unknown SHAR field preset {preset!r}; expected one of {sorted(presets)}"
            raise ValueError(msg)
        return replace(presets[preset], **overrides)


@dataclass
class SharEntry:
    """One target/context pair ready to be packed into a shard."""

    language: str
    sampling_rate: int
    video_id: str
    speaker_id: str
    audio_filepath: str
    target_cut: dict[str, Any]
    context_cut: dict[str, Any]
    tar_path: str
    target_offset: int
    target_size: int
    context_offset: int
    context_size: int
    target_file: str | None = None
    context_file: str | None = None
    audio_source: str = "original"
    context_speaker_id: str | None = None  # None: context segment no longer in the row (filtered out upstream)

    @property
    def cut_id(self) -> str:
        return str(self.target_cut["id"])

    @property
    def mos(self) -> float:
        try:
            value = float(self.target_cut["supervisions"][0]["custom"].get("MOS"))
        except (TypeError, ValueError, KeyError, IndexError):
            return float("-inf")
        return value if math.isfinite(value) else float("-inf")

    def identity(self) -> tuple[Any, ...]:
        return (
            self.sampling_rate,
            self.audio_filepath,
            self.tar_path,
            self.target_offset,
            self.target_size,
            self.target_file,
        )


class SkipEntry(Exception):  # noqa: N818
    """A segment cannot become a SHAR entry; ``reason`` is counted in the registry (``skipped_<reason>``)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class AudioChoice:
    """Which audio a cut is materialized from: original tar members or enhanced files (target + context)."""

    source: str = "original"
    target_file: str | None = None
    context_file: str | None = None
    sampling_rate: int | None = None
    method: str | None = None


def _enhanced_copy(segment: dict[str, Any] | None, key: str) -> tuple[str, int, str | None] | None:
    if segment is None:
        return None
    path = enhanced_audio_path(segment, key)
    if path is None:
        return None
    annotation = get_dotted(segment, key) or {}
    rate = annotation.get("sampling_rate")
    if not rate:
        import soundfile as sf

        rate = sf.info(path).samplerate
    return path, int(rate), annotation.get("method")


def _segment_by_member(sample: dict[str, Any], member: str, fields: SharFieldMap) -> dict[str, Any] | None:
    for candidate in sample.get("segments") or []:
        if isinstance(candidate, dict) and str(get_dotted(candidate, fields.tar_member) or "") == member:
            return candidate
    return None


def choose_audio(
    sample: dict[str, Any],
    segment: dict[str, Any],
    context_member: str,
    fields: SharFieldMap,
    audio_config: TarAudioConfig | None,
) -> AudioChoice:
    """Original audio, or (``audio_source="enhanced"``) the enhanced copies of the target and of the context segment
    (looked up in the same row). A missing copy raises ``SkipEntry`` unless ``enhanced_fallback`` is set."""
    if audio_config is None or audio_config.audio_source != "enhanced":
        return AudioChoice()
    target = _enhanced_copy(segment, audio_config.enhanced_key)
    context = _enhanced_copy(_segment_by_member(sample, context_member, fields), audio_config.enhanced_key)
    if target is not None and context is not None:
        return AudioChoice("enhanced", target[0], context[0], target[1], target[2])
    if audio_config.enhanced_fallback:
        return AudioChoice("original")
    raise SkipEntry(MISSING_ENHANCED_AUDIO)


# --------------------------------------------------------------------------------------------- entry construction


def get_recording_id(relative_path: str) -> str:
    return "rec-" + relative_path.rsplit(".", 1)[0].replace("/", "-")


def normalize_tar_member(name: str) -> str:
    name = name.replace("\\", "/")
    while name.startswith("./"):
        name = name[2:]
    return name.lstrip("/")


def check_speaker_format(speaker: str) -> bool:
    return isinstance(speaker, str) and bool(_SPEAKER_RE.match(speaker))


def get_annotated(payload: dict[str, Any], path: str) -> Any:  # noqa: ANN401
    """``get_dotted`` that returns ``MISSING`` when any dict on the path carries a non-null ``error``."""
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return MISSING
        if current is not payload and current.get("error") is not None:
            return MISSING
        current = current[part]
    if isinstance(current, dict) and current.get("error") is not None:
        return MISSING
    return current


def _lookup(sample: dict[str, Any], segment: dict[str, Any], path: str) -> Any:  # noqa: ANN401
    if path.startswith("const:"):
        return json.loads(path[len("const:") :])
    if path.startswith("sample:"):
        return get_annotated(sample, path[len("sample:") :])
    return get_annotated(segment, path)


def _resolve(value: Any, name: str, policy: str) -> Any:  # noqa: ANN401
    if value is not MISSING:
        return value
    if policy == "null":
        return None
    msg = f"Required SHAR field {name!r} is missing or its annotation has an error"
    raise KeyError(msg)


def _context_tn_raw(sample: dict[str, Any], context_member: str, fields: SharFieldMap) -> str | None:
    """Spoken-form text of the context segment (same sample row), as the reference flags rewrite does."""
    for segment in sample.get("segments") or []:
        if isinstance(segment, dict) and str(get_dotted(segment, fields.tar_member) or "") == context_member:
            value = get_dotted(segment, fields.text)
            return value if isinstance(value, str) else None
    return None


def _ipa_value(segment: dict[str, Any], fields: SharFieldMap, policy: str) -> Any:  # noqa: ANN401
    """IPA string; ``None`` when the IPA annotation failed (``valid_ipa`` is then false); absent -> policy."""
    parent_path = fields.ipa.rsplit(".", 1)[0] if "." in fields.ipa else ""
    parent = get_dotted(segment, parent_path) if parent_path else None
    if isinstance(parent, dict) and parent.get("error") is not None:
        return None
    return _resolve(get_dotted(segment, fields.ipa, MISSING), "ipa", policy)


def build_custom_fields(
    sample: dict[str, Any], segment: dict[str, Any], fields: SharFieldMap, missing_field_policy: str = "fail"
) -> dict[str, Any]:
    """Final ``supervision.custom`` in the published key order."""
    values = {
        name: _resolve(_lookup(sample, segment, path), name, missing_field_policy)
        for name, path in fields.custom_fields
    }
    custom: dict[str, Any] = {}
    for leading in ("sampling_rate", "context_audio_duration"):
        if leading in values:
            custom[leading] = values[leading]
    for name, value in values.items():
        if name not in custom and name != "context_audio_text":
            custom[name] = value

    context_member = values.get("context_recording_id")
    context_text = None
    if context_member is not None:
        custom["context_recording_id"] = get_recording_id(str(context_member))
        context_text = _context_tn_raw(sample, str(context_member), fields)
    custom["context_audio_text"] = context_text if context_text is not None else values.get("context_audio_text")

    custom["ipa"] = _ipa_value(segment, fields, missing_field_policy)
    flags = get_dotted(segment, fields.flags, MISSING)
    flags = flags if isinstance(flags, dict) else {}
    for name in FLAG_FIELDS:
        custom[name] = _resolve(flags.get(name, MISSING), f"{fields.flags}.{name}", missing_field_policy)
    if "context_audio_text_overlap" in flags:
        custom["context_audio_text_overlap"] = flags["context_audio_text_overlap"]
    return custom


def _recording(recording_id: str, source: str, sampling_rate: int, duration: float) -> Any:  # noqa: ANN401
    from lhotse import AudioSource, Recording, compute_num_samples

    return Recording(
        id=recording_id,
        sources=[AudioSource(type="file", channels=[0], source=source)],
        sampling_rate=sampling_rate,
        num_samples=compute_num_samples(duration, sampling_rate),
        duration=duration,
    )


def build_entry(  # noqa: PLR0913
    sample: dict[str, Any],
    segment: dict[str, Any],
    *,
    language: str,
    dataset: str,
    fields: SharFieldMap,
    audio_config: TarAudioConfig | None = None,
    missing_field_policy: str = "fail",
) -> SharEntry | None:
    """Build target/context cuts for one segment; ``None`` when the segment has no usable context pairing."""
    from lhotse import MonoCut, SupervisionSegment

    context = get_dotted(segment, fields.context)
    if not isinstance(context, dict) or context.get("error") is not None:
        return None
    speaker_id = _lookup(sample, segment, fields.speaker_id)
    if speaker_id is MISSING or speaker_id is None:
        return None
    custom = build_custom_fields(sample, segment, fields, missing_field_policy)
    missing = [name for name in _TRANSPORT_FIELDS if custom.get(name) is None]
    if missing:
        msg = f"SHAR transport values are missing: {missing}"
        raise KeyError(msg)

    member = _resolve(get_dotted(segment, fields.tar_member, MISSING), fields.tar_member, "fail")
    tar_key = _resolve(get_dotted(segment, fields.tar_key, MISSING), fields.tar_key, "fail")
    tar_path = resolve_tar_path(str(tar_key), audio_config)
    text = get_dotted(segment, fields.text)
    if not isinstance(text, str) or not text.strip():
        text = get_dotted(segment, fields.text_fallback)
    duration = float(segment["duration"])
    context_duration = float(custom["context_audio_duration"])
    context_member = str(get_dotted(segment, f"{fields.context}.context_recording_id"))
    context_segment = _segment_by_member(sample, context_member, fields)
    context_speaker = _lookup(sample, context_segment, fields.speaker_id) if context_segment is not None else None
    context_speaker = None if context_speaker is MISSING else context_speaker
    if context_speaker is not None and str(context_speaker) != str(speaker_id):
        logger.warning(
            "Context {} of {} belongs to speaker {} != {}", context_member, member, context_speaker, speaker_id
        )
        raise SkipEntry(CONTEXT_SPEAKER_MISMATCH)
    choice = choose_audio(sample, segment, context_member, fields, audio_config)
    if audio_config is not None and audio_config.audio_source == "enhanced":
        # appended keys only in enhanced mode: original-mode custom stays byte-identical to the published format
        custom["audio_source"] = choice.source
        if choice.source == "enhanced":
            custom["enhancement_method"] = choice.method
            custom["sampling_rate"] = choice.sampling_rate
    sampling_rate = int(custom["sampling_rate"])
    speaker = SPEAKER_TEMPLATE.format(language=language, dataset=dataset, speaker_id=speaker_id)
    if not text or not check_speaker_format(speaker):
        logger.warning("Skipping segment {}: missing text or malformed speaker {!r}", member, speaker)
        return None

    target_rec_id = get_recording_id(str(member))
    context_rec_id = get_recording_id(context_member)
    target_cut_id = f"cut-{target_rec_id}-{0.0:.2f}-{duration:.2f}"
    context_cut_id = f"context_cut-{context_rec_id}-{0.0:.2f}-{context_duration:.2f}"
    if context_cut_id.split("-", 1)[1] == target_cut_id.split("-", 1)[1]:
        logger.warning("Context equals target for {}; skipping", target_cut_id)
        return None

    lang_match = _LANGUAGE_RE.search(speaker)
    supervision = SupervisionSegment(
        id=f"sup-{target_rec_id}",
        recording_id=target_rec_id,
        start=0.0,
        duration=duration,
        channel=0,
        text=text,
        language=lang_match.group(1) if lang_match else None,
        speaker=speaker,
        custom=custom,
    )
    target_source = choice.target_file or f"{tar_path}:{normalize_tar_member(str(member))}"
    context_source = choice.context_file or f"{tar_path}:{normalize_tar_member(context_member)}"
    target_cut = MonoCut(
        id=target_cut_id,
        start=0.0,
        duration=duration,
        channel=0,
        recording=_recording(target_rec_id, target_source, sampling_rate, duration),
        supervisions=[supervision],
    )
    context_cut = MonoCut(
        id=context_cut_id,
        start=0.0,
        duration=context_duration,
        channel=0,
        recording=_recording(context_rec_id, context_source, sampling_rate, context_duration),
    )
    video_id = _lookup(sample, segment, f"sample:{fields.sample_id}")
    return SharEntry(
        language=language,
        sampling_rate=sampling_rate,
        video_id=str(video_id) if video_id not in (MISSING, None, "") else str(speaker_id),
        speaker_id=str(speaker_id),
        audio_filepath=str(member),
        target_cut=target_cut.to_dict(),
        context_cut=context_cut.to_dict(),
        tar_path=str(tar_path),
        target_offset=int(custom["tar_member_offset"]),
        target_size=int(custom["tar_member_bytes"]),
        context_offset=int(custom["context_tar_member_offset"]),
        context_size=int(custom["context_tar_member_bytes"]),
        target_file=choice.target_file,
        context_file=choice.context_file,
        audio_source=choice.source,
        context_speaker_id=None if context_speaker is None else str(context_speaker),
    )


def row_language(sample: dict[str, Any], fields: SharFieldMap, language: str | None) -> str:
    """Explicit ``language`` wins; otherwise the row's language field; neither -> error (no path inference)."""
    if language:
        return str(language)
    value = get_dotted(sample, fields.language)
    if isinstance(value, str) and value.strip():
        return value.strip()
    msg = (
        f"No language for sample {sample.get(fields.sample_id)!r}: pass `language` or set "
        f"the row field {fields.language!r}"
    )
    raise ValueError(msg)


def build_shar_entries(  # noqa: PLR0913
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    language: str | None = None,
    fields: SharFieldMap | None = None,
    audio_config: TarAudioConfig | None = None,
    missing_field_policy: str = "fail",
) -> tuple[list[SharEntry], dict[str, int]]:
    """Flatten sample rows into deduplicated SHAR entries; returns ``(entries, counts)``."""
    fields = fields or SharFieldMap()
    entries: list[SharEntry] = []
    counts = {"segments": 0, "without_context": 0, "duplicates": 0}
    for sample in rows:
        segments = sample.get("segments")
        if not isinstance(segments, list):
            continue
        sample_language = row_language(sample, fields, language)
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            counts["segments"] += 1
            try:
                entry = build_entry(
                    sample,
                    segment,
                    language=sample_language,
                    dataset=dataset,
                    fields=fields,
                    audio_config=audio_config,
                    missing_field_policy=missing_field_policy,
                )
            except SkipEntry as skip:
                key = f"skipped_{skip.reason}"
                counts[key] = counts.get(key, 0) + 1
                continue
            if entry is None:
                counts["without_context"] += 1
                continue
            entries.append(entry)
    entries, counts["duplicates"] = deduplicate_entries(entries)
    return entries, counts


def deduplicate_entries(entries: list[SharEntry]) -> tuple[list[SharEntry], int]:
    """Keep one entry per cut id, preferring the higher MOS; conflicting physical targets are an error."""
    selected: dict[str, SharEntry] = {}
    duplicates = 0
    for entry in entries:
        existing = selected.get(entry.cut_id)
        if existing is None:
            selected[entry.cut_id] = entry
            continue
        if existing.identity() != entry.identity():
            msg = f"Conflicting segments produced the same cut ID: {entry.cut_id}"
            raise ValueError(msg)
        duplicates += 1
        if entry.mos > existing.mos:
            selected[entry.cut_id] = entry
    return list(selected.values()), duplicates


def apply_training_filter(cut: dict[str, Any] | Any, max_context_overlap: float = 0.5) -> bool:  # noqa: ANN401
    """Mirror of the EasyMagpie v2610.granary runtime supervision filter for a cut (dict or ``MonoCut``)."""
    supervisions = cut.get("supervisions") if isinstance(cut, dict) else [s.to_dict() for s in cut.supervisions]
    if not supervisions:
        return False
    custom = supervisions[0].get("custom") or {}
    overlap = custom.get("context_audio_text_overlap")
    return (
        custom.get("valid_tn_raw_transcript") is True
        and custom.get("valid_ipa") is True
        and custom.get("valid_context_audio_text") is True
        and custom.get("early_cut_off") is False
        and overlap is not None
        and float(overlap) < max_context_overlap
    )


# --------------------------------------------------------------------------------------------- registry + stage


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def manifest_output_dir(output_dir: str | Path, manifest: str | Path) -> Path:
    """``<output_dir>/<manifest stem>-<8-hex sha1 of the manifest's absolute path>`` (unique per manifest)."""
    absolute = os.path.abspath(os.path.expanduser(str(manifest)))
    tag = hashlib.sha1(absolute.encode("utf-8")).hexdigest()[:8]  # noqa: S324
    return Path(os.path.abspath(os.path.expanduser(str(output_dir)))) / f"{Path(absolute).name.split('.')[0]}-{tag}"


def _cuts_duration(path: Path) -> float:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return round(sum(float(json.loads(line)["duration"]) for line in stream if line.strip()), 6)


def build_shars_for_manifest(  # noqa: PLR0913
    manifest: str | Path,
    output_dir: str | Path,
    *,
    dataset: str,
    language: str | None = None,
    fields: SharFieldMap | None = None,
    audio_config: TarAudioConfig | None = None,
    missing_field_policy: str = "fail",
    max_per_shard: int = 4096,
    seed: int = 17,
    audio_format: str = "wav",
    keep_speaker_in_one_shard: bool = True,
) -> Path:
    """Plan + materialize all shards of one annotated manifest into ``output_dir``; returns the registry path.

    Resumable: a shard is reused only if its completion marker's content digest equals the digest of the currently
    planned entries, so re-annotated rows (flags, IPA, MOS, context pairing) rebuild the affected shards.
    """
    if missing_field_policy not in MISSING_FIELD_POLICIES:
        msg = f"missing_field_policy must be one of {MISSING_FIELD_POLICIES}"
        raise ValueError(msg)
    output_dir = Path(os.path.abspath(os.path.expanduser(str(output_dir))))
    output_dir.mkdir(parents=True, exist_ok=True)
    entries, counts = build_shar_entries(
        read_jsonl(manifest),
        dataset=dataset,
        language=language,
        fields=fields,
        audio_config=audio_config,
        missing_field_policy=missing_field_policy,
    )
    shards: list[dict[str, Any]] = []
    totals = {"reused_shards": 0, "built_shards": 0, "empty_shards": 0, "skipped_cuts": 0, "written_cuts": 0}
    for (shard_language, rate), packed in plan_shards(
        entries, max_per_shard, seed, keep_speaker_in_one_shard=keep_speaker_in_one_shard
    ).items():
        rate_dir = output_dir / shard_language / str(rate)
        for shard_id, shard_entries in enumerate(packed):
            digest = entries_digest(shard_entries)
            marker = shard_marker(rate_dir, shard_id, digest)
            if marker is None:
                marker = materialize_shard(shard_entries, rate_dir, shard_id, audio_format, digest)
                totals["built_shards"] += 1
            else:
                totals["reused_shards"] += 1
            totals["skipped_cuts"] += len(marker["skipped"])
            if not marker["cut_ids"]:
                totals["empty_shards"] += 1
                continue
            totals["written_cuts"] += len(marker["cut_ids"])
            cuts, target, context = shard_paths(rate_dir, shard_id)
            shards.append(
                {
                    "language": shard_language,
                    "sampling_rate": str(rate),
                    "shard_id": shard_id,
                    "dir": str(rate_dir),
                    "num_items": len(marker["cut_ids"]),
                    "digest": digest,
                    "skipped": marker["skipped"],
                    "duration_seconds": _cuts_duration(cuts),
                    "cuts": str(cuts),
                    "target_audio": str(target),
                    "context_audio": str(context),
                }
            )
    registry = {
        "version": REGISTRY_VERSION,
        "source_manifest": os.path.abspath(str(manifest)),
        "dataset": dataset,
        "output_dir": str(output_dir),
        "max_samples_per_shard": max_per_shard,
        "keep_speaker_in_one_shard": keep_speaker_in_one_shard,
        "audio_source": audio_config.audio_source if audio_config is not None else "original",
        "counts": {**counts, "planned_cuts": len(entries), **totals},
        "shards": shards,
    }
    registry_path = output_dir / REGISTRY_NAME
    write_json_atomic(registry_path, registry)
    logger.info("SHAR build {} -> {}: {}", manifest, output_dir, registry["counts"])
    return registry_path


@dataclass
class LhotseSharBuildStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    """Materialize annotated sample-row manifests into aligned Lhotse SHAR shards.

    Input: ``FileGroupTask`` of annotated JSONL manifests. Output: ``FileGroupTask`` of ``shar_registry.json`` paths,
    one per manifest, under ``<output_dir>/<manifest stem>-<hash>/``. Segments without a valid ``context`` annotation
    are skipped (the reference ``filter_context`` step).

    Args:
        output_dir: root for all shards (made absolute; symlinks are not resolved).
        dataset: dataset string of the speaker label, e.g. ``"GranaryHiFi_YTC (orig_audio)"``.
        language: explicit language; overrides the row ``source_lang``. Required if rows carry none.
        preset: ``"default"`` or ``"librilight"`` field mapping; ignored when ``field_map`` is given.
        field_map: full ``SharFieldMap`` override.
        missing_field_policy: ``"fail"`` (raise on a missing / errored annotation) or ``"null"``.
        tar_root / strip_tar_key_prefixes: resolve relative ``tar_key`` values (see ``TarAudioConfig``).
        audio_source / enhanced_key / enhanced_fallback: ``"enhanced"`` materializes target and context audio from
            the segments' enhanced copies (at the enhanced sample rate; recording sources point at the files);
            cuts lacking a copy are skipped (``skipped_missing_enhanced_audio``) or fall back to the original.
        keep_speaker_in_one_shard: never split a speaker across shards (oversized shards for huge speakers).
    """

    output_dir: str = ""
    dataset: str = "GranaryHiFi_YTC (orig_audio)"
    language: str | None = None
    preset: str = "default"
    field_map: SharFieldMap | None = None
    missing_field_policy: str = "fail"
    max_samples_per_shard: int = 4096
    shuffle_seed: int = 17
    audio_format: str = "wav"
    tar_root: str | None = None
    strip_tar_key_prefixes: tuple[str, ...] = ()
    audio_source: str = "original"
    enhanced_key: str = "enhanced_audio"
    enhanced_fallback: bool = False
    keep_speaker_in_one_shard: bool = True
    name: str = "LhotseSharBuild"
    resources: Resources = field(default_factory=lambda: Resources(cpus=2.0))
    batch_size: int = 1

    def __post_init__(self) -> None:
        super().__init__()
        if not self.output_dir:
            msg = "output_dir is required"
            raise ValueError(msg)
        if self.missing_field_policy not in MISSING_FIELD_POLICIES:
            msg = f"missing_field_policy must be one of {MISSING_FIELD_POLICIES}"
            raise ValueError(msg)
        if self.max_samples_per_shard <= 0:
            msg = "max_samples_per_shard must be positive"
            raise ValueError(msg)
        self.output_dir = os.path.abspath(os.path.expanduser(self.output_dir))
        if self.tar_root:
            self.tar_root = os.path.abspath(os.path.expanduser(self.tar_root))
        if self.field_map is None:
            self.field_map = SharFieldMap.for_preset(self.preset)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: FileGroupTask) -> FileGroupTask:
        audio_config = make_audio_config(self)
        registries = [
            str(
                build_shars_for_manifest(
                    manifest,
                    manifest_output_dir(self.output_dir, manifest),
                    dataset=self.dataset,
                    language=self.language,
                    fields=self.field_map,
                    audio_config=audio_config,
                    missing_field_policy=self.missing_field_policy,
                    max_per_shard=self.max_samples_per_shard,
                    seed=self.shuffle_seed,
                    audio_format=self.audio_format,
                    keep_speaker_in_one_shard=self.keep_speaker_in_one_shard,
                )
            )
            for manifest in task.data
        ]
        return FileGroupTask(
            dataset_name=task.dataset_name,
            data=registries,
            _metadata={**task._metadata, "shar_registries": registries},
            _stage_perf=task._stage_perf,
        )
