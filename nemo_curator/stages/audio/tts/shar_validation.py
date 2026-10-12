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

"""Structural validation and hard-link publication of GranaryHiFi SHAR shards.

A shard is valid when its five streams (cuts, target/context audio, target/context codes) describe the same
cut ids in the same order, cut ids are unique, the count matches the registry and every text-bearing supervision has
IPA (or is flagged ``valid_ipa == False``). Publishing renumbers shards globally per ``(language, sampling_rate)`` and
hard-links them into the training layout::

    <root>/<lang>/<collection>/<dataset>/<rate>/<cuts_dir>/cuts.N.jsonl.gz
    <root>/<lang>/<collection>/<dataset>/<rate>/{target_audio,context_audio}/recording.N.tar
    <root>/<lang>/<collection>/<dataset>/<rate>/<codec>/{target_codes,context_codes}/codes.N.tar
    <root>/index.jsonl, <root>/cfgs/<lang>/<collection>_<dataset>_<rate>_<split>.yaml
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from nemo_curator.stages.audio.tts.codec import DEFAULT_CODEC_NAME, codes_paths
from nemo_curator.stages.audio.tts.shar_materialize import (
    AUDIO_EXTENSIONS,
    load_registry,
    marker_path,
    read_json,
    read_tar_ids,
    write_json_atomic,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask

STREAMS = ("cuts", "target_audio", "context_audio", "target_codes", "context_codes")
DEFAULT_CUTS_DIR = "cuts_with_ipaNG_early_cut_off_filter"
VALIDATION_REPORT = "shar_validation.json"


class SharValidationError(ValueError):
    """A SHAR shard failed structural validation."""


def read_cut_rows(path: Path) -> tuple[list[str], float, list[str]]:
    """Return ``(cut ids, total duration, problems)`` for a cuts file (IPA presence checked per supervision)."""
    ids: list[str] = []
    duration = 0.0
    problems: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            cut = json.loads(line)
            cut_id = cut.get("id")
            if not isinstance(cut_id, str) or not cut_id:
                problems.append(f"{path}:{line_no}: missing cut id")
                continue
            ids.append(cut_id)
            duration += float(cut.get("duration") or 0.0)
            for supervision in cut.get("supervisions") or []:
                custom = supervision.get("custom") or {}
                has_text = bool(custom.get("normalized_text") or supervision.get("text"))
                if has_text and not str(custom.get("ipa") or "").strip() and custom.get("valid_ipa") is not False:
                    problems.append(f"{path}:{line_no}: supervision has text but no IPA and is not flagged invalid")
    return ids, duration, problems


def _id_mismatch(expected: list[str], observed: list[str], label: str) -> str | None:
    if expected == observed:
        return None
    if len(expected) != len(observed):
        return f"{label}: count mismatch expected={len(expected)} observed={len(observed)}"
    index = next(i for i, (a, b) in enumerate(zip(expected, observed, strict=True)) if a != b)
    return f"{label}: index={index} expected={expected[index]!r} observed={observed[index]!r}"


def shard_stream_paths(shard: dict[str, Any], codec_name: str = DEFAULT_CODEC_NAME) -> dict[str, Path]:
    target_codes, context_codes = codes_paths(Path(shard["dir"]), int(shard["shard_id"]), codec_name)
    return {
        "cuts": Path(shard["cuts"]),
        "target_audio": Path(shard["target_audio"]),
        "context_audio": Path(shard["context_audio"]),
        "target_codes": target_codes,
        "context_codes": context_codes,
    }


def validate_shard(
    shard: dict[str, Any], codec_name: str = DEFAULT_CODEC_NAME, *, require_codes: bool = True
) -> list[str]:
    """Return a list of problems (empty when the shard is valid)."""
    paths = shard_stream_paths(shard, codec_name)
    streams = STREAMS if require_codes else STREAMS[:3]
    missing = [name for name in streams if not paths[name].is_file()]
    if missing:
        return [f"shard {shard['shard_id']} ({shard['dir']}): missing streams {missing}"]
    cut_ids, _, problems = read_cut_rows(paths["cuts"])
    if len(cut_ids) != len(set(cut_ids)):
        problems.append(f"{paths['cuts']}: duplicate cut ids")
    if len(cut_ids) != int(shard["num_items"]):
        problems.append(f"{paths['cuts']}: registry num_items={shard['num_items']} cuts={len(cut_ids)}")
    for name in streams[1:]:
        extensions = {".npy"} if name.endswith("codes") else AUDIO_EXTENSIONS
        payload, metadata = read_tar_ids(paths[name], extensions)
        for label, observed in ((f"{name} payload", payload), (f"{name} metadata", metadata)):
            mismatch = _id_mismatch(cut_ids, observed, label)
            if mismatch:
                problems.append(f"{paths[name]}: {mismatch}")
    return problems


_SPEAKER_RE = re.compile(r"Speaker:(.+?) \|$")


def _shard_speaker_index(shard: dict[str, Any]) -> list[tuple[str, str, str | None]]:
    """``(recording id, speaker id, context recording id)`` per cut of a shard."""
    rows = []
    with gzip.open(shard["cuts"], "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            cut = json.loads(line)
            supervision = (cut.get("supervisions") or [{}])[0]
            match = _SPEAKER_RE.search(str(supervision.get("speaker") or ""))
            custom = supervision.get("custom") or {}
            rows.append(
                (
                    str((cut.get("recording") or {}).get("id") or cut["id"]),
                    match.group(1) if match else "",
                    custom.get("context_recording_id"),
                )
            )
    return rows


def _check_contexts(
    shard: dict[str, Any],
    rows: list[tuple[str, str, str | None]],
    recording_owner: dict[str, tuple[int, str]],
    label: str,
    require_one_shard: bool,
) -> list[str]:
    shard_id = int(shard["shard_id"])
    marker = read_json(marker_path(Path(shard["dir"]), shard_id)) or {}
    context_speakers = marker.get("context_speakers") or {}
    cut_ids = marker.get("cut_ids") or []
    problems: list[str] = []
    for position, (recording_id, speaker, context_id) in enumerate(rows):
        where = f"{label} shard {shard_id}: context {context_id} of {recording_id}"
        if context_id is None:
            problems.append(f"{label} shard {shard_id}: {recording_id} has no context")
            continue
        owner = recording_owner.get(str(context_id))
        if owner is not None:
            if owner[1] != speaker:
                problems.append(f"{where} belongs to speaker {owner[1]!r}, not {speaker!r}")
            elif require_one_shard and owner[0] != shard_id:
                problems.append(f"{where} is in shard {owner[0]}")
            continue
        cut_id = cut_ids[position] if position < len(cut_ids) else None
        recorded = context_speakers.get(cut_id) if cut_id else None
        if recorded is not None and recorded != speaker:
            problems.append(f"{where} (filtered-out segment) belongs to speaker {recorded!r}, not {speaker!r}")
    return problems


def validate_speaker_grouping(registry: dict[str, Any], *, require_one_shard: bool = True) -> list[str]:
    """Speaker/context integrity of a registry.

    * ``require_one_shard``: every speaker appears in exactly one shard per ``(language, sampling_rate)``.
    * Every cut's context recording is either a cut of the *same speaker in the same shard*, or a segment that is
      not a target anywhere (filtered out upstream) whose recorded speaker (shard marker) is the same or unknown.
    """
    problems: list[str] = []
    shards = [s for s in registry["shards"] if Path(s["cuts"]).is_file()]
    index = {id(s): _shard_speaker_index(s) for s in shards}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for shard in shards:
        groups[(str(shard["language"]), str(shard["sampling_rate"]))].append(shard)
    for (language, rate), group in groups.items():
        speaker_shards: dict[str, set[int]] = defaultdict(set)
        recording_owner: dict[str, tuple[int, str]] = {}
        for shard in group:
            for recording_id, speaker, _ in index[id(shard)]:
                speaker_shards[speaker].add(int(shard["shard_id"]))
                recording_owner[recording_id] = (int(shard["shard_id"]), speaker)
        if require_one_shard:
            problems.extend(
                f"{language}/{rate}: speaker {speaker!r} spans shards {sorted(ids)}"
                for speaker, ids in sorted(speaker_shards.items())
                if len(ids) > 1
            )
        for shard in group:
            problems.extend(
                _check_contexts(shard, index[id(shard)], recording_owner, f"{language}/{rate}", require_one_shard)
            )
    return problems


def validate_registry(
    registry_path: str | Path, codec_name: str = DEFAULT_CODEC_NAME, *, require_codes: bool = True
) -> dict[str, Any]:
    """Validate every shard of a registry (streams + speaker/context grouping); writes ``shar_validation.json``."""
    registry = load_registry(registry_path)
    problems: list[str] = []
    for shard in registry["shards"]:
        problems.extend(validate_shard(shard, codec_name, require_codes=require_codes))
    problems.extend(
        validate_speaker_grouping(registry, require_one_shard=bool(registry.get("keep_speaker_in_one_shard", True)))
    )
    report = {
        "registry": os.path.abspath(str(registry_path)),
        "codec_name": codec_name,
        "require_codes": require_codes,
        "num_shards": len(registry["shards"]),
        "num_cuts": sum(int(s["num_items"]) for s in registry["shards"]),
        "valid": not problems,
        "problems": problems[:100],
    }
    write_json_atomic(Path(registry_path).with_name(VALIDATION_REPORT), report)
    return report


@dataclass
class SharValidationStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    """Validate SHAR shards listed in ``shar_registry.json`` files; raises on invalid shards by default."""

    codec_name: str = DEFAULT_CODEC_NAME
    require_codes: bool = True
    fail_on_error: bool = True
    name: str = "SharValidation"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: FileGroupTask) -> FileGroupTask:
        reports = [validate_registry(r, self.codec_name, require_codes=self.require_codes) for r in task.data]
        invalid = [r for r in reports if not r["valid"]]
        if invalid and self.fail_on_error:
            msg = f"{len(invalid)} SHAR registries failed validation; first problems: {invalid[0]['problems'][:3]}"
            raise SharValidationError(msg)
        for report in invalid:
            logger.error("SHAR validation failed for {}: {}", report["registry"], report["problems"][:3])
        return FileGroupTask(
            dataset_name=task.dataset_name,
            data=[r["registry"] for r in reports if r["valid"]],
            _metadata={**task._metadata, "shar_validation": reports},
            _stage_perf=task._stage_perf,
        )


# --------------------------------------------------------------------------------------------- publishing


@dataclass(frozen=True)
class PublishLayout:
    """Published directory layout (dataset name, speech domain, codec and cut variants)."""

    collection: str = "GranaryHiFi"
    dataset: str = "YTC"
    # (inventory variant name, cuts dir); the single final cuts file is hard-linked into every listed dir.
    cut_variants: tuple[tuple[str, str], ...] = (("audioContext_with_ipa_wo_early_cut_off", DEFAULT_CUTS_DIR),)
    codec_name: str = DEFAULT_CODEC_NAME
    codec_key: str = "spectralCodecBWE25fps"
    audio_root_env: str = "AUDIO_ROOT"
    split: str = "train"
    proprietary: bool = False
    speech_domain: str = ""
    shard_digits: int = 6

    @classmethod
    def for_dataset(cls, dataset: str, speech_domain: str | None = None, **overrides: Any) -> PublishLayout:  # noqa: ANN401
        """Layout for a GranaryHiFi source: ``YTC``, ``YODAS``, ``VoxPopuli`` or ``LibriLight`` (case-insensitive)."""
        key = dataset.lower()
        if key not in _DATASETS:
            msg = f"Unknown GranaryHiFi dataset {dataset!r}; expected one of {[n for n, _ in _DATASETS.values()]}"
            raise ValueError(msg)
        name, default_domain = _DATASETS[key]
        return cls(dataset=name, speech_domain=default_domain if speech_domain is None else speech_domain, **overrides)


# canonical dataset name and default ``speech_domain`` per source
_DATASETS = {
    "ytc": ("YTC", "YouTube speech"),
    "yodas": ("YODAS", "YouTube speech"),
    "voxpopuli": ("VoxPopuli", "European Parliament speech"),
    "librilight": ("LibriLight", "Audiobooks"),
}


def _publication_rows(
    registries: list[dict[str, Any]], output_root: Path, layout: PublishLayout, *, include_codes: bool = True
) -> list[dict]:
    grouped: dict[tuple[str, str], list[tuple[tuple[str, str, int], dict[str, Any]]]] = defaultdict(list)
    for registry in registries:
        for shard in registry["shards"]:
            key = (str(registry["source_manifest"]), str(shard["sampling_rate"]), int(shard["shard_id"]))
            grouped[(str(shard["language"]), str(shard["sampling_rate"]))].append((key, shard))
    rows = []
    for (language, rate), items in sorted(grouped.items()):
        pair_root = output_root / language / layout.collection / layout.dataset / rate
        for new_id, (key, shard) in enumerate(sorted(items, key=lambda item: item[0])):
            n = f"{new_id:0{layout.shard_digits}d}"
            sources = shard_stream_paths(shard, layout.codec_name)
            destinations = {
                "target_audio": pair_root / "target_audio" / f"recording.{n}.tar",
                "context_audio": pair_root / "context_audio" / f"recording.{n}.tar",
            }
            if include_codes:
                destinations["target_codes"] = pair_root / layout.codec_name / "target_codes" / f"codes.{n}.tar"
                destinations["context_codes"] = pair_root / layout.codec_name / "context_codes" / f"codes.{n}.tar"
            links = [(sources[name], dest) for name, dest in destinations.items()]
            links += [
                (sources["cuts"], pair_root / cuts_dir / f"cuts.{n}.jsonl.gz") for _, cuts_dir in layout.cut_variants
            ]
            rows.append(
                {
                    "language": language,
                    "sampling_rate": rate,
                    "new_shard_id": new_id,
                    "source_manifest": key[0],
                    "source_shard_id": key[2],
                    "num_items": int(shard["num_items"]),
                    "duration_seconds": float(shard.get("duration_seconds") or 0.0),
                    "links": [[str(src), str(dst.relative_to(output_root))] for src, dst in links],
                }
            )
    return rows


def _inventory(
    rows: list[dict[str, Any]], layout: PublishLayout, *, include_codes: bool = True
) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["language"], row["sampling_rate"])].append(row)
    files = {}
    for (language, rate), group in sorted(grouped.items()):
        relative = f"cfgs/{language}/{layout.collection}_{layout.dataset}_{rate}_{layout.split}.yaml"
        files[relative] = {
            "shar_base": f"${{oc.env:{layout.audio_root_env}}}/{language}/{layout.collection}/{layout.dataset}/{rate}",
            "num_shards": len(group),
            "duration_in_hours": round(sum(r["duration_seconds"] for r in group) / 3600.0, 2),
            "proprietary": layout.proprietary,
            "speech_domain": layout.speech_domain,
            "codecs": {layout.codec_key: layout.codec_name} if include_codes else {},
            "cut_variants": {
                variant: {"cuts_dir": cuts_dir, "has_audio_context": True} for variant, cuts_dir in layout.cut_variants
            },
        }
    return files


def _nearest_existing_parent(path: Path) -> Path:
    while not path.exists():
        path = path.parent
    return path


def _publication_matches(output_root: Path, rows: list[dict[str, Any]], inventory: dict[str, dict]) -> bool:
    import yaml

    index = output_root / "index.jsonl"
    if not index.is_file():
        return False
    observed = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]
    if observed != rows:
        return False
    for row in rows:
        for src, dst in row["links"]:
            left, right = Path(src), output_root / dst
            if not left.is_file() or not right.is_file():
                return False
            if (left.stat().st_dev, left.stat().st_ino) != (right.stat().st_dev, right.stat().st_ino):
                return False
    for relative, payload in inventory.items():
        path = output_root / relative
        if not path.is_file() or yaml.safe_load(path.read_text(encoding="utf-8")) != payload:
            return False
    return True


def _require_valid(registry_paths: list[str | Path], codec_name: str, *, require_codes: bool) -> None:
    for path in registry_paths:
        report = validate_registry(path, codec_name, require_codes=require_codes)
        if not report["valid"]:
            msg = f"Refusing to publish invalid shards from {path}: {report['problems'][:3]}"
            raise SharValidationError(msg)


def _require_same_filesystem(rows: list[dict[str, Any]], output_root: Path) -> None:
    device = _nearest_existing_parent(output_root.parent).stat().st_dev
    for row in rows:
        for src, _ in row["links"]:
            if Path(src).stat().st_dev != device:
                msg = f"Hard link would cross filesystems: {src} -> {output_root}"
                raise OSError(msg)


def _hard_link(src: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(src, destination)
    left, right = Path(src).stat(), destination.stat()
    if (left.st_dev, left.st_ino) != (right.st_dev, right.st_ino):
        msg = f"Destination is not a hard link to source: {src} -> {destination}"
        raise OSError(msg)


def _write_staging(staging: Path, rows: list[dict[str, Any]], inventory: dict[str, dict]) -> None:
    import yaml

    for row in rows:
        for src, dst in row["links"]:
            _hard_link(src, staging / dst)
    (staging / "index.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    for relative, payload in inventory.items():
        path = staging / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def publish_shars(
    registry_paths: list[str | Path],
    output_root: str | Path,
    layout: PublishLayout | None = None,
    *,
    require_valid: bool = True,
    require_codes: bool = True,
) -> dict[str, Any]:
    """Hard-link validated shards into the training layout (atomic staging dir + ``os.replace``).

    ``require_codes=False`` publishes audio + cuts only (e.g. a build without codec extraction); validation then uses
    the same setting, so the ``shar_validation.json`` written here stays consistent with ``SharValidationStage``.
    Re-publishing identical content is a no-op; an existing, different publication raises ``FileExistsError``.
    """
    layout = layout or PublishLayout()
    output_root = Path(os.path.abspath(os.path.expanduser(str(output_root))))
    registries = [load_registry(path) for path in registry_paths]
    if require_valid:
        _require_valid(registry_paths, layout.codec_name, require_codes=require_codes)
    rows = _publication_rows(registries, output_root, layout, include_codes=require_codes)
    inventory = _inventory(rows, layout, include_codes=require_codes)
    _require_same_filesystem(rows, output_root)
    if output_root.exists():
        if _publication_matches(output_root, rows, inventory):
            return {"output_root": str(output_root), "shards": len(rows), "status": "already_published"}
        msg = f"Publication root already exists and does not match: {output_root}"
        raise FileExistsError(msg)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp.", dir=output_root.parent))
    try:
        _write_staging(staging, rows, inventory)
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if not _publication_matches(output_root, rows, inventory):
        msg = f"Published output failed hard-link verification: {output_root}"
        raise RuntimeError(msg)
    return {"output_root": str(output_root), "shards": len(rows), "status": "published"}
