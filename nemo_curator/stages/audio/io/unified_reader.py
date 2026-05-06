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

"""Unified audio reader using lhotse CutSet as internal bridge.

Handles both NeMo tarred (Granary YAML) and non-tarred S3 (corpusview
YAML) datasets through a single code path:

    manifest (local/S3) -> lhotse CutSet -> cut.load_audio() -> AudioTask

Decomposes into:
1. ``UnifiedDiscoveryStage`` — parses YAML, expands shards, checks .done
2. ``UnifiedReaderStage`` — manifest -> CutSet -> AudioTask
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

try:
    from nemo_curator.backends.experimental.utils import RayStageSpecKeys
except ModuleNotFoundError:
    RayStageSpecKeys = None

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.tasks import AudioTask, FileGroupTask, _EmptyTask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expand_nemo_path(pattern: str) -> list[str]:
    """Expand NeMo brace patterns like ``__OP_0..N_CL_``."""
    match = re.search(r"_OP_(\d+)\.\.(\d+)_CL_", pattern)
    if not match:
        return [pattern]
    start, end = int(match.group(1)), int(match.group(2))
    prefix = pattern[: match.start()]
    suffix = pattern[match.end() :]
    return [f"{prefix}{i}{suffix}" for i in range(start, end + 1)]


def _read_text(path: str, s3_config: dict[str, Any] | None = None) -> str:
    """Read text from a local path or S3 path."""
    if path.startswith("s3://"):
        from urllib.parse import urlparse

        import boto3

        cfg = s3_config or {}
        session = boto3.Session(profile_name=cfg.get("profile", "maglev"))
        client = session.client("s3", endpoint_url=cfg.get("endpoint_url", "https://pdx.s8k.io"))
        parsed = urlparse(path)
        resp = client.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
        return resp["Body"].read().decode("utf-8")
    with open(path, encoding="utf-8") as f:
        return f.read()


def _manifest_to_shard_key(manifest_path: str, corpus: str, language: str = "", shard_type: str = "") -> str:
    """Derive a unique shard key from a manifest path.

    For tarred (Granary) data the key is extracted from the filesystem
    path starting at the corpus directory — this matches the existing
    ``NemoTarredAudioReader`` convention and requires the YAML ``corpus``
    field to match a directory component (case-insensitive).

    For non-tarred (corpusview / S3) data the corpus name typically does
    not appear in the path, so the key is constructed as
    ``{corpus}/{language}/{manifest_stem}`` instead.
    """
    if shard_type == "nemo_tarred":
        parts = manifest_path.replace("\\", "/").split("/")
        parts_lower = [p.lower() for p in parts]
        corpus_lower = corpus.lower()
        matches = [i for i, p in enumerate(parts_lower) if p == corpus_lower]
        if matches:
            rel = "/".join(parts[matches[0]:])
            for ext in (".jsonl", ".json", ".jsonl.gz"):
                if rel.endswith(ext):
                    rel = rel[: -len(ext)]
                    break
            return rel

    basename = manifest_path.replace("\\", "/").rsplit("/", 1)[-1]
    for ext in (".jsonl", ".json", ".jsonl.gz"):
        if basename.endswith(ext):
            basename = basename[: -len(ext)]
            break
    parts = [corpus]
    if language:
        parts.append(language)
    parts.append(basename)
    return "/".join(parts)


# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------


def _parse_granary_yaml(config: list[dict], corpus_filter: list[str] | None) -> list[dict[str, Any]]:
    """Parse Granary-style YAML (input_cfg with manifest + tar pairs)."""
    shards = []
    for group in config:
        for cfg in group.get("input_cfg", []):
            corpus = cfg.get("corpus", "unknown")
            if corpus_filter and corpus not in corpus_filter:
                continue
            if cfg.get("type", "nemo_tarred") != "nemo_tarred":
                continue
            manifest_paths = _expand_nemo_path(cfg["manifest_filepath"])
            tar_paths = _expand_nemo_path(cfg["tarred_audio_filepaths"])
            if len(manifest_paths) != len(tar_paths):
                msg = f"Manifest/tar count mismatch for {corpus}: {len(manifest_paths)} vs {len(tar_paths)}"
                raise ValueError(msg)
            for mp, tp in zip(manifest_paths, tar_paths, strict=False):
                shards.append({"type": "nemo_tarred", "corpus": corpus, "manifest_path": mp, "tar_path": tp, "language": cfg.get("language", "")})
    return shards


def _parse_corpusview_yaml(
    config: list[dict], corpus_filter: list[str] | None, site: str,
) -> list[dict[str, Any]]:
    """Parse corpusview-style YAML (flat list with paths.{site})."""
    shards = []
    for entry in config:
        corpus = entry.get("corpus", "unknown")
        if corpus_filter and corpus not in corpus_filter:
            continue
        paths = entry.get("paths", {})
        site_paths = paths.get(site, paths.get("default", {}))
        manifest_pattern = site_paths.get("manifest_filepath", "") if isinstance(site_paths, dict) else str(site_paths)
        if not manifest_pattern or manifest_pattern == site:
            default_ref = paths.get("default", {})
            resolved_site = default_ref.get("manifest_filepath", site) if isinstance(default_ref, dict) else str(default_ref)
            actual = paths.get(resolved_site, {})
            manifest_pattern = actual.get("manifest_filepath", "") if isinstance(actual, dict) else str(actual)
        if not manifest_pattern:
            continue
        for mp in _expand_nemo_path(manifest_pattern):
            shards.append({"type": entry.get("type", "nemo"), "corpus": corpus, "manifest_path": mp, "language": entry.get("language", "")})
    return shards


def _detect_and_parse_yaml(yaml_text: str, corpus_filter: list[str] | None, site: str) -> list[dict[str, Any]]:
    """Auto-detect YAML format and parse into shard descriptors."""
    import yaml

    config = yaml.safe_load(yaml_text)
    if not isinstance(config, list) or not config:
        msg = f"Expected a YAML list, got {type(config)}"
        raise ValueError(msg)
    if "input_cfg" in config[0]:
        return _parse_granary_yaml(config, corpus_filter)
    return _parse_corpusview_yaml(config, corpus_filter, site)


# ---------------------------------------------------------------------------
# Stage 1: Discovery
# ---------------------------------------------------------------------------


@dataclass
class UnifiedDiscoveryStage(ProcessingStage[_EmptyTask, FileGroupTask]):
    """Parse YAML config and emit one ``FileGroupTask`` per shard.

    Auto-detects Granary (tarred) vs corpusview (S3 individual) format.
    """

    name: str = "unified_discovery"
    yaml_path: str = ""
    site: str = "pdx"
    corpus_filter: list[str] | None = None
    output_dir: str | None = None
    s3_config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.yaml_path:
            msg = "yaml_path is required"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers_per_node": 1}

    def _scan_completed_shards(self) -> set[str]:
        if not self.output_dir or not os.path.isdir(self.output_dir):
            return set()
        completed: set[str] = set()
        for root, _dirs, files in os.walk(self.output_dir):
            for fname in files:
                if fname.endswith(".jsonl.done"):
                    rel = os.path.relpath(os.path.join(root, fname), self.output_dir)
                    completed.add(rel[: -len(".jsonl.done")])
        return completed

    def process(self, _task: _EmptyTask) -> list[FileGroupTask]:
        yaml_text = _read_text(self.yaml_path, self.s3_config)
        shard_descs = _detect_and_parse_yaml(yaml_text, self.corpus_filter, self.site)

        completed = self._scan_completed_shards()
        if completed:
            logger.info(f"Checkpoint: {len(completed)} shards already completed, first 10: {sorted(completed)[:10]}")

        tasks: list[FileGroupTask] = []
        skipped = 0
        for desc in shard_descs:
            corpus = desc["corpus"]
            language = desc.get("language", "")
            shard_key = _manifest_to_shard_key(desc["manifest_path"], corpus, language, desc["type"])
            if shard_key in completed:
                skipped += 1
                continue
            if self.output_dir:
                partial = os.path.join(self.output_dir, f"{shard_key}.jsonl")
                if os.path.exists(partial):
                    os.remove(partial)
                    logger.info(f"Removed partial output for {shard_key}")

            data = [desc["manifest_path"]]
            if "tar_path" in desc:
                data.append(desc["tar_path"])

            tasks.append(FileGroupTask(
                task_id=shard_key,
                dataset_name=corpus,
                data=data,
                reader_config={"type": desc["type"], "corpus": corpus, "shard_key": shard_key, "language": desc.get("language", "")},
            ))

        logger.info(f"UnifiedDiscovery: {len(tasks)} shards to process, {skipped} skipped")
        return tasks


# ---------------------------------------------------------------------------
# Stage 2: Reader (CutSet bridge)
# ---------------------------------------------------------------------------


@dataclass
class UnifiedReaderStage(ProcessingStage[FileGroupTask, AudioTask]):
    """Read a manifest shard and emit AudioTasks.

    For tarred data (``task.data = [manifest, tar_path]``), streams the tar
    via lhotse ``open_best`` and decodes audio with soundfile — same
    approach as ``NemoTarShardReaderStage``.

    For individual files (local or S3), builds a lhotse CutSet and uses
    ``cut.load_audio()`` which natively handles S3 URLs via
    ``AudioSource(type="url")``.
    """

    name: str = "unified_reader"
    filepath_key: str = "audio_filepath"
    s3_config: dict[str, Any] | None = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["waveform", "sampling_rate", "sample_rate", "corpus", "num_channels"]

    def ray_stage_spec(self) -> dict[str, Any]:
        if RayStageSpecKeys is not None:
            return {RayStageSpecKeys.IS_FANOUT_STAGE: True}
        return {"is_fanout_stage": True}

    # -- tarred path: stream tar, decode with soundfile -------------------

    def _process_tarred(  # noqa: PLR0913
        self, entries: list[dict[str, Any]], tar_path: str,
        corpus: str, shard_key: str, language: str, metadata: dict,
    ) -> list[AudioTask]:
        from io import BytesIO

        import soundfile as sf

        from nemo_curator.stages.audio.io.nemo_tarred_reader import _open_tar

        manifest = {e[self.filepath_key]: e for e in entries}
        ais_endpoint = (self.s3_config or {}).get("ais_endpoint_url")
        tar = _open_tar(tar_path, ais_endpoint)

        results: list[AudioTask] = []
        loaded = 0
        for tar_info in tar:
            if not tar_info.isfile() or tar_info.name not in manifest:
                continue
            raw = tar.extractfile(tar_info).read()
            try:
                audio, sr = sf.read(BytesIO(raw), dtype="float32")
            except Exception:  # noqa: BLE001
                logger.warning(f"Skipping corrupt audio {tar_info.name} in {tar_path}")
                continue

            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            loaded += 1
            if loaded % max(1, len(manifest) // 10) == 0 or loaded == 1:
                logger.info(f"  [{shard_key}] loaded {loaded}/{len(manifest)}")

            entry_data = dict(manifest[tar_info.name])
            entry_data["waveform"] = audio
            entry_data["sampling_rate"] = sr
            entry_data["sample_rate"] = sr
            entry_data["num_channels"] = 1
            entry_data["corpus"] = corpus
            if language and "source_lang" not in entry_data:
                entry_data["source_lang"] = language

            results.append(AudioTask(
                task_id=f"{shard_key}_{tar_info.name}",
                dataset_name=corpus,
                data=entry_data,
                _metadata={**metadata, "_shard_key": shard_key},
            ))

        tar.close()
        return results

    # -- individual files path: lhotse CutSet -----------------------------

    def _build_cutset(self, entries: list[dict[str, Any]]) -> Any:  # noqa: ANN401
        """Build a lhotse CutSet from manifest entries (individual files)."""
        from lhotse import CutSet, MonoCut, Recording
        from lhotse.audio import AudioSource

        cuts = []
        for i, entry in enumerate(entries):
            audio_path = entry.get(self.filepath_key, "")
            duration = entry.get("duration", 0.0)
            sr = entry.get("sample_rate", 16000)
            sample_id = entry.get("id", str(i))

            if audio_path.startswith("s3://"):
                source = AudioSource(type="url", channels=[0], source=audio_path)
            else:
                source = AudioSource(type="file", channels=[0], source=audio_path)

            recording = Recording(
                id=sample_id, sources=[source], sampling_rate=sr,
                num_samples=int(duration * sr), duration=duration,
            )
            cuts.append(MonoCut(id=sample_id, start=0.0, duration=duration, channel=0, recording=recording))

        return CutSet.from_cuts(cuts)

    def _process_individual(
        self, entries: list[dict[str, Any]],
        corpus: str, shard_key: str, language: str, metadata: dict,
    ) -> list[AudioTask]:
        cutset = self._build_cutset(entries)
        results: list[AudioTask] = []

        n_entries = len(entries)
        for i, (cut, entry) in enumerate(zip(cutset, entries, strict=False)):
            try:
                audio = cut.load_audio().squeeze()
            except Exception:  # noqa: BLE001
                logger.warning(f"Skipping unreadable audio: {entry.get(self.filepath_key, cut.id)}")
                continue

            if (i + 1) % max(1, n_entries // 10) == 0 or i == 0:
                logger.info(f"  [{shard_key}] loaded {i + 1}/{n_entries}")

            if audio.ndim > 1:
                audio = audio.mean(axis=0)

            entry_data = dict(entry)
            entry_data["waveform"] = audio.astype(np.float32)
            entry_data["sampling_rate"] = cut.recording.sampling_rate
            entry_data["sample_rate"] = cut.recording.sampling_rate
            entry_data["num_channels"] = 1
            entry_data["corpus"] = corpus
            if language and "source_lang" not in entry_data:
                entry_data["source_lang"] = language

            results.append(AudioTask(
                task_id=f"{shard_key}_{cut.id}",
                dataset_name=corpus,
                data=entry_data,
                _metadata={**metadata, "_shard_key": shard_key},
            ))

        return results

    # -- dispatch ---------------------------------------------------------

    def process(self, task: FileGroupTask) -> list[AudioTask]:
        manifest_path = task.data[0]
        tar_path = task.data[1] if len(task.data) >= 2 else None  # noqa: PLR2004
        corpus = task.reader_config.get("corpus", "unknown")
        shard_key = task.reader_config.get("shard_key", task.task_id)
        language = task.reader_config.get("language", "")

        text = _read_text(manifest_path, self.s3_config)
        entries = [json.loads(line) for line in text.splitlines() if line.strip()]

        if tar_path:
            logger.info(f"Reading shard {shard_key}: {len(entries)} entries via tar stream")
            results = self._process_tarred(entries, tar_path, corpus, shard_key, language, dict(task._metadata))
        else:
            logger.info(f"Reading shard {shard_key}: {len(entries)} entries via CutSet")
            results = self._process_individual(entries, corpus, shard_key, language, dict(task._metadata))

        for r in results:
            r._metadata["_shard_total"] = len(results)
            r._stage_perf = list(task._stage_perf)

        logger.info(f"Shard {shard_key}: emitted {len(results)} AudioTasks")
        return results


# ---------------------------------------------------------------------------
# Composite stage (user-facing API)
# ---------------------------------------------------------------------------


@dataclass
class UnifiedAudioReader(CompositeStage[_EmptyTask, AudioTask]):
    """Unified reader for NeMo audio datasets from local or S3.

    Auto-detects YAML format and data type.  Uses lhotse CutSet
    internally for audio loading.
    """

    name: str = "unified_audio_reader"
    yaml_path: str = ""
    site: str = "pdx"
    corpus_filter: list[str] | None = None
    output_dir: str | None = None
    filepath_key: str = "audio_filepath"
    s3_config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        super().__init__()
        if not self.yaml_path:
            msg = "yaml_path is required for UnifiedAudioReader"
            raise ValueError(msg)
        self._stages: list[ProcessingStage] = [
            UnifiedDiscoveryStage(
                yaml_path=self.yaml_path, site=self.site,
                corpus_filter=self.corpus_filter, output_dir=self.output_dir,
                s3_config=self.s3_config,
            ),
            UnifiedReaderStage(
                filepath_key=self.filepath_key,
                s3_config=self.s3_config,
            ),
        ]

    def inputs(self) -> tuple[list[str], list[str]]:
        return self._stages[0].inputs()

    def outputs(self) -> tuple[list[str], list[str]]:
        return self._stages[-1].outputs()

    def decompose(self) -> list[ProcessingStage]:
        return self._stages
