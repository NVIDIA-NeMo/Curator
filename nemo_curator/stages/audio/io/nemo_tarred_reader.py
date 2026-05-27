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

"""Reader for NeMo-style tarred audio datasets (e.g. Granary YAML configs).

Decomposes into a shard-discovery stage that parses the YAML config and a
shard-reader stage that streams each local tar, decodes audio in memory via
lhotse/soundfile, and emits one ``AudioTask`` per utterance with the waveform
as a numpy array; no files are written to disk.
"""

from __future__ import annotations

import json
import os
import re
import tarfile
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import soundfile as sf
import yaml
from loguru import logger

from nemo_curator.backends.utils import RayStageSpecKeys
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


def _open_tar(tar_path: str) -> tarfile.TarFile:
    """Open a local tar file via lhotse's ``open_best``.

    Uses streaming mode (``r|*``) so the tar is read sequentially without
    seeking.
    """
    from lhotse.serialization import open_best

    if not os.path.exists(tar_path):
        msg = f"Tar file not found: {tar_path}"
        raise FileNotFoundError(msg)
    fileobj = open_best(tar_path, mode="rb")
    return tarfile.open(fileobj=fileobj, mode="r|*")


def _open_text_stream(path: str) -> Any:
    """Open a local text file as a binary stream."""
    from lhotse.serialization import open_best

    if not os.path.exists(path):
        msg = f"Text file not found: {path}"
        raise FileNotFoundError(msg)
    return open_best(path, mode="rb")


# ---------------------------------------------------------------------------
# Stage 1: Shard discovery
# ---------------------------------------------------------------------------

@dataclass
class NemoTarShardDiscoveryStage(ProcessingStage[_EmptyTask, FileGroupTask]):
    """Parse a Granary YAML config and emit one ``FileGroupTask`` per shard.

    Each emitted task has ``data = [manifest_path, tar_path]``.

    Args:
        yaml_path: Path to the Granary YAML data config.
        corpus_filter: Only include shards whose ``corpus`` field is in this
            list.  ``None`` means include everything.
    """

    name: str = "nemo_tar_shard_discovery"
    yaml_path: str = ""
    corpus_filter: list[str] | None = None
    output_dir: str | None = None

    def __post_init__(self) -> None:
        if not self.yaml_path:
            msg = "yaml_path is required for NemoTarShardDiscoveryStage"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers_per_node": 1}

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def _scan_completed_shards(self) -> set[str]:
        """Scan output_dir recursively for .done marker files and return completed shard keys.

        Keys are relative paths like ``yodas/0_from_captions/en/sharded_manifests/manifest_42``.
        """
        if not self.output_dir:
            return set()
        completed: set[str] = set()
        if not os.path.isdir(self.output_dir):
            return completed
        for root, _dirs, files in os.walk(self.output_dir):
            for fname in files:
                if fname.endswith(".jsonl.done"):
                    rel = os.path.relpath(os.path.join(root, fname), self.output_dir)
                    shard_key = rel[:-len(".jsonl.done")]
                    completed.add(shard_key)
        return completed

    @staticmethod
    def _manifest_to_rel_path(manifest_path: str, corpus: str) -> str:
        """Extract relative output path from a manifest path, starting at the corpus name.

        Example::

            manifest_path = "/data/yodas/0_from_captions/en/sharded_manifests/manifest_42.jsonl"
            corpus = "yodas"
            -> "yodas/0_from_captions/en/sharded_manifests/manifest_42"

        The ``.jsonl`` extension is stripped so it can be used as a shard key.
        """
        parts = manifest_path.replace("\\", "/").split("/")
        parts_lower = [p.lower() for p in parts]
        corpus_lower = corpus.lower()
        matches = [i for i, p in enumerate(parts_lower) if p == corpus_lower]
        if len(matches) == 0:
            msg = (
                f"Corpus name '{corpus}' not found in manifest path: {manifest_path}. "
                f"The YAML 'corpus' field must match a directory component in the manifest path (case-insensitive)."
            )
            raise ValueError(msg)
        if len(matches) > 1:
            msg = (
                f"Corpus name '{corpus}' appears {len(matches)} times in manifest path: {manifest_path}. "
                f"It must appear exactly once for unambiguous path extraction."
            )
            raise ValueError(msg)
        idx = matches[0]
        rel = "/".join(parts[idx:])
        if rel.endswith(".jsonl"):
            rel = rel[:-len(".jsonl")]
        elif rel.endswith(".json"):
            rel = rel[:-len(".json")]
        return rel

    def process(self, _task: _EmptyTask) -> list[FileGroupTask]:  # noqa: C901
        t0 = time.perf_counter()
        completed = self._scan_completed_shards()
        if completed:
            logger.info(f"Checkpoint: {len(completed)} shards already completed, will skip them")
            logger.info(f"Completed shard keys (first 10): {sorted(completed)[:10]}")

        with open(self.yaml_path) as f:
            config = yaml.safe_load(f)

        tasks: list[FileGroupTask] = []
        corpora_seen = 0
        shards_seen = 0
        skipped = 0
        for group in config:
            for cfg in group.get("input_cfg", []):
                corpus = cfg.get("corpus", "unknown")
                corpora_seen += 1
                if self.corpus_filter and corpus not in self.corpus_filter:
                    continue
                if cfg.get("type", "nemo_tarred") != "nemo_tarred":
                    logger.warning(f"Skipping non-nemo_tarred corpus {corpus} (type={cfg.get('type')})")
                    continue
                manifest_paths = _expand_nemo_path(cfg["manifest_filepath"])
                tar_paths = _expand_nemo_path(cfg["tarred_audio_filepaths"])
                if len(manifest_paths) != len(tar_paths):
                    msg = (
                        f"Manifest/tar count mismatch for corpus={corpus}: "
                        f"{len(manifest_paths)} manifests vs {len(tar_paths)} tars"
                    )
                    raise ValueError(msg)
                for mp, tp in zip(manifest_paths, tar_paths, strict=False):
                    shards_seen += 1
                    shard_key = self._manifest_to_rel_path(mp, corpus)
                    if shard_key in completed:
                        skipped += 1
                        continue
                    if self.output_dir:
                        partial = os.path.join(self.output_dir, f"{shard_key}.jsonl")
                        if os.path.exists(partial):
                            os.remove(partial)
                            logger.info(f"Removed partial output for {shard_key}")
                    tasks.append(
                        FileGroupTask(
                            task_id=shard_key,
                            dataset_name=corpus,
                            data=[mp, tp],
                            reader_config={"corpus": corpus, "shard_key": shard_key},
                        )
                    )

        logger.info(
            f"NemoTarShardDiscoveryStage: found {len(tasks)} shards, skipped {skipped} completed (corpus_filter={self.corpus_filter})"
        )
        self._log_metrics({
            "input_tasks": 1.0,
            "output_tasks": float(len(tasks)),
            "corpora_seen": float(corpora_seen),
            "shards_seen": float(shards_seen),
            "shards_emitted": float(len(tasks)),
            "shards_skipped_completed": float(skipped),
            "discovery_time_s": time.perf_counter() - t0,
        })
        return tasks


# ---------------------------------------------------------------------------
# Stage 2: Shard reader (in-memory via lhotse/soundfile)
# ---------------------------------------------------------------------------

@dataclass
class NemoTarShardReaderStage(ProcessingStage[FileGroupTask, AudioTask]):
    """Read a single NeMo tar shard and emit one ``AudioTask`` per utterance.

    Expects ``task.data = [manifest_path, tar_path]`` as produced by
    ``NemoTarShardDiscoveryStage``.

    Audio is decoded entirely in memory using lhotse's ``open_best`` for
    tar streaming and ``soundfile`` for audio decoding.  No files are
    written to disk.  Each emitted ``AudioTask`` carries the waveform as
    a 1-D numpy float32 array (mono) in ``task.data["waveform"]`` and the
    native sample rate in ``task.data["sample_rate"]``. A
    ``task.data["sampling_rate"]`` alias is also emitted for compatibility
    with Granary v2 reference scripts.

    Args:
        filepath_key: Manifest key that identifies the audio filename
            inside the tar archive.
        duration_key: Manifest key containing utterance duration in seconds.
        max_duration_s: Optional upper bound for emitted utterance duration.
    """

    name: str = "nemo_tar_shard_reader"
    filepath_key: str = "audio_filepath"
    duration_key: str = "duration"
    max_duration_s: float | None = None
    max_utterances_per_shard: int | None = None
    num_workers_override: int | None = None
    num_workers_per_node: int | None = None

    def __post_init__(self) -> None:
        if self.max_duration_s is not None and self.max_duration_s <= 0:
            msg = "max_duration_s must be positive when set"
            raise ValueError(msg)
        if self.max_utterances_per_shard is not None and self.max_utterances_per_shard <= 0:
            msg = "max_utterances_per_shard must be positive when set"
            raise ValueError(msg)
        if self.num_workers_override is not None and self.num_workers_override <= 0:
            msg = "num_workers_override must be positive when set"
            raise ValueError(msg)
        if self.num_workers_per_node is not None and self.num_workers_per_node <= 0:
            msg = "num_workers_per_node must be positive when set"
            raise ValueError(msg)

    def num_workers(self) -> int | None:
        return self.num_workers_override

    def xenna_stage_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if self.num_workers_override is not None:
            spec["num_workers"] = self.num_workers_override
        if self.num_workers_per_node is not None:
            spec["num_workers_per_node"] = self.num_workers_per_node
        return spec

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["waveform", "sample_rate", "sampling_rate", "corpus", "num_channels"]

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    @staticmethod
    def _manifest_lookup_keys(path: str) -> set[str]:
        stripped = path.lstrip("./")
        return {path, stripped, os.path.basename(stripped)}

    def _read_manifest(self, path: str) -> tuple[dict[str, dict], int]:
        entries: dict[str, dict] = {}
        duplicate_keys: set[str] = set()
        entry_count = 0
        with _open_text_stream(path) as f:
            for raw_line in f:
                if isinstance(raw_line, bytes):
                    raw_line = raw_line.decode("utf-8")
                stripped = raw_line.strip()
                if stripped:
                    entry_count += 1
                    entry = json.loads(stripped)
                    audio_path = str(entry[self.filepath_key])
                    for lookup_key in self._manifest_lookup_keys(audio_path):
                        if lookup_key in entries and entries[lookup_key] != entry:
                            duplicate_keys.add(lookup_key)
                        else:
                            entries[lookup_key] = entry
        for lookup_key in duplicate_keys:
            entries.pop(lookup_key, None)
        return entries, entry_count

    def process(self, task: FileGroupTask) -> list[AudioTask]:
        t0 = time.perf_counter()
        manifest_path, tar_path = task.data[0], task.data[1]
        corpus = task.reader_config.get("corpus", "unknown")
        shard_key = task.reader_config.get("shard_key", task.task_id)

        manifest_t0 = time.perf_counter()
        manifest, manifest_entry_count = self._read_manifest(manifest_path)
        manifest_elapsed = time.perf_counter() - manifest_t0

        logger.info(f"Reading shard {shard_key}: {tar_path} ({manifest_entry_count} manifest entries)")

        open_t0 = time.perf_counter()
        tar = _open_tar(tar_path)
        tar_open_elapsed = time.perf_counter() - open_t0
        results: list[AudioTask] = []
        tar_members_seen = 0
        audio_members_matched = 0
        corrupt_audio_count = 0
        duration_filtered_count = 0
        decoded_audio_seconds = 0.0
        decoded_waveform_bytes = 0.0
        decode_elapsed = 0.0

        try:
            for tar_info in tar:
                tar_members_seen += 1
                if not tar_info.isfile():
                    continue
                manifest_entry = next(
                    (manifest[key] for key in self._manifest_lookup_keys(tar_info.name) if key in manifest),
                    None,
                )
                if manifest_entry is None:
                    continue

                entry = dict(manifest_entry)
                if self.max_duration_s is not None and self.duration_key in entry:
                    try:
                        duration_s = float(entry[self.duration_key])
                    except (TypeError, ValueError):
                        duration_s = None
                    if duration_s is not None and duration_s > self.max_duration_s:
                        duration_filtered_count += 1
                        continue

                fobj = tar.extractfile(tar_info)
                if fobj is None:
                    corrupt_audio_count += 1
                    logger.warning(f"Skipping non-regular tar member {tar_info.name} in {tar_path}")
                    continue
                raw_audio = fobj.read()
                try:
                    decode_t0 = time.perf_counter()
                    audio, sample_rate = sf.read(BytesIO(raw_audio), dtype="float32")
                    decode_elapsed += time.perf_counter() - decode_t0
                except Exception:  # noqa: BLE001
                    corrupt_audio_count += 1
                    logger.warning(f"Skipping corrupt audio {tar_info.name} in {tar_path}")
                    continue

                audio_members_matched += 1
                num_channels = audio.shape[1] if audio.ndim > 1 else 1

                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                decoded_waveform_bytes += float(getattr(audio, "nbytes", 0))
                if sample_rate:
                    decoded_audio_seconds += float(audio.shape[0]) / float(sample_rate)

                entry["waveform"] = audio
                entry["sample_rate"] = sample_rate
                entry["sampling_rate"] = sample_rate
                entry["num_channels"] = num_channels
                entry["corpus"] = corpus

                results.append(
                    AudioTask(
                        task_id=f"{shard_key}_{tar_info.name}",
                        dataset_name=corpus,
                        data=entry,
                        _metadata={**task._metadata, "_shard_key": shard_key},
                        _stage_perf=list(task._stage_perf),
                    )
                )
                if self.max_utterances_per_shard and len(results) >= self.max_utterances_per_shard:
                    logger.info(
                        "Shard {}: reached max_utterances_per_shard={} and stopped early",
                        shard_key,
                        self.max_utterances_per_shard,
                    )
                    break
        finally:
            tar.close()

        shard_total = len(results)
        for result_task in results:
            result_task._metadata["_shard_total"] = shard_total

        logger.info(f"Shard {shard_key}: emitted {shard_total} AudioTasks")
        self._log_metrics({
            "input_shards": 1.0,
            "output_tasks": float(shard_total),
            "output_utterances": float(shard_total),
            "manifest_entries": float(manifest_entry_count),
            "tar_members_seen": float(tar_members_seen),
            "audio_members_decoded": float(audio_members_matched),
            "corrupt_audio_count": float(corrupt_audio_count),
            "duration_filtered_count": float(duration_filtered_count),
            "utterances_emitted": float(shard_total),
            "utterance_limit_hit": float(
                bool(self.max_utterances_per_shard and shard_total >= self.max_utterances_per_shard)
            ),
            "audio_duration_s": decoded_audio_seconds,
            "waveform_bytes": decoded_waveform_bytes,
            "manifest_read_time_s": manifest_elapsed,
            "tar_open_time_s": tar_open_elapsed,
            "audio_decode_time_s": decode_elapsed,
            "reader_total_time_s": time.perf_counter() - t0,
        })
        return results


# ---------------------------------------------------------------------------
# Composite stage (user-facing API)
# ---------------------------------------------------------------------------

@dataclass
class NemoTarredAudioReader(CompositeStage[_EmptyTask, AudioTask]):
    """Read NeMo-style tarred audio datasets from a Granary YAML config.

    Decomposes into:

    1. ``NemoTarShardDiscoveryStage`` - parses the YAML, emits one
       ``FileGroupTask`` per shard.
    2. ``NemoTarShardReaderStage`` - streams each tar via lhotse, decodes
       audio in memory, emits ``AudioTask`` objects with waveform arrays.

    Args:
        yaml_path: Path to the Granary YAML data config.
        corpus_filter: Only process shards whose ``corpus`` matches.
        filepath_key: Manifest key for audio filenames inside tar archives.
        duration_key: Manifest key containing utterance duration in seconds.
        max_duration_s: Optional upper bound for emitted utterance duration.
    """

    name: str = "nemo_tarred_audio_reader"
    yaml_path: str = ""
    corpus_filter: list[str] | None = None
    filepath_key: str = "audio_filepath"
    duration_key: str = "duration"
    max_duration_s: float | None = None
    output_dir: str | None = None
    max_utterances_per_shard: int | None = None
    reader_num_workers: int | None = None
    reader_num_workers_per_node: int | None = None

    def __post_init__(self) -> None:
        super().__init__()
        if not self.yaml_path:
            msg = "yaml_path is required for NemoTarredAudioReader"
            raise ValueError(msg)

        self._stages: list[ProcessingStage] = [
            NemoTarShardDiscoveryStage(
                yaml_path=self.yaml_path,
                corpus_filter=self.corpus_filter,
                output_dir=self.output_dir,
            ),
            NemoTarShardReaderStage(
                filepath_key=self.filepath_key,
                duration_key=self.duration_key,
                max_duration_s=self.max_duration_s,
                max_utterances_per_shard=self.max_utterances_per_shard,
                num_workers_override=self.reader_num_workers,
                num_workers_per_node=self.reader_num_workers_per_node,
            ),
        ]

    def inputs(self) -> tuple[list[str], list[str]]:
        return self._stages[0].inputs()

    def outputs(self) -> tuple[list[str], list[str]]:
        return self._stages[-1].outputs()

    def decompose(self) -> list[ProcessingStage]:
        return self._stages
