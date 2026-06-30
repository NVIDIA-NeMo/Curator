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

Decomposes into a shard-discovery stage (parses the YAML) and a shard-reader
stage (streams each local tar, decodes audio in memory via lhotse/soundfile,
emits one ``AudioTask`` per utterance; nothing is written to disk).
"""

from __future__ import annotations

import json
import os
import re
import tarfile
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, BinaryIO

import soundfile as sf
import yaml
from loguru import logger

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.tasks import AudioTask, EmptyTask, FileGroupTask


def _expand_nemo_path(pattern: str) -> list[str]:
    """Expand NeMo brace patterns like ``__OP_0..N_CL_``."""
    match = re.search(r"_OP_(\d+)\.\.(\d+)_CL_", pattern)
    if not match:
        return [pattern]
    start, end = int(match.group(1)), int(match.group(2))
    if end < start:
        msg = f"NeMo brace range must be ascending, got {start}..{end} in {pattern!r}"
        raise ValueError(msg)
    prefix = pattern[: match.start()]
    suffix = pattern[match.end() :]
    return [f"{prefix}{i}{suffix}" for i in range(start, end + 1)]


def _open_tar(tar_path: str) -> tarfile.TarFile:
    """Open a local tar via lhotse's ``open_best`` in streaming mode (``r|*``)."""
    from lhotse.serialization import open_best

    if not os.path.exists(tar_path):
        msg = f"Tar file not found: {tar_path}"
        raise FileNotFoundError(msg)
    fileobj = open_best(tar_path, mode="rb")
    return tarfile.open(fileobj=fileobj, mode="r|*")


def _open_text_stream(path: str) -> BinaryIO:
    """Open a local text file as a binary stream."""
    from lhotse.serialization import open_best

    if not os.path.exists(path):
        msg = f"Text file not found: {path}"
        raise FileNotFoundError(msg)
    return open_best(path, mode="rb")


def _normalize_audio_path(path: str) -> str:
    """Strip leading ``./`` so manifest and tar-member paths compare consistently."""
    return path.lstrip("./")


def _path_suffix_overlap(a: list[str], b: list[str]) -> int:
    """Count shared trailing path components between two split paths."""
    n = 0
    for x, y in zip(reversed(a), reversed(b), strict=False):
        if x != y:
            break
        n += 1
    return n


class _ManifestIndex:
    """Resolve a tar member name to its manifest entry, collision-safe.

    Entries are keyed by normalized full path. Lookup prefers an exact
    full-path match, then falls back to basename. When several entries share a
    basename, the candidate with the longest trailing path-component overlap
    wins; genuine ties resolve to no match rather than an arbitrary one.
    """

    def __init__(self) -> None:
        self._by_path: dict[str, dict] = {}
        self._dup_paths: set[str] = set()
        self._basename_to_paths: dict[str, list[str]] = {}

    def add(self, audio_path: str, entry: dict) -> None:
        norm = _normalize_audio_path(audio_path)
        if norm in self._by_path and self._by_path[norm] != entry:
            self._dup_paths.add(norm)  # same path, different content -> ambiguous
        else:
            self._by_path[norm] = entry

    def finalize(self) -> None:
        for path in self._dup_paths:
            self._by_path.pop(path, None)
        self._basename_to_paths = {}
        for norm in self._by_path:
            self._basename_to_paths.setdefault(os.path.basename(norm), []).append(norm)

    def match(self, member_name: str) -> dict | None:
        norm = _normalize_audio_path(member_name)
        entry = self._by_path.get(norm)
        if entry is not None:
            return entry
        candidates = self._basename_to_paths.get(os.path.basename(norm))
        if not candidates:
            return None
        if len(candidates) == 1:
            return self._by_path.get(candidates[0])
        member_parts = norm.split("/")
        best, best_overlap, tied = None, 0, False
        for cand in candidates:
            overlap = _path_suffix_overlap(member_parts, cand.split("/"))
            if overlap > best_overlap:
                best, best_overlap, tied = cand, overlap, False
            elif overlap == best_overlap:
                tied = True
        if best is None or tied:
            return None
        return self._by_path.get(best)

    def __getitem__(self, member_name: str) -> dict:
        entry = self.match(member_name)
        if entry is None:
            raise KeyError(member_name)
        return entry

    def __contains__(self, member_name: str) -> bool:
        return self.match(member_name) is not None


def _iter_discovery_groups(config: object, yaml_path: str) -> list[dict[str, Any]]:
    """Validate the Granary discovery YAML root and return corpus-group dicts.

    Require a list of mappings at the top level (safe_load can return None /
    scalar / string, which would crash or silently mis-parse on iteration);
    skip non-mapping entries with a warning.
    """
    if config is None:
        msg = f"Granary YAML at {yaml_path} is empty (safe_load returned None)"
        raise ValueError(msg)
    if not isinstance(config, list):
        msg = f"Granary YAML at {yaml_path} must be a list of corpus-group mappings, got {type(config).__name__}"
        raise TypeError(msg)

    groups: list[dict[str, Any]] = []
    for idx, group in enumerate(config):
        if not isinstance(group, dict):
            logger.warning(
                "Skipping non-mapping entry at index {} in {} (got {})",
                idx,
                yaml_path,
                type(group).__name__,
            )
            continue
        groups.append(group)
    return groups


def _iter_input_cfg_entries(group: dict[str, Any], yaml_path: str) -> list[dict[str, Any]]:
    """Return validated ``input_cfg`` corpus entries from one top-level group."""
    raw = group.get("input_cfg", [])
    if raw is None:
        return []
    if not isinstance(raw, list):
        logger.warning(
            "Skipping corpus group in {} with non-list input_cfg (got {})",
            yaml_path,
            type(raw).__name__,
        )
        return []

    entries: list[dict[str, Any]] = []
    for idx, cfg in enumerate(raw):
        if not isinstance(cfg, dict):
            logger.warning(
                "Skipping non-mapping input_cfg entry at index {} in {}",
                idx,
                yaml_path,
            )
            continue
        entries.append(cfg)
    return entries


@dataclass
class NemoTarShardDiscoveryStage(ProcessingStage[EmptyTask, FileGroupTask]):
    """Parse a Granary YAML config and emit one ``FileGroupTask`` per shard.

    Each emitted task has ``data = [manifest_path, tar_path]``.

    Args:
        yaml_path: Path to the Granary YAML data config.
        corpus_filter: Include only these corpora (``None`` = all).
    """

    yaml_path: str
    name: str = "nemo_tar_shard_discovery"
    corpus_filter: list[str] | None = None
    output_dir: str | None = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers_per_node": 1}

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def _scan_completed_shards(self) -> set[str]:
        """Return shard keys with a ``.done`` marker under output_dir (resume skip set).

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
                    shard_key = rel[: -len(".jsonl.done")]
                    completed.add(shard_key)
        return completed

    @staticmethod
    def _manifest_to_rel_path(manifest_path: str, corpus: str) -> str:
        """Extract the shard-key path from a manifest path, starting at the corpus name.

        Example: ``/data/yodas/.../manifest_42.jsonl`` + corpus ``yodas`` ->
        ``yodas/.../manifest_42`` (the ``.jsonl`` extension is stripped).
        The corpus name must appear exactly once for unambiguous extraction.
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
            rel = rel[: -len(".jsonl")]
        elif rel.endswith(".json"):
            rel = rel[: -len(".json")]
        return rel

    def process(self, _task: EmptyTask) -> list[FileGroupTask]:  # noqa: C901, PLR0915
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
        invalid_corpora = 0
        invalid_shards = 0
        for group in _iter_discovery_groups(config, self.yaml_path):
            for cfg in _iter_input_cfg_entries(group, self.yaml_path):
                corpus = cfg.get("corpus", "unknown")
                corpora_seen += 1
                if self.corpus_filter and corpus not in self.corpus_filter:
                    continue
                if cfg.get("type", "nemo_tarred") != "nemo_tarred":
                    logger.warning(f"Skipping non-nemo_tarred corpus {corpus} (type={cfg.get('type')})")
                    continue
                manifest_pattern = cfg.get("manifest_filepath")
                tar_pattern = cfg.get("tarred_audio_filepaths")
                if not isinstance(manifest_pattern, str) or not isinstance(tar_pattern, str):
                    invalid_corpora += 1
                    logger.warning(
                        "Skipping corpus {} in {}: manifest_filepath and tarred_audio_filepaths are required strings",
                        corpus,
                        self.yaml_path,
                    )
                    continue
                manifest_paths = _expand_nemo_path(manifest_pattern)
                tar_paths = _expand_nemo_path(tar_pattern)
                if len(manifest_paths) != len(tar_paths):
                    msg = (
                        f"Manifest/tar count mismatch for corpus={corpus}: "
                        f"{len(manifest_paths)} manifests vs {len(tar_paths)} tars"
                    )
                    raise ValueError(msg)
                for mp, tp in zip(manifest_paths, tar_paths, strict=False):
                    shards_seen += 1
                    try:
                        shard_key = self._manifest_to_rel_path(mp, corpus)
                    except ValueError as exc:
                        invalid_shards += 1
                        logger.warning("Skipping manifest {} for corpus {}: {}", mp, corpus, exc)
                        continue
                    if shard_key in completed:
                        skipped += 1
                        continue
                    if self.output_dir:
                        partial = os.path.join(self.output_dir, f"{shard_key}.jsonl")
                        if os.path.exists(partial):
                            os.remove(partial)
                            logger.info(f"Removed partial output for {shard_key}")
                    shard_task = FileGroupTask(
                        dataset_name=corpus,
                        data=[mp, tp],
                        reader_config={"corpus": corpus, "shard_key": shard_key},
                    )
                    shard_task.task_id = shard_key
                    tasks.append(shard_task)

        logger.info(
            f"NemoTarShardDiscoveryStage: found {len(tasks)} shards, skipped {skipped} completed (corpus_filter={self.corpus_filter})"
        )
        # ``total_items_emitted``: framework ``num_items_processed`` counts 0 for
        # stages that synthesise work from config, so the summary builder falls
        # back to this when the framework count is 0.
        self._log_metrics(
            {
                "input_tasks": 1.0,
                "output_tasks": float(len(tasks)),
                "total_items_emitted": float(len(tasks)),
                "corpora_seen": float(corpora_seen),
                "shards_seen": float(shards_seen),
                "shards_emitted": float(len(tasks)),
                "shards_skipped_completed": float(skipped),
                "corpora_skipped_invalid": float(invalid_corpora),
                "shards_skipped_invalid": float(invalid_shards),
                "discovery_time_s": time.perf_counter() - t0,
            }
        )
        return tasks


@dataclass
class NemoTarShardReaderStage(ProcessingStage[FileGroupTask, AudioTask]):
    """Read a single NeMo tar shard and emit one ``AudioTask`` per utterance.

    Expects ``task.data = [manifest_path, tar_path]`` from
    ``NemoTarShardDiscoveryStage``. Audio is decoded in memory (lhotse
    ``open_best`` for tar streaming, ``soundfile`` for decode); nothing is
    written to disk. Each ``AudioTask`` carries a 1-D mono float32 waveform in
    ``task.data["waveform"]`` and the native rate in ``sample_rate`` (plus a
    ``sampling_rate`` alias for Granary v2 reference-script compatibility).

    Args:
        filepath_key: Manifest key for the audio filename inside the tar.
        duration_key: Manifest key for utterance duration (seconds).
        max_duration_s: Optional upper bound on emitted utterance duration.
    """

    name: str = "nemo_tar_shard_reader"
    filepath_key: str = "audio_filepath"
    duration_key: str = "duration"
    max_duration_s: float | None = None
    max_utterances_per_shard: int | None = None
    # Used ONLY to write a ``.done`` marker for zero-utterance shards, which
    # never reach the writer and would otherwise be re-queued forever on resume.
    # ``None`` -> no checkpoint dir (single-rank tutorial); resume not a concern.
    output_dir: str | None = None

    def __post_init__(self) -> None:
        if self.max_duration_s is not None and self.max_duration_s <= 0:
            msg = "max_duration_s must be positive when set"
            raise ValueError(msg)
        if self.max_utterances_per_shard is not None and self.max_utterances_per_shard <= 0:
            msg = "max_utterances_per_shard must be positive when set"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["waveform", "sample_rate", "sampling_rate", "corpus", "num_channels"]

    def num_workers(self) -> int | None:
        # Pin to 1 to bound memory (a shard's waveforms are held in memory).
        return 1

    def ray_stage_spec(self) -> dict[str, Any]:
        # IS_ACTOR_STAGE makes Ray Data honor num_workers()=1 (cluster-wide
        # analog of the per-node Xenna pin); IS_FANOUT_STAGE keeps 1-row/block.
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True, RayStageSpecKeys.IS_ACTOR_STAGE: True}

    def xenna_stage_spec(self) -> dict[str, Any]:
        # One reader actor per node: node-local decode, ~1 shard of memory/node.
        return {"num_workers_per_node": 1}

    def _read_manifest(self, path: str) -> tuple[_ManifestIndex, int]:
        index = _ManifestIndex()
        entry_count = 0
        skipped_lines = 0
        with _open_text_stream(path) as f:
            for raw_line in f:
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                stripped = line.strip()
                if not stripped:
                    continue
                entry_count += 1
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError:
                    skipped_lines += 1
                    logger.warning("Skipping invalid JSON line in manifest {}", path)
                    continue
                audio_path = entry.get(self.filepath_key)
                if audio_path is None:
                    skipped_lines += 1
                    logger.warning(
                        "Skipping manifest line missing {!r} in {}",
                        self.filepath_key,
                        path,
                    )
                    continue
                index.add(str(audio_path), entry)
        if skipped_lines:
            logger.warning(
                "Manifest {}: skipped {} line(s) (invalid JSON or missing {})",
                path,
                skipped_lines,
                self.filepath_key,
            )
        index.finalize()
        return index, entry_count

    def _mark_empty_shard_done(self, shard_key: str) -> None:
        """Write a ``<shard_key>.jsonl.done`` (count 0) for a zero-utterance shard.

        Mirrors the writer's marker so discovery skips the shard on resume; no
        ``.jsonl`` is written. No-op when ``output_dir`` is unset.
        """
        if not self.output_dir:
            return
        done_path = os.path.join(self.output_dir, f"{shard_key}.jsonl.done")
        try:
            os.makedirs(os.path.dirname(done_path), exist_ok=True)
            with open(done_path, "w") as f:
                f.write("0\n")
            logger.info(f"Shard {shard_key}: 0 utterances, wrote empty-shard marker {done_path}")
        except OSError as exc:
            logger.warning("Failed to write empty-shard marker for {}: {}", shard_key, exc)

    def process(self, task: FileGroupTask) -> list[AudioTask]:  # noqa: C901, PLR0912, PLR0915
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
                manifest_entry = manifest.match(tar_info.name)
                if manifest_entry is None:
                    continue

                entry = dict(manifest_entry)
                # Cheap pre-decode skip when the manifest has a usable duration.
                # Missing / non-numeric durations are re-checked post-decode below
                # so they cannot bypass the cap.
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
                # Only genuine decode/format failures count as "corrupt audio";
                # resource/dependency errors are excluded so they propagate
                # instead of being mislabeled and skipped.
                except (sf.LibsndfileError, ValueError, EOFError) as exc:
                    corrupt_audio_count += 1
                    logger.warning(f"Skipping corrupt audio {tar_info.name} in {tar_path}: {exc}")
                    continue

                num_channels = audio.shape[1] if audio.ndim > 1 else 1
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                utt_seconds = float(audio.shape[0]) / float(sample_rate) if sample_rate else 0.0
                # Post-decode duration enforcement using the authoritative decoded
                # length (covers rows the pre-decode skip could not bound).
                if self.max_duration_s is not None and utt_seconds > self.max_duration_s:
                    duration_filtered_count += 1
                    continue

                audio_members_matched += 1
                decoded_waveform_bytes += float(getattr(audio, "nbytes", 0))
                decoded_audio_seconds += utt_seconds

                entry["waveform"] = audio
                entry["sample_rate"] = sample_rate
                entry["sampling_rate"] = sample_rate
                entry["num_channels"] = num_channels
                entry["corpus"] = corpus

                audio_task = AudioTask(
                    dataset_name=corpus,
                    data=entry,
                    _metadata={**task._metadata, "_shard_key": shard_key},
                    _stage_perf=list(task._stage_perf),
                )
                audio_task.task_id = f"{shard_key}_{tar_info.name}"
                results.append(audio_task)
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

        # Empty / fully-filtered shard never reaches the writer, so mark it done
        # here so resume skips it instead of re-queueing it indefinitely.
        if shard_total == 0:
            self._mark_empty_shard_done(shard_key)

        logger.info(f"Shard {shard_key}: emitted {shard_total} AudioTasks")
        self._log_metrics(
            {
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
            }
        )
        return results


@dataclass
class NemoTarredAudioReader(CompositeStage[EmptyTask, AudioTask]):
    """Read NeMo-style tarred audio datasets from a Granary YAML config.

    Decomposes into ``NemoTarShardDiscoveryStage`` (parse YAML -> one
    ``FileGroupTask`` per shard) then ``NemoTarShardReaderStage`` (stream each
    tar, decode in memory -> ``AudioTask`` with waveform arrays).

    Args:
        yaml_path: Path to the Granary YAML data config.
        corpus_filter: Process only these corpora (``None`` = all).
        filepath_key: Manifest key for audio filenames inside tar archives.
        duration_key: Manifest key for utterance duration (seconds).
        max_duration_s: Optional upper bound on emitted utterance duration.
    """

    yaml_path: str
    name: str = "nemo_tarred_audio_reader"
    corpus_filter: list[str] | None = None
    filepath_key: str = "audio_filepath"
    duration_key: str = "duration"
    max_duration_s: float | None = None
    output_dir: str | None = None
    max_utterances_per_shard: int | None = None

    def __post_init__(self) -> None:
        super().__init__()

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
                output_dir=self.output_dir,
            ),
        ]

    def inputs(self) -> tuple[list[str], list[str]]:
        return self._stages[0].inputs()

    def outputs(self) -> tuple[list[str], list[str]]:
        return self._stages[-1].outputs()

    def decompose(self) -> list[ProcessingStage]:
        return self._stages
