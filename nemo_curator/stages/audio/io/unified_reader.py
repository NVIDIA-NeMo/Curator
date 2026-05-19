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

"""Unified audio reader using NeMo's lhotse adapters.

Handles NeMo ``input_cfg`` YAML configs (``nemo_tarred`` and ``nemo``
types) through NeMo's ``LazyNeMoIterator`` and
``LazyNeMoTarredIterator``:

    YAML (input_cfg) -> discovery (shard expansion + checkpointing)
                     -> NeMo lhotse adapter -> CutSet -> cut.load_audio() -> AudioTask

Decomposes into:
1. ``UnifiedDiscoveryStage`` — parses ``input_cfg`` YAML, expands shards, checks .done
2. ``UnifiedReaderStage`` — manifest -> NeMo CutSet -> AudioTask (format-agnostic)
"""

from __future__ import annotations

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
    """Expand **all** NeMo brace patterns like ``_OP_0..N_CL_`` recursively."""
    match = re.search(r"_OP_(\d+)\.\.(\d+)_CL_", pattern)
    if not match:
        return [pattern]
    start, end = int(match.group(1)), int(match.group(2))
    prefix = pattern[: match.start()]
    suffix = pattern[match.end() :]
    results: list[str] = []
    for i in range(start, end + 1):
        results.extend(_expand_nemo_path(f"{prefix}{i}{suffix}"))
    return results


def _manifest_to_shard_key(manifest_path: str, corpus: str) -> str:
    """Derive a shard key from a manifest path starting at the corpus directory.

    Matches the logic in ``NemoTarShardDiscoveryStage._manifest_to_rel_path``:
    finds the corpus name (case-insensitive, must appear exactly once) in
    the path components and returns everything from that point onward with
    the file extension stripped.
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
    if rel.endswith(".jsonl.gz"):
        rel = rel[: -len(".jsonl.gz")]
    elif rel.endswith(".jsonl"):
        rel = rel[: -len(".jsonl")]
    elif rel.endswith(".json"):
        rel = rel[: -len(".json")]
    return rel


def _s3_to_pipe(s3_path: str) -> str:
    """Convert ``s3://BUCKET/KEY`` to a ``pipe:curl`` command for AIS.

    Uses ``AIS_ENDPOINT`` and ``AIS_AUTHN_TOKEN`` environment variables.
    """
    from urllib.parse import urlparse

    parsed = urlparse(s3_path)
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    endpoint = os.environ.get("AIS_ENDPOINT", "")
    if not endpoint:
        msg = "AIS_ENDPOINT env var required for s3:// tar paths"
        raise RuntimeError(msg)
    url = f"{endpoint.rstrip('/')}/v1/objects/{bucket}/{key}?provider=s3"
    token = os.environ.get("AIS_AUTHN_TOKEN", "")
    ca_cert = os.environ.get("AIS_CLIENT_CA", "")
    cacert_flag = f" --cacert '{ca_cert}'" if ca_cert else ""
    if token:
        return f"pipe:curl -sL{cacert_flag} -H 'Authorization: Bearer {token}' '{url}'"
    return f"pipe:curl -sL{cacert_flag} '{url}'"


# ---------------------------------------------------------------------------
# YAML parsing (input_cfg format only)
# ---------------------------------------------------------------------------


def _parse_input_cfg(yaml_path: str, corpus_filter: list[str] | None) -> list[dict[str, Any]]:
    """Parse a NeMo ``input_cfg`` YAML into shard descriptors.

    Each descriptor has ``manifest_path``, optional ``tar_path``,
    ``corpus``, and ``language``.

    Only supports the standard NeMo config format with ``input_cfg``
    entries of type ``nemo_tarred`` or ``nemo``.
    """
    import yaml

    with open(yaml_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, list) or not config:
        msg = f"Expected a YAML list, got {type(config)}"
        raise ValueError(msg)

    shards: list[dict[str, Any]] = []
    for group in config:
        for cfg in group.get("input_cfg", [group]):
            corpus = cfg.get("corpus", "unknown")
            if corpus_filter and corpus not in corpus_filter:
                continue

            language = cfg.get("language", "")

            if "tarred_audio_filepaths" in cfg:
                manifest_paths = _expand_nemo_path(cfg["manifest_filepath"])
                tar_paths = _expand_nemo_path(cfg["tarred_audio_filepaths"])
                if len(manifest_paths) != len(tar_paths):
                    msg = f"Manifest/tar count mismatch for {corpus}: {len(manifest_paths)} vs {len(tar_paths)}"
                    raise ValueError(msg)
                for mp, tp in zip(manifest_paths, tar_paths, strict=False):
                    shards.append({"corpus": corpus, "manifest_path": mp, "tar_path": tp, "language": language})
            elif "manifest_filepath" in cfg:
                for mp in _expand_nemo_path(cfg["manifest_filepath"]):
                    shards.append({"corpus": corpus, "manifest_path": mp, "language": language})

    return shards


# ---------------------------------------------------------------------------
# Stage 1: Discovery
# ---------------------------------------------------------------------------


@dataclass
class UnifiedDiscoveryStage(ProcessingStage[_EmptyTask, FileGroupTask]):
    """Parse ``input_cfg`` YAML and emit one ``FileGroupTask`` per shard.

    Supports NeMo ``input_cfg`` format with ``nemo_tarred`` and ``nemo``
    types.  Handles shard expansion and ``.done``-file checkpointing.
    """

    name: str = "unified_discovery"
    yaml_path: str = ""
    corpus_filter: list[str] | None = None
    output_dir: str | None = None

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
        shard_descs = _parse_input_cfg(self.yaml_path, self.corpus_filter)

        completed = self._scan_completed_shards()
        if completed:
            logger.info(f"Checkpoint: {len(completed)} shards already completed, first 10: {sorted(completed)[:10]}")

        tasks: list[FileGroupTask] = []
        skipped = 0
        for desc in shard_descs:
            corpus = desc["corpus"]
            shard_key = _manifest_to_shard_key(desc["manifest_path"], corpus)
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
                reader_config={"corpus": corpus, "shard_key": shard_key, "language": desc.get("language", "")},
            ))

        logger.info(f"UnifiedDiscovery: {len(tasks)} shards to process, {skipped} skipped")
        return tasks


# ---------------------------------------------------------------------------
# Stage 2: Reader (format-agnostic, converts CutSet -> AudioTask)
# ---------------------------------------------------------------------------


@dataclass
class UnifiedReaderStage(ProcessingStage[FileGroupTask, AudioTask]):
    """Read a manifest shard and emit AudioTasks via NeMo lhotse adapters.

    Format-agnostic: uses ``LazyNeMoTarredIterator`` when a tar path
    is present, ``LazyNeMoIterator`` otherwise.  Both produce a lhotse
    ``CutSet`` iterated to load audio and emit ``AudioTask`` objects.

    The reader does not parse manifests itself — NeMo's adapters handle
    all I/O (including lazy line-by-line streaming for large files).
    """

    name: str = "unified_reader"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["waveform", "sampling_rate", "sample_rate", "corpus", "num_channels"]

    def setup(self, _worker_metadata: Any = None) -> None:
        self._patch_nemo_adapters()

    def ray_stage_spec(self) -> dict[str, Any]:
        if RayStageSpecKeys is not None:
            return {
                RayStageSpecKeys.IS_FANOUT_STAGE: True,
                RayStageSpecKeys.IS_ACTOR_STAGE: True,
            }
        return {"is_fanout_stage": True, "is_actor_stage": True}

    @staticmethod
    def _patch_nemo_adapters() -> None:
        """Apply two NeMo fixes for S3-based non-tarred manifests.

        Fix 1 (get_full_path): NeMo uses os.path.isabs() which returns False for
        s3://, http://, gs:// URIs — causing them to be joined with the manifest
        directory.  Adds NeMo's own is_datastore_path() check.

        Fix 2 (sampling_rate field): LazyNeMoIterator expects "sampling_rate" in
        manifest entries but many datasets use "sample_rate".  Without it, NeMo
        calls Recording.from_file() which can't read S3 URIs.  With it, NeMo
        creates Recording(source_type="url") using lhotse's native URL backend.
        """
        import nemo.collections.common.data.lhotse.nemo_adapters as _adapters
        import nemo.collections.common.parts.preprocessing.manifest as _manifest

        if getattr(_manifest, "_nemo_s3_patched", False):
            return

        from nemo.utils.data_utils import is_datastore_path

        # --- Fix 1: get_full_path URI recognition ---
        _original_get_full_path = _manifest.get_full_path

        def _patched_get_full_path(
            audio_file, manifest_file=None, data_dir=None, audio_file_len_limit=255, force_cache=True
        ):
            if isinstance(audio_file, str) and is_datastore_path(audio_file):
                return audio_file
            if isinstance(audio_file, list):
                return [
                    _patched_get_full_path(a, manifest_file, data_dir, audio_file_len_limit, force_cache)
                    for a in audio_file
                ]
            return _original_get_full_path(audio_file, manifest_file, data_dir, audio_file_len_limit, force_cache)

        _manifest.get_full_path = _patched_get_full_path
        _adapters.get_full_path = _patched_get_full_path

        # --- Fix 2: alias sample_rate -> sampling_rate in LazyNeMoIterator ---
        # Wrap the source iterator to normalize field names before NeMo processes them.
        _OriginalIterInit = _adapters.LazyNeMoIterator.__init__

        def _patched_init(self, *args, **kwargs):
            _OriginalIterInit(self, *args, **kwargs)
            self.source = _FieldNormalizingSource(self.source)

        class _FieldNormalizingSource:
            """Wraps a lazy JSONL source to alias sample_rate -> sampling_rate."""

            def __init__(self, source):
                self._source = source

            def __iter__(self):
                for data in self._source:
                    if "sampling_rate" not in data and "sample_rate" in data:
                        data["sampling_rate"] = data["sample_rate"]
                    yield data

            def __len__(self):
                return len(self._source)

            def __add__(self, other):
                return self._source.__add__(other)

        _adapters.LazyNeMoIterator.__init__ = _patched_init
        _manifest._nemo_s3_patched = True

    @staticmethod
    def _make_cutset(manifest_path: str, tar_path: str | None) -> Any:  # noqa: ANN401
        """Build a lhotse CutSet using NeMo adapters."""
        from lhotse import CutSet
        from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoIterator, LazyNeMoTarredIterator

        if tar_path:
            resolved_tar = _s3_to_pipe(tar_path) if tar_path.startswith("s3://") else tar_path
            # Temporarily disable NeMo's shard ID validation — it fails when
            # manifest is a local path but tar is a pipe:curl URL (string vs int
            # ID comparison). Data integrity is still verified during iteration.
            _orig_validate = LazyNeMoTarredIterator._validate
            LazyNeMoTarredIterator._validate = lambda self: None
            try:
                iterator = LazyNeMoTarredIterator(
                    manifest_path=manifest_path,
                    tar_paths=resolved_tar,
                    skip_missing_manifest_entries=True,
                )
            finally:
                LazyNeMoTarredIterator._validate = _orig_validate
            return CutSet(iterator)

        return CutSet(LazyNeMoIterator(manifest_path))

    def process(self, task: FileGroupTask) -> list[AudioTask]:
        manifest_path = task.data[0]
        tar_path = task.data[1] if len(task.data) >= 2 else None  # noqa: PLR2004
        corpus = task.reader_config.get("corpus", "unknown")
        shard_key = task.reader_config.get("shard_key", task.task_id)
        language = task.reader_config.get("language", "")
        metadata = dict(task._metadata)

        cutset = self._make_cutset(manifest_path, tar_path)

        mode = "tarred" if tar_path else "non-tarred"
        logger.info(f"Reading shard {shard_key} via NeMo {mode} adapter")

        results: list[AudioTask] = []
        loaded = 0
        for cut in cutset:
            try:
                audio = cut.load_audio().squeeze()
            except Exception:  # noqa: BLE001
                logger.warning(f"Skipping unreadable audio: {cut.id}")
                continue

            if audio.ndim > 1:
                audio = audio.mean(axis=0)

            loaded += 1
            if loaded % 100 == 0 or loaded == 1:
                logger.info(f"  [{shard_key}] loaded {loaded}")

            entry_data = dict(cut.custom) if cut.custom else {}
            entry_data["waveform"] = audio.astype(np.float32)
            entry_data["sampling_rate"] = cut.recording.sampling_rate
            entry_data["sample_rate"] = cut.recording.sampling_rate
            entry_data["duration"] = cut.duration
            entry_data["num_channels"] = 1
            entry_data["corpus"] = corpus
            if language and "source_lang" not in entry_data:
                entry_data["source_lang"] = language

            results.append(AudioTask(
                task_id=f"{shard_key}_{cut.id}",
                dataset_name=corpus,
                data=entry_data,
                _metadata={**metadata, "_shard_key": shard_key},
                _stage_perf=list(task._stage_perf),
            ))

        for r in results:
            r._metadata["_shard_total"] = len(results)

        logger.info(f"Shard {shard_key}: emitted {len(results)} AudioTasks")
        return results


# ---------------------------------------------------------------------------
# Composite stage (user-facing API)
# ---------------------------------------------------------------------------


@dataclass
class UnifiedAudioReader(CompositeStage[_EmptyTask, AudioTask]):
    """Unified reader for NeMo audio datasets.

    Reads NeMo ``input_cfg`` YAML configs and uses NeMo's lhotse
    adapters (``LazyNeMoIterator`` / ``LazyNeMoTarredIterator``)
    for audio loading.
    """

    name: str = "unified_audio_reader"
    yaml_path: str = ""
    corpus_filter: list[str] | None = None
    output_dir: str | None = None

    def __post_init__(self) -> None:
        super().__init__()
        if not self.yaml_path:
            msg = "yaml_path is required for UnifiedAudioReader"
            raise ValueError(msg)
        self._stages: list[ProcessingStage] = [
            UnifiedDiscoveryStage(
                yaml_path=self.yaml_path,
                corpus_filter=self.corpus_filter,
                output_dir=self.output_dir,
            ),
            UnifiedReaderStage(),
        ]

    def inputs(self) -> tuple[list[str], list[str]]:
        return self._stages[0].inputs()

    def outputs(self) -> tuple[list[str], list[str]]:
        return self._stages[-1].outputs()

    def decompose(self) -> list[ProcessingStage]:
        return self._stages
