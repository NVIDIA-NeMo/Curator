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
shard-reader stage that streams each tar (local or S3/AIS), decodes audio
in memory via lhotse/soundfile, and emits one ``AudioTask`` per utterance
with the waveform as a numpy array — no files are written to disk.
"""

from __future__ import annotations

import json
import os
import re
import tarfile
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import soundfile as sf
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


def _s3_to_pipe(tar_path: str, s3_endpoint_url: str | None = None) -> str:
    """Convert ``s3://BUCKET/KEY`` to a ``pipe:`` command for lhotse."""
    from urllib.parse import urlparse

    parsed = urlparse(tar_path)
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    endpoint = s3_endpoint_url or os.environ.get("AIS_ENDPOINT")
    if not endpoint:
        msg = "Set AIS_ENDPOINT env var or pass s3_endpoint_url for s3:// paths"
        raise RuntimeError(msg)
    url = f"{endpoint.rstrip('/')}/v1/objects/{bucket}/{key}?provider=s3"
    token = os.environ.get("AIS_AUTHN_TOKEN")
    if token:
        return f"pipe:curl -sL -H 'Authorization: Bearer {token}' '{url}'"
    return f"pipe:curl -sL '{url}'"


def _open_tar(tar_path: str, s3_endpoint_url: str | None = None) -> tarfile.TarFile:
    """Open a tar file from local disk or S3/AIS via lhotse's ``open_best``.

    Uses streaming mode (``r|*``) so the tar is read sequentially without
    seeking — works with pipes and cloud storage.
    """
    from lhotse.serialization import open_best

    if tar_path.startswith("s3://"):
        pipe_path = _s3_to_pipe(tar_path, s3_endpoint_url)
        fileobj = open_best(pipe_path, mode="rb")
    else:
        if not os.path.exists(tar_path):
            msg = f"Tar file not found: {tar_path}"
            raise FileNotFoundError(msg)
        fileobj = open_best(tar_path, mode="rb")
    return tarfile.open(fileobj=fileobj, mode="r|*")


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
            → "yodas/0_from_captions/en/sharded_manifests/manifest_42"

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

    def process(self, _task: _EmptyTask) -> list[FileGroupTask]:
        import yaml

        completed = self._scan_completed_shards()
        if completed:
            logger.info(f"Checkpoint: {len(completed)} shards already completed, will skip them")
            logger.info(f"Completed shard keys (first 10): {sorted(completed)[:10]}")

        with open(self.yaml_path) as f:
            config = yaml.safe_load(f)

        tasks: list[FileGroupTask] = []
        skipped = 0
        for group in config:
            for cfg in group.get("input_cfg", []):
                corpus = cfg.get("corpus", "unknown")
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
    native sample rate in ``task.data["sample_rate"]``.

    Args:
        filepath_key: Manifest key that identifies the audio filename
            inside the tar archive.
        s3_endpoint_url: Override for the AIS/S3 endpoint.  Falls back to
            the ``AIS_ENDPOINT`` environment variable.
    """

    name: str = "nemo_tar_shard_reader"
    filepath_key: str = "audio_filepath"
    s3_endpoint_url: str | None = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["waveform", "sample_rate", "corpus", "num_channels"]

    def ray_stage_spec(self) -> dict[str, Any]:
        if RayStageSpecKeys is not None:
            return {RayStageSpecKeys.IS_FANOUT_STAGE: True}
        return {"is_fanout_stage": True}

    def _read_manifest(self, path: str) -> dict[str, dict]:
        entries: dict[str, dict] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    entries[entry[self.filepath_key]] = entry
        return entries

    def process(self, task: FileGroupTask) -> list[AudioTask]:
        manifest_path, tar_path = task.data[0], task.data[1]
        corpus = task.reader_config.get("corpus", "unknown")
        shard_key = task.reader_config.get("shard_key", task.task_id)

        manifest = self._read_manifest(manifest_path)

        logger.info(f"Reading shard {shard_key}: {tar_path} ({len(manifest)} manifest entries)")

        tar = _open_tar(tar_path, self.s3_endpoint_url)
        results: list[AudioTask] = []

        for tar_info in tar:
            if not tar_info.isfile() or tar_info.name not in manifest:
                continue

            # Decode audio in memory — no disk writes
            raw_audio = tar.extractfile(tar_info).read()
            try:
                audio, sample_rate = sf.read(BytesIO(raw_audio), dtype="float32")
            except Exception:
                logger.warning(f"Skipping corrupt audio {tar_info.name} in {tar_path}")
                continue

            num_channels = audio.shape[1] if audio.ndim > 1 else 1

            # Convert to mono 1-D array
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            entry = dict(manifest[tar_info.name])
            entry["waveform"] = audio
            entry["sample_rate"] = sample_rate
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

        tar.close()

        shard_total = len(results)
        for result_task in results:
            result_task._metadata["_shard_total"] = shard_total

        logger.info(f"Shard {shard_key}: emitted {shard_total} AudioTasks")
        return results


# ---------------------------------------------------------------------------
# Composite stage (user-facing API)
# ---------------------------------------------------------------------------

@dataclass
class NemoTarredAudioReader(CompositeStage[_EmptyTask, AudioTask]):
    """Read NeMo-style tarred audio datasets from a Granary YAML config.

    Decomposes into:

    1. ``NemoTarShardDiscoveryStage`` — parses the YAML, emits one
       ``FileGroupTask`` per shard.
    2. ``NemoTarShardReaderStage`` — streams each tar via lhotse, decodes
       audio in memory, emits ``AudioTask`` objects with waveform arrays.

    Args:
        yaml_path: Path to the Granary YAML data config.
        corpus_filter: Only process shards whose ``corpus`` matches.
        filepath_key: Manifest key for audio filenames inside tar archives.
        s3_endpoint_url: Override for AIS/S3 endpoint.
    """

    name: str = "nemo_tarred_audio_reader"
    yaml_path: str = ""
    corpus_filter: list[str] | None = None
    filepath_key: str = "audio_filepath"
    s3_endpoint_url: str | None = None
    output_dir: str | None = None

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
                s3_endpoint_url=self.s3_endpoint_url,
            ),
        ]

    def inputs(self) -> tuple[list[str], list[str]]:
        return self._stages[0].inputs()

    def outputs(self) -> tuple[list[str], list[str]]:
        return self._stages[-1].outputs()

    def decompose(self) -> list[ProcessingStage]:
        return self._stages
