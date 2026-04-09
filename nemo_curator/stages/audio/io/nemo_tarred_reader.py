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
shard-reader stage that streams each tar (local or S3/AIS), extracts audio
to a shared cache directory, and emits one ``AudioTask`` per utterance.
"""

from __future__ import annotations

import json
import os
import re
import tarfile
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nemo_curator.backends.experimental.utils import RayStageSpecKeys
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
    """Open a tar file from local disk or S3/AIS via lhotse's ``open_best``."""
    from lhotse.serialization import open_best

    if tar_path.startswith("s3://"):
        pipe_path = _s3_to_pipe(tar_path, s3_endpoint_url)
        fileobj = open_best(pipe_path, mode="rb")
    else:
        if not os.path.exists(tar_path):
            raise FileNotFoundError(f"Tar file not found: {tar_path}")
        fileobj = open(tar_path, "rb")  # noqa: SIM115
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

    def process(self, _task: _EmptyTask) -> list[FileGroupTask]:
        import yaml

        with open(self.yaml_path) as f:
            config = yaml.safe_load(f)

        tasks: list[FileGroupTask] = []
        for group in config:
            for cfg in group.get("input_cfg", []):
                corpus = cfg.get("corpus", "unknown")
                if self.corpus_filter and corpus not in self.corpus_filter:
                    continue
                manifest_paths = _expand_nemo_path(cfg["manifest_filepath"])
                tar_paths = _expand_nemo_path(cfg["tarred_audio_filepaths"])
                if len(manifest_paths) != len(tar_paths):
                    msg = (
                        f"Manifest/tar count mismatch for corpus={corpus}: "
                        f"{len(manifest_paths)} manifests vs {len(tar_paths)} tars"
                    )
                    raise ValueError(msg)
                for i, (mp, tp) in enumerate(zip(manifest_paths, tar_paths)):
                    tasks.append(
                        FileGroupTask(
                            task_id=f"{corpus}_{i}",
                            dataset_name=corpus,
                            data=[mp, tp],
                            reader_config={"corpus": corpus, "shard_idx": i},
                        )
                    )

        logger.info(
            "NemoTarShardDiscoveryStage: found %d shards (corpus_filter=%s)",
            len(tasks), self.corpus_filter,
        )
        return tasks


# ---------------------------------------------------------------------------
# Stage 2: Shard reader
# ---------------------------------------------------------------------------

@dataclass
class NemoTarShardReaderStage(ProcessingStage[FileGroupTask, AudioTask]):
    """Read a single NeMo tar shard and emit one ``AudioTask`` per utterance.

    Expects ``task.data = [manifest_path, tar_path]`` as produced by
    ``NemoTarShardDiscoveryStage``.

    Audio files are extracted to ``audio_cache_dir/<corpus>_<shard_idx>/``
    so that downstream GPU stages can access them on the shared filesystem.

    Args:
        audio_cache_dir: Shared directory for extracted audio files.
        filepath_key: Key used for the audio path in each ``AudioTask``.
        s3_endpoint_url: Override for the AIS/S3 endpoint.  Falls back to
            the ``AIS_ENDPOINT`` environment variable.
    """

    name: str = "nemo_tar_shard_reader"
    audio_cache_dir: str = "/tmp/nemo_tar_audio_cache"  # noqa: S108
    filepath_key: str = "audio_filepath"
    s3_endpoint_url: str | None = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.filepath_key]

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

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
        shard_idx = task.reader_config.get("shard_idx", 0)
        shard_key = f"{corpus}_{shard_idx}"

        manifest = self._read_manifest(manifest_path)

        shard_dir = os.path.join(self.audio_cache_dir, shard_key)
        os.makedirs(shard_dir, exist_ok=True)

        logger.info("Reading shard %s: %s (%d manifest entries)", shard_key, tar_path, len(manifest))

        tar = _open_tar(tar_path, self.s3_endpoint_url)
        results: list[AudioTask] = []

        for tar_info in tar:
            if not tar_info.isfile() or tar_info.name not in manifest:
                continue

            audio_path = os.path.join(shard_dir, tar_info.name)
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            with open(audio_path, "wb") as af:
                af.write(tar.extractfile(tar_info).read())

            entry = manifest[tar_info.name]
            entry[self.filepath_key] = audio_path

            results.append(
                AudioTask(
                    task_id=f"{shard_key}_{tar_info.name}",
                    dataset_name=corpus,
                    data=entry,
                    filepath_key=self.filepath_key,
                    _metadata={**task._metadata, "_shard_key": shard_key},
                    _stage_perf=list(task._stage_perf),
                )
            )

        tar.close()
        logger.info("Shard %s: emitted %d AudioTasks", shard_key, len(results))
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
    2. ``NemoTarShardReaderStage`` — streams each tar, extracts audio,
       emits ``AudioTask`` objects.

    Args:
        yaml_path: Path to the Granary YAML data config.
        corpus_filter: Only process shards whose ``corpus`` matches.
        audio_cache_dir: Shared directory for extracted audio.
        filepath_key: Key for audio paths in emitted tasks.
        s3_endpoint_url: Override for AIS/S3 endpoint.
    """

    name: str = "nemo_tarred_audio_reader"
    yaml_path: str = ""
    corpus_filter: list[str] | None = None
    audio_cache_dir: str = "/tmp/nemo_tar_audio_cache"  # noqa: S108
    filepath_key: str = "audio_filepath"
    s3_endpoint_url: str | None = None

    def __post_init__(self) -> None:
        super().__init__()
        if not self.yaml_path:
            msg = "yaml_path is required for NemoTarredAudioReader"
            raise ValueError(msg)

        self._stages: list[ProcessingStage] = [
            NemoTarShardDiscoveryStage(
                yaml_path=self.yaml_path,
                corpus_filter=self.corpus_filter,
            ),
            NemoTarShardReaderStage(
                audio_cache_dir=self.audio_cache_dir,
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
