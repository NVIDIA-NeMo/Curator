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

"""Curator stage for creating NeMo-compatible tarred ASR datasets."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nemo_curator.stages.audio.asr.io.convert_to_tarred_audio_dataset import create_tar_datasets
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import _EmptyTask


@dataclass
class TarredAudioDatasetWriterStage(ProcessingStage[_EmptyTask, _EmptyTask]):
    """Create one tarred ASR dataset per manifest/target directory pair.

    This stage wraps NeMo's ``convert_to_tarred_audio_dataset.py`` helper for
    YAML-driven Curator pipelines. It is a terminal stage and does not emit
    downstream tasks.
    """

    manifest_paths: str | list[str]
    target_dirs: str | list[str]
    num_shards: int
    max_duration: float | None
    name: str = "tarred_audio_dataset_writer"
    min_duration: float | None = None
    shuffle: bool = False
    keep_files_together: bool = False
    sort_in_shards: bool = False
    buckets_num: int = 1
    dynamic_buckets_num: int = 30
    shuffle_seed: int | None = None
    no_shard_manifests: bool = False
    force_codec: str | None = None
    workers: int = 1
    slice_with_offset: bool = False
    only_manifests: bool = False
    dry_run: bool = False
    is_sink_stage: bool = True
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self) -> None:
        self.manifest_paths = _coerce_path_list(self.manifest_paths)
        self.target_dirs = _coerce_path_list(self.target_dirs)
        if not self.manifest_paths:
            msg = "manifest_paths is required for TarredAudioDatasetWriterStage"
            raise ValueError(msg)
        if not self.target_dirs:
            msg = "target_dirs is required for TarredAudioDatasetWriterStage"
            raise ValueError(msg)
        if len(self.manifest_paths) != len(self.target_dirs):
            msg = "manifest_paths and target_dirs must have the same length"
            raise ValueError(msg)
        if self.num_shards < 1:
            msg = "num_shards must be >= 1 for TarredAudioDatasetWriterStage"
            raise ValueError(msg)
        self.resources = Resources(cpus=float(max(self.workers, 1)))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def num_workers(self) -> int | None:
        return 1

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers": 1}

    def process(self, _: _EmptyTask) -> list[_EmptyTask]:
        start = time.perf_counter()
        for manifest_path, target_dir in zip(self.manifest_paths, self.target_dirs, strict=True):
            logger.info(f"[{self.name}] creating tarred dataset from {manifest_path} -> {target_dir}")
            create_tar_datasets(
                manifest_path=manifest_path,
                target_dir=target_dir,
                num_shards=self.num_shards,
                max_duration=self.max_duration,
                min_duration=self.min_duration,
                shuffle=self.shuffle,
                keep_files_together=self.keep_files_together,
                sort_in_shards=self.sort_in_shards,
                buckets_num=self.buckets_num,
                dynamic_buckets_num=self.dynamic_buckets_num,
                shuffle_seed=self.shuffle_seed,
                no_shard_manifests=self.no_shard_manifests,
                force_codec=self.force_codec,
                workers=self.workers,
                slice_with_offset=self.slice_with_offset,
                only_manifests=self.only_manifests,
                dry_run=self.dry_run,
            )
        self._log_metrics(
            {
                "process_time": time.perf_counter() - start,
                "input_manifests": len(self.manifest_paths),
                "emitted_tasks": 0,
            }
        )
        return []


def _coerce_path_list(paths: str | list[str]) -> list[str]:
    if isinstance(paths, str):
        return [paths]
    return [str(path) for path in paths]
