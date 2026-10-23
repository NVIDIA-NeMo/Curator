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

"""Pretraining artifact finalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nemo_curator.stages.audio.audio_agent import _verb_evidence


@dataclass(frozen=True)
class _PretrainFinalizer:
    """Driver lifecycle required by the shard-writing ALM pretrain stages."""

    manifest_path: str
    metrics_path: str
    audio_tar_path: str
    audio_filepath_key: str
    id_key: str = "id"
    segments_key: str = "segments"
    duration_key: str = "duration"

    def prepare(self) -> None:
        from nemo_curator.stages.audio.alm.pretrain import (
            prepare_audio_pretrain_outputs,
        )

        prepare_audio_pretrain_outputs(
            self.manifest_path,
            self.metrics_path,
            self.audio_tar_path,
        )

    def finalize(self) -> int:
        """Merge this attempt's shards and return its serialized snippet count.

        Only ever called after a SUCCESSFUL attempt. ``finalize_audio_pretrain_outputs``
        merges unconditionally, so calling it when a failed attempt produced no shards
        would overwrite a previously completed output bundle with empty files -- which is
        why the failure path deliberately does not finalize, and any partial shards a
        crashed run left behind are abandoned rather than merged.
        """
        from nemo_curator.stages.audio.alm.pretrain import (
            finalize_audio_pretrain_outputs,
        )

        finalize_audio_pretrain_outputs(
            self.manifest_path,
            self.metrics_path,
            self.audio_tar_path,
            audio_filepath_key=self.audio_filepath_key,
            id_key=self.id_key,
            segments_key=self.segments_key,
            duration_key=self.duration_key,
        )
        return _verb_evidence._count_output_rows(self.manifest_path)


def _pretrain_finalizer(
    stages: list[Any],
) -> tuple[_PretrainFinalizer | None, str]:
    """Resolve a complete ALM shard lifecycle, or reject a partial recipe."""
    wanted: dict[str, list[Any]] = {
        "SnippetExtractionStage": [],
        "SnippetManifestWriterStage": [],
        "PretrainMetricsAggregatorStage": [],
    }
    for stage in stages:
        name = type(stage).__name__
        if name in wanted:
            wanted[name].append(stage)
    present = {name for name, matches in wanted.items() if matches}
    if not present:
        return None, ""
    if present != set(wanted):
        missing = sorted(set(wanted) - present)
        return None, (
            "ALM pretrain shard outputs require SnippetExtractionStage, "
            "SnippetManifestWriterStage, and PretrainMetricsAggregatorStage "
            f"in one recipe; missing {missing}"
        )
    duplicates = sorted(name for name, matches in wanted.items() if len(matches) != 1)
    if duplicates:
        return None, (
            f"ALM pretrain finalization requires exactly one of each shard stage; ambiguous stage(s): {duplicates}"
        )

    extraction = wanted["SnippetExtractionStage"][0]
    writer = wanted["SnippetManifestWriterStage"][0]
    aggregator = wanted["PretrainMetricsAggregatorStage"][0]
    manifest_path = getattr(writer, "output_path", None)
    metrics_path = getattr(aggregator, "output_path", None)
    audio_tar_path = getattr(extraction, "output_audio_tar_path", None)
    missing_paths = sorted(
        key
        for key, value in {
            "manifest_path": manifest_path,
            "metrics_path": metrics_path,
            "audio_tar_path": audio_tar_path,
        }.items()
        if not isinstance(value, str) or not value
    )
    if missing_paths:
        return None, (f"ALM pretrain finalization is missing required output path(s): {missing_paths}")
    shared_keys = {}
    for key in ("id_key", "segments_key", "duration_key"):
        default = key.removesuffix("_key")
        value = getattr(extraction, key, default)
        if value != getattr(aggregator, key, default):
            return None, f"ALM extraction and metrics must agree on {key}"
        shared_keys[key] = value
    return (
        _PretrainFinalizer(
            manifest_path=str(manifest_path),
            metrics_path=str(metrics_path),
            audio_tar_path=str(audio_tar_path),
            audio_filepath_key=str(getattr(extraction, "audio_filepath_key", "audio_filepath")),
            **shared_keys,
        ),
        "",
    )
