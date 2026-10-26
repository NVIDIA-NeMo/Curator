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

import json
import os
import tarfile
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from nemo_curator.stages.audio.audio_agent import _verb_evidence

if TYPE_CHECKING:
    from nemo_curator.stages.audio.audio_agent._verb_outputs import _StagedFileOutput


_OUTPUT_PARAMS = {
    "SnippetExtractionStage": ("output_audio_tar_path",),
    "SnippetManifestWriterStage": ("output_path",),
    "PretrainMetricsAggregatorStage": ("output_path",),
}


def _atomic_output_params(stage: Any) -> tuple[str, ...]:  # noqa: ANN401
    """ALM bundle members published together by the agent's driver lifecycle."""
    if type(stage).__name__ == "SnippetExtractionStage" and getattr(stage, "dry_run", False):
        return ()
    return _OUTPUT_PARAMS.get(type(stage).__name__, ())


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
    dry_run: bool = False

    def staged(self, outputs: list[_StagedFileOutput]) -> _PretrainFinalizer:
        paths = {output.final_path: output.staged_path for output in outputs}
        manifest = paths.get(self.manifest_path, self.manifest_path)
        return replace(
            self,
            manifest_path=manifest,
            metrics_path=paths.get(self.metrics_path, self.metrics_path),
            audio_tar_path=f"{manifest}.dry-run.tar"
            if self.dry_run
            else paths.get(self.audio_tar_path, self.audio_tar_path),
        )

    def discard_shards(self) -> None:
        from nemo_curator.stages.audio.alm.pretrain import prepare_audio_pretrain_outputs

        prepare_audio_pretrain_outputs(self.manifest_path, self.metrics_path, self.audio_tar_path)

    def prepare(self) -> None:
        """Initialize this attempt's isolated bundle, including the empty-result case."""
        from nemo_curator.stages.audio.alm.pretrain.finalize import _build_final_summary

        self.discard_shards()
        for path in (self.manifest_path, self.metrics_path, self.audio_tar_path):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8"):
            pass
        with open(self.metrics_path, "w", encoding="utf-8") as stream:
            json.dump(_build_final_summary({}, []), stream)
        if not self.dry_run:
            with tarfile.open(self.audio_tar_path, "w"):
                pass

    def finalize(self) -> int:
        """Merge only a successful attempt's isolated shards and count its snippets."""
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
    wanted: dict[str, list[Any]] = {name: [] for name in _OUTPUT_PARAMS}
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
            dry_run=bool(getattr(extraction, "dry_run", False)),
            **shared_keys,
        ),
        "",
    )
