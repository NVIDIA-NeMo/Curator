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

"""Split/language-aware JSONL manifest writer for ASR pipelines.

Routes each ``AudioTask`` to ``{output_dir}/{lang}/{split_type}.jsonl`` so a single
pipeline run produces the per-language train/dev/test manifests directly. Runs as a
single worker so the per-split files are written without cross-worker contention.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata


@dataclass
class SplitAwareManifestWriter(ProcessingStage[AudioTask, AudioTask]):
    """Write each entry to ``{output_dir}/{lang}/{split_type}.jsonl``.

    Args:
        output_dir: Destination root for the manifests.
        lang_key: Data key holding the language.
        split_key: Data key holding the split type (``train``/``dev``/``test``).
        langs: Optional languages to pre-create empty manifests for.
        splits: Optional split types to pre-create empty manifests for.
            When both ``langs`` and ``splits`` are given, all combinations are
            created (and truncated) up front, guaranteeing the files exist even
            if a split receives no entries.
    """

    output_dir: str = ""
    name: str = "split_manifest_writer"
    lang_key: str = "lang"
    split_key: str = "split_type"
    langs: list[str] | None = None
    splits: list[str] | None = None
    is_sink_stage: bool = True
    _handles: dict[tuple[str, str], Any] = field(default_factory=dict, init=False, repr=False)
    _counts: dict[tuple[str, str], int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.output_dir:
            msg = "output_dir is required for SplitAwareManifestWriter"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.lang_key, self.split_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def num_workers(self) -> int | None:
        return 1

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers": 1}

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        self._handles = {}
        self._counts = {}
        # Pre-create (truncate) declared manifests so all expected files exist.
        if self.langs and self.splits:
            for lang in self.langs:
                for split in self.splits:
                    self._open(lang, split)

    def _path(self, lang: str, split: str) -> str:
        return os.path.join(self.output_dir, lang, f"{split}.jsonl")

    def _open(self, lang: str, split: str) -> Any:  # noqa: ANN401
        key = (lang, split)
        handle = self._handles.get(key)
        if handle is None:
            path = self._path(lang, split)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            handle = open(path, "w", encoding="utf-8")  # noqa: SIM115
            self._handles[key] = handle
            self._counts[key] = 0
            logger.info(f"[{self.name}] writing manifest -> {path}")
        return handle

    def process(self, task: AudioTask) -> AudioTask:
        lang = str(task.data.get(self.lang_key, "unknown"))
        split = str(task.data.get(self.split_key, "unknown"))
        handle = self._open(lang, split)
        handle.write(json.dumps(task.data, ensure_ascii=False) + "\n")
        # Flush every write: the executor may terminate the worker without
        # invoking teardown(), so we cannot rely on close() to flush the buffer.
        handle.flush()
        self._counts[(lang, split)] += 1
        return task

    def teardown(self) -> None:
        for (lang, split), handle in self._handles.items():
            handle.close()
            logger.info(f"[{self.name}] {lang}/{split}.jsonl: {self._counts.get((lang, split), 0)} entries")
        self._handles = {}
