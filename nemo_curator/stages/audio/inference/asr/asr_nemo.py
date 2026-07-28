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

"""Curator stage wrapper for the NeMo ASR model adapter."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from nemo_curator.models.asr_nemo import NeMoASRAdapter
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata


@dataclass
class InferenceAsrNemoStage(ProcessingStage[AudioTask, AudioTask]):
    """Transcribe audio files with a registry model or local ``.nemo`` checkpoint."""

    name: str = "ASR_inference"
    model_name: str = ""
    model_path: str | None = None
    cache_dir: str | None = None
    asr_model: Any | None = field(default=None, repr=False)
    filepath_key: str = "audio_filepath"
    pred_text_key: str = "pred_text"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 16
    _adapter: NeMoASRAdapter | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.model_name and not self.model_path and self.asr_model is None:
            msg = "Either model_name, model_path, or asr_model is required for InferenceAsrNemoStage"
            raise ValueError(msg)
        if self.model_name and self.model_path:
            msg = "model_name and model_path are mutually exclusive"
            raise ValueError(msg)

    def check_cuda(self) -> torch.device:
        return torch.device("cuda") if self.resources.gpus > 0 else torch.device("cpu")

    def _create_adapter(self) -> NeMoASRAdapter:
        return NeMoASRAdapter(
            model_name=self.model_name,
            model_path=self.model_path,
            cache_dir=self.cache_dir,
            map_location=self.check_cuda(),
            model=self.asr_model,
        )

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        if self.asr_model is not None:
            return
        try:
            adapter = self._create_adapter()
            adapter.download_weights_on_node()
        except Exception as error:
            identifier = self.model_path or self.model_name
            msg = f"Failed to prepare {identifier}"
            raise RuntimeError(msg) from error

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self.asr_model is not None:
            self._adapter = self._create_adapter()
            return
        try:
            self._adapter = self._create_adapter()
            self._adapter.setup()
            self.asr_model = self._adapter.model
        except Exception as error:
            identifier = self.model_path or self.model_name
            msg = f"Failed to load {identifier}"
            raise RuntimeError(msg) from error

    def teardown(self) -> None:
        if self._adapter is not None:
            self._adapter.teardown()
        self._adapter = None
        self.asr_model = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.filepath_key, self.pred_text_key]

    def transcribe(self, files: list[str]) -> list[str]:
        adapter = self._adapter or self._create_adapter()
        return adapter.transcribe(files)

    def process(self, task: AudioTask) -> AudioTask:
        msg = "InferenceAsrNemoStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if not tasks:
            return []
        start = time.perf_counter()
        for task in tasks:
            if not self.validate_input(task):
                msg = f"Task {task.task_id} missing required columns for {type(self).__name__}: {self.inputs()}"
                raise ValueError(msg)

        files = [str(task.data[self.filepath_key]) for task in tasks]
        texts = self.transcribe(files)
        if len(texts) != len(tasks):
            msg = f"NeMo ASR returned {len(texts)} transcriptions for {len(tasks)} files"
            raise RuntimeError(msg)
        for task, text in zip(tasks, texts, strict=True):
            task.data[self.pred_text_key] = text

        self._log_metrics(
            {
                "process_time": time.perf_counter() - start,
                "files_transcribed": len(files),
            }
        )
        return tasks
