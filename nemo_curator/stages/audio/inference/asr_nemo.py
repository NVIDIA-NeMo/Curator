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

from dataclasses import dataclass, field
from typing import Any

import nemo.collections.asr as nemo_asr
import torch

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class InferenceAsrNemoStage(ProcessingStage[AudioTask, AudioTask]):
    """Speech recognition inference using a NeMo ASR model.

    Overrides ``process_batch`` for batched GPU inference.

    Args:
        model_name: Pretrained NeMo ASR model name.
            See full list at https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/all_chkpt.html
        filepath_key: Key in the entry dict pointing to the audio file.
        pred_text_key: Key where the predicted transcription is stored.
    """

    name: str = "ASR_inference"
    model_name: str = ""
    asr_model: Any | None = field(default=None, repr=False)
    filepath_key: str = "audio_filepath"
    pred_text_key: str = "pred_text"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 16

    def __post_init__(self) -> None:
        if not self.model_name and not self.asr_model:
            msg = "Either model_name or asr_model is required for InferenceAsrNemoStage"
            raise ValueError(msg)

    def check_cuda(self) -> torch.device:
        return torch.device("cuda") if self.resources.gpus > 0 else torch.device("cpu")

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        if self.asr_model:
            return
        try:
            nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name, return_model_file=True)
        except Exception as e:
            msg = f"Failed to download {self.model_name}"
            raise RuntimeError(msg) from e

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if not self.asr_model:
            try:
                map_location = self.check_cuda()
                self.asr_model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=self.model_name, map_location=map_location
                )
            except Exception as e:
                msg = f"Failed to load {self.model_name}"
                raise RuntimeError(msg) from e

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.filepath_key, self.pred_text_key]

    def transcribe(self, files: list[str]) -> list[str]:
        outputs = self.asr_model.transcribe(files)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        if outputs and isinstance(outputs[0], list):
            if outputs[0] and hasattr(outputs[0][0], "text"):
                return [inner[0].text for inner in outputs]
            return [inner[0] for inner in outputs]

        return [output.text for output in outputs]

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []
        for task in tasks:
            if not self.validate_input(task):
                msg = f"Task {task.task_id} missing required columns for {type(self).__name__}: {self.inputs()}"
                raise ValueError(msg)
        files = [t.data[self.filepath_key] for t in tasks]
        texts = self.transcribe(files)
        for task, text in zip(tasks, texts, strict=True):
            task.data[self.pred_text_key] = text
        return tasks
