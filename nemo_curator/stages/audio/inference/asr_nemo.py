# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from nemo_curator.tasks import AudioBatch, DocumentBatch, FileGroupTask


@dataclass
class InferenceAsrNemoStage(ProcessingStage[FileGroupTask | DocumentBatch | AudioBatch, AudioBatch]):
    """Stage that do speech recognition inference using NeMo model.

    Args:
        model_name (str): name of the speech recognition NeMo model. See full list at https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/all_chkpt.html
        asr_model (Any): ASR model object. Defaults to None
        filepath_key (str): which key of the data object should be used to find the path to audiofile. Defaults to “audio_filepath”
        pred_text_key (str): key is used to identify the field containing the predicted transcription associated with a particular audio sample. Defaults to “pred_text”
        name (str): Stage name. Defaults to "ASR_inference"
    """

    model_name: str
    asr_model: Any | None = None
    filepath_key: str = "audio_filepath"
    pred_text_key: str = "pred_text"
    _name: str = "ASR_inference"
    _batch_size: int = 16
    _resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def check_cuda(self) -> torch.device:
        return torch.device("cuda") if self.resources.gpus > 0 else torch.device("cpu")

    def setup_on_node(self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata = None) -> None:
        self.setup()

    def setup(self, _worker_metadata: WorkerMetadata = None) -> None:
        """Initialise heavy object self.asr_model: nemo_asr.models.ASRModel"""
        if not self.asr_model:
            try:
                map_location = self.check_cuda()
                self.asr_model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=self.model_name, map_location=map_location
                )
            except Exception as e:
                msg = f"Failed to download {self.model_name}"
                raise RuntimeError(msg) from e

    def inputs(self) -> tuple[list[str], list[str]]:
        """Define the input attributes required by this stage.

        Returns:
            Tuple of (top_level_attrs, data_attrs) where:
            - top_level_attrs: ["data"] - requires FileGroupTask.data to be populated
        """
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define the output attributes produced by this stage.

        Returns:
            Tuple of (top_level_attrs, data_attrs) where:
            - top_level_attrs: ["data"] - populates FileGroupTask.data
            - data_attrs: [self.filepath_key, self.pred_text_key] - audiofile path and predicted text.
        """
        return ["data"], [self.filepath_key, self.pred_text_key]

    def transcribe(self, files: list[str]) -> list[str]:
        """Run inference for speech recognition model
         Args:
            files: list of audio file paths.

        Returns:
            list of predicted texts.
        """

        outputs = self.asr_model.transcribe(files)

        # Tuple (hyps, all_hyps) noqa: ERA001
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        # list[list[Hypothesis]] noqa: ERA001
        if outputs and isinstance(outputs[0], list):
            if outputs[0] and hasattr(outputs[0][0], "text"):
                return [inner[0].text for inner in outputs]
            return [inner[0] for inner in outputs]

        return [output.text for output in outputs]

    def process(self, task: FileGroupTask | DocumentBatch | AudioBatch) -> AudioBatch:
        """Process a audio task by reading audio file and do ASR inference.


        Args:
            tasks: List of FileGroupTask containing a path to audop file for inference.

        Returns:
            List of SpeechObject with self.filepath_key .
            If errors occur, the task is returned with error information stored.
        """
        files = []
        audio_items = []

        if not self.validate_input(task):
            msg = f"Task {task!s} failed validation for stage {self}"
            raise ValueError(msg)
        if isinstance(task, FileGroupTask):
            files = [task.data[0]]
        elif isinstance(task, DocumentBatch):
            files = list(task.data[self.filepath_key])
        elif isinstance(task, AudioBatch):
            files = [item[self.filepath_key] for item in task.data]
        else:
            raise TypeError(str(task))

        outputs = self.transcribe(files)

        for i, text in enumerate(outputs):
            entry = task.data[i]
            file_path = files[i]

            if isinstance(entry, dict):
                item = entry
                item[self.pred_text_key] = text
            else:
                item = {self.filepath_key: file_path, self.pred_text_key: text}
            audio_items.append(item)

        return AudioBatch(
            task_id=f"task_id_{self.model_name}",
            dataset_name=f"{self.model_name}_inference",
            filepath_key=self.filepath_key,
            data=audio_items,
        )
