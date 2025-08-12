import time
from dataclasses import dataclass, field
from typing import ClassVar

import nemo.collections.asr as nemo_asr
import torch
from loguru import logger

from ray_curator.backends.base import NodeInfo, WorkerMetadata
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.resources import Resources
from ray_curator.tasks import DocumentBatch, FileGroupTask, SpeechObject


@dataclass
class InferenceAsrNemoStage(ProcessingStage[FileGroupTask | DocumentBatch | SpeechObject, SpeechObject]):
    """Stage that do speech recognition inference using NeMo model.

    Args:
        model_name (str): name of the speech recognition NeMo model. See full list at https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/all_chkpt.html
        filepath_key (str): which key of the data object should be used to find the path to audiofile. Defaults to “audio_filepath”
        pred_text_key (str): key is used to identify the field containing the predicted transcription associated with a particular audio sample. Defaults to “pred_text”
        cuda (str): device to run inference on it. Could be cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, mtia or cuda number (digit). Defaults to “” (empty string)
        name (str): Stage name. Defaults to "ASR_inference"
    """

    model_name: str
    filepath_key: str = "audio_filepath"
    pred_text_key: str = "pred_text"
    cuda: str = ""
    name: str = "ASR_inference"
    _start_time = time.time()
    _metrics: ClassVar[dict] = {}
    _batch_size: int = 16
    _resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def check_cuda(self) -> torch.device:
        self.cuda = str(self.cuda)
        if self.cuda != "":
            map_location = torch.device(f"cuda:{self.cuda}") if self.cuda.isdigit() else torch.device(self.cuda)
        elif torch.cuda.is_available():
            map_location = torch.device("cuda:0")
        else:
            map_location = torch.device("cpu")
        return map_location

    def setup_on_node(self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata = None) -> None:
        # TODO: load asr_model file only once per node
        pass

    def setup(self, _worker_metadata: WorkerMetadata = None) -> None:
        """Initialise heavy object self.asr_model: nemo_asr.models.ASRModel"""
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
        return [output.text for output in outputs]

    def process_batch(self, tasks: list[FileGroupTask | DocumentBatch | SpeechObject]) -> list[SpeechObject]:
        """Process a audio task by reading audio file and do ASR inference.


        Args:
            tasks: List of FileGroupTask containing a path to audop file for inference.

        Returns:
            List of SpeechObject with self.filepath_key .
            If errors occur, the task is returned with error information stored.
        """

        if isinstance(tasks[0], FileGroupTask):
            files = [task.data[0] for task in tasks]
        elif isinstance(tasks[0], DocumentBatch):
            files = []
            for task in tasks:
                files.extend(list(task.data[self.filepath_key]))
        elif isinstance(tasks[0], SpeechObject):
            files = [task.data[self.filepath_key] for task in tasks]
        else:
            raise TypeError(str(tasks[0]))

        outputs = self.transcribe(files)

        audio_tasks = []
        for i in range(len(outputs)):
            entry = tasks[i].data
            text = outputs[i]
            file_path = files[i]

            if isinstance(entry, dict):
                item = entry
                item[self.pred_text_key] = text
            else:
                item = {self.filepath_key: file_path, self.pred_text_key: text}

            audio_task = SpeechObject(
                task_id=f"task_id_{file_path}",
                dataset_name=f"{self.model_name}_inference",
                filepath_key=self.filepath_key,
                data=item,
            )
            audio_tasks.append(audio_task)
        self.finalize()
        return audio_tasks

    def process(self, task: FileGroupTask) -> list[SpeechObject]:
        pass

    def finalize(self) -> None:
        elapsed = time.time() - self._start_time
        logger.info(f"Stage {self.name} completed in {elapsed:.2f} seconds.")
        self._metrics["elapsed"] = elapsed
