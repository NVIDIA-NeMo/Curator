from dataclasses import dataclass, field

import nemo.collections.asr as nemo_asr

from ray_curator.stages.base import CompositeStage, ProcessingStage
from ray_curator.stages.io.reader.file_partitioning import FilePartitioningStage
from ray_curator.tasks import SpeechObject, _EmptyTask
from ray_curator.tasks.file_group import FileGroupTask


@dataclass
class AsrNemoInferenceStage(ProcessingStage[FileGroupTask, SpeechObject]):
    """Stage that do speech recognition inference using NeMo model."""

    model_name: str
    filepath_key: str = "audio_filepath"
    text_key: str = "text"
    _name: str = "audio_inference"

    def setup(self) -> None:
        """Initialise heavy object self.asr_model: nemo_asr.models.ASRModel"""
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)

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
            - data_attrs: [self.filepath_key, self.text_key] - audiofile path and predicted text.
        """
        return ["data"], [self.filepath_key, self.text_key]

    def transcribe(self, files: list[str]) -> list[str]:
        """Run inference for speech recognition model
         Args:
            files: list of audio file paths.

        Returns:
            list of predicted texts.
        """
        outputs = self.asr_model.transcribe(files)
        return [output.text for output in outputs]

    def process_batch(self, tasks: list[FileGroupTask]) -> list[SpeechObject]:
        """Process a audio task by reading audio file and do ASR inference.


        Args:
            tasks: List of FileGroupTask containing a path to audop file for inference.

        Returns:
            List of SpeechObject with self.filepath_key .
            If errors occur, the task is returned with error information stored.
        """
        files = [task.data[0] for task in tasks]
        outputs = self.transcribe(files)

        audio_tasks = []
        for i in range(len(outputs)):
            text = outputs[i]
            file_path = files[i]

            entry = {self.filepath_key: file_path, self.text_key: text}

            audio_task = SpeechObject(
                task_id=f"{file_path}_task_id",
                dataset_name=f"{self.model_name}_inference",
                filepath_key=self.filepath_key,
                data=entry,
            )
            audio_tasks.append(audio_task)
        return audio_tasks

    def process(self, task: FileGroupTask) -> list[SpeechObject]:
        pass


@dataclass
class AsrNemoInference(CompositeStage[_EmptyTask, SpeechObject]):
    """Composite stage that read audio files and do speech recognition inference using NeMo model.

    This stage combines FilePartitioningStage and AsrNemoInferenceStage into a single
    high-level operation for reading audiop files from a directory and processing
    them.

    Args:
        input_audio_path: Path to the directory containing audio files
        model_name: name of the speech recognition NeMo model
        audio_limit: Maximum number of audios to process (None for unlimited)
        file_extensions: file name extensions of files to process
        verbose: Whether to enable verbose logging during download/processing
    """

    input_audio_path: str
    model_name: str
    audio_limit: int | None = None
    file_extensions: list[str] = field(default_factory=lambda: [".wav", ".mp3", ".flac", ".ogg", ".opus"])
    verbose: bool = False

    def __post_init__(self):
        """Initialize the parent CompositeStage after dataclass initialization."""
        super().__init__()

    @property
    def name(self) -> str:
        return "audio_inference"

    def decompose(self) -> list[ProcessingStage]:
        """Decompose into constituent execution stages.

        Returns:
            List of processing stages: [FilePartitioningStage, AsrNemoInferenceStage]
        """
        reader_stage = FilePartitioningStage(
            file_paths=self.input_audio_path,
            files_per_partition=1,
            file_extensions=self.file_extensions,
            limit=self.audio_limit,
        )

        download_stage = AsrNemoInferenceStage(
            model_name=self.model_name,
        )

        return [reader_stage, download_stage]

    def get_description(self) -> str:
        """Get a description of what this composite stage does."""
        return (
            f"Reads audio files from '{self.input_audio_path}' "
            f"(limit: {self.audio_limit if self.audio_limit > 0 else 'unlimited'}) "
            f"and do inference by speech recognition NeMo model."
        )
