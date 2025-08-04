from dataclasses import dataclass, field

import nemo.collections.asr as nemo_asr

from ray_curator.stages.base import CompositeStage, ProcessingStage
from ray_curator.stages.io.reader.file_partitioning import FilePartitioningStage
from ray_curator.tasks import SpeechEntry, _EmptyTask
from ray_curator.tasks.file_group import FileGroupTask


@dataclass
class AsrNemoInferenceStage(ProcessingStage[FileGroupTask, SpeechEntry]):
    """Stage that reads video files from storage and extracts metadata."""

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
            - top_level_attrs: ["data"] - requires VideoTask.data to be populated
        """
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define the output attributes produced by this stage.

        Returns:
            Tuple of (top_level_attrs, data_attrs) where:
            - top_level_attrs: ["data"] - populates VideoTask.data
            - data_attrs: ["text"] - predicted text
        """
        return ["data"], ["text"]

    def transcribe(self, files: list[str]) -> list[str]:
        """Run inference for speech recognition model
         Args:
            files: list of audio file paths.

        Returns:
            list of predicted texts.
        """
        outputs = self.asr_model.transcribe(files)
        return [output.text for output in outputs]

    def process_batch(self, tasks: list[FileGroupTask]) -> list[SpeechEntry]:
        """Process a audio task by reading file bytes and extracting metadata.


        Args:
            task: VideoTask containing a Video object with input_video path set.

        Returns:
            The same VideoTask with video.source_bytes and video.metadata populated.
            If errors occur, the task is returned with error information stored.
        """
        files = [task.data[0] for task in tasks]
        outputs = self.transcribe(files)

        audio_tasks = []
        for i in range(len(outputs)):
            text = outputs[i]
            file_path = files[i]

            entry = {self.filepath_key: file_path, self.filepath_key: text}

            audio_task = SpeechEntry(
                task_id=f"{file_path}_task_id",
                dataset_name=f"{self.model_name}_inference",
                data=entry,
            )
            audio_tasks.append(audio_task)
        return audio_tasks

    def process(self, task: FileGroupTask) -> list[SpeechEntry]:
        pass


@dataclass
class AsrNemoInference(CompositeStage[_EmptyTask, SpeechEntry]):
    """Composite stage that reads video files from storage and downloads/processes them.

    This stage combines FilePartitioningStage and VideoReaderStage into a single
    high-level operation for reading video files from a directory and processing
    them with metadata extraction.

    Args:
        input_video_path: Path to the directory containing video files
        video_limit: Maximum number of videos to process (-1 for unlimited)
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
            List of processing stages: [FilePartitioningStage, VideoReaderStage]
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
            f"Reads video files from '{self.input_audio_path}' "
            f"(limit: {self.audio_limit if self.audio_limit > 0 else 'unlimited'}) "
            f"and downloads/processes them with metadata extraction"
        )
