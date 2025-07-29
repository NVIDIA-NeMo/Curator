import pathlib
from dataclasses import dataclass

from loguru import logger

from ray_curator.stages.base import ProcessingStage
from ray_curator.tasks import DataEntry, _EmptyTask
from ray_curator.utils.file_utils import get_all_files_paths_under


@dataclass
class AudioReaderStage(ProcessingStage[_EmptyTask, DataEntry]):
    """Stage that reads video files from storage and extracts metadata."""

    input_audio_path: str
    audio_limit: int = -1
    filepath_key: str = "audio_filepath"
    _name: str = "audio_reader"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["input_audio"]

    def process(self, _: _EmptyTask) -> list[DataEntry]:
        """Process a single group of audio files."""
        if self.input_audio_path is None:
            msg = "input_audio_path is not set"
            raise ValueError(msg)
        files = get_all_files_paths_under(
            self.input_audio_path,
            recurse_subdirectories=True,
            keep_extensions=[".wav", ".mp3", ".flac", ".ogg", ".opus"],
        )
        logger.info(f"Found {len(files)} files under {self.input_audio_path}")

        if self.audio_limit > 0:
            files = files[: self.audio_limit]
            logger.info(
                f"Using first {len(files)} files under {self.input_audio_path} since audio_limit is set to {self.audio_limit}"
            )

        audio_tasks = []
        for fp in files:
            file_path = fp
            if isinstance(file_path, str):
                file_path = pathlib.Path(file_path)

            audio = {self.filepath_key: file_path}
            audio_task = DataEntry(
                task_id=f"{file_path}_processed",
                dataset_name=self.input_audio_path,
                data=audio,
            )
            audio_tasks.append(audio_task)

        return audio_tasks
