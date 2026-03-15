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

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from operator import eq, ge, gt, le, lt, ne
from typing import Any

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioEntry


class AudioEntryStage(ProcessingStage[AudioEntry, AudioEntry]):
    """Base class for stages that process one audio manifest entry at a time.

    Subclasses implement ``process_dataset_entry`` which receives a plain ``dict``
    (the manifest entry) and returns either:

    * a ``dict`` — the (possibly modified) entry, or
    * ``None``  — to filter the entry out of the pipeline.

    All undeclared columns silently pass through.  Perf / metadata
    propagation is handled automatically by the base ``process`` method.
    """

    @abstractmethod
    def process_dataset_entry(self, data_entry: dict) -> dict | None:
        """Process a single manifest entry dict.

        Returns:
            dict: processed entry (may be the same object, mutated in-place).
            None: to drop / filter out this entry.
        """

    def process(self, task: AudioEntry) -> AudioEntry | list[AudioEntry]:
        if not self.validate_input(task):
            msg = f"Task {task.task_id} missing required columns for {type(self).__name__}: {self.inputs()}"
            raise ValueError(msg)

        result = self.process_dataset_entry(task.data)
        if result is None:
            return []

        return AudioEntry(
            data=result,
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            filepath_key=task.filepath_key,
            _stage_perf=list(task._stage_perf),
            _metadata=task._metadata.copy(),
        )


# ---------------------------------------------------------------------------
# Concrete stages
# ---------------------------------------------------------------------------


@dataclass
class GetAudioDurationStage(AudioEntryStage):
    """Compute audio duration from the file at *audio_filepath_key* and
    store the result under *duration_key*.

    Args:
        audio_filepath_key: Key to get path to wav file.
        duration_key: Key to put audio duration.
    """

    name: str = "GetAudioDurationStage"
    audio_filepath_key: str = "audio_filepath"
    duration_key: str = "duration"

    def setup(self, worker_metadata: Any = None) -> None:  # noqa: ARG002, ANN401
        import soundfile

        self._soundfile = soundfile

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.duration_key]

    def process_dataset_entry(self, data: dict) -> dict:
        audio_filepath = data[self.audio_filepath_key]
        try:
            raw, samplerate = self._soundfile.read(audio_filepath)
            data[self.duration_key] = raw.shape[0] / samplerate
        except self._soundfile.SoundFileError as e:
            logger.warning(str(e) + " file: " + audio_filepath)
            data[self.duration_key] = -1.0
        return data


class PreserveByValueStage(AudioEntryStage):
    """Filter entries by comparing *input_value_key* against *target_value*.

    Args:
        input_value_key: The field in the dataset entries to evaluate.
        target_value: The value to compare with.
        operator: Comparison operator (lt, le, eq, ne, ge, gt).
    """

    name: str = "PreserveByValueStage"

    def __init__(
        self,
        input_value_key: str,
        target_value: int | str,
        operator: str = "eq",
    ):
        self.input_value_key = input_value_key
        self.target_value = target_value
        ops = {"lt": lt, "le": le, "eq": eq, "ne": ne, "ge": ge, "gt": gt}
        if operator not in ops:
            msg = f"Operator must be one of: {', '.join(ops)}"
            raise ValueError(msg)
        self.operator = ops[operator]

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.input_value_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.input_value_key]

    def process_dataset_entry(self, data: dict) -> dict | None:
        if self.operator(data[self.input_value_key], self.target_value):
            return data
        return None
