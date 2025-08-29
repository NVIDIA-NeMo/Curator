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

from abc import abstractmethod
from dataclasses import dataclass
from operator import eq, ge, gt, le, lt, ne

import soundfile
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioBatch, Task


class LegacySpeechStage(ProcessingStage[Task, Task]):
    """
    LegacySpeechStage for SDP processors inherited from BaseParallelProcessor

    """

    def process(self, task: AudioBatch) -> list[Task]:
        result = []
        for entry in task.data:
            result.extend(self.process_dataset_entry(entry))
        return result

    @abstractmethod
    def process_dataset_entry(self, data_entry: AudioBatch) -> list[AudioBatch]:
        return [data_entry]


@dataclass
class GetAudioDurationStage(LegacySpeechStage):
    """
    Stage that computes the duration of the file in ``audio_filepath_key`` (using soundfile)
    and saves the duration in ``duration_key``. If there is an error computing the duration,
    the value at ``duration_key`` will be updated with the value -1.0.

    Args:
        audio_filepath_key (str): Key to get path to wav file.
        duration_key (str): Key to put to audio duration.
    Returns:
        All the same fields as in the input manifest plus duration_key
    """

    audio_filepath_key: str
    duration_key: str

    def process_dataset_entry(self, data_entry: dict) -> list[AudioBatch]:
        audio_filepath = data_entry[self.audio_filepath_key]
        try:
            data, samplerate = soundfile.read(audio_filepath)
            data_entry[self.duration_key] = data.shape[0] / samplerate
        except soundfile.SoundFileError as e:
            logger.warning(str(e) + " file: " + audio_filepath)
            data_entry[self.duration_key] = -1.0
        return [AudioBatch(data=data_entry)]


class PreserveByValueStage(LegacySpeechStage):
    """
    Processor for preserving dataset entries based on a specified condition involving a target value and an input field.

    Args:
        input_value_key (str): The field in the dataset entries to be evaluated.
        target_value (Union[int, str]): The value to compare with the input field.
        operator (str): (Optional) The operator to apply for comparison. Options: "lt" (less than), "le" (less than or equal to), "eq" (equal to), "ne" (not equal to), "ge" (greater than or equal to), "gt" (greater than). Defaults to "eq".
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.

    """

    def __init__(
        self,
        input_value_key: str,
        target_value: int | str,
        operator: str = "eq",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_value_key = input_value_key
        self.target_value = target_value
        if operator == "lt":
            self.operator = lt
        elif operator == "le":
            self.operator = le
        elif operator == "eq":
            self.operator = eq
        elif operator == "ne":
            self.operator = ne
        elif operator == "ge":
            self.operator = ge
        elif operator == "gt":
            self.operator = gt
        else:
            msg = 'Operator must be one from the list: "lt" (less than), "le" (less than or equal to), "eq" (equal to), "ne" (not equal to), "ge" (greater than or equal to), "gt" (greater than)'
            raise ValueError(msg)

    def process_dataset_entry(self, data_entry: AudioBatch) -> list[AudioBatch]:
        input_value = data_entry[self.input_value_key]
        target = self.target_value
        if self.operator(input_value, target):
            return [AudioBatch(data=data_entry)]
        else:
            return []
