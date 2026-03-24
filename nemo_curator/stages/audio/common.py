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

import os
from abc import abstractmethod
from dataclasses import dataclass
from operator import eq, ge, gt, le, lt, ne
from typing import Any

import soundfile
import torch
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
            entries = self.process_dataset_entry(entry)
            for r in entries:
                if r is not task and not r._stage_perf:
                    r._stage_perf = list(task._stage_perf)
            result.extend(entries)
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

    name = "GetAudioDurationStage"
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

    name = "PreserveByValueStage"

    def __init__(
        self,
        input_value_key: str,
        target_value: int | str,
        operator: str = "eq",
    ):
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


def load_audio_file(audio_path: str, mono: bool = True) -> tuple[torch.Tensor, int]:
    """Load audio file and return waveform tensor (channels, samples) and sample rate."""
    data, sample_rate = soundfile.read(audio_path, dtype="float32")
    waveform = torch.from_numpy(data)
    waveform = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform.T
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sample_rate


def ensure_waveform_2d(waveform: Any) -> torch.Tensor:
    """Ensure waveform is a torch.Tensor in 2D (channels, samples) format."""
    if not torch.is_tensor(waveform):
        waveform = torch.as_tensor(waveform, dtype=torch.float32)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    return waveform


def ensure_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Convert multi-channel waveform to mono. Assumes 2D (channels, samples) input."""
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def resolve_waveform_from_item(
    item: dict[str, Any], task_id: str, mono: bool = True
) -> tuple[torch.Tensor, int] | None:
    """
    Resolve (waveform, sample_rate) from an item dict, loading from file if needed.

    Checks item['waveform'] + item['sample_rate'], falls back to loading from
    item['audio_filepath'], resolves missing sample_rate from file header.
    Updates item in-place when loading from file.
    Returns None if resolution fails.
    """
    waveform = item.get("waveform")
    sample_rate = item.get("sample_rate")

    if waveform is None:
        audio_filepath = item.get("audio_filepath")
        if audio_filepath and os.path.exists(audio_filepath):
            try:
                waveform, sample_rate = load_audio_file(audio_filepath, mono=mono)
                item["waveform"] = waveform
                item["sample_rate"] = sample_rate
            except Exception as e:
                logger.error(f"[{task_id}] Failed to load audio file: {e}")
                return None
        else:
            logger.warning(f"[{task_id}] No waveform or valid audio_filepath found")
            return None
    elif sample_rate is None:
        audio_filepath = item.get("audio_filepath")
        if audio_filepath and os.path.exists(audio_filepath):
            try:
                info = soundfile.info(audio_filepath)
                sample_rate = info.samplerate
                item["sample_rate"] = sample_rate
            except Exception as e:
                logger.error(f"[{task_id}] Waveform present but sample_rate missing "
                             f"and could not read from '{audio_filepath}': {e}")
                return None
        else:
            logger.error(f"[{task_id}] Waveform present but 'sample_rate' missing "
                         "and no audio_filepath available.")
            return None

    waveform = ensure_waveform_2d(waveform)
    if mono:
        waveform = ensure_mono(waveform)

    return waveform, sample_rate


def resolve_model_path(model_path: str, reference_file: str, module_subdir: str) -> str:
    """Resolve a relative model path using the reference file's directory and module subdirectory."""
    if os.path.isabs(model_path):
        return model_path
    current_dir = os.path.dirname(os.path.abspath(reference_file))
    module_dir = os.path.join(current_dir, module_subdir)
    for base in (module_dir, current_dir):
        resolved = os.path.join(base, model_path)
        if os.path.exists(resolved):
            return resolved
    return os.path.join(module_dir, model_path)
