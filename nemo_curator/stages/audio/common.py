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

from dataclasses import dataclass
from operator import eq, ge, gt, le, lt, ne
from typing import Any

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


class AudioTaskStage(ProcessingStage[AudioTask, AudioTask]):
    """Base class for audio processing stages.

    Three-method hierarchy (cannot be reduced further):

    ``process_dataset_entry(data: dict) -> dict | None``
        CPU-stage hook.  Receives the raw manifest dict, returns the
        (possibly mutated) dict or ``None`` to drop the entry.  Five+
        concrete stages use this; it eliminates AudioTask
        unwrap/rewrap boilerplate for simple per-entry transforms.

    ``process_batch(tasks: list[AudioTask]) -> list[AudioTask]``
        The real entry point called by backends (Xenna / Ray).
        Default implementation validates inputs then loops via
        ``process_dataset_entry``, wrapping each returned dict in
        a new ``AudioTask``.
        **GPU stages and IO stages override this** for batched
        processing (e.g. batched inference, aggregated conversion).

    ``process(task: AudioTask) -> AudioTask | list[AudioTask]``
        Required by abstract ``ProcessingStage``.  Delegates to
        ``process_batch([task])`` so every code path (single or
        batched) goes through one validation gateway.

    Subclass contract:
        - CPU stage  → override ``process_dataset_entry``
        - GPU / IO stage  → override ``process_batch``
    """

    def process_dataset_entry(self, data_entry: dict) -> dict | None:
        """Process a single manifest entry dict.

        CPU stages override this.  GPU stages override
        ``process_batch`` instead.

        Returns:
            dict: processed entry (may be the same object, mutated in-place).
            None: to drop / filter out this entry.
        """
        msg = f"{type(self).__name__} must implement process_dataset_entry or override process_batch"
        raise NotImplementedError(msg)

    def _validate_batch(self, tasks: list[AudioTask]) -> None:
        """Validate that every task in the batch has the required columns.

        Raises ``ValueError`` on the first task that fails.
        """
        for task in tasks:
            if not self.validate_input(task):
                msg = f"Task {task.task_id} missing required columns for {type(self).__name__}: {self.inputs()}"
                raise ValueError(msg)

    def process(self, task: AudioTask) -> AudioTask | list[AudioTask]:
        results = self.process_batch([task])
        if not results:
            return []
        return results[0] if len(results) == 1 else results

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Validate then process via ``process_dataset_entry``.

        GPU and IO stages override this entirely for batched
        processing.
        """
        if len(tasks) == 0:
            return []
        self._validate_batch(tasks)
        results: list[AudioTask] = []
        for task in tasks:
            result = self.process_dataset_entry(task.data)
            if result is None:
                continue
            results.append(
                AudioTask(
                    data=result,
                    task_id=task.task_id,
                    dataset_name=task.dataset_name,
                    filepath_key=task.filepath_key,
                    _stage_perf=list(task._stage_perf),
                    _metadata=task._metadata.copy(),
                )
            )
        return results


# ---------------------------------------------------------------------------
# Concrete stages
# ---------------------------------------------------------------------------


@dataclass
class GetAudioDurationStage(AudioTaskStage):
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


class PreserveByValueStage(AudioTaskStage):
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
