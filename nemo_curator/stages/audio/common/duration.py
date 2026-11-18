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

import time
from dataclasses import KW_ONLY, dataclass
from typing import Any, Literal

import soundfile
from loguru import logger

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, IOSpec, StageContract
from nemo_curator.stages.audio.common.helpers import ensure_waveform_2d
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


def get_audio_duration(audio_filepath: str) -> float:
    """Get the duration of the audio file in seconds."""
    try:
        info = soundfile.info(audio_filepath)
        return info.frames / info.samplerate
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to get duration for audio file {audio_filepath}: {e}")
        return -1.0


@dataclass
class GetAudioDurationStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """Compute audio duration from the file at *audio_filepath_key* and
    store the result under *duration_key*.

    Args:
        audio_filepath_key: Key to get path to wav file.
        duration_key: Key to put audio duration.
        waveform_key: Key for an in-memory waveform tensor.
        sample_rate_key: Key for the in-memory waveform sample rate.
        input_residency: Which input to use — "file" (audio_filepath only; default,
            unchanged), "waveform" (in-memory only), or "auto" (waveform first, file fallback).
    """

    name: str = "GetAudioDurationStage"
    audio_filepath_key: str = "audio_filepath"
    duration_key: str = "duration"
    _: KW_ONLY
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    input_residency: Literal["file", "waveform", "auto"] = "file"

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        import soundfile

        self._soundfile = soundfile

    def inputs(self) -> tuple[list[str], list[str]]:
        if self.input_residency == "waveform":
            return [], [self.waveform_key, self.sample_rate_key]
        return [], [self.audio_filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.duration_key]

    def describe(self) -> StageContract:
        from nemo_curator.stages.audio._agent._residency import residency_read_specs

        return StageContract(
            preferred_reads=(
                IOSpec(data_keys=[self.waveform_key, self.sample_rate_key], accepts=["waveform"])
                if self.input_residency == "auto"
                else None
            ),
            reads_one_of=residency_read_specs(
                self.input_residency,
                audio_filepath_key=self.audio_filepath_key,
                waveform_key=self.waveform_key,
                sample_rate_key=self.sample_rate_key,
            ),
            writes=IOSpec(data_keys=[self.duration_key]),
            key_roles={self.duration_key: "duration"},
            gates=Gates(per_row_independent=True),
        )

    def __post_init__(self) -> None:
        from nemo_curator.stages.audio._agent._residency import validate_input_residency

        validate_input_residency(self.input_residency, stage_name=self.name)
        if self.input_residency != "file":
            from nemo_curator.stages.audio._agent._residency import validate_audio_key_configuration

            inputs = {"waveform_key": self.waveform_key, "sample_rate_key": self.sample_rate_key}
            if self.input_residency == "auto":
                inputs["audio_filepath_key"] = self.audio_filepath_key
            validate_audio_key_configuration(
                self.name, input_keys=inputs, output_keys={"duration_key": self.duration_key}
            )

    def validate_input(self, task: AudioTask) -> bool:
        if not super().validate_input(task):
            return False
        return self.input_residency != "waveform" or self._resident_rate(task.data) is not None

    def input_spec_for_task(self, task: AudioTask) -> tuple[list[str], list[str]]:
        attrs, keys = super().input_spec_for_task(task)
        if self.input_residency == "auto" and self._resident_rate(task.data) is not None:
            keys = [key for key in keys if key != self.audio_filepath_key]
            keys.extend([self.waveform_key, self.sample_rate_key])
        return attrs, keys

    def _resident_rate(self, data: dict[str, Any]) -> int | None:
        from nemo_curator.stages.audio._agent._residency import resident_sample_rate

        if data.get(self.waveform_key) is None or data.get(self.sample_rate_key) is None:
            return None
        try:
            return resident_sample_rate(
                data[self.sample_rate_key], sample_rate_key=self.sample_rate_key, stage_name=self.name
            )
        except ValueError:
            return None

    def _resolve_duration(self, data: dict[str, Any]) -> float:
        """Prefer valid resident audio only in the explicitly selected residency modes."""
        if self.input_residency != "file":
            rate = self._resident_rate(data)
            if rate is not None:
                return ensure_waveform_2d(data[self.waveform_key]).shape[-1] / rate
            if self.input_residency == "waveform":
                msg = f"[{self.name}] Missing waveform or invalid resident sample rate"
                raise ValueError(msg)
        return get_audio_duration(data[self.audio_filepath_key])

    def process(self, task: AudioTask) -> AudioTask:
        t0 = time.perf_counter()
        duration = self._resolve_duration(task.data)
        task.data[self.duration_key] = duration
        self._log_metrics({"process_time": time.perf_counter() - t0, "duration": max(duration, 0.0)})
        return task
