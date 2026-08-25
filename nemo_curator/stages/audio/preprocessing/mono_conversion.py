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

"""
Audio mono conversion stage.

Converts multi-channel audio to mono and verifies sample rate.
Typically the first stage in an audio processing pipeline.

Example:
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.audio.preprocessing import MonoConversionStage

    pipeline = Pipeline(name="audio_pipeline")
    pipeline.add_stage(MonoConversionStage(output_sample_rate=48000))
"""

import os
from dataclasses import dataclass, field

import torch
from loguru import logger

from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, IOSpec, StageContract
from nemo_curator.stages.audio._agent._residency import (
    InputResidency,
    produce_audio_filepath,
    residency_read_specs,
    resolve_audio,
    write_audio_stable,
)
from nemo_curator.stages.audio.common import ensure_waveform_2d, load_audio_file
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class MonoConversionStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """
    Audio mono conversion and sample rate verification stage.

    Converts multi-channel audio to mono by averaging channels.
    Optionally verifies that audio matches expected sample rate.

    Args:
        output_sample_rate: Expected sample rate in Hz (default: 48000)
        audio_filepath_key: Key in data dict for audio file path
        strict_sample_rate: If True, reject audio with wrong sample rate
        waveform_key: Key in data dict for the in-memory mono waveform tensor.
        sample_rate_key: Key in data dict for the waveform sample rate.
        is_mono_key: Key where the mono flag is written.
        duration_key: Key where the audio duration in seconds is written.
        num_samples_key: Key where the number of samples is written.
        output_audio_filepath_key: Key where the written mono WAV path is stored
            (write_to_disk=True only).
        original_audio_filepath_key: Key preserving the pre-conversion path when
            update_audio_filepath=True.
        input_residency: Which input to use — "file" (audio_filepath only; default,
            matching this stage's pre-agent behavior), "waveform" (in-memory only), or
            "auto" (waveform first, file fallback).
        keep_waveform_in_task: If True (default), store the mono waveform and sample
            rate in task.data for downstream in-memory consumers.
        write_to_disk: If True, write the converted mono audio to a WAV file.
            write_to_disk without output_dir writes WAV files to the system temp dir
            and nothing cleans them up; in multi-node runs point output_dir at a
            shared filesystem.
        update_audio_filepath: If True (with write_to_disk), repoint audio_filepath_key
            at the written mono WAV and keep the old path under original_audio_filepath_key.
        output_dir: Directory for the written WAV files (default: system temp dir).
    """

    output_sample_rate: int = 48000
    audio_filepath_key: str = "audio_filepath"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    is_mono_key: str = "is_mono"
    duration_key: str = "duration"
    num_samples_key: str = "num_samples"
    output_audio_filepath_key: str = "mono_audio_filepath"
    original_audio_filepath_key: str = "original_audio_filepath"
    strict_sample_rate: bool = True
    # "file", not "auto": this stage only ever read ``audio_filepath`` before the agent work
    # (``load_audio_file(audio_filepath, mono=False)``), unlike sigmos/utmos/band/vad, whose
    # own resolvers already preferred a resident waveform and so default to "auto" honestly.
    # Defaulting to "auto" here silently changed which input a default pipeline reads.
    input_residency: InputResidency = "file"
    keep_waveform_in_task: bool = True
    write_to_disk: bool = False
    update_audio_filepath: bool = False
    output_dir: str | None = None

    name: str = "MonoConversion"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self):
        super().__init__()

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        outputs = [
            self.waveform_key,
            self.sample_rate_key,
            self.is_mono_key,
            self.duration_key,
            self.num_samples_key,
        ]
        if self.write_to_disk:
            outputs.append(self.output_audio_filepath_key)
            if self.update_audio_filepath:
                outputs.append(self.audio_filepath_key)
        return [], outputs

    def describe(self) -> StageContract:
        produces = []
        if self.keep_waveform_in_task:
            produces.append("tensor")
        if self.write_to_disk:
            produces.append("disk")
        writes = [
            self.is_mono_key,
            self.duration_key,
            self.num_samples_key,
        ]
        if self.keep_waveform_in_task:
            writes.extend([self.waveform_key, self.sample_rate_key])
        if self.write_to_disk:
            writes.append(self.output_audio_filepath_key)
            if self.update_audio_filepath:
                writes.append(self.audio_filepath_key)
        return StageContract(
            reads_one_of=residency_read_specs(
                self.input_residency,
                audio_filepath_key=self.audio_filepath_key,
                waveform_key=self.waveform_key,
                sample_rate_key=self.sample_rate_key,
            ),
            writes=IOSpec(data_keys=writes, produces=produces),
            gates=Gates(
                writes_to_disk=self.write_to_disk,
                output_path_params=["output_dir"],
                per_row_independent=True,
            ),
            # With ``strict_sample_rate`` -- the DEFAULT -- a row whose rate differs from
            # ``output_sample_rate`` returns ``[]``. That is row-dropping, and undeclared it
            # made this stage the one place the rule was invisible: the newer stages doing the
            # same thing say so, so validation treated a 48 kHz default silently discarding a
            # 16 kHz corpus as a pass-through, while flagging its neighbours for less.
            cardinality="filter" if self.strict_sample_rate else "1:1",
        )

    def _write_audio(self, waveform: torch.Tensor, sample_rate: int, task: AudioTask) -> str:
        stem = os.path.splitext(os.path.basename(str(task.data.get(self.audio_filepath_key, "audio"))))[0]
        return write_audio_stable(
            waveform,
            sample_rate,
            output_dir=self.output_dir,
            stem=stem,
            tag="mono",
        )

    def process(self, task: AudioTask) -> AudioTask | list[AudioTask]:  # noqa: C901 (complexity accepted: residency/sample-rate branch matrix; no refactor pre-PR)
        """
        Convert audio to mono and verify sample rate.

        Mutates task.data in-place with waveform data.
        Returns task if successful, [] if doesn't meet requirements.
        """
        try:
            resolved = resolve_audio(
                task.data,
                residency=self.input_residency,  # type: ignore[arg-type]
                audio_filepath_key=self.audio_filepath_key,
                waveform_key=self.waveform_key,
                sample_rate_key=self.sample_rate_key,
                mono=False,
                loader=load_audio_file,  # module-level symbol: patchable at this module, as pre-residency
            )
        except (OSError, RuntimeError) as e:  # corrupt/unreadable audio -> skip the row, don't crash the batch
            logger.error(f"Failed to load audio for {task.data.get(self.audio_filepath_key)!r}: {e}")
            return []
        if resolved is None:
            logger.error(f"Audio input not found for key {self.audio_filepath_key!r}")
            return []

        try:
            waveform, sample_rate = resolved
            waveform = ensure_waveform_2d(waveform)

            if sample_rate <= 0:
                logger.error(f"Invalid sample rate ({sample_rate}) in audio input")
                return []

            num_channels = waveform.shape[0]

            if self.strict_sample_rate and sample_rate != self.output_sample_rate:
                audio_source = task.data.get(self.audio_filepath_key, self.waveform_key)
                logger.warning(f"Sample rate {sample_rate}Hz != expected {self.output_sample_rate}Hz: {audio_source}")
                return []

            if num_channels > 1:
                mono_waveform = torch.mean(waveform, dim=0, keepdim=True)
                logger.debug(f"Converted {num_channels} channels to mono")
            else:
                mono_waveform = waveform

            if self.keep_waveform_in_task:
                task.data[self.waveform_key] = mono_waveform
                task.data[self.sample_rate_key] = sample_rate
            task.data[self.is_mono_key] = True
            task.data[self.duration_key] = mono_waveform.shape[1] / sample_rate
            task.data[self.num_samples_key] = mono_waveform.shape[1]

            if self.write_to_disk:
                path = self._write_audio(mono_waveform, sample_rate, task)
                task.data[self.output_audio_filepath_key] = path
                if self.update_audio_filepath:
                    produce_audio_filepath(
                        task.data,
                        path,
                        key=self.audio_filepath_key,
                        original_key=self.original_audio_filepath_key,
                    )

        except (OSError, RuntimeError) as e:
            logger.error(f"Error processing audio input: {e}")
            return []
        else:
            return task
