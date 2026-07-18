# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
Resample Audio Stage

Resamples audio files to a target sample rate and format.
Follows the exact pattern from NeMo Curator:
https://github.com/NVIDIA-NeMo/Curator/blob/main/nemo_curator/stages/audio/common.py

"""

import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass

from fsspec.core import url_to_fs

from nemo_curator.stages.audio._agent_ready import AgentReady, Gates, IOSpec, StageContract
from nemo_curator.stages.audio._residency import (
    InputResidency,
    cleanup_temp_files,
    produce_audio_filepath,
    residency_read_specs,
    resolve_audio_path,
)
from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.audio.common import get_audio_duration, load_audio_file
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


@dataclass
class ResampleAudioStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """
    Stage for resampling audio files in a TTS/ALM dataset.

    Takes a manifest containing audio file paths and resamples them to
    target sample rate and format, while creating a new manifest with
    updated paths.

    """

    # Processing parameters
    resampled_audio_dir: str
    input_format: str = "wav"
    target_sample_rate: int = 16000
    target_format: str = "wav"
    target_nchannels: int = 1

    # Key names
    audio_filepath_key: str = "audio_filepath"
    resampled_audio_filepath_key: str = "resampled_audio_filepath"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    duration_key: str = "duration"
    audio_item_id_key: str = "audio_item_id"
    original_audio_filepath_key: str = "original_audio_filepath"

    input_residency: InputResidency = "file"
    keep_waveform_in_task: bool = False
    write_to_disk: bool = True
    update_audio_filepath: bool = False

    # Stage metadata
    name: str = "ResampleAudio"

    def __post_init__(self) -> None:
        if not (self.keep_waveform_in_task or self.write_to_disk):
            msg = "At least one of keep_waveform_in_task or write_to_disk must be True"
            raise ValueError(msg)

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        if not shutil.which("ffmpeg"):
            msg = "ResampleAudioStage requires 'ffmpeg'. Install with: sudo apt-get install -y ffmpeg"
            raise RuntimeError(msg)
        fs, path = url_to_fs(self.resampled_audio_dir)
        fs.makedirs(path, exist_ok=True)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        outputs = [self.audio_item_id_key, self.duration_key]
        if self.write_to_disk:
            outputs.append(self.resampled_audio_filepath_key)
        if self.keep_waveform_in_task:
            outputs.extend([self.waveform_key, self.sample_rate_key])
        if self.update_audio_filepath:
            outputs.append(self.audio_filepath_key)
        return [], outputs

    def describe(self) -> StageContract:
        writes = [self.audio_item_id_key, self.duration_key]
        produces = []
        if self.write_to_disk:
            writes.append(self.resampled_audio_filepath_key)
            produces.append("disk")
        if self.keep_waveform_in_task:
            writes.extend([self.waveform_key, self.sample_rate_key])
            produces.append("tensor")
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
            gates=Gates(writes_to_disk=self.write_to_disk, requires_ffmpeg=True),
        )

    def process(self, task: AudioTask) -> AudioTask:
        """
        Process a single task by resampling the audio file.

        Args:
            task: AudioTask with data dict containing audio_filepath and audio_item_id(optional)

        Returns:
            AudioTask with updated metadata
        """
        t0 = time.perf_counter()
        data_entry = task.data

        temp_paths: list[str] = []
        input_audio_path = resolve_audio_path(
            data_entry,
            residency=self.input_residency,  # type: ignore[arg-type]
            audio_filepath_key=self.audio_filepath_key,
            waveform_key=self.waveform_key,
            sample_rate_key=self.sample_rate_key,
            register_temp=temp_paths,
        )
        if input_audio_path is None:
            msg = "Audio file path or waveform/sample_rate is required"
            raise ValueError(msg)

        original_audio_filepath = data_entry.get(self.audio_filepath_key)
        _, local_audio_path = url_to_fs(input_audio_path)
        if self.audio_item_id_key not in data_entry:
            stem = os.path.splitext(os.path.basename(local_audio_path))[0]
            path_hash = hashlib.sha256(local_audio_path.encode()).hexdigest()[:8]
            data_entry[self.audio_item_id_key] = f"{stem}_{path_hash}"

        if self.write_to_disk:
            output_audio_path = os.path.join(
                self.resampled_audio_dir,
                data_entry[self.audio_item_id_key] + "." + self.target_format,
            )
        else:
            fd, output_audio_path = tempfile.mkstemp(suffix=f".{self.target_format}")
            os.close(fd)

        # Convert audio file if not already done
        fs, output_path = url_to_fs(output_audio_path)
        skipped_conversion = self.write_to_disk and fs.exists(output_path)
        if not skipped_conversion:
            cmd = [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                input_audio_path,
                "-ar",
                str(self.target_sample_rate),
                "-ac",
                str(self.target_nchannels),
                "-acodec",
                "pcm_s16le",
                output_audio_path,
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
            except subprocess.CalledProcessError as e:
                cleanup_temp_files(temp_paths)
                msg = f"Error converting {input_audio_path}: {e}"
                raise RuntimeError(msg) from e

        # Input temp WAV (materialized from a waveform) is no longer needed after conversion.
        cleanup_temp_files(temp_paths)

        # Update metadata — preserve original URL for cloud paths.
        if original_audio_filepath is not None:
            data_entry[self.audio_filepath_key] = original_audio_filepath
        if self.write_to_disk:
            data_entry[self.resampled_audio_filepath_key] = output_audio_path
            if self.update_audio_filepath:
                produce_audio_filepath(
                    data_entry,
                    output_audio_path,
                    key=self.audio_filepath_key,
                    original_key=self.original_audio_filepath_key,
                )
        if self.keep_waveform_in_task:
            waveform, sample_rate = load_audio_file(output_audio_path, mono=False)
            data_entry[self.waveform_key] = waveform
            data_entry[self.sample_rate_key] = sample_rate
        duration = get_audio_duration(output_audio_path)
        data_entry[self.duration_key] = duration
        if not self.write_to_disk:
            try:
                os.remove(output_audio_path)
            except OSError:
                pass

        self._log_metrics(
            {
                "process_time": time.perf_counter() - t0,
                "duration": max(duration, 0.0),
                "skipped_conversion": float(skipped_conversion),
            }
        )
        return task
