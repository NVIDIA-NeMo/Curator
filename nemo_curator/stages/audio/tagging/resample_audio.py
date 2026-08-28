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

import soundfile
from fsspec.core import url_to_fs

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, IOSpec, StageContract
from nemo_curator.stages.audio._agent._residency import (
    InputResidency,
    cleanup_temp_files,
    produce_audio_filepath,
    residency_read_specs,
    resolve_audio_path,
)
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
            gates=Gates(
                writes_to_disk=self.write_to_disk,
                requires_ffmpeg=True,
                output_path_params=["resampled_audio_dir"],
                per_row_independent=True,
            ),
        )

    def _audio_digest(self, local_audio_path: str) -> str:
        """A short digest of this audio and the settings about to be applied to it."""
        digest = hashlib.sha256()
        with open(local_audio_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        digest.update(f"|{self.target_sample_rate}|{self.target_nchannels}|{self.target_format}".encode())
        return digest.hexdigest()[:16]

    def _item_id(self, local_audio_path: str, *, from_scratch_file: bool, source: str | None) -> str:
        """The output filename for a row that does not already carry an id.

        A real input path is stable across runs, so it can name the output. A scratch path
        materialised from a waveform is not: naming a persistent output after it made every run
        write a fresh set of files (measured: 129 on disk against 65 manifest rows). So hash the
        audio instead, which is the identity the path was standing in for -- and folding in the
        settings makes the "already converted, skip it" check below correct, not merely fast.

        The path branch is unchanged, so pipelines reading real files keep their output names.
        """
        if not from_scratch_file:
            stem = os.path.splitext(os.path.basename(local_audio_path))[0]
            return f"{stem}_{hashlib.sha256(local_audio_path.encode()).hexdigest()[:8]}"
        # Keep the source name on the front so a clip stays traceable by eye.
        stem = os.path.splitext(os.path.basename(str(source)))[0] if source else "clip"
        return f"{stem}_{self._audio_digest(local_audio_path)}"

    def _matches_target(self, path: str) -> bool:
        """Whether the file at the output path really holds the conversion asked for.

        The file-route name carries the source path, never the settings, so a name hit is not
        evidence the work is done: a second run at a different ``target_sample_rate`` used to skip
        and serve the old rate, with the duration measured off the stale file. Reading the header
        is free beside spawning ffmpeg. A header-valid but truncated file still passes this, which
        is why the conversion below writes to a temp name and renames.
        """
        try:
            info = soundfile.info(path)
        except Exception:  # noqa: BLE001 - unreadable or not-audio -> convert it again
            return False
        return info.samplerate == self.target_sample_rate and info.channels == self.target_nchannels

    def process(self, task: AudioTask) -> AudioTask:  # noqa: C901, PLR0912, PLR0915
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
        inherited_id = self.audio_item_id_key in data_entry
        if not inherited_id:
            data_entry[self.audio_item_id_key] = self._item_id(
                local_audio_path,
                from_scratch_file=bool(temp_paths),
                source=original_audio_filepath,
            )
        output_stem = data_entry[self.audio_item_id_key]
        if inherited_id and temp_paths:
            # A fan-out gives every child the parent's id, so an inherited id is not a filename:
            # 26 VAD segments of utt1.wav once collapsed onto one, with all 26 rows pointing at
            # the survivor. The FILE takes a digest; the row keeps the id its producer gave it,
            # which downstream stages read as the shared ``item_id`` role.
            output_stem = f"{output_stem}_{self._audio_digest(local_audio_path)}"

        if self.write_to_disk:
            output_audio_path = os.path.join(
                self.resampled_audio_dir,
                output_stem + "." + self.target_format,
            )
        else:
            fd, output_audio_path = tempfile.mkstemp(suffix=f".{self.target_format}")
            os.close(fd)

        # Convert audio file if not already done
        fs, output_path = url_to_fs(output_audio_path)
        skipped_conversion = self.write_to_disk and fs.exists(output_path) and self._matches_target(output_path)
        if not skipped_conversion:
            # ffmpeg used to write straight to the deliverable. That was survivable while every
            # run picked a new output name, but the name is stable now, so a run killed mid-write
            # leaves a stump the NEXT run finds -- and a truncated WAV keeps a valid header, so
            # the skip above waves it through and a fragment's duration lands in the manifest.
            # Convert to a sibling temp name and rename, which is atomic on POSIX.
            staging_dir = os.path.dirname(output_audio_path)
            if staging_dir:
                # setup_on_node makes this, but process() must not depend on having been through it.
                os.makedirs(staging_dir, exist_ok=True)
            staged_fd, staged = tempfile.mkstemp(prefix=".", suffix=f".{self.target_format}", dir=staging_dir or None)
            os.close(staged_fd)
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
                staged,
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
                os.replace(staged, output_audio_path)
            except subprocess.CalledProcessError as e:
                cleanup_temp_files([staged, *temp_paths])
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
            try:  # noqa: SIM105
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
