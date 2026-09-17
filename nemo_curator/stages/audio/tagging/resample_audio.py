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
import uuid
from dataclasses import KW_ONLY, dataclass, field
from typing import ClassVar

import soundfile
from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.audio._agent._agent_ready import (
    AgentReady,
    ConditionalWrite,
    Gates,
    IOSpec,
    StageContract,
    StaticHints,
)
from nemo_curator.stages.audio._agent._residency import (
    InputResidency,
    cleanup_temp_files,
    drop_resident_audio,
    produce_audio_filepath,
    reject_sinkless_conversion,
    residency_read_specs,
    resolve_audio_path,
    validate_audio_key_configuration,
    validate_input_residency,
)
from nemo_curator.stages.audio.common import get_audio_duration, load_audio_file
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


def _is_usable_sample_rate(value: object) -> bool:
    """Whether a resident ``sample_rate`` can actually time a waveform.

    Rejects ``None``, ``bool`` (an accidental ``int`` subclass), non-positive rates, and
    fractional floats such as ``16000.5``; accepts a positive int or an integer-valued float.
    """
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, int):
        return value > 0
    if isinstance(value, float):
        return value > 0 and value.is_integer()
    return False


@dataclass
class ResampleAudioStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """
    Stage for resampling audio files in a TTS/ALM dataset.

    Takes a manifest containing audio file paths and resamples them to
    target sample rate and format, while creating a new manifest with
    updated paths.

    """

    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(
        gates=Gates(
            writes_to_disk=True,
            requires_ffmpeg=True,
            output_path_params=["resampled_audio_dir"],
            per_row_independent=True,
        )
    )

    # Processing parameters (legacy positional order preserved)
    resampled_audio_dir: str
    input_format: str = "wav"
    target_sample_rate: int = 16000
    target_format: str = "wav"
    target_nchannels: int = 1

    # Key names (legacy positional slots)
    audio_filepath_key: str = "audio_filepath"
    resampled_audio_filepath_key: str = "resampled_audio_filepath"
    duration_key: str = "duration"
    audio_item_id_key: str = "audio_item_id"

    # Stage metadata (legacy positional slot)
    name: str = "ResampleAudio"

    # Agent-added knobs are keyword-only (KW_ONLY sentinel) so the legacy positional
    # slots above keep their historical order and meaning.
    _: KW_ONLY
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    original_audio_filepath_key: str = "original_audio_filepath"
    input_residency: InputResidency = "file"
    keep_waveform_in_task: bool = False
    write_to_disk: bool = True
    update_audio_filepath: bool = False

    # Per-worker record of which source each inherited-id output name belongs to; see process().
    _stem_owners: dict[str, str] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        validate_audio_key_configuration(
            type(self).__name__,
            input_keys={
                "waveform_key": self.waveform_key,
                "sample_rate_key": self.sample_rate_key,
            },
            output_keys={},
        )
        validate_input_residency(self.input_residency, stage_name=type(self).__name__)
        reject_sinkless_conversion(
            stage=type(self).__name__,
            keep_waveform_in_task=self.keep_waveform_in_task,
            write_to_disk=self.write_to_disk,
            update_audio_filepath=self.update_audio_filepath,
        )

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

    def validate_input(self, task: AudioTask) -> bool:
        """Validate the configured file/waveform residency alternative.

        A resident waveform is only a usable alternative when its ``sample_rate`` is a
        positive, non-boolean, losslessly-integer value: a missing/zero/negative rate cannot
        time the audio, ``True`` is an accident of ``bool`` being an ``int`` subclass, and a
        fractional rate (e.g. ``16000.5``) is not a frame rate the converter can honor.
        """
        data = task.data
        has_file = bool(data.get(self.audio_filepath_key))
        has_waveform = data.get(self.waveform_key) is not None and _is_usable_sample_rate(
            data.get(self.sample_rate_key)
        )
        if self.input_residency == "file":
            return has_file
        if self.input_residency == "waveform":
            return has_waveform
        return has_file or has_waveform

    def outputs(self) -> tuple[list[str], list[str]]:
        # Keep the pre-PR public tuple (audio_filepath, audio_item_id, resampled_audio_filepath,
        # duration) for the default/file-compatible route, since the runtime preserves those
        # keys, and extend it for the newer resident/rename sink modes.
        outputs = [self.audio_filepath_key, self.audio_item_id_key]
        if self.write_to_disk:
            outputs.append(self.resampled_audio_filepath_key)
        outputs.append(self.duration_key)
        if self.keep_waveform_in_task:
            outputs.extend([self.waveform_key, self.sample_rate_key])
        if self.update_audio_filepath:
            outputs.append(self.original_audio_filepath_key)
        return [], outputs

    def describe(self) -> StageContract:
        writes = [self.audio_item_id_key, self.duration_key]
        produces = []
        conditional_writes = []
        if self.write_to_disk:
            writes.append(self.resampled_audio_filepath_key)
            produces.append("disk")
        if self.keep_waveform_in_task:
            writes.extend([self.waveform_key, self.sample_rate_key])
            produces.append("tensor")
        if self.update_audio_filepath:
            writes.append(self.audio_filepath_key)
            conditional_writes.append(
                ConditionalWrite(
                    writes=IOSpec(data_keys=[self.original_audio_filepath_key]),
                    condition=(
                        f"'{self.audio_filepath_key}' exists and "
                        f"'{self.original_audio_filepath_key}' is not already present"
                    ),
                )
            )
        return StageContract(
            reads_one_of=residency_read_specs(
                self.input_residency,
                audio_filepath_key=self.audio_filepath_key,
                waveform_key=self.waveform_key,
                sample_rate_key=self.sample_rate_key,
            ),
            writes=IOSpec(data_keys=writes, produces=produces),
            # Only a resident-source config (waveform/auto) supersedes and drops the resident
            # pair. The legacy file route never touches it, so its contract stays empty.
            removes_keys=(
                [self.waveform_key, self.sample_rate_key]
                if self.write_to_disk and not self.keep_waveform_in_task and self.input_residency != "file"
                else []
            ),
            conditional_writes=conditional_writes,
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
            # 16 hex (64 bits) matches _audio_digest: an 8-hex (32-bit) suffix made two distinct
            # paths sharing a basename stem collide far too readily onto one output stem.
            return f"{stem}_{hashlib.sha256(local_audio_path.encode()).hexdigest()[:16]}"
        # Keep the source name on the front so a clip stays traceable by eye.
        stem = os.path.splitext(os.path.basename(str(source)))[0] if source else "clip"
        return f"{stem}_{self._audio_digest(local_audio_path)}"

    # Resampling may land a handful of frames off the exact ratio; anything beyond this is a
    # different recording or a truncated one, never a rounding artefact.
    _FRAME_TOLERANCE_FLOOR = 64
    _FRAME_TOLERANCE_RATIO = 0.01

    def _matches_target(self, path: str, input_audio_path: str | None = None) -> bool:
        """Whether the file at the output path really holds the conversion asked for.

        The file-route name carries the source path, never the settings, so a name hit is not
        evidence the work is done: a second run at a different ``target_sample_rate`` used to skip
        and serve the old rate, with the duration measured off the stale file. Reading the header
        is free beside spawning ffmpeg.

        The header alone is not enough either: a WAV truncated by a killed writer -- or one that
        is a complete conversion of a DIFFERENT recording that happened to claim the same output
        name -- keeps a valid header at the right rate and channel count. So when the source is
        readable, the frame count must also agree with the source's length at the target rate
        (within resampling tolerance); otherwise the file is converted again.
        """
        try:
            info = soundfile.info(path)
        except Exception:  # noqa: BLE001 - unreadable or not-audio -> convert it again
            return False
        if info.samplerate != self.target_sample_rate or info.channels != self.target_nchannels:
            return False
        if input_audio_path is None:
            return True
        try:
            source = soundfile.info(input_audio_path)
        except Exception:  # noqa: BLE001 - a source libsndfile cannot read (e.g. mp3): header check only
            return True
        expected_frames = round(source.frames * self.target_sample_rate / source.samplerate)
        tolerance = max(self._FRAME_TOLERANCE_FLOOR, int(expected_frames * self._FRAME_TOLERANCE_RATIO))
        return abs(info.frames - expected_frames) <= tolerance

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
        elif inherited_id and self.write_to_disk:
            # The legacy file route names the output after the inherited id, which keeps every
            # tutorial's output names stable -- but two DIFFERENT recordings carrying the same id
            # would then share one output path, and the second would either overwrite the first
            # or (if the lengths agree) be served the first recording as a finished conversion.
            # Keep the legacy name for the first source seen under an id in this worker; a later
            # source with the same id gets a path digest so it can never alias the first.
            owner = self._stem_owners.setdefault(str(output_stem), local_audio_path)
            if owner != local_audio_path:
                digest = hashlib.sha256(local_audio_path.encode()).hexdigest()[:16]
                logger.warning(
                    f"[{self.name}] {self.audio_item_id_key}={output_stem!r} names both {owner!r} and "
                    f"{local_audio_path!r}; writing the latter as {output_stem}_{digest} so it cannot alias the first"
                )
                output_stem = f"{output_stem}_{digest}"

        if self.write_to_disk:
            output_audio_path = os.path.join(
                self.resampled_audio_dir,
                output_stem + "." + self.target_format,
            )
        else:
            fd, output_audio_path = tempfile.mkstemp(suffix=f".{self.target_format}")
            os.close(fd)

        try:
            return self._convert_and_update(
                task,
                input_audio_path=input_audio_path,
                output_audio_path=output_audio_path,
                original_audio_filepath=original_audio_filepath,
                # A materialized temp path means the source was a resident waveform, so the
                # resident pair is now stale and must be dropped. The file route never is.
                used_resident_source=bool(temp_paths),
                started_at=t0,
            )
        finally:
            cleanup_temp_files(temp_paths)
            if not self.write_to_disk:
                cleanup_temp_files([output_audio_path])

    def _convert_and_update(  # noqa: PLR0913 - keyword-only per-conversion inputs, not unrelated knobs
        self,
        task: AudioTask,
        *,
        input_audio_path: str,
        output_audio_path: str,
        original_audio_filepath: str | None,
        used_resident_source: bool,
        started_at: float,
    ) -> AudioTask:
        """Convert one resolved input and update its task metadata."""
        data_entry = task.data

        # Convert audio file if not already done
        fs, output_path = url_to_fs(output_audio_path)
        skipped_conversion = (
            self.write_to_disk and fs.exists(output_path) and self._matches_target(output_path, input_audio_path)
        )
        if not skipped_conversion:
            # ffmpeg used to write straight to the deliverable. That was survivable while every
            # run picked a new output name, but the name is stable now, so a run killed mid-write
            # leaves a stump the NEXT run finds -- and a truncated WAV keeps a valid header, so
            # the skip above waves it through and a fragment's duration lands in the manifest.
            # Convert to a sibling temp name and rename, which is atomic on POSIX. Upstream
            # landed the same fix independently; this keeps its naming so the two do not drift.
            staging_dir = os.path.dirname(output_audio_path)
            if staging_dir:
                # setup_on_node makes this, but process() must not depend on having been through it.
                os.makedirs(staging_dir, exist_ok=True)
            output_stem, output_extension = os.path.splitext(output_audio_path)
            temporary_audio_path = f"{output_stem}.{uuid.uuid4().hex}.tmp{output_extension}"
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
                temporary_audio_path,
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
                os.replace(temporary_audio_path, output_audio_path)
            except subprocess.CalledProcessError as e:
                msg = f"Error converting {input_audio_path}: {e}"
                raise RuntimeError(msg) from e
            finally:
                cleanup_temp_files([temporary_audio_path])

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
        elif self.write_to_disk and used_resident_source:
            # Drop the stale resident pair only when the conversion actually consumed a
            # resident/materialized source; a plain file-route conversion leaves the row's
            # own pre-existing pair (e.g. a carried sample_rate) untouched.
            drop_resident_audio(
                data_entry,
                waveform_key=self.waveform_key,
                sample_rate_key=self.sample_rate_key,
            )
        duration = get_audio_duration(output_audio_path)
        data_entry[self.duration_key] = duration

        self._log_metrics(
            {
                "process_time": time.perf_counter() - started_at,
                "duration": max(duration, 0.0),
                "skipped_conversion": float(skipped_conversion),
            }
        )
        return task
