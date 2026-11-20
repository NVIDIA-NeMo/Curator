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

"""Record, select on, or convert an audio row's sample rate and channel count."""

import os
from dataclasses import dataclass, field
from typing import ClassVar

import fsspec
import soundfile as sf
import torch
from loguru import logger

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
    accepts_for_residency,
    drop_resident_audio,
    normalize_audio_waveform,
    produce_audio_filepath,
    reject_sinkless_conversion,
    residency_read_specs,
    resident_sample_rate,
    resolve_audio,
    validate_audio_key_configuration,
    validate_input_residency,
    write_audio_stable,
)
from nemo_curator.stages.audio.common import ensure_waveform_2d, load_audio_file, resample_waveform
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class AudioFormatStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """Record, select on, or convert each row's sample rate and channel count.

    Every row gets its sample rate and channel count recorded. Rows outside the configured
    ``allowed_*``/``min_*``/``max_*`` constraints are dropped. Without a ``target_*`` that is
    all: the values come from a resident waveform or the file header, and no samples are
    decoded. With ``target_sample_rate`` and/or ``target_channels`` the audio is decoded and
    converted (downmix to mono, duplicate mono into N channels, resample), then kept in the
    task and/or written to a WAV file.

    Args:
        allowed_sample_rates / min_sample_rate / max_sample_rate: Keep rows whose sample rate is
            one of / at least / at most these (inclusive). None = no constraint.
        allowed_channels / min_channels / max_channels: The same for the channel count.
        target_sample_rate: Resample to this rate. None keeps the rate.
        target_channels: Convert to this many channels: 1 downmixes by averaging, N > 1
            duplicates mono. Other layouts (e.g. 6 -> 2) are dropped. None keeps the count.
        audio_filepath_key / waveform_key / sample_rate_key: Where the audio lives.
        num_channels_key / duration_key: Where the channel count and duration (seconds,
            conversion only) are written. The sample rate is written to ``sample_rate_key``.
        output_audio_filepath_key: Key for the written WAV path (``write_to_disk``).
        original_audio_filepath_key: Key preserving the old path (``update_audio_filepath``).
        input_residency: "auto" (resident waveform first, file fallback), "waveform" or "file".
        keep_waveform_in_task: Keep the converted waveform and rate in task.data.
        write_to_disk: Write the converted audio to a WAV file under ``output_dir`` (the
            system temp dir when None; nothing cleans it up there).
        update_audio_filepath: Point ``audio_filepath_key`` at the written file.
    """

    allowed_sample_rates: list[int] | None = None
    min_sample_rate: int | None = None
    max_sample_rate: int | None = None
    allowed_channels: list[int] | None = None
    min_channels: int | None = None
    max_channels: int | None = None
    target_sample_rate: int | None = None
    target_channels: int | None = None

    audio_filepath_key: str = "audio_filepath"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    num_channels_key: str = "num_channels"
    duration_key: str = "duration"
    output_audio_filepath_key: str = "converted_audio_filepath"
    original_audio_filepath_key: str = "original_audio_filepath"

    input_residency: InputResidency = "auto"
    keep_waveform_in_task: bool = True
    write_to_disk: bool = False
    update_audio_filepath: bool = False
    output_dir: str | None = None

    name: str = "AudioFormat"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    AGENT_STATIC: ClassVar[StaticHints] = StaticHints(cardinality_options=["1:1", "filter"])
    # Own bookkeeping: nothing routes on the channel count, so it needs no shared role.
    INTERNAL_KEY_FIELDS: ClassVar[frozenset[str]] = frozenset({"num_channels_key"})

    def __post_init__(self) -> None:  # noqa: C901 (one check per parameter group)
        super().__init__()
        validate_input_residency(self.input_residency, stage_name=self.name)
        for name in (
            "allowed_sample_rates",
            "min_sample_rate",
            "max_sample_rate",
            "allowed_channels",
            "min_channels",
            "max_channels",
            "target_sample_rate",
            "target_channels",
        ):
            value = getattr(self, name)
            if isinstance(value, list) and not value:
                msg = f"{name} must name at least one value, or be None for no constraint"
                raise ValueError(msg)
            for item in value if isinstance(value, list) else [value]:
                if item is not None and (isinstance(item, bool) or not isinstance(item, int) or item < 1):
                    msg = f"{name} must be positive whole numbers, got {value!r}"
                    raise ValueError(msg)
        for low, high in (("min_sample_rate", "max_sample_rate"), ("min_channels", "max_channels")):
            if None not in (getattr(self, low), getattr(self, high)) and getattr(self, low) > getattr(self, high):
                msg = f"{low} is above {high}, so nothing can pass"
                raise ValueError(msg)
        if not self._converts and (self.write_to_disk or self.update_audio_filepath or self.output_dir):
            msg = "write_to_disk, update_audio_filepath and output_dir need target_sample_rate or target_channels"
            raise ValueError(msg)
        if self._converts:
            reject_sinkless_conversion(
                stage=self.name,
                keep_waveform_in_task=self.keep_waveform_in_task,
                write_to_disk=self.write_to_disk,
                update_audio_filepath=self.update_audio_filepath,
            )
        output_keys = {"num_channels_key": self.num_channels_key}
        if self._converts:
            output_keys["duration_key"] = self.duration_key
            if self.write_to_disk:
                output_keys["output_audio_filepath_key"] = self.output_audio_filepath_key
            if self.update_audio_filepath:
                output_keys["original_audio_filepath_key"] = self.original_audio_filepath_key
        validate_audio_key_configuration(
            self.name,
            input_keys={
                "audio_filepath_key": self.audio_filepath_key,
                "waveform_key": self.waveform_key,
                "sample_rate_key": self.sample_rate_key,
            },
            output_keys=output_keys,
        )

    @property
    def _converts(self) -> bool:
        return self.target_sample_rate is not None or self.target_channels is not None

    @property
    def _selects(self) -> bool:
        constraints = (
            self.allowed_sample_rates,
            self.min_sample_rate,
            self.max_sample_rate,
            self.allowed_channels,
            self.min_channels,
            self.max_channels,
        )
        # A channel target above 1 cannot be reached from every layout, so those rows drop too.
        return any(c is not None for c in constraints) or (self.target_channels or 1) > 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], self._written_keys()

    def _written_keys(self) -> list[str]:
        keys = [self.sample_rate_key, self.num_channels_key]
        if not self._converts:
            return keys
        keys.append(self.duration_key)
        if self.keep_waveform_in_task:
            keys.append(self.waveform_key)
        if self.write_to_disk:
            keys.append(self.output_audio_filepath_key)
            if self.update_audio_filepath:
                keys.append(self.audio_filepath_key)
        return keys

    def describe(self) -> StageContract:
        cardinality = "filter" if self._selects else "1:1"
        if not self._converts:
            forms = accepts_for_residency(self.input_residency)
            reads_one_of = []
            if "waveform" in forms:
                reads_one_of.append(IOSpec(data_keys=[self.waveform_key, self.sample_rate_key], accepts=["waveform"]))
            if "file" in forms:
                reads_one_of.append(IOSpec(data_keys=[self.audio_filepath_key], accepts=["file"]))
            return StageContract(
                reads_one_of=reads_one_of,
                writes=IOSpec(data_keys=self._written_keys()),
                cardinality=cardinality,
                cardinality_options=["1:1", "filter"],
                gates=Gates(per_row_independent=True),
            )
        produces = (["tensor"] if self.keep_waveform_in_task else []) + (["disk"] if self.write_to_disk else [])
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
            writes=IOSpec(data_keys=self._written_keys(), produces=produces),
            conditional_writes=(
                [
                    ConditionalWrite(
                        writes=IOSpec(data_keys=[self.original_audio_filepath_key]),
                        condition="a prior audio path is present and no original path has already been preserved",
                        requires_keys=[self.audio_filepath_key],
                    )
                ]
                if self.update_audio_filepath
                else []
            ),
            # A disk-only conversion ends the resident audio, so the keys leave the task.
            removes_keys=[] if self.keep_waveform_in_task else [self.waveform_key, self.sample_rate_key],
            cardinality=cardinality,
            cardinality_options=["1:1", "filter"],
            gates=Gates(
                writes_to_disk=self.write_to_disk,
                output_path_params=["output_dir"],
                per_row_independent=True,
            ),
        )

    def _accepts(self, value: int, allowed: list[int] | None, low: int | None, high: int | None) -> bool:
        return (
            (allowed is None or value in allowed) and (low is None or value >= low) and (high is None or value <= high)
        )

    def _observe(self, task: AudioTask) -> tuple[int, int] | None:
        """(sample_rate, channels) from resident audio, else the file header; None drops the row."""
        resident = task.data.get(self.waveform_key)
        if self.input_residency != "file" and resident is not None:
            try:
                rate = resident_sample_rate(
                    task.data.get(self.sample_rate_key), sample_rate_key=self.sample_rate_key, stage_name=self.name
                )
                return rate, int(ensure_waveform_2d(resident).shape[0])
            except (RuntimeError, TypeError, ValueError, IndexError) as e:
                logger.error(f"[{self.name}] unusable resident audio: {e}")
                return None
        path = task.data.get(self.audio_filepath_key)
        if self.input_residency == "waveform" or not path:
            logger.error(f"[{self.name}] no resident waveform and no path under {self.audio_filepath_key!r}")
            return None
        try:
            # Header only: rate and channel count are metadata, decoding samples to read them is waste.
            with fsspec.open(os.path.expanduser(str(path)), "rb") as f:
                info = sf.info(f)
        except (OSError, RuntimeError) as e:
            logger.error(f"[{self.name}] could not read the header of {path!r}: {e}")
            return None
        return int(info.samplerate), int(info.channels)

    def _convert_channels(self, waveform: torch.Tensor, source: str) -> torch.Tensor | None:
        channels, target = waveform.shape[0], self.target_channels
        if target is None or channels == target:
            return waveform
        if target == 1:
            return torch.mean(waveform, dim=0, keepdim=True)
        if channels == 1:
            return waveform.repeat(target, 1)
        logger.warning(
            f"[{self.name}] cannot downmix {channels} channels to {target} without a channel layout: {source}"
        )
        return None

    def process(self, task: AudioTask) -> AudioTask | list[AudioTask]:
        observed = self._observe(task)
        if observed is None:
            return []
        sample_rate, channels = observed
        task.data[self.sample_rate_key] = sample_rate
        task.data[self.num_channels_key] = channels
        source = task.data.get(self.audio_filepath_key, self.waveform_key)
        if not self._accepts(sample_rate, self.allowed_sample_rates, self.min_sample_rate, self.max_sample_rate):
            logger.warning(f"[{self.name}] sample rate {sample_rate}Hz is not allowed: {source}")
            return []
        if not self._accepts(channels, self.allowed_channels, self.min_channels, self.max_channels):
            logger.warning(f"[{self.name}] channel count {channels} is not allowed: {source}")
            return []
        if not self._converts:
            return task
        return self._convert_row(task, str(source))

    def _convert_row(self, task: AudioTask, source: str) -> AudioTask | list[AudioTask]:
        try:
            resolved = resolve_audio(
                task.data,
                residency=self.input_residency,
                audio_filepath_key=self.audio_filepath_key,
                waveform_key=self.waveform_key,
                sample_rate_key=self.sample_rate_key,
                mono=False,
                preserve_pcm_dtype=True,
                loader=load_audio_file,
            )
            if resolved is None:
                logger.error(f"[{self.name}] audio input not found for {source}")
                return []
            waveform, sample_rate = resolved
            waveform = self._convert_channels(
                normalize_audio_waveform(waveform, stage_name=self.name, mono=False), source
            )
            if waveform is None:
                return []
            if self.target_sample_rate is not None:
                waveform = resample_waveform(waveform, sample_rate, self.target_sample_rate)
                sample_rate = self.target_sample_rate

            if self.write_to_disk:
                stem = os.path.splitext(os.path.basename(source))[0]
                path = write_audio_stable(waveform, sample_rate, output_dir=self.output_dir, stem=stem, tag="fmt")
                task.data[self.output_audio_filepath_key] = path
                if self.update_audio_filepath:
                    produce_audio_filepath(
                        task.data, path, key=self.audio_filepath_key, original_key=self.original_audio_filepath_key
                    )
            if self.keep_waveform_in_task:
                task.data[self.waveform_key] = waveform
            else:
                # After the file exists, so a failed write leaves the row's audio as it arrived.
                drop_resident_audio(task.data, waveform_key=self.waveform_key, sample_rate_key=self.sample_rate_key)
            if self.keep_waveform_in_task or self.sample_rate_key in task.data:
                task.data[self.sample_rate_key] = sample_rate
            task.data[self.num_channels_key] = waveform.shape[0]
            task.data[self.duration_key] = waveform.shape[1] / sample_rate
        except (OSError, RuntimeError, TypeError, ValueError) as e:
            logger.error(f"[{self.name}] could not convert {source}: {e}")
            return []
        return task
