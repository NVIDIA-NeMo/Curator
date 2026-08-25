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
Audio channel-count stage: record it, select on it, or change it.

``action`` picks which of the three, in the vocabulary ``UTMOSFilterStage``, ``BandFilterStage``
and ``SIGMOSFilterStage`` already use for measure-and-keep versus measure-and-drop, extended
with the one thing a channel count can do that a quality score cannot: be changed.

Sample rate, format and file layout are never touched, so a pipeline sets its rate policy
separately -- ``SampleRateFilterStage`` to select rates, ``ResampleAudioStage`` to convert
them -- or sets none at all.

Example:
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.audio.preprocessing import ChannelCountStage

    pipeline = Pipeline(name="audio_pipeline")
    pipeline.add_stage(ChannelCountStage())                                    # record the count
    pipeline.add_stage(ChannelCountStage(action="filter", allowed_channels=[1]))  # keep mono only
    pipeline.add_stage(ChannelCountStage(action="convert", target_channels=1))    # make it mono
"""

import os
import tempfile
from dataclasses import dataclass, field, fields
from typing import ClassVar, Literal

import soundfile as sf
import torch
from loguru import logger

from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, IOSpec, StageContract
from nemo_curator.stages.audio._agent._residency import (
    InputResidency,
    accepts_for_residency,
    produce_audio_filepath,
    residency_read_specs,
    resolve_audio,
    write_audio_stable,
)
from nemo_curator.stages.audio.common import ensure_waveform_2d, load_audio_file
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

ChannelAction = Literal["annotate", "filter", "convert"]


@dataclass
class ChannelCountStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """
    Record, select on, or change the number of audio channels.

    ============  =================================================================
    action        behaviour
    ============  =================================================================
    ``annotate``  records ``num_channels`` on every row and keeps them all (default)
    ``filter``    records it, then drops the rows whose count is not allowed
    ``convert``   brings the audio to ``target_channels``, in memory
    ============  =================================================================

    Selecting and converting are deliberately separate actions rather than one parameter that
    means both. They do opposite things to a corpus -- ``convert`` changes every row and keeps
    all of them, ``filter`` changes nothing and keeps a subset -- so a single knob spelling both
    makes "mono" ambiguous between "make it mono" and "keep only what already is". Parameters
    belonging to an action you did not choose are refused at construction rather than ignored.

    ``annotate`` and ``filter`` never decode. A channel count sits in the file header, which
    ``soundfile.info`` reads without touching samples, so putting either in front of a decoding
    stage costs almost nothing and spares the rejected rows entirely. ``convert`` must decode,
    because it has to rewrite the samples.

    **What ``num_channels`` means depends on the action.** Under ``annotate``/``filter`` it is
    the count OBSERVED in the source audio. Under ``convert`` it is the count RESULTING from the
    conversion. ``sample_rate`` carries exactly this trap between ``SampleRateFilterStage`` (a
    measurement) and ``ResampleAudioStage`` (a target); reading a pipeline's final
    ``num_channels`` without knowing which stage last wrote it inverts its meaning.

    Under ``convert``, what happens depends on how many channels the input actually has:

    ============  ==================  ==================================================
    input         target              behaviour
    ============  ==================  ==================================================
    ``N``         ``N``               passed through unchanged
    ``N > 1``     ``1``               averaged into one channel (standard mono downmix)
    ``1``         ``T > 1``           duplicated into ``T`` identical channels
    ``N > T > 1`` ``T``               REFUSED -- the row is dropped
    ============  ==================  ==================================================

    That refusal is deliberate. A correct downmix to more than one channel needs ITU-R BS.775
    coefficients *and* the file's channel order, and a bare ``(channels, samples)`` tensor
    carries neither -- WAV channel order comes from the file's channel mask, which is gone by
    the time the audio is a tensor. Averaging 5.1 into two channels does not produce stereo, it
    produces a phase-smeared mix that sounds plausible and is wrong. ``ResampleAudioStage``
    drives ffmpeg, which does know layouts, so that is the honest tool for those conversions.
    Downmix to ``1`` is not the same problem: averaging every channel together IS what mono
    means, so it needs no layout knowledge.

    Nothing here ever resamples. When a rate must actually change, ``ResampleAudioStage``
    converts it via ffmpeg.

    Args:
        action: "annotate" records num_channels and keeps every row (default); "filter" also
            drops rows whose count is not allowed; "convert" brings the audio to
            target_channels. Parameters of the other actions are refused, not ignored.
        allowed_channels: Counts to keep, e.g. [1] for mono only (action="filter"). None = no
            constraint from this parameter.
        min_channels: Lowest acceptable count, inclusive (action="filter"). None = unbounded.
        max_channels: Highest acceptable count, inclusive (action="filter"). None = unbounded.
        target_channels: Channel count to produce (action="convert"). None means 1 (mono).
        audio_filepath_key: Key in data dict for the audio file path.
        waveform_key: Key in data dict for the in-memory waveform tensor.
        sample_rate_key: Key in data dict for the waveform sample rate.
        num_channels_key: Key where the channel count is written -- observed under
            annotate/filter, resulting under convert.
        duration_key: Key where the audio duration in seconds is written (convert only).
        output_audio_filepath_key: Key where the written WAV path is stored
            (action="convert", write_to_disk=True only).
        original_audio_filepath_key: Key preserving the pre-conversion path when
            update_audio_filepath=True.
        input_residency: Which input to use -- "waveform" (in-memory only), "file"
            (audio_filepath only), or "auto" (waveform first, file fallback; default).
        keep_waveform_in_task: If True (default), store the converted waveform and sample
            rate in task.data for downstream in-memory consumers (convert only).
        write_to_disk: If True, write the converted audio to a WAV file (action="convert").
            Without output_dir this writes to the system temp dir and nothing cleans it up; in
            multi-node runs point output_dir at shared storage.
        update_audio_filepath: If True, repoint audio_filepath_key at the written file and
            preserve the original under original_audio_filepath_key.
        output_dir: Directory for written audio (action="convert", write_to_disk=True only).
    """

    action: ChannelAction = "annotate"

    allowed_channels: list[int] | None = None
    min_channels: int | None = None
    max_channels: int | None = None

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

    # Own bookkeeping: the channel count, recorded for readers and reports. Nothing routes on
    # it, so it needs no shared role -- and declaring it here means adding this stage did not
    # require touching the central role table.
    INTERNAL_KEY_FIELDS: ClassVar[frozenset[str]] = frozenset({"num_channels_key"})

    # Which action each parameter belongs to. A parameter left at its default is "not asked
    # for", which is why every one of these defaults to None or False: it lets the constructor
    # tell "I want mono" from "I never mentioned channels" and refuse the former under the
    # wrong action instead of silently dropping the request.
    _ACTION_PARAMS: ClassVar[dict[str, tuple[str, ...]]] = {
        "filter": ("allowed_channels", "min_channels", "max_channels"),
        "convert": ("target_channels", "write_to_disk", "update_audio_filepath", "output_dir"),
    }

    name: str = "ChannelCount"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self):
        super().__init__()
        if self.action not in ("annotate", "filter", "convert"):
            msg = f"action must be one of ('annotate', 'filter', 'convert'), got {self.action!r}"
            raise ValueError(msg)
        self._reject_other_actions_params()
        if self.action == "filter":
            self._validate_filter()
        if self.action == "convert":
            self._validate_convert()

    def _reject_other_actions_params(self) -> None:
        """Refuse parameters belonging to an action other than the configured one.

        Ignoring them is what makes the two intents blur: ``action="convert",
        allowed_channels=[1]`` reads as "make everything mono AND only keep mono", and whichever
        half is silently dropped, the corpus that comes out is not the one that was asked for.
        """
        defaults = {item.name: item.default for item in fields(self)}
        foreign = [
            (name, owner)
            for owner, names in self._ACTION_PARAMS.items()
            if owner != self.action
            for name in names
            if getattr(self, name) != defaults[name]
        ]
        if not foreign:
            return
        listed = ", ".join(f"{name} (action={owner!r})" for name, owner in sorted(foreign))
        msg = (
            f"action={self.action!r} does not use {listed}. Selecting a channel count and "
            f"converting to one are separate intents: convert changes every row and keeps all "
            f"of them, filter changes nothing and keeps a subset. Set the action those "
            f"parameters belong to, or drop them."
        )
        raise ValueError(msg)

    def _validate_filter(self) -> None:
        if self.allowed_channels is not None and not self.allowed_channels:
            msg = "allowed_channels must name at least one count, or be None for no constraint"
            raise ValueError(msg)
        for name in ("allowed_channels", "min_channels", "max_channels"):
            value = getattr(self, name)
            counts = value if isinstance(value, list) else [value]
            for count in counts:
                if count is None:
                    continue
                if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                    msg = f"{name} must be whole channel counts of at least 1, got {value!r}"
                    raise ValueError(msg)
        low, high = self.min_channels, self.max_channels
        if low is not None and high is not None and low > high:
            msg = f"min_channels ({low}) is above max_channels ({high}), so nothing can pass"
            raise ValueError(msg)
        if self.allowed_channels is None and low is None and high is None:
            msg = (
                "action='filter' needs allowed_channels, min_channels or max_channels -- with no "
                "constraint it would declare a filter that drops nothing. Use action='annotate' "
                "to only record the count."
            )
            raise ValueError(msg)

    def _validate_convert(self) -> None:
        # Type as well as range. YAML reads ``target_channels: 2.0`` as a float, which used to
        # construct fine and then die inside a worker at ``waveform.repeat(2.0, 1)`` with a
        # TypeError -- not one of the (OSError, RuntimeError) this stage drops rows for, so it
        # propagated and took the run down mid-corpus instead of being caught at the recipe.
        target = self.target_channels
        if target is None:
            return
        if isinstance(target, bool) or not isinstance(target, int):
            msg = f"target_channels must be a whole number of channels, got {target!r} ({type(target).__name__})"
            raise ValueError(msg)  # noqa: TRY004
        if target < 1:
            msg = f"target_channels must be at least 1, got {target}"
            raise ValueError(msg)

    @property
    def _target(self) -> int:
        """The channel count ``convert`` produces; unset means mono."""
        return 1 if self.target_channels is None else self.target_channels

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], self._written_keys()

    def _written_keys(self) -> list[str]:
        if self.action != "convert":
            return [self.num_channels_key]
        keys = [
            self.num_channels_key,
            self.duration_key,
        ]
        if self.keep_waveform_in_task:
            keys.extend([self.waveform_key, self.sample_rate_key])
        if self.write_to_disk:
            keys.append(self.output_audio_filepath_key)
            if self.update_audio_filepath:
                keys.append(self.audio_filepath_key)
        return keys

    def describe(self) -> StageContract:
        if self.action == "convert":
            return self._convert_contract()
        return self._observe_contract()

    def _observe_contract(self) -> StageContract:
        """The contract for ``annotate``/``filter``: a count in, a count out, no samples read."""
        forms = accepts_for_residency(self.input_residency)
        reads_one_of: list[IOSpec] = []
        if "waveform" in forms:
            # The count is ``waveform.shape[0]``. Unlike a conversion this needs no sample rate,
            # and asking for one would make a resident-waveform row look unsatisfiable for a
            # stage that can in fact answer from the waveform alone.
            reads_one_of.append(IOSpec(data_keys=[self.waveform_key], accepts=["waveform"]))
        if "file" in forms:
            reads_one_of.append(IOSpec(data_keys=[self.audio_filepath_key], accepts=["file"]))
        return StageContract(
            reads_one_of=reads_one_of,
            writes=IOSpec(data_keys=[self.num_channels_key]),
            cardinality="filter" if self.action == "filter" else "1:1",
            # The two row cardinalities this stage can take across its actions. ``convert``
            # names no third one: it is "1:1" for mono and "filter" for the targets it refuses.
            cardinality_options=["filter", "annotate"],
            # Each row is judged against the configured counts, not against the corpus.
            gates=Gates(per_row_independent=True),
        )

    def _convert_contract(self) -> StageContract:
        produces = []
        if self.keep_waveform_in_task:
            produces.append("tensor")
        if self.write_to_disk:
            produces.append("disk")
        return StageContract(
            reads_one_of=residency_read_specs(
                self.input_residency,
                audio_filepath_key=self.audio_filepath_key,
                waveform_key=self.waveform_key,
                sample_rate_key=self.sample_rate_key,
            ),
            writes=IOSpec(data_keys=self._written_keys(), produces=produces),
            # Downmixing to mono always succeeds, but any other target refuses the conversions
            # it cannot do correctly (N > target > 1) and drops those rows. That makes the stage
            # a filter for those configurations, and saying so is what puts a seam in the
            # semantic review packet for a reviewer to ask about.
            cardinality="filter" if self._target > 1 else "1:1",
            cardinality_options=["filter", "annotate"],
            # Declared here, by the stage that owns the parameter, so a caller running
            # this in a sandbox knows what to redirect without a central table entry.
            gates=Gates(
                writes_to_disk=self.write_to_disk,
                output_path_params=["output_dir"],
                # Every row is converted on its own terms, so a delta run gives the changed
                # files the same answer a full run would have given them.
                per_row_independent=True,
            ),
        )

    def accepts(self, num_channels: int) -> bool:
        """Whether ``num_channels`` satisfies every constraint that was configured."""
        if self.allowed_channels is not None and num_channels not in self.allowed_channels:
            return False
        if self.min_channels is not None and num_channels < self.min_channels:
            return False
        return not (self.max_channels is not None and num_channels > self.max_channels)

    def _requirement(self) -> str:
        """The configured constraint, phrased for a log line."""
        parts = []
        if self.allowed_channels is not None:
            parts.append(f"one of {sorted(self.allowed_channels)}")
        if self.min_channels is not None:
            parts.append(f">= {self.min_channels}")
        if self.max_channels is not None:
            parts.append(f"<= {self.max_channels}")
        return " and ".join(parts) or "any channel count"

    def _observed_channels(self, task: AudioTask) -> int | None:  # noqa: PLR0911 (complexity accepted: one early return per input/error condition)
        """The row's channel count, from resident audio if present, else from the file header.

        Resident audio is asked first because it is the audio this pipeline is carrying: after a
        conversion the file on disk still has its original channels while the waveform in the
        task has the converted ones, so the header would answer about audio nobody is using any
        more. Without a waveform the header is read rather than an existing ``num_channels``
        column believed -- standing alone that column is manifest metadata about a file nobody
        re-opened, and trusting it lets a stale value decide the filter.
        """
        resident = task.data.get(self.waveform_key)
        if self.input_residency != "file" and resident is not None:
            try:
                return int(ensure_waveform_2d(resident).shape[0])
            except (RuntimeError, TypeError, ValueError, IndexError) as e:
                logger.error(f"Could not read the channel count of the resident waveform: {e}")
                return None
        if self.input_residency == "waveform":
            logger.error(
                f"No resident waveform under {self.waveform_key!r} and input_residency='waveform', "
                "so there is nothing to count without touching disk"
            )
            return None

        declared = task.data.get(self.num_channels_key)
        declared = int(declared) if isinstance(declared, (int, float)) and int(declared) > 0 else None
        path = task.data.get(self.audio_filepath_key)
        if not path:
            if declared is None:
                logger.error(f"No channel count and no audio path under {self.audio_filepath_key!r}")
                return None
            # Nothing to verify against, so the declared count is all there is. Say so rather
            # than dropping a row that may well be fine.
            logger.warning(
                f"Filtering on an unverified channel count ({declared}): no resident waveform "
                f"and no path under {self.audio_filepath_key!r}"
            )
            return declared
        try:
            # Header only: the count is metadata, so decoding samples to reach it would cost
            # orders of magnitude more per file for something in the first few bytes.
            return int(sf.info(os.path.expanduser(str(path))).channels)
        except (OSError, RuntimeError) as e:
            logger.error(f"Could not read the channel count of {path!r}: {e}")
            return None

    def _convert(self, waveform: torch.Tensor, source: str) -> torch.Tensor | None:
        """Bring ``waveform`` to ``target_channels``, or None when that cannot be done right."""
        num_channels = waveform.shape[0]
        target = self._target
        if num_channels == target:
            return waveform
        if target == 1:
            logger.debug(f"Averaging {num_channels} channels to mono")
            return torch.mean(waveform, dim=0, keepdim=True)
        if num_channels == 1:
            # Duplicate the single channel. This adds no information -- the result is the
            # same signal N times -- but it is what `ffmpeg -ac` does and what a consumer
            # expecting a fixed channel count needs.
            logger.debug(f"Duplicating mono into {target} channels")
            return waveform.repeat(target, 1)
        logger.warning(
            f"Cannot downmix {num_channels} channels to {target} without the "
            f"file's channel layout, which a waveform tensor does not carry: {source}. "
            "Use ResampleAudioStage (ffmpeg) for a layout-aware downmix, or "
            "target_channels=1."
        )
        return None

    def _write_audio(self, waveform: torch.Tensor, sample_rate: int, task: AudioTask) -> str:
        stem = os.path.splitext(os.path.basename(str(task.data.get(self.audio_filepath_key, "audio"))))[0]
        return write_audio_stable(
            waveform,
            sample_rate,
            output_dir=self.output_dir or tempfile.gettempdir(),
            stem=stem,
            tag=f"ch{self._target}",
        )

    def process(self, task: AudioTask) -> AudioTask | list[AudioTask]:
        """Record, select on, or change the channel count, per ``action``."""
        if self.action == "convert":
            return self._convert_row(task)
        return self._observe_row(task)

    def _observe_row(self, task: AudioTask) -> AudioTask | list[AudioTask]:
        """Record the channel count, and under ``filter`` keep the row only if it is allowed."""
        num_channels = self._observed_channels(task)
        if num_channels is None:
            return []
        task.data[self.num_channels_key] = num_channels

        if self.action == "filter" and not self.accepts(num_channels):
            logger.warning(
                f"Channel count {num_channels} does not satisfy {self._requirement()}: "
                f"{task.data.get(self.audio_filepath_key, self.waveform_key)}"
            )
            return []
        return task

    def _convert_row(self, task: AudioTask) -> AudioTask | list[AudioTask]:
        """Convert the audio's channel count. Returns [] for a row that cannot be converted."""
        try:
            resolved = resolve_audio(
                task.data,
                residency=self.input_residency,  # type: ignore[arg-type]
                audio_filepath_key=self.audio_filepath_key,
                waveform_key=self.waveform_key,
                sample_rate_key=self.sample_rate_key,
                mono=False,
                loader=load_audio_file,  # module-level symbol: patchable at this module
            )
        except (OSError, RuntimeError) as e:  # corrupt/unreadable audio -> skip the row
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

            source = str(task.data.get(self.audio_filepath_key, self.waveform_key))
            converted = self._convert(waveform, source)
            if converted is None:
                return []

            if self.keep_waveform_in_task:
                task.data[self.waveform_key] = converted
                task.data[self.sample_rate_key] = sample_rate
            task.data[self.num_channels_key] = converted.shape[0]
            task.data[self.duration_key] = converted.shape[1] / sample_rate

            if self.write_to_disk:
                path = self._write_audio(converted, sample_rate, task)
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
