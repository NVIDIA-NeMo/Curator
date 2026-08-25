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
Sample-rate selection stage.

Records every row's sample rate and keeps only the rows whose rate is acceptable. Channels,
format and file layout are untouched, so a pipeline pairs this with whatever channel policy
it wants -- or with none at all.

It never resamples: ``ResampleAudioStage`` is the stage that converts a rate, this one only
decides which rates are allowed through and writes down what it saw.

Reading rather than decoding is the point. Determining a rate needs only the file header,
so this stage never loads samples -- measured on 30s stereo WAVs, a header read is ~0.03ms
against ~5.3ms for a full decode (~186x). Placing it before any decoding stage means rows
that will be rejected are never decoded at all.

Example:
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.audio.preprocessing import SampleRateFilterStage

    pipeline = Pipeline(name="audio_pipeline")
    pipeline.add_stage(SampleRateFilterStage(allowed_sample_rates=[16000, 22050]))
    pipeline.add_stage(SampleRateFilterStage(min_sample_rate=16000))
"""

import os
from dataclasses import dataclass, field

import soundfile as sf
from loguru import logger

from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, IOSpec, StageContract
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class SampleRateFilterStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """
    Keep rows whose audio sample rate is acceptable, and record the rate on every row.

    Acceptance is expressed two ways, deliberately as separate parameters rather than one
    that means both. A single ``sample_rates=[16000, 48000]`` is genuinely ambiguous --
    "these two rates" or "everything between them"? -- and the two readings filter very
    different corpora.

    * ``allowed_sample_rates``: an explicit set. A rate must be one of these.
    * ``min_sample_rate`` / ``max_sample_rate``: inclusive bounds.

    Set either, both (a rate must then satisfy both), or neither -- with nothing set the
    stage keeps every row and acts purely as an observer that annotates the rate.

    Nothing is resampled here. A corpus with mixed rates that must be uniform needs
    ``ResampleAudioStage``; this stage only decides what is allowed through.

    Args:
        allowed_sample_rates: Explicit rates to keep, e.g. [16000, 22050]. None = no
            constraint from this parameter.
        min_sample_rate: Lowest acceptable rate, inclusive. None = unbounded below.
        max_sample_rate: Highest acceptable rate, inclusive. None = unbounded above.
        audio_filepath_key: Key in data dict for the audio file path.
        sample_rate_key: Key where the observed sample rate is written. Reused without a
            disk read only when a resident waveform backs it (see ``waveform_key``).
        waveform_key: Key a resident waveform would occupy. Consulted to know whether the
            sample rate in task.data belongs to resident audio; a rate standing alone is
            manifest metadata and is re-read from the file header instead of trusted.
    """

    allowed_sample_rates: list[int] | None = None
    min_sample_rate: int | None = None
    max_sample_rate: int | None = None

    audio_filepath_key: str = "audio_filepath"
    sample_rate_key: str = "sample_rate"
    waveform_key: str = "waveform"

    name: str = "SampleRateFilter"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self):
        super().__init__()
        if self.allowed_sample_rates is not None and not self.allowed_sample_rates:
            msg = "allowed_sample_rates must name at least one rate, or be None for no constraint"
            raise ValueError(msg)
        low, high = self.min_sample_rate, self.max_sample_rate
        if low is not None and high is not None and low > high:
            msg = f"min_sample_rate ({low}) is above max_sample_rate ({high}), so nothing can pass"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.sample_rate_key]

    def describe(self) -> StageContract:
        # Either a resident rate or a readable path satisfies this stage; it needs no samples,
        # so the waveform form asks for the rate alone rather than the waveform with it.
        return StageContract(
            reads_one_of=[
                IOSpec(data_keys=[self.sample_rate_key], accepts=["waveform"]),
                IOSpec(data_keys=[self.audio_filepath_key], accepts=["file"]),
            ],
            writes=IOSpec(data_keys=[self.sample_rate_key]),
            # Dropping rows is this stage's whole purpose, and "filter" is what puts a seam in
            # the semantic review packet. Left at the 1:1 default the reviewer is never told the
            # corpus can shrink here, so nobody asks how much of it survives.
            cardinality="filter",
            # Each row is judged against the configured rates, not against the corpus.
            gates=Gates(per_row_independent=True),
        )

    def accepts(self, sample_rate: int) -> bool:
        """Whether ``sample_rate`` satisfies every constraint that was configured."""
        if self.allowed_sample_rates is not None and sample_rate not in self.allowed_sample_rates:
            return False
        if self.min_sample_rate is not None and sample_rate < self.min_sample_rate:
            return False
        return not (self.max_sample_rate is not None and sample_rate > self.max_sample_rate)

    def _requirement(self) -> str:
        """The configured constraint, phrased for a log line."""
        parts = []
        if self.allowed_sample_rates is not None:
            parts.append(f"one of {sorted(self.allowed_sample_rates)}")
        if self.min_sample_rate is not None:
            parts.append(f">= {self.min_sample_rate}")
        if self.max_sample_rate is not None:
            parts.append(f"<= {self.max_sample_rate}")
        return " and ".join(parts) or "any rate"

    def _observed_rate(self, task: AudioTask) -> int | None:
        """The row's sample rate, from resident audio if present, else from the file header.

        An existing ``sample_rate_key`` is only believed when a resident waveform is there to
        back it, because then it describes audio this pipeline is carrying. Standing alone it
        is manifest metadata about a file nobody re-read, and trusting it lets a stale or wrong
        column decide the filter: a genuinely 48 kHz file labelled 16000 would be kept for a
        16 kHz-only corpus AND re-stamped with the wrong rate. The header read is cheap enough
        that guessing is never worth it.
        """
        declared = task.data.get(self.sample_rate_key)
        declared = int(declared) if isinstance(declared, (int, float)) and int(declared) > 0 else None
        if declared is not None and self.waveform_key in task.data:
            return declared

        path = task.data.get(self.audio_filepath_key)
        if not path:
            if declared is None:
                logger.error(f"No sample rate and no audio path under {self.audio_filepath_key!r}")
                return None
            # Nothing to verify against, so the declared rate is all there is. Say so rather
            # than dropping a row that may well be fine.
            logger.warning(
                f"Filtering on an unverified sample rate ({declared}Hz): no resident waveform "
                f"and no path under {self.audio_filepath_key!r}"
            )
            return declared
        try:
            # Header only: the rate is metadata, so decoding samples to read it would cost
            # ~186x more per file for information already sitting in the first few bytes.
            # ``expanduser`` matches ChannelCountStage: a manifest written with ``~/audio/x.wav``
            # is otherwise unreadable here, and the failure path drops the row rather than
            # raising, so the whole corpus would disappear through this stage in silence.
            return int(sf.info(os.path.expanduser(str(path))).samplerate)
        except (OSError, RuntimeError) as e:
            logger.error(f"Could not read the sample rate of {path!r}: {e}")
            return None

    def process(self, task: AudioTask) -> AudioTask | list[AudioTask]:
        """Record the sample rate and keep the row only if that rate is acceptable."""
        sample_rate = self._observed_rate(task)
        if sample_rate is None:
            return []
        if sample_rate <= 0:
            logger.error(f"Invalid sample rate ({sample_rate}) for {task.data.get(self.audio_filepath_key)!r}")
            return []

        task.data[self.sample_rate_key] = sample_rate

        if not self.accepts(sample_rate):
            logger.warning(
                f"Sample rate {sample_rate}Hz does not satisfy {self._requirement()}: "
                f"{task.data.get(self.audio_filepath_key, self.waveform_key)}"
            )
            return []
        return task
