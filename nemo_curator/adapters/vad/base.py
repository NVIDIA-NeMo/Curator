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

"""Stage-adapter contract for voice activity detection (SDP-V2 design doc §4).

Mirrors the diarization / ASR contract pattern:

* :class:`~nemo_curator.stages.audio.inference.vad.VADStage` owns the
  Curator-side glue (``task.data`` reads, duration-skip rule, metric
  logging).
* :class:`VADAdapter` owns the model-side library call (weight prefetch,
  model setup, the actual VAD invocation - WhisperX / Silero / PyAnnote
  VAD - and packing results into the canonical :class:`VADResult` shape).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class VADInterval:
    """Canonical per-VAD-region dataclass.

    Attributes:
        start: Interval start time in seconds (clip coordinates).
        end: Interval end time in seconds.
    """

    start: float
    end: float


@dataclass
class VADResult:
    """Canonical per-task VAD adapter output.

    Attributes:
        intervals: One :class:`VADInterval` per emitted speech region.
            Empty list when the adapter could not process the input.
        model_id: The actual model identifier the adapter ran (mirrors
            the stage's ``model_id`` field; populated by the adapter).
        extras: Adapter-specific scalar / structured diagnostics that
            do not fit the canonical shape. Stage never reads inside
            this dict.
    """

    intervals: list[VADInterval]
    model_id: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class VADAdapter(Protocol):
    """Structural protocol every VAD adapter must implement.

    Constructor contract: adapters are constructed by the stage as
    ``cls(model_id=..., revision=..., **adapter_kwargs)``. Tier-2 knobs
    are adapter-specific.

    Per-batch contract: :meth:`detect_batch` receives a list of dicts
    (Tier-3 per-task knobs unpacked from ``task.data`` by the stage)
    and returns one :class:`VADResult` per input, in the same order.

    Expected per-item keys (the stage populates these; the adapter
    reads whichever it needs):

    * ``audio_filepath`` (``str``): Path to a decodable audio file.
    * ``waveform`` (``numpy.ndarray | None``): Optional in-memory
      waveform.
    * ``sample_rate`` (``int | None``): Sample rate of ``waveform``.
    * ``duration`` (``float | None``): Optional clip duration in
      seconds.
    * ``task_id`` (``str | None``): Diagnostic only.

    Attributes:
        model_id: Identifier of the underlying model checkpoint.
        last_metrics: Scalar metrics from the last :meth:`detect_batch`
            call. The stage merges these into ``_log_metrics`` output
            under ``model_<key>`` aliases.
    """

    model_id: str
    last_metrics: dict[str, float]

    @classmethod
    def prefetch_weights(cls, model_id: str, revision: str | None = None) -> None:
        """Download weights to local cache without allocating a GPU."""
        ...

    def setup(self) -> None:
        """Load the model into the worker's process."""
        ...

    def teardown(self) -> None:
        """Release GPU memory and worker-local state."""
        ...

    def detect_batch(self, items: list[dict[str, Any]]) -> list[VADResult]:
        """Run VAD on a batch of per-task dicts.

        Args:
            items: One dict per task with the keys documented on the
                class docstring. Length matches the batch size.

        Returns:
            One :class:`VADResult` per input, in the same order.
        """
        ...
