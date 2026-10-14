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

"""Stage-adapter contract for speaker-diarization inference.

The Curator stage owns task I/O, input-mode selection, in-memory waveform
normalization, batching, resume behavior, and RTTM output. A diarization
adapter owns provider-specific weight resolution, worker-local model state,
inference, and direct-path decoding or bounded streaming when it accepts a
file-backed item. Changing ``adapter_target`` therefore does not change the
task schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypedDict, runtime_checkable


class DiarizationSegment(TypedDict):
    """One JSON-serializable ``who-spoke-when`` interval in seconds."""

    start: float
    end: float
    speaker: str


class DiarizationInputError(RuntimeError):
    """One or more file-backed inputs could not be read or decoded."""


@dataclass(frozen=True)
class DiarizationResult:
    """Canonical adapter output for one recording.

    ``segments`` is deliberately manifest-ready: it contains only dictionaries
    of built-in scalar values, not provider segment objects or tensors.
    """

    segments: list[DiarizationSegment]


@runtime_checkable
class DiarizationAdapter(Protocol):
    """Structural protocol implemented by speaker-diarization adapters.

    Constructor contract: a stage creates an adapter with ``model_id``,
    ``sample_rate``, and explicitly configured adapter keyword arguments.

    ``diarize_batch`` receives canonical items in input order. Each item has
    either a contiguous mono, one-dimensional float32 ``waveform`` plus its
    integer ``sample_rate``, or an ``audio_filepath`` for adapters that can
    preserve provider-native or bounded file streaming. The adapter must
    return exactly one :class:`DiarizationResult` per input item in the same
    order, including an empty result for an empty waveform. File-backed
    adapters raise :class:`DiarizationInputError` only for task-local input or
    decode failures; resource and provider failures propagate unchanged.
    """

    model_id: str
    sample_rate: int

    def download_weights_on_node(self) -> None:
        """Cache provider weights without allocating worker-local model state."""
        ...

    def load_model(self, *, num_gpus: int) -> None:
        """Load worker-local model state for the requested physical GPU count."""
        ...

    def unload_model(self) -> None:
        """Release worker-local model and accelerator state."""
        ...

    def diarize_batch(self, items: list[dict[str, Any]]) -> list[DiarizationResult]:
        """Return one canonical diarization result per prepared item, in order."""
        ...
