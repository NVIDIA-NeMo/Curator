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

"""Stage-adapter contract for forced alignment (SDP-V2 design doc §13).

Mirrors the ASR / diarization / VAD contract pattern:

* :class:`~nemo_curator.stages.audio.inference.alignment.ForcedAlignmentStage`
  owns Curator-side glue (task.data reads, split-filepath fan-out + scatter,
  segment cut, time-offset adjustment, metric logging).
* :class:`ForcedAlignmentAdapter` owns the model-side library call
  (weight prefetch, model setup, decoder configuration, the actual
  ``transcribe(...)`` invocation, hypothesis-to-word-alignment
  conversion) and packs results into the canonical
  :class:`AlignmentResult` shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class WordAlignment:
    """Canonical per-word alignment dataclass.

    Attributes:
        word: The aligned word (or character, when the adapter uses
            char-level timestamps).
        start: Word start time in seconds (clip / segment coordinates,
            see :class:`AlignmentResult`).
        end: Word end time in seconds.
        confidence: Optional adapter-supplied per-word confidence
            score in ``[0, 1]``. ``None`` when the adapter doesn't
            surface one.
    """

    word: str
    start: float
    end: float
    confidence: float | None = None


@dataclass
class AlignmentResult:
    """Canonical per-input alignment adapter output.

    Attributes:
        alignments: One :class:`WordAlignment` per emitted word
            (or char). The stage applies any necessary time-offset
            shift before writing this onto ``task.data``. Empty list
            when the adapter could not process the input.
        text: Concatenated transcription text. The stage writes this
            onto ``task.data[text_key]``.
        model_id: The actual model identifier the adapter ran (mirrors
            the stage's ``model_id`` field; populated by the adapter).
        extras: Adapter-specific scalar / structured diagnostics that
            do not fit the canonical shape. Stage never reads inside
            this dict.
    """

    alignments: list[WordAlignment]
    text: str = ""
    model_id: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ForcedAlignmentAdapter(Protocol):
    """Structural protocol every forced-alignment adapter must implement.

    Constructor contract: adapters are constructed by the stage as
    ``cls(model_id=..., revision=..., **adapter_kwargs)``. Tier-2 knobs
    are adapter-specific (decoder type, FastConformer toggle, batch
    sizes, ...).

    Per-batch contract: :meth:`align_batch` receives a list of dicts
    (Tier-3 per-task knobs unpacked from ``task.data`` by the stage)
    and returns one :class:`AlignmentResult` per input, in the same
    order.

    Expected per-item keys (the stage populates these; the adapter
    reads whichever is present):

    * ``audio_path`` (``str | None``): Path to a decodable audio file.
      Used for full-audio / split-filepath inference.
    * ``audio_segment`` (``numpy.ndarray | None``): In-memory mono
      audio array, one segment cut. Used for segment-only inference.
    * ``sample_rate`` (``int | None``): Sample rate of
      ``audio_segment`` (only meaningful in segment-mode).
    * ``task_id`` (``str | None``): Carried through for diagnostics.

    A batch must be homogeneous - either all items have ``audio_path``
    OR all have ``audio_segment``; the stage guarantees this.

    Attributes:
        model_id: Identifier of the underlying model checkpoint.
        last_metrics: Scalar metrics from the last :meth:`align_batch`
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

    def align_batch(self, items: list[dict[str, Any]]) -> list[AlignmentResult]:
        """Run forced alignment on a batch of per-task dicts.

        Args:
            items: One dict per task with the keys documented on the
                class docstring. Length matches the batch size.

        Returns:
            One :class:`AlignmentResult` per input, in the same order.
            Items the adapter could not process must still appear with
            empty ``alignments`` and ``text=""`` so the stage can
            scatter results 1:1.
        """
        ...
