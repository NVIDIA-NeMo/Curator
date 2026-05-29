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

"""Stage-adapter contract for speaker diarization (SDP-V2 design doc §3).

Mirrors the ASR contract pattern (``nemo_curator.adapters.asr.base``):

* :class:`~nemo_curator.stages.audio.inference.speaker_diarization.DiarizationStage`
  owns Curator-side glue - ``task.data`` reads, batching, ``min_length``
  / ``max_length`` filtering, non-speaker gap fill, metric logging.
* :class:`DiarizationAdapter` owns the model-side library call - weight
  prefetch, model setup, the actual diarizer invocation
  (``pyannote.audio.Pipeline(...)``, ``SortformerEncLabelModel.diarize(...)``,
  ...) and packing results into the canonical :class:`DiarizationResult`
  shape.

This split lets the stage swap diarizers with a single ``adapter_target:``
line in YAML without rewriting the stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class DiarSegment:
    """Canonical per-speaker-turn dataclass.

    Attributes:
        start: Turn start time in seconds (clip coordinates).
        end: Turn end time in seconds.
        speaker: Speaker identifier as emitted by the adapter
            (e.g. ``"speaker_0"`` or ``"<audio_item_id>_speaker_0"``).
            The stage does NOT remap speaker ids - the adapter is
            responsible for any namespacing.
        confidence: Optional adapter-supplied per-turn confidence.
            ``None`` when the adapter doesn't surface one (matches
            PyAnnote / Sortformer defaults).
    """

    start: float
    end: float
    speaker: str
    confidence: float | None = None


@dataclass
class DiarizationResult:
    """Canonical per-task diarization adapter output.

    Identical across every diarization adapter so the stage's schema
    mutation code path stays constant when the adapter is swapped.

    Attributes:
        diar_segments: One :class:`DiarSegment` per speaker turn the
            adapter emitted. The stage writes these onto
            ``task.data[segments_key]``.
        overlap_segments: Optional list of cross-speaker overlap turns.
            Populated by adapters that surface overlap detection (e.g.
            PyAnnote); empty list otherwise. The stage writes these
            onto ``task.data[overlap_segments_key]`` when the key is
            configured.
        model_id: The actual model identifier the adapter ran (mirrors
            the stage's ``model_id`` field; populated by the adapter so
            downstream consumers see the live value).
        extras: Adapter-specific scalar / structured diagnostics that
            do not fit the canonical shape (e.g. raw RTTM path, per-turn
            embedding tensors). Stage never reads inside this dict.
    """

    diar_segments: list[DiarSegment]
    overlap_segments: list[DiarSegment] = field(default_factory=list)
    model_id: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DiarizationAdapter(Protocol):
    """Structural protocol every diarization adapter must implement.

    Constructor contract: adapters are constructed by the stage as
    ``cls(model_id=..., revision=..., **adapter_kwargs)`` - so every
    adapter must accept ``model_id`` and ``revision`` keyword arguments,
    plus whatever Tier-2 knobs that adapter exposes.

    Per-batch contract: :meth:`diarize_batch` receives a list of dicts
    (Tier-3 per-task knobs unpacked from ``task.data`` by the stage)
    and returns one :class:`DiarizationResult` per input, in the same
    order.

    Expected per-item keys (the stage populates these; the adapter
    reads whichever it needs):

    * ``audio_filepath`` (``str``): Path to a decodable audio file
      (typically the §1.2 resampled 16 kHz mono WAV). Always present.
    * ``waveform`` (``numpy.ndarray | None``): Optional in-memory
      waveform; adapters MAY use it instead of re-decoding from disk
      when present.
    * ``sample_rate`` (``int | None``): Sample rate of ``waveform``;
      ignored when ``waveform`` is ``None``.
    * ``audio_item_id`` (``str | None``): Carried through for diagnostic
      / speaker-id namespacing.
    * ``duration`` (``float | None``): Optional clip duration in
      seconds. The stage uses it for non-speaker gap fill if the
      adapter doesn't echo it back via ``extras``.
    * ``task_id`` (``str | None``): Carried through for diagnostics.

    Attributes:
        model_id: Identifier of the underlying model checkpoint.
        last_metrics: Scalar metrics from the last
            :meth:`diarize_batch` call (per-clip timings, speaker
            counts, ...). The stage merges these into its
            ``_log_metrics`` output under ``model_<key>`` aliases.
    """

    model_id: str
    last_metrics: dict[str, float]

    @classmethod
    def prefetch_weights(cls, model_id: str, revision: str | None = None) -> None:
        """Download weights to local cache without allocating a GPU.

        Called once per node from
        :meth:`DiarizationStage.setup_on_node` before any worker
        starts. Must be a classmethod so the stage can call it without
        instantiating the adapter (which may import heavy GPU
        libraries at construction time).
        """
        ...

    def setup(self) -> None:
        """Load the model into the worker's process.

        Called once per worker from :meth:`DiarizationStage.setup`. May
        allocate GPU memory, build pipelines, instantiate processors.
        """
        ...

    def teardown(self) -> None:
        """Release GPU memory and worker-local state."""
        ...

    def diarize_batch(self, items: list[dict[str, Any]]) -> list[DiarizationResult]:
        """Run diarization on a batch of per-task dicts.

        Args:
            items: One dict per task with the keys documented on the
                class docstring. Length matches the batch size.

        Returns:
            One :class:`DiarizationResult` per input, in the same
            order. Items the adapter could not process (e.g. corrupt
            audio) must still appear in the output list with an empty
            ``diar_segments`` list so the stage can preserve task
            ordering.
        """
        ...
