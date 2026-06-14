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

"""Stage-adapter contract for audio speech-recognition.

Mirrors the diarization/LID/VAD contract: ``ASRStage`` owns Curator-side glue
(``task.data`` reads, batching, ISO language mapping, ``_skip_me``, metrics),
while ``ASRAdapter`` (this module) owns the model-side call (prefetch, setup,
generation, packing into ``ASRResult``). The split lets the stage swap models
via a single YAML ``adapter_target:`` line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ASRResult:
    """Canonical per-utterance ASR adapter output.

    Identical across every adapter so the stage's schema-mutation path stays
    constant when the adapter is swapped.

    Attributes:
        text: Primary transcription (Turn-1 / sole output). Empty if skipped.
        secondary_text: Optional Turn-2 / disfluency-preserved output;
            ``None`` for single-turn or skipped Turn-2. Written to
            ``task.data`` only when ``ASRStage.disfluency_text_key`` is set.
        skipped: True when the item could not be processed (e.g. empty/corrupt
            waveform); the stage then sets ``skip_me_key = "empty_audio"``.
        model_id: Identifier of the model actually run (populated by adapter).
        extras: Adapter-specific diagnostics outside the canonical shape; the
            stage never reads inside this dict.
    """

    text: str
    secondary_text: str | None = None
    skipped: bool = False
    model_id: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ASRAdapter(Protocol):
    """Structural protocol every ASR adapter must implement.

    Constructor contract: the stage builds adapters as
    ``cls(model_id=..., revision=..., **adapter_kwargs)``, so every adapter
    must accept ``model_id`` and ``revision`` keyword args plus its own knobs.

    Per-batch contract: ``transcribe_batch`` receives a list of per-task dicts
    (unpacked from ``task.data``) and returns one ``ASRResult`` per input, in
    order. Expected per-item keys (stage-populated):

    * ``waveform``: canonical Curator waveform object from the stage
      (typically a torch tensor shaped ``(channels, samples)``); adapters own
      any model-specific conversion such as squeezing to 1-D numpy.
    * ``sample_rate`` (``int``): source rate; adapter handles any resampling.
    * ``language`` (``str | None``): human-readable name (e.g. ``"English"``).
    * ``task_id`` (``str | None``): carried through for diagnostics.

    Attributes:
        model_id: Identifier of the underlying model checkpoint.
        last_metrics: Scalar metrics from the last ``transcribe_batch`` call;
            the stage merges these under ``model_<key>`` aliases.
    """

    model_id: str
    last_metrics: dict[str, float]

    @classmethod
    def prefetch_weights(cls, model_id: str, revision: str | None = None) -> None:
        """Download weights to local cache without allocating a GPU.

        Classmethod so the stage can call it (once per node) without
        instantiating the adapter or importing heavy GPU libraries.
        """
        ...

    def setup(self) -> None:
        """Load the model into the worker process (once per worker)."""
        ...

    def teardown(self) -> None:
        """Release GPU memory and worker-local state."""
        ...

    def transcribe_batch(self, items: list[dict[str, Any]]) -> list[ASRResult]:
        """Run inference on a batch of per-task dicts.

        Returns one ``ASRResult`` per input, in order; skipped items must
        still appear with ``skipped=True`` to preserve task ordering.
        """
        ...
