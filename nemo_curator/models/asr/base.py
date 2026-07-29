# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stage-adapter contract for audio speech-recognition.

``ASRStage`` owns Curator-side glue (``task.data`` reads, batching, ISO
language mapping, ``_skipme``), while ``ASRAdapter`` owns the model-side call
(weight download, model loading, generation, and packing into ``ASRResult``).
The split lets the stage swap models via a single YAML ``adapter_target:`` line.
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
        text: Transcription text. Empty if skipped.
        skipped: True when the item could not be processed. The stage writes
            ``skip_reason`` to ``_skipme`` and falls back to ``"empty_audio"``
            when no reason is supplied.
        skip_reason: Optional machine-readable reason written to ``_skipme``
            when ``skipped`` is true. Defaults to ``"empty_audio"`` in the stage.
        unsupported_language: Optional normalized language code used by the
            stage to annotate items excluded by its language allowlist.
        extras: Adapter-specific diagnostics outside the canonical shape; the
            stage never reads inside this dict.
    """

    text: str
    skipped: bool = False
    skip_reason: str | None = None
    unsupported_language: str | None = None
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

    * ``waveform``: contiguous, mono, 1-D float32 NumPy samples normalized by
      ``ASRStage`` from a file or a reader-provided in-memory waveform.
    * ``sample_rate`` (``int``): the stage's configured target sample rate.
    * ``language`` (``str | None``): human-readable name (e.g. ``"English"``).
    * ``language_code`` (``str | None``): original language code from the
      configured stage input column.
    * ``task_id`` (``str | None``): carried through for diagnostics.

    Attributes:
        model_id: Identifier of the underlying model checkpoint.
    """

    model_id: str

    @classmethod
    def download_weights_on_node(cls, model_id: str, revision: str | None = None) -> None:
        """Download weights to local cache without allocating a GPU.

        Classmethod so the stage can call it (once per node) without
        instantiating the adapter or importing heavy GPU libraries.
        """
        ...

    def load_model(self, *, num_gpus: int) -> None:
        """Load the model into the worker process using the stage-owned GPU count.

        ``ASRStage`` derives ``num_gpus`` from its Curator resource request so
        adapters do not expose a second, independently configurable GPU count.
        """
        ...

    def unload_model(self) -> None:
        """Release GPU memory and worker-local state."""
        ...

    def transcribe_batch(self, items: list[dict[str, Any]]) -> list[ASRResult]:
        """Run inference on a batch of per-task dicts.

        Returns one ``ASRResult`` per input, in order; skipped items must
        still appear with ``skipped=True`` to preserve task ordering.
        """
        ...
