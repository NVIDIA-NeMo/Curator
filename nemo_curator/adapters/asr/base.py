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

Mirrors the diarization/LID/VAD contract pattern from the SDP-V2 design:

* ``ASRStage`` (in ``nemo_curator/stages/audio/inference/asr.py``) owns
  Curator-side glue - ``task.data`` reads, batching, ISO language-code
  mapping, ``_skip_me`` handling, metric logging.
* ``ASRAdapter`` (this module) owns the model-side library call - weight
  prefetch, model setup, tokenizer/processor wiring, the actual
  ``vllm.LLM.generate(...)`` / ``nemo_asr.transcribe(...)`` invocation,
  and packing results into the canonical ``ASRResult`` shape.

This split lets the stage swap models with a single ``adapter_target:``
line in YAML without rewriting the stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ASRResult:
    """Canonical per-utterance ASR adapter output.

    Identical across every ASR adapter so the stage's schema mutation
    code path stays constant when the adapter is swapped.

    Attributes:
        text: Primary transcription (Turn-1 output for two-turn adapters,
            the only output for single-turn adapters). Empty string for
            skipped items.
        secondary_text: Optional Turn-2 / refined / disfluency-preserved
            output. ``None`` for single-turn adapters or when Turn-2 was
            skipped. The stage writes this onto ``task.data`` only when
            ``ASRStage.secondary_text_key`` is set.
        skipped: True when the adapter could not process this item
            (e.g. empty/corrupt waveform). The stage marks
            ``task.data[skip_me_key] = "empty_audio"`` in that case.
        model_id: The actual model identifier the adapter ran (mirrors
            the stage's ``model_id`` field; populated by the adapter so
            downstream consumers see the live value).
        extras: Adapter-specific scalar / structured diagnostics that do
            not fit the canonical shape (e.g. per-result token counts,
            raw scores). Stage never reads inside this dict.
    """

    text: str
    secondary_text: str | None = None
    skipped: bool = False
    model_id: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ASRAdapter(Protocol):
    """Structural protocol every ASR adapter must implement.

    Constructor contract: adapters are constructed by the stage as
    ``cls(model_id=..., revision=..., **adapter_kwargs)`` - so every
    adapter must accept ``model_id`` and ``revision`` keyword arguments,
    plus whatever Tier-2 knobs that adapter exposes.

    Per-batch contract: ``transcribe_batch`` receives a list of dicts
    (Tier-3 per-task knobs unpacked from ``task.data`` by the stage)
    and returns one ``ASRResult`` per input, in the same order.

    Expected per-item keys (the stage populates these; the adapter
    reads whichever it needs):

    * ``waveform`` (``numpy.ndarray``): 1-D mono float32 array.
    * ``sample_rate`` (``int``): Source sample rate; the adapter is
      responsible for any resampling.
    * ``language`` (``str | None``): Human-readable language name
      (e.g. ``"English"``), already mapped from ISO code by the stage.
    * ``task_id`` (``str | None``): Carried through for diagnostics.

    Attributes:
        model_id: Identifier of the underlying model checkpoint.
        last_metrics: Scalar metrics from the last ``transcribe_batch``
            call (token counts, prep/generation timings, ...). The stage
            merges these into its ``_log_metrics`` output under
            ``model_<key>`` aliases.
    """

    model_id: str
    last_metrics: dict[str, float]

    @classmethod
    def prefetch_weights(cls, model_id: str, revision: str | None = None) -> None:
        """Download weights to local cache without allocating a GPU.

        Called once per node from ``ASRStage.setup_on_node`` before any
        worker starts. Must be a classmethod so the stage can call it
        without instantiating the adapter (which may import heavy GPU
        libraries at construction time).
        """
        ...

    def setup(self) -> None:
        """Load the model into the worker's process.

        Called once per worker from ``ASRStage.setup``. May allocate GPU
        memory, build vLLM engines, instantiate processors, etc.
        """
        ...

    def teardown(self) -> None:
        """Release GPU memory and worker-local state."""
        ...

    def transcribe_batch(self, items: list[dict[str, Any]]) -> list[ASRResult]:
        """Run inference on a batch of per-task dicts.

        Args:
            items: One dict per task with the keys documented on the
                class docstring. Length matches the batch size.

        Returns:
            One ``ASRResult`` per input, in the same order. Skipped
            items must still appear in the output list with
            ``skipped=True`` so the stage can preserve task ordering.
        """
        ...
