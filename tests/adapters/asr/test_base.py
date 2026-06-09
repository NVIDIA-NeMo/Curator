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

"""Tests for the generic stage-adapter split contract.

These pin the seam between ``ASRStage`` (Curator-side glue) and any
``ASRAdapter`` (model-side library call), using a minimal **non-Qwen** fake
adapter so the split is proven model-agnostic:

    * the canonical ``ASRResult`` shape/defaults the stage relies on;
    * ``@runtime_checkable`` protocol conformance for an arbitrary adapter;
    * end-to-end swappability -- ``ASRStage`` constructs, sets up, and
      delegates to any conforming adapter resolved from ``adapter_target``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np

from nemo_curator.adapters.asr.base import ASRAdapter, ASRResult
from nemo_curator.stages.audio.inference.asr import ASRStage
from nemo_curator.tasks import AudioTask

_SR = 16000


class _FakeASRAdapter:
    """Minimal non-Qwen adapter implementing the ``ASRAdapter`` protocol.

    Echoes each item's stage-mapped ``language`` back as the transcription so
    a test can prove the stage both constructed this adapter and forwarded the
    per-item dicts through to it.
    """

    def __init__(self, model_id: str, revision: str | None = None, **adapter_kwargs: object) -> None:
        self.model_id = model_id
        self.revision = revision
        self.adapter_kwargs = adapter_kwargs
        self.last_metrics: dict[str, float] = {}
        self.setup_called = False
        self.seen_items: list[dict[str, Any]] = []

    @classmethod
    def prefetch_weights(cls, _model_id: str, _revision: str | None = None) -> None:
        return None

    def setup(self) -> None:
        self.setup_called = True

    def teardown(self) -> None:
        return None

    def transcribe_batch(self, items: list[dict[str, Any]]) -> list[ASRResult]:
        self.seen_items = items
        return [ASRResult(text=f"fake:{it.get('language')}", model_id=self.model_id) for it in items]


# ----------------------------------------------------------------------
# ASRResult: canonical adapter-output shape
# ----------------------------------------------------------------------


def test_asr_result_defaults() -> None:
    """The shape every adapter returns and the stage reads must stay stable."""
    r = ASRResult(text="hello")
    assert r.text == "hello"
    assert r.secondary_text is None
    assert r.skipped is False
    assert r.model_id == ""
    assert r.extras == {}


# ----------------------------------------------------------------------
# Protocol conformance for an arbitrary (non-Qwen) adapter
# ----------------------------------------------------------------------


def test_fake_adapter_conforms_to_asr_protocol() -> None:
    """isinstance() works only because ASRAdapter is @runtime_checkable;
    a minimal hand-written adapter must satisfy the structural contract."""
    adapter = _FakeASRAdapter(model_id="fake/model")
    assert isinstance(adapter, ASRAdapter)


# ----------------------------------------------------------------------
# Swappability: ASRStage drives ANY conforming adapter end-to-end
# ----------------------------------------------------------------------


def test_asr_stage_drives_arbitrary_conforming_adapter() -> None:
    """The split's core promise: ASRStage resolves ``adapter_target``,
    constructs the adapter with model_id+revision, calls its ``setup()``,
    and delegates ``process_batch`` to it -- with no Qwen-specific coupling.
    """
    stage = ASRStage(
        adapter_target="tests.adapters.asr.test_base._FakeASRAdapter",
        model_id="fake/model",
        pred_text_key="pred_text",
    )

    # adapter_target resolution is patched so the dotted string need not be
    # importable; the stage still does the construct -> setup -> store wiring.
    with patch("hydra.utils.get_class", return_value=_FakeASRAdapter):
        stage.setup()

    assert isinstance(stage._adapter, _FakeASRAdapter)
    assert stage._adapter.setup_called is True

    task = AudioTask(data={"waveform": np.zeros(_SR, dtype=np.float32), "sample_rate": _SR, "source_lang": "es"})
    results = stage.process_batch([task])

    # Stage mapped ISO "es" -> "Spanish", forwarded it to the fake adapter,
    # and packaged the adapter's ASRResult.text under the configured key.
    assert results[0].data["pred_text"] == "fake:Spanish"
