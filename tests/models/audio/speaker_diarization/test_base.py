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

"""Tests for the shared speaker-diarization adapter contract."""

from dataclasses import FrozenInstanceError

import pytest

from nemo_curator.models.audio.speaker_diarization.base import (
    DiarizationAdapter,
    DiarizationResult,
)


class _AdapterStub:
    model_id = "stub/model"
    sample_rate = 16_000

    def download_weights_on_node(self) -> None:
        pass

    def load_model(self, *, num_gpus: int) -> None:
        del num_gpus

    def unload_model(self) -> None:
        pass

    def diarize_batch(self, items: list[dict]) -> list[DiarizationResult]:
        return [DiarizationResult(segments=[]) for _ in items]


def test_structural_adapter_protocol_accepts_complete_adapter() -> None:
    assert isinstance(_AdapterStub(), DiarizationAdapter)


def test_result_is_frozen_and_manifest_serializable() -> None:
    result = DiarizationResult(segments=[{"start": 0.0, "end": 1.25, "speaker": "speaker_0"}])

    assert result.segments == [{"start": 0.0, "end": 1.25, "speaker": "speaker_0"}]
    with pytest.raises(FrozenInstanceError):
        result.segments = []  # type: ignore[misc]
