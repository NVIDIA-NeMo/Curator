# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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


import pytest

from nemo_curator.stages.audio.agent import pipeline_identity
from nemo_curator.stages.audio.common import GetAudioDurationStage


def test_pipeline_identity_tracks_configured_semantics() -> None:
    assert pipeline_identity([GetAudioDurationStage()]) == pipeline_identity([GetAudioDurationStage()])
    assert pipeline_identity([GetAudioDurationStage()]) != pipeline_identity(
        [GetAudioDurationStage(duration_key="seconds")]
    )
    assert pipeline_identity(
        [GetAudioDurationStage(), GetAudioDurationStage(duration_key="seconds")]
    ) != pipeline_identity([GetAudioDurationStage(duration_key="seconds"), GetAudioDurationStage()])


def test_pipeline_identity_rejects_unserializable_configuration() -> None:
    with pytest.raises(TypeError, match="Cannot fingerprint"):
        pipeline_identity([GetAudioDurationStage(duration_key=object())])


@pytest.mark.parametrize("comparison", ["lt", "le", "eq", "ne", "ge", "gt"])
def test_pipeline_identity_supports_value_filters(comparison: str) -> None:
    from nemo_curator.stages.audio.common import ManifestWriterStage, PreserveByValueStage

    def identity(operator: str, target: float = 1) -> str:
        return pipeline_identity(
            [
                GetAudioDurationStage(),
                PreserveByValueStage("duration", target, operator=operator),
                ManifestWriterStage("output.jsonl"),
            ]
        )

    assert identity(comparison) == identity(comparison)
    assert identity(comparison) != identity(comparison, 2)
    assert identity(comparison) != identity("ne" if comparison == "eq" else "eq")


def test_pipeline_identity_rejects_arbitrary_callable() -> None:
    with pytest.raises(TypeError, match="Cannot fingerprint"):
        pipeline_identity([GetAudioDurationStage(duration_key=lambda: "duration")])
