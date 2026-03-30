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

from collections.abc import Callable

from nemo_curator.stages.audio.tagging.text.itn import InverseTextNormalizationStage
from nemo_curator.tasks import AudioTask


class TestInverseTextNormalizationStage:
    """Tests for InverseTextNormalizationStage."""

    def test_process(self, audio_task: Callable[..., AudioTask]) -> None:
        stage = InverseTextNormalizationStage(language="en", text_key="text")
        stage.setup()
        task = audio_task(
            segments=[
                {"text": "hello", "start": 0.0, "end": 0.5},
                {"text": "the answer is forty two", "start": 0.5, "end": 1.0},
            ],
        )
        result = stage.process(task)
        assert stage._normalizer_initialized
        out = result.data
        assert len(out["segments"]) == 2
        assert out["segments"][0]["text_ITN"] == "hello"
        assert out["segments"][1]["text_ITN"] == "the answer is 42"
