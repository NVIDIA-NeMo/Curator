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

import subprocess
import sys

import pytest

from nemo_curator.models.audio.lid.base import AudioLIDAdapter, AudioLIDResult
from nemo_curator.models.audio.lid.speechbrain import SpeechBrainLIDAdapter


def test_result_normalizes_language_and_confidence() -> None:
    result = AudioLIDResult("  EN-us ", 1)

    assert result == AudioLIDResult(language="en-us", confidence=1.0)


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("inf"), float("nan")])
def test_result_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        AudioLIDResult("en", confidence)


def test_provider_adapter_conforms_to_shared_protocol() -> None:
    assert isinstance(SpeechBrainLIDAdapter(), AudioLIDAdapter)


def test_package_import_does_not_load_optional_model_runtimes() -> None:
    probe = (
        "import sys;"
        "import nemo_curator.models.audio.lid;"
        "blocked=('speechbrain','whisper','nemo.collections.asr','tensorrt_llm');"
        "print(','.join(name for name in blocked if name in sys.modules))"
    )
    result = subprocess.run(  # noqa: S603 - fixed probe under the test interpreter
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == ""
