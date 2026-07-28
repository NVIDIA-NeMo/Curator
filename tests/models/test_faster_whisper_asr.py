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

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from nemo_curator.models.faster_whisper_asr import FasterWhisperASR


def test_adapter_transcribes_waveform_and_forces_language() -> None:
    model = FasterWhisperASR(device="cpu", compute_type="int8")
    model._model = MagicMock()
    model._model.transcribe.return_value = (
        iter([SimpleNamespace(text="hello"), SimpleNamespace(text="world")]),
        object(),
    )

    texts, languages = model.generate(
        [np.zeros(160, dtype=np.float32)],
        [16000],
        ["en"],
    )

    assert texts == ["hello world"]
    assert languages == ["en"]
    assert model._model.transcribe.call_args.kwargs["language"] == "en"


def test_adapter_rejects_mismatched_cardinality() -> None:
    model = FasterWhisperASR()
    model._model = MagicMock()

    with pytest.raises(ValueError, match="equal lengths"):
        model.generate([np.zeros(1)], [], ["en"])
