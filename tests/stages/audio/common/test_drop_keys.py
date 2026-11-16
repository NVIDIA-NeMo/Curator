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

import json

import torch

from nemo_curator.stages.audio.common import DropKeysStage
from nemo_curator.tasks import AudioTask


def test_drops_listed_keys_and_ignores_missing_ones() -> None:
    task = AudioTask(data={"waveform": torch.ones(1, 8), "sample_rate": 16000, "text": "hi"})
    out = DropKeysStage(keys=["waveform", "not_there"]).process(task)
    assert out is task
    assert out.data == {"sample_rate": 16000, "text": "hi"}
    json.dumps(out.data)
