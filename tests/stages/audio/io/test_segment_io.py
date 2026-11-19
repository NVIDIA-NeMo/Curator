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

from pathlib import Path

import fsspec
import numpy as np
import pytest
import soundfile as sf

from nemo_curator.stages.audio.io.segment_io import write_audio

_RAMP = (np.arange(1600, dtype=np.float32) / 1600) - 0.5


@pytest.mark.parametrize("root", ["local", "memory://segment-io"])
def test_write_audio_round_trips_locally_and_to_cloud(tmp_path: Path, root: str) -> None:
    path = f"{tmp_path if root == 'local' else root}/nested/a.wav"
    write_audio(path, _RAMP, 16000, file_format="wav", subtype="FLOAT")

    with fsspec.open(path, "rb") as f:
        audio, sample_rate = sf.read(f, dtype="float32")
    assert sample_rate == 16000
    np.testing.assert_array_equal(audio, _RAMP)


def test_failed_encode_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="subtype"):
        write_audio(str(tmp_path / "a.wav"), _RAMP, 16000, file_format="wav", subtype="NOPE")
    assert not list(tmp_path.iterdir())
