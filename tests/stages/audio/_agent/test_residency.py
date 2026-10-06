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


from pathlib import Path

import numpy as np
import pytest

from nemo_curator.stages.audio._agent import _residency


@pytest.mark.parametrize("explicit_dir", [False, True])
def test_failed_audio_write_removes_only_owned_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit_dir: bool
) -> None:
    unrelated = tmp_path / "existing.wav"
    unrelated.write_bytes(b"keep")
    monkeypatch.setattr(_residency.tempfile, "tempdir", str(tmp_path))

    def fail_write(path: str, *_args: object) -> None:
        Path(path).write_bytes(b"partial")
        message = "disk write failed"
        raise OSError(message)

    monkeypatch.setattr(_residency.sf, "write", fail_write)
    with pytest.raises(OSError, match="disk write failed"):
        _residency.write_audio_stable(
            np.zeros((1, 16), dtype=np.float32), 16000, output_dir=str(tmp_path) if explicit_dir else None
        )
    assert list(tmp_path.iterdir()) == [unrelated]
    assert unrelated.read_bytes() == b"keep"
