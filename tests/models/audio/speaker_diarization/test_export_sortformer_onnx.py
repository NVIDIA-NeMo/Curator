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

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import torch

from nemo_curator.models.audio.speaker_diarization.export_sortformer_onnx import (
    RivaStreamingExportMixin,
    _is_high_resolution,
    save_learnable_silence,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_missing_high_resolution_attribute_uses_standard_resolution() -> None:
    assert not _is_high_resolution(SimpleNamespace())
    assert not _is_high_resolution(SimpleNamespace(high_resolution=False))
    assert _is_high_resolution(SimpleNamespace(high_resolution=True))


def test_concat_and_pad_preserves_each_ragged_prefix() -> None:
    speaker_cache = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
    fifo = torch.tensor([[[10.0], [11.0]], [[12.0], [13.0]]])
    chunk = torch.tensor([[[100.0], [101.0], [102.0]], [[200.0], [201.0], [202.0]]])

    output, lengths = RivaStreamingExportMixin.concat_and_pad(
        [speaker_cache, fifo, chunk],
        [torch.tensor([1, 2]), torch.tensor([1, 0]), torch.tensor([2, 1])],
    )

    assert lengths.tolist() == [4, 3]
    torch.testing.assert_close(output[0, :, 0], torch.tensor([1.0, 10.0, 100.0, 101.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(output[1, :, 0], torch.tensor([3.0, 4.0, 200.0, 0.0, 0.0, 0.0, 0.0]))


def test_save_learnable_silence_writes_float32_numpy_artifact(tmp_path: Path) -> None:
    model = SimpleNamespace(sortformer_modules=SimpleNamespace(learnable_sil_emb=torch.arange(4, dtype=torch.float16)))
    output = tmp_path / "learnable_sil_emb.npy"

    assert save_learnable_silence(model, output)

    value = np.load(output, allow_pickle=False)
    assert value.dtype == np.float32
    np.testing.assert_array_equal(value, np.arange(4, dtype=np.float32))


def test_save_learnable_silence_reports_absent_checkpoint_parameter(tmp_path: Path) -> None:
    model = SimpleNamespace(sortformer_modules=SimpleNamespace())
    output = tmp_path / "learnable_sil_emb.npy"

    assert not save_learnable_silence(model, output)
    assert not output.exists()
