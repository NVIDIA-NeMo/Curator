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

"""CPU-only unit tests for the vendored Indic Canary TensorRT-LLM runtime."""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch

from nemo_curator.stages.audio.inference import indic_canary_trtllm_runtime as runtime


@pytest.mark.parametrize("array_type", [np.asarray, torch.as_tensor])
def test_pad_or_trim_truncates_and_pads(
    array_type: Callable[[list[float]], np.ndarray | torch.Tensor],
) -> None:
    values = array_type([1.0, 2.0, 3.0])

    truncated = runtime.pad_or_trim(values, length=2)
    padded = runtime.pad_or_trim(values, length=5)

    np.testing.assert_array_equal(np.asarray(truncated), [1.0, 2.0])
    np.testing.assert_array_equal(np.asarray(padded), [1.0, 2.0, 3.0, 0.0, 0.0])


def test_pad_or_trim_honors_nonfinal_axis() -> None:
    values = np.arange(6).reshape(2, 3)

    result = runtime.pad_or_trim(values, length=3, axis=0)

    np.testing.assert_array_equal(result, [[0, 1, 2], [3, 4, 5], [0, 0, 0]])


def test_unpack_tensors_uses_each_row_length() -> None:
    values = torch.tensor([[1, 2, 3], [4, 5, 6]])

    result = runtime.unpack_tensors(values, torch.tensor([1, 2]))

    assert [row.tolist() for row in result] == [[1], [4, 5]]


def test_read_config_merges_decoder_sections_in_order(tmp_path: Path) -> None:
    decoder_dir = tmp_path / "decoder"
    decoder_dir.mkdir()
    (decoder_dir / "config.json").write_text(
        json.dumps(
            {
                "pretrained_config": {"dtype": "float16", "shared": "pretrained"},
                "build_config": {"max_batch_size": 8, "shared": "build"},
            }
        )
    )

    config = runtime.read_config("decoder", tmp_path)

    assert list(config) == ["dtype", "shared", "max_batch_size"]
    assert config == {"dtype": "float16", "shared": "build", "max_batch_size": 8}


def test_read_config_preserves_encoder_config(tmp_path: Path) -> None:
    encoder_dir = tmp_path / "encoder"
    encoder_dir.mkdir()
    (encoder_dir / "config.json").write_text(json.dumps({"max_batch_size": 4, "precision": "fp16"}))

    config = runtime.read_config("encoder", tmp_path, mode="encoder")

    assert config == {"max_batch_size": 4, "precision": "fp16"}


def test_missing_optional_runtime_is_reported_at_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runtime, "_TRTLLM_IMPORT_ERROR", ImportError("not installed"))

    with pytest.raises(ImportError, match="tensorrt_llm is required"):
        runtime.CanaryTRTLLM(tmp_path)
