# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import torch

from nemo_curator.stages.text.models.utils import clip_tokens, format_name_with_suffix


def test_format_name_with_suffix() -> None:
    assert format_name_with_suffix("nvidia/quality-classifier-deberta") == "quality_classifier_deberta_classifier"
    assert format_name_with_suffix("nViDiA/qUaLiTy-ClAsSiFiEr-DeBeRtA", "_test") == "quality_classifier_deberta_test"


# Test modified from CrossFit: https://github.com/rapidsai/crossfit/blob/main/tests/op/test_tokenize.py
def test_clip_tokens_right_padding():
    input_ids = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 3, 4, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    token_o = {"input_ids": input_ids, "attention_mask": attention_mask}

    result = clip_tokens(token_o, padding_side="right")

    assert isinstance(result["input_ids"], torch.Tensor)
    assert isinstance(result["attention_mask"], torch.Tensor)
    assert result["input_ids"].shape == (2, 4)
    assert result["attention_mask"].shape == (2, 4)
    assert torch.equal(result["input_ids"].to("cpu"), torch.tensor([[1, 2, 3, 0], [1, 2, 3, 4]]))
    assert torch.equal(
        result["attention_mask"].to("cpu"), torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    )


# Test modified from CrossFit: https://github.com/rapidsai/crossfit/blob/main/tests/op/test_tokenize.py
def test_clip_tokens_left_padding():
    input_ids = torch.tensor([[0, 0, 1, 2, 3], [0, 1, 2, 3, 4]])
    attention_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
    token_o = {"input_ids": input_ids, "attention_mask": attention_mask}

    result = clip_tokens(token_o, padding_side="left")

    assert isinstance(result["input_ids"], torch.Tensor)
    assert isinstance(result["attention_mask"], torch.Tensor)
    assert result["input_ids"].shape == (2, 4)
    assert result["attention_mask"].shape == (2, 4)
    assert torch.equal(result["input_ids"].to("cpu"), torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]))
    assert torch.equal(
        result["attention_mask"].to("cpu"), torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
    )


# Test modified from CrossFit: https://github.com/rapidsai/crossfit/blob/main/tests/op/test_tokenize.py
def test_clip_tokens_no_clipping_needed():
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 1]])
    token_o = {"input_ids": input_ids, "attention_mask": attention_mask}

    result = clip_tokens(token_o, padding_side="right")

    assert isinstance(result["input_ids"], torch.Tensor)
    assert isinstance(result["attention_mask"], torch.Tensor)
    assert result["input_ids"].shape == (2, 3)
    assert result["attention_mask"].shape == (2, 3)
    assert torch.equal(result["input_ids"].to("cpu"), torch.tensor([[1, 2, 3], [4, 5, 6]]))
    assert torch.equal(result["attention_mask"].to("cpu"), torch.tensor([[1, 1, 1], [1, 1, 1]]))
