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

import os
from typing import Literal

os.environ["RAPIDS_NO_INITIALIZE"] = "1"
import torch

INPUT_ID_COLUMN = "input_ids"
ATTENTION_MASK_COLUMN = "attention_mask"
SEQ_ORDER_COLUMN = "_curator_seq_order"
TOKEN_LENGTH_COLUMN = "_curator_token_length"  # noqa: S105


def format_name_with_suffix(model_identifier: str, suffix: str = "_classifier") -> str:
    return model_identifier.split("/")[-1].replace("-", "_").lower() + suffix


def clip_tokens(token_o: dict, padding_side: Literal["left", "right"] = "right") -> dict[str, torch.Tensor]:
    """
    Clip the tokens to the smallest size possible.

    Args:
        token_o: The dictionary containing the input tokens (input_ids, attention_mask).
        padding_side: The side to pad the input tokens. Defaults to "right".

    Returns:
        The clipped tokens (input_ids, attention_mask).

    """
    clip_len = token_o[ATTENTION_MASK_COLUMN].sum(axis=1).max()

    if padding_side == "right":
        token_o[INPUT_ID_COLUMN] = token_o[INPUT_ID_COLUMN][:, :clip_len]
        token_o[ATTENTION_MASK_COLUMN] = token_o[ATTENTION_MASK_COLUMN][:, :clip_len]
    else:
        token_o[INPUT_ID_COLUMN] = token_o[INPUT_ID_COLUMN][:, -clip_len:]
        token_o[ATTENTION_MASK_COLUMN] = token_o[ATTENTION_MASK_COLUMN][:, -clip_len:]

    token_o.pop("metadata", None)

    return token_o
