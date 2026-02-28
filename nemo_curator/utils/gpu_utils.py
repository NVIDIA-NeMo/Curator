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

import math

import torch
from loguru import logger
from transformers import AutoConfig


def get_gpu_count() -> int:
    """
    Get number of available CUDA GPUs as a power of 2.

    Many models require tensor parallelism to use power-of-2 GPU counts.
    This returns the largest power of 2 <= available GPU count.

    Returns:
        Power of 2 GPU count, minimum 1.
    """
    count = torch.cuda.device_count()
    tp_size = 2 ** int(math.log2(count)) if count >= 2 else 1  # noqa: PLR2004
    logger.info(
        f"Detected {count} GPU(s), using tensor_parallel_size={tp_size}"
    )
    return tp_size


def get_max_model_len_from_config(model: str) -> int | None:
    """
    Try to get max model length from HuggingFace AutoConfig.

    Args:
        model: Model identifier (e.g., "microsoft/phi-4")

    Returns:
        Max model length if found, None otherwise.
    """
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    max_len = (
        getattr(config, "max_position_embeddings", None)
        or getattr(config, "n_positions", None)
        or getattr(config, "max_sequence_length", None)
    )
    if max_len:
        logger.info(f"Auto-detected max_model_len={max_len} for {model}")

    return max_len
