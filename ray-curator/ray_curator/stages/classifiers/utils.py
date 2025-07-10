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


def _get_total_memory_per_gpu() -> int:
    # 0 grabs the first GPU available
    # This will raise a RuntimeError if no GPUs are available,
    # which is desired behavior since the script is GPU-dependent
    total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
    # Convert memory from bytes to GB
    return total_gpu_memory / (1024**3)


def _get_suggest_memory_for_classifier() -> int:
    total_gpu_memory = _get_total_memory_per_gpu()
    # Subtract 4GB from the total
    # to leave room for other operations
    # like cuDF operations
    classifier_gpu_memory_gb = total_gpu_memory - 4
    return int(classifier_gpu_memory_gb)


# TODO: This is not customizable for the user anywhere
def _get_suggest_memory_for_tokenizer() -> int:
    total_gpu_memory = _get_total_memory_per_gpu()
    classifier_gpu_memory_gb = _get_suggest_memory_for_classifier()
    # TODO: The result is 4GB, but more finetuning is needed here
    tokenizer_gpu_memory_gb = total_gpu_memory - classifier_gpu_memory_gb
    return int(tokenizer_gpu_memory_gb)
