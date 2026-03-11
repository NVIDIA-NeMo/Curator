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

import pynvml
import ray


def align_down_to_256(memory_size: int) -> int:
    """
    Aligns a memory size down to the nearest multiple of 256.
    """
    return (memory_size // 256) * 256


def get_device_free_memory() -> int | None:
    """
    Return total memory of the first GPU the caller has access to.
    Returns None if the GPU is not available or information could not be retrieved.
    """
    try:
        index = int(ray.get_gpu_ids()[0]) if ray.is_initialized() else 0
    except IndexError:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).free
    except pynvml.NVMLError:
        return None
