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

from dataclasses import dataclass


def _get_gpu_memory_gb() -> float:
    """Get GPU memory in GB for the current device."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Get first GPU
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return float(info.total) / (1024**3)  # Convert bytes to GB
    except Exception:  # noqa: BLE001
        # Fallback to 24GB if detection fails
        return 24.0


@dataclass
class Resources:
    """Define resource requirements for a processing stage.

    Attributes:
        cpus: Number of CPU cores required
        gpu_memory_gb: GPU memory required in GB (Only for single-GPU stages)
        nvdecs: Number of NVDEC units required
        nvencs: Number of NVENC units required
        entire_gpu: Whether to allocate entire GPU regardless of memory (This also gives you nvdecs and nvencs of that GPU)
        gpus: Number of GPUs required (Only for multi-GPU stages)
    """

    # TODO : Revisit this gpu_memory_gb, gpus, entire_gpu too many variables for gpu
    cpus: float = 1.0
    gpu_memory_gb: float = 0.0
    nvdecs: int = 0
    nvencs: int = 0
    entire_gpu: bool = False
    gpus: float = 0.0

    def __post_init__(self):
        """Calculate GPU count based on memory requirements."""

        if self.gpus > 0 and self.gpu_memory_gb > 0:
            error_message = "Cannot specify both gpus and gpu_memory_gb. "
            error_message += "Please use gpus for multi-GPU stages and "
            error_message += "gpu_memory_gb for single-GPU stages."
            raise ValueError(error_message)

        if self.gpu_memory_gb > 0:
            # Get actual GPU memory for current device
            gpu_memory_per_device = _get_gpu_memory_gb()
            # Calculate required GPUs and round to 1 decimal place
            required_gpus = self.gpu_memory_gb / gpu_memory_per_device
            self.gpus = round(required_gpus, 1)
            if self.gpus > 1:
                error_message = "gpu_memory_gb is too large for a single GPU. "
                error_message += "Please use gpus for multi-GPU stages."
                raise ValueError(error_message)

    @property
    def requires_gpu(self) -> bool:
        """Check if this stage requires GPU resources."""
        return self.gpus > 0 or self.gpu_memory_gb > 0 or self.entire_gpu
