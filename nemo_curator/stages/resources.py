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

# Smallest fractional GPU a memory-based request may resolve to. Matches the
# one-decimal rounding used below so a positive gpu_memory_gb never collapses to
# a zero-GPU (CPU-only) reservation on very large devices.
MIN_GPU_FRACTION = 0.1


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
        gpus: Number of GPUs required (Only for multi-GPU stages)
    """

    cpus: float = 1.0
    gpu_memory_gb: float = 0.0
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
            # Calculate required GPUs and round to 1 decimal place. Floor the
            # result at 0.1: on large devices (e.g. GB300, ~250 GB) a small
            # request such as gpu_memory_gb=10 rounds to 0.0, which makes the
            # scheduler treat the stage as CPU-only and clear
            # CUDA_VISIBLE_DEVICES, so the stage later fails with
            # "RuntimeError: No CUDA GPUs are available" even though
            # requires_gpu is True. A memory request must always reserve a
            # non-zero GPU fraction.
            required_gpus = self.gpu_memory_gb / gpu_memory_per_device
            self.gpus = max(MIN_GPU_FRACTION, round(required_gpus, 1))
            if self.gpus > 1:
                error_message = "gpu_memory_gb is too large for a single GPU. "
                error_message += "Please use gpus for multi-GPU stages."
                raise ValueError(error_message)

    @property
    def requires_gpu(self) -> bool:
        """Check if this stage requires GPU resources."""
        return self.gpus > 0 or self.gpu_memory_gb > 0
