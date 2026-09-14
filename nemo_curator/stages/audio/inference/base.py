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

"""Shared infrastructure for audio inference stages backed by model adapters."""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

import hydra.utils
import numpy as np
import soundfile
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata
    from nemo_curator.stages.resources import Resources


_CHANNEL_FIRST_DIMENSIONS = 2


class InferenceAdapter(Protocol):
    """Lifecycle shared by model adapters hosted in an inference stage."""

    def download_weights_on_node(self) -> None:
        """Cache model weights without allocating worker-local model state."""
        ...

    def load_model(self, *, num_gpus: int) -> None:
        """Load worker-local model state."""
        ...

    def unload_model(self) -> None:
        """Release worker-local model state."""
        ...


AdapterT = TypeVar("AdapterT", bound=InferenceAdapter)


class AdapterInferenceStage(ProcessingStage[AudioTask, AudioTask], Generic[AdapterT]):
    """Own adapter lifecycle and file-input behavior shared by audio stages.

    Subclasses retain responsibility for constructing the adapter and for
    model-specific waveform normalization, inference, and result assembly.
    """

    adapter_target: str
    waveform_key: str | None
    sample_rate_key: str
    audio_filepath_key: str
    resources: Resources
    prefetch_fail_on_error: bool
    _adapter: AdapterT | None

    def __post_init__(self) -> None:
        self._adapter = None

    def _adapter_class(self) -> type:
        """Resolve the configured adapter without importing it eagerly."""
        return hydra.utils.get_class(self.adapter_target)

    def _adapter_gpu_count(self) -> int:
        """Return the physical GPU count represented by the resource request."""
        requested_gpus = float(self.resources.gpus)
        if requested_gpus < 0 or not math.isfinite(requested_gpus):
            msg = f"{type(self).__name__}.resources.gpus must be a finite non-negative value, got {requested_gpus}"
            raise ValueError(msg)
        return math.ceil(requested_gpus)

    @abstractmethod
    def _create_adapter(self) -> AdapterT:
        """Construct one unloaded adapter from the subclass configuration."""
        ...

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        """Cache adapter-owned model weights once per node."""
        try:
            self._create_adapter().download_weights_on_node()
            logger.info("{} weights cached on node ({})", type(self).__name__, self.adapter_target)
        except Exception as exc:
            msg = f"{type(self).__name__}: download_weights_on_node failed for {self.adapter_target}"
            if self.prefetch_fail_on_error:
                raise RuntimeError(msg) from exc
            logger.warning("{}; setup() will retry: {}", msg, exc)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        """Construct and load the worker-local adapter once."""
        if self._adapter is not None:
            return
        adapter = self._create_adapter()
        try:
            adapter.load_model(num_gpus=self._adapter_gpu_count())
        except Exception:
            try:
                adapter.unload_model()
            except Exception as teardown_exc:  # noqa: BLE001
                logger.warning("Adapter cleanup after setup failure also failed: {}", teardown_exc)
            raise
        self._adapter = adapter
        logger.info("{} adapter ready on worker ({})", type(self).__name__, self.adapter_target)

    def teardown(self) -> None:
        """Unload any initialized worker-local adapter."""
        if self._adapter is not None:
            self._adapter.unload_model()
            self._adapter = None

    def inputs(self) -> tuple[list[str], list[str]]:
        """Declare either the configured in-memory waveform or file input."""
        if self.waveform_key:
            return [], [self.waveform_key, self.sample_rate_key]
        return [], [self.audio_filepath_key]

    @staticmethod
    def _load_audio(audio_filepath: str) -> tuple[np.ndarray, int]:
        """Load one file as contiguous mono or channel-first float32 audio."""
        waveform, sample_rate = soundfile.read(audio_filepath, dtype="float32")
        if waveform.ndim == _CHANNEL_FIRST_DIMENSIONS:
            waveform = waveform.T
        return np.ascontiguousarray(waveform, dtype=np.float32), int(sample_rate)
