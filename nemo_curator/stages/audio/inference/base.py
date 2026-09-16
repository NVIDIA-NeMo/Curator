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

import hashlib
import math
import os
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

import hydra.utils
import numpy as np
import soundfile
from loguru import logger

from nemo_curator.stages.audio._agent._agent_ready import IOSpec
from nemo_curator.stages.audio._agent._residency import (
    InputResidency,
    accepts_for_residency,
    residency_read_specs,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, Task

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata
    from nemo_curator.stages.resources import Resources


_CHANNEL_FIRST_DIMENSIONS = 2
_CANONICAL_AUDIO_FILEPATH_KEY = "audio_filepath"
_RESAMPLED_AUDIO_FILEPATH_KEY = "resampled_audio_filepath"


def _inference_audio_input_spec(
    residency: InputResidency,
    *,
    audio_filepath_key: str,
    waveform_key: str,
    sample_rate_key: str,
) -> tuple[list[str], list[str]]:
    """Return the legacy input tuple for one configured residency.

    ``ProcessingStage.inputs`` cannot encode ``auto`` alternatives. The custom
    validator below handles that mode; an empty tuple prevents the inherited
    batch loop from incorrectly requiring both forms.
    """
    if residency == "waveform":
        return [], [waveform_key, sample_rate_key]
    if residency == "file":
        return [], [audio_filepath_key]
    return [], []


def _inference_audio_read_specs(
    residency: InputResidency,
    *,
    audio_filepath_key: str,
    waveform_key: str,
    sample_rate_key: str,
    fallback_audio_filepath_keys: tuple[str, ...] = (),
) -> list[IOSpec]:
    """Build residency reads, including intentional canonical path fallbacks."""
    specs = residency_read_specs(
        residency,
        audio_filepath_key=audio_filepath_key,
        waveform_key=waveform_key,
        sample_rate_key=sample_rate_key,
    )
    if "file" not in accepts_for_residency(residency):
        return specs
    declared_paths = {audio_filepath_key}
    for key in fallback_audio_filepath_keys:
        if key not in declared_paths:
            specs.append(IOSpec(data_keys=[key], accepts=["file"]))
            declared_paths.add(key)
    return specs


def _validate_inference_audio_input(  # noqa: PLR0913
    task: Task,
    *,
    stage_name: str,
    residency: InputResidency,
    audio_filepath_keys: tuple[str, ...],
    waveform_key: str,
    sample_rate_key: str,
) -> bool:
    """Validate the configured audio source and complete-pair policy.

    File mode ignores resident fields entirely. Waveform mode requires both
    resident keys. Auto mode prefers a complete pair, but deliberately rejects
    an orphaned resident key instead of falling back to a file and allowing
    stale resident state to describe audio that was not consumed.
    """
    data = task.data
    has_file = any(data.get(key) for key in audio_filepath_keys)
    if residency == "file":
        return has_file

    waveform_present = data.get(waveform_key) is not None
    sample_rate_present = data.get(sample_rate_key) is not None
    if waveform_present != sample_rate_present:
        present = waveform_key if waveform_present else sample_rate_key
        missing = sample_rate_key if waveform_present else waveform_key
        msg = (
            f"[{stage_name}] incomplete resident audio for task {task.task_id!r}: "
            f"found {present!r} but missing {missing!r}; "
            f"{waveform_key!r} and {sample_rate_key!r} must be provided together"
        )
        raise ValueError(msg)

    has_waveform = waveform_present and sample_rate_present
    if residency == "waveform":
        return has_waveform
    return has_waveform or has_file


def _channel_first_waveform(waveform: Any) -> np.ndarray:  # noqa: ANN401
    """Return canonical channel-first float32 audio."""
    if hasattr(waveform, "detach"):
        waveform = waveform.detach()
    if hasattr(waveform, "cpu"):
        waveform = waveform.cpu()
    if hasattr(waveform, "numpy"):
        try:
            waveform = waveform.numpy()
        except (RuntimeError, TypeError) as exc:
            dtype = getattr(waveform, "dtype", type(waveform).__name__)
            msg = f"unsupported resident waveform dtype {dtype}"
            raise ValueError(msg) from exc
    array = np.asarray(waveform)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != _CHANNEL_FIRST_DIMENSIONS:
        msg = f"waveform must be 1-D mono or 2-D channel-first audio, got shape {array.shape}"
        raise ValueError(msg)
    if array.dtype in {np.dtype(np.int16), np.dtype(np.int32)}:
        scale = float(1 << (array.dtype.itemsize * 8 - 1))
        return np.ascontiguousarray(array.astype(np.float32) / scale)
    if np.issubdtype(array.dtype, np.integer):
        msg = f"unsupported resident waveform integer dtype {array.dtype}; supported PCM dtypes are int16 and int32"
        raise ValueError(msg)
    if not np.issubdtype(array.dtype, np.floating):
        msg = f"unsupported resident waveform dtype {array.dtype}; expected floating-point or signed PCM int16/int32"
        raise ValueError(msg)
    return np.ascontiguousarray(array, dtype=np.float32)


def _fanout_audio_slice(
    waveform: Any,  # noqa: ANN401
    sample_rate: int,
    *,
    start: float,
    end: float,
) -> np.ndarray:
    """Clone one bounded channel-first segment from a parent waveform."""
    segment, _start, _end = _fanout_audio_segment(waveform, sample_rate, start=start, end=end)
    return segment


def _fanout_audio_segment(
    waveform: Any,  # noqa: ANN401
    sample_rate: int,
    *,
    start: float,
    end: float,
) -> tuple[np.ndarray, float, float]:
    """Return a cloned slice and sample-aligned normalized boundaries."""
    audio = _channel_first_waveform(waveform)
    rate = int(sample_rate)
    if rate <= 0:
        msg = f"sample rate must be > 0, got {rate}"
        raise ValueError(msg)
    if not math.isfinite(start) or not math.isfinite(end):
        msg = f"segment boundaries must be finite, got start={start}, end={end}"
        raise ValueError(msg)
    if end < start:
        msg = f"segment end must be >= start, got start={start}, end={end}"
        raise ValueError(msg)
    sample_count = audio.shape[1]
    start_sample = max(0, min(sample_count, int(max(0.0, start) * rate)))
    end_sample = max(start_sample, min(sample_count, int(max(0.0, end) * rate)))
    segment = np.array(audio[:, start_sample:end_sample], copy=True, order="C")
    return segment, start_sample / rate, end_sample / rate


def _resident_audio_duration(waveform: np.ndarray, sample_rate: int) -> float:
    """Derive duration from the selected resident waveform."""
    rate = int(sample_rate)
    if rate <= 0:
        msg = f"sample rate must be > 0, got {rate}"
        raise ValueError(msg)
    return float(waveform.shape[-1]) / rate


def _stable_source_path(item: dict[str, Any], *path_keys: str) -> str | None:
    """Return source provenance from task data, never a materialized temp path."""
    for key in path_keys:
        value = item.get(key)
        if value:
            return str(value)
    return None


def _fanout_original_file(
    item: dict[str, Any],
    *,
    original_file_key: str,
    source_path: str | None,
    stable_identity: str,
) -> Any:  # noqa: ANN401
    """Preserve existing provenance before falling back to source identity."""
    if original_file_key in item:
        return item[original_file_key]
    return source_path if source_path is not None else stable_identity


def _stable_audio_identity(  # noqa: PLR0913
    item: dict[str, Any],
    waveform: Any,  # noqa: ANN401
    sample_rate: int,
    *,
    source_path: str | None,
    explicit_keys: tuple[str, ...] = ("audio_item_id", "session_name"),
    fallback_keys: tuple[str, ...] = (),
) -> str:
    """Return a preferred explicit, path, fallback, or content identity."""
    for key in explicit_keys:
        value = item.get(key)
        if value is not None and str(value):
            return str(value)
    if source_path:
        source_name = source_path.rstrip("/").rsplit("/", 1)[-1]
        stem, _suffix = os.path.splitext(source_name)
        return stem or source_name
    for key in fallback_keys:
        value = item.get(key)
        if value is not None and str(value):
            return str(value)

    audio = _channel_first_waveform(waveform)
    digest = hashlib.sha256()
    digest.update(memoryview(audio).cast("B"))
    digest.update(f"|{audio.shape!r}|{audio.dtype.str}|{int(sample_rate)}".encode())
    return f"audio_{digest.hexdigest()}"


def _fanout_path_keys(audio_filepath_key: str) -> list[str]:
    """All consumable recording paths forbidden on waveform-only children."""
    return list(
        dict.fromkeys(
            [
                audio_filepath_key,
                _CANONICAL_AUDIO_FILEPATH_KEY,
                _RESAMPLED_AUDIO_FILEPATH_KEY,
            ]
        )
    )


def _validate_fanout_key_contract(
    *,
    stage_name: str,
    audio_filepath_key: str,
    output_keys: list[str],
    removed_container_keys: tuple[str, ...] = (),
) -> None:
    """Reject fan-out key aliases that contradict writes/removals."""
    if not audio_filepath_key:
        msg = f"[{stage_name}] audio filepath key must be non-empty when fanout=True"
        raise ValueError(msg)
    if any(not key for key in removed_container_keys):
        msg = f"[{stage_name}] removed parent container keys must be non-empty"
        raise ValueError(msg)
    if any(not key for key in output_keys):
        msg = f"[{stage_name}] fan-out output keys must be non-empty"
        raise ValueError(msg)
    if len(output_keys) != len(set(output_keys)):
        msg = f"[{stage_name}] fan-out output keys must be distinct"
        raise ValueError(msg)
    collisions = set(output_keys) & set(_fanout_path_keys(audio_filepath_key))
    if collisions:
        msg = f"[{stage_name}] fan-out output keys {sorted(collisions)} collide with removed full-recording path keys"
        raise ValueError(msg)
    container_collisions = set(output_keys) & set(removed_container_keys)
    if container_collisions:
        msg = (
            f"[{stage_name}] fan-out output keys {sorted(container_collisions)} "
            "collide with removed parent container keys"
        )
        raise ValueError(msg)


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
